# src/utils/validation_utils.py
import os
import numpy as np
from nuscenes.nuscenes import NuScenes # For type hinting
from tqdm import tqdm
import json
from scipy.spatial import KDTree # For efficient point matching
from typing import Dict, List, Tuple, Any, Optional, Union

from ..core.constants import POINT_LABEL_DTYPE 

def get_gt_dynamic_points_for_sweep(
    nusc: NuScenes,
    sweep_data_dict: Dict, # Expects dict from get_scene_sweep_data_sequence
    all_points_global: np.ndarray, # Global XYZ of all points in current sweep (N,3)
    label_base_dir: str,
    velocity_threshold: float
) -> Dict[str, np.ndarray]: # Return a dict for consistency with mdet_points
    """
    Loads point labels for a sweep, filters dynamic points based on velocity,
    and returns their global coordinates along with all static points.

    Args:
        nusc: NuScenes API instance.
        sweep_data_dict (Dict): Dictionary for the current sweep from get_scene_sweep_data_sequence.
                                Must contain 'lidar_sd_token' and 'scene_token'.
        all_points_global (np.ndarray): Global XYZ coords of all points in the current sweep (N, 3).
        label_base_dir (str): Base directory where label files are stored.
        velocity_threshold (float): Minimum velocity magnitude to be considered dynamic.

    Returns:
        Dict[str, np.ndarray]: {'dynamic': (M,3), 'static': (K,3)}
    """
    lidar_sd_token = sweep_data_dict['lidar_sd_token']
    # Scene name is needed for the path. Get it from scene_token.
    scene_rec = nusc.get('scene', sweep_data_dict['scene_token'])
    scene_name = scene_rec['name']
    
    label_filename = f"{lidar_sd_token}_pointlabels.npy"
    label_filepath = os.path.join(label_base_dir, scene_name, label_filename)

    empty_result = {'dynamic': np.empty((0, 3)), 'static': np.empty((0,3))}

    if not os.path.exists(label_filepath):
        # tqdm.write(f"Info: GT Label file not found: {label_filepath}")
        # If no label file, consider all points static for GT visualization purposes
        empty_result['static'] = all_points_global 
        return empty_result

    try:
        point_labels = np.load(label_filepath)
        if point_labels.dtype != POINT_LABEL_DTYPE:
            try:
                point_labels = point_labels.astype(POINT_LABEL_DTYPE)
            except ValueError as ve:
                tqdm.write(f"Critical DTYPE mismatch for {label_filepath}. Error: {ve}. Skipping GT labels.")
                empty_result['static'] = all_points_global
                return empty_result
    except Exception as e:
        tqdm.write(f"Error loading GT label file {label_filepath}: {e}")
        empty_result['static'] = all_points_global
        return empty_result
    
    if all_points_global.shape[0] == 0: return empty_result
    if all_points_global.shape[0] != point_labels.shape[0]:
        tqdm.write(f"Warning: Mismatch in point count for {lidar_sd_token}. Points: {all_points_global.shape[0]}, Labels: {point_labels.shape[0]}.")
        # Fallback: consider all points static if counts don't match, or try to reconcile
        # For simplicity, let's return all as static to avoid crashing.
        # A more robust solution would be to align them if possible.
        min_count = min(all_points_global.shape[0], point_labels.shape[0])
        if min_count == 0: return empty_result
        all_points_global = all_points_global[:min_count]
        point_labels = point_labels[:min_count]

    velocities = np.stack([
        point_labels['velocity_x'], point_labels['velocity_y'], point_labels['velocity_z']
    ], axis=-1)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    is_valid_instance = np.array([token_bytes != b'' for token_bytes in point_labels['instance_token']])
    dynamic_mask = (velocity_magnitudes >= velocity_threshold) & is_valid_instance
    
    gt_dynamic_points = all_points_global[dynamic_mask]
    gt_static_points = all_points_global[~dynamic_mask] # Points not meeting dynamic criteria
    
    return {'dynamic': gt_dynamic_points, 'static': gt_static_points}

# --- Configuration Loading from NPZ ---
def load_config_from_npz(npz_path: str) -> Optional[Dict]:
    """Loads a configuration dictionary saved as a JSON string from an NPZ file."""
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if '_config_json_str' in data:
                config_str = data['_config_json_str'].item() # .item() to extract from 0-d array
                return json.loads(config_str)
            # Fallback for older format if config was saved as a pickled object array
            elif 'config_object' in data and data['config_object'].size == 1:
                 loaded_item = data['config_object'].item()
                 if isinstance(loaded_item, dict):
                    return loaded_item
                 else:
                    print(f"Warning: 'config_object' in {npz_path} is not a dict.")
                    return None
            print(f"Warning: Configuration key ('_config_json_str' or 'config_object') not found in NPZ: {npz_path}")
            return None
    except FileNotFoundError:
        # print(f"Error: NPZ file not found when trying to load config: {npz_path}") # Can be noisy if file is optional
        return None
    except Exception as e:
        print(f"Error loading config from NPZ {npz_path}: {e}")
        return None

# --- Ground Truth Processing ---
def get_gt_dynamic_status(
    gt_velocities_xyz: np.ndarray, 
    gt_velocity_threshold: float
) -> np.ndarray:
    """Determines dynamic status of GT points based on velocity magnitude."""
    if gt_velocities_xyz.ndim != 2 or gt_velocities_xyz.shape[0] == 0 or gt_velocities_xyz.shape[1] < 2:
        return np.zeros(gt_velocities_xyz.shape[0], dtype=bool)
    gt_speeds_sq = np.sum(gt_velocities_xyz[:, :2]**2, axis=1) 
    return gt_speeds_sq > gt_velocity_threshold**2

# --- Core Sweep Metrics Calculation ---
def calculate_point_metrics_for_sweep(
    gt_points_xyz: np.ndarray,          # Shape (N, 3)
    gt_is_dynamic: np.ndarray,          # Shape (N,) boolean, True if GT point is dynamic
    pred_points_xyz: np.ndarray,        # Shape (N, 3) or (M,3)
    pred_is_dynamic: np.ndarray,        # Shape (N,) or (M,) boolean, True if predicted dynamic
    # matching_distance_threshold: float = 0.1, # Less critical if index matching works
    # use_kdtree_matching: bool = False # Parameter to force KDTree if needed
    coordinate_tolerance: float = 1e-3 # Meters, for verifying point correspondence
) -> Dict[str, Union[int, float]]:
    """
    Calculates TP, FP, FN, TN for a single sweep assuming point-wise correspondence.
    Verifies that GT and Prediction point clouds match if lengths are the same.
    """
    n_gt_points = gt_points_xyz.shape[0]
    n_pred_points = pred_points_xyz.shape[0]

    # --- Sanity Check for Index-Based Matching ---
    if n_gt_points != n_pred_points:
        print(f"    Warning: Sweep has different number of GT points ({n_gt_points}) and Pred points ({n_pred_points}). Cannot use index matching. Metrics will be unreliable for this sweep or KDTree fallback needed.")
        # For now, return empty/error metrics if counts don't match, as index matching is the goal.
        # Or, implement KDTree as a fallback here if use_kdtree_matching is True.
        return {"TP": 0, "FP": 0, "FN": np.sum(gt_is_dynamic), "TN": 0, 
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Accuracy_approx": 0.0,
                "Error_Msg": "Point count mismatch"}

    if n_gt_points == 0: # Both are 0 due to above check
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0, "Accuracy_approx": 1.0}

    # Verify coordinate correspondence (optional but recommended)
    if not np.allclose(gt_points_xyz, pred_points_xyz, atol=coordinate_tolerance):
        max_diff = np.max(np.abs(gt_points_xyz - pred_points_xyz)) if gt_points_xyz.size > 0 else 0
        print(f"    Warning: Point coordinates for GT and Pred differ more than tolerance ({coordinate_tolerance}m). Max diff: {max_diff:.4f}m. Proceeding with index matching, but investigate data generation.")
        # Depending on severity, you might choose to error out or proceed.

    # --- Index-based Metrics Calculation ---
    tp = np.sum(gt_is_dynamic & pred_is_dynamic)
    fp = np.sum(~gt_is_dynamic & pred_is_dynamic)
    fn = np.sum(gt_is_dynamic & ~pred_is_dynamic)
    tn = np.sum(~gt_is_dynamic & ~pred_is_dynamic) # True negatives (correctly static)
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0) # Handle TP=0, FP=0 case
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)    # Handle TP=0, FN=0 case
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy_approx = (tp + tn) / n_gt_points if n_gt_points > 0 else 1.0 # Approx, as it's per-point
    
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Precision": precision, "Recall": recall, "F1": f1, "Accuracy_approx": accuracy_approx,
        "N_GT_Points": n_gt_points, "N_Pred_Points": n_pred_points, # Should be same
        "N_GT_Dynamic": np.sum(gt_is_dynamic),
        "N_Pred_Dynamic": np.sum(pred_is_dynamic)
    }

# --- Scene and Experiment Level Metrics ---
def calculate_metrics_for_scene(
    gt_scene_npz_path: str, 
    mdet_scene_npz_path: str, 
    eval_params: Dict # Contains GT velocity threshold, MDet label mapping, etc.
) -> Optional[Dict[str, Any]]:
    """Calculates aggregated metrics for an entire scene by processing each sweep using index matching."""
    try:
        gt_data = np.load(gt_scene_npz_path, allow_pickle=True)
        mdet_data = np.load(mdet_scene_npz_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: NPZ file not found. GT: {gt_scene_npz_path} or MDet: {mdet_scene_npz_path}")
        return None

    mdet_run_config = load_config_from_npz(mdet_scene_npz_path)
    
    gt_vel_thresh = eval_params.get('gt_velocity_threshold', 0.5)
    coord_tol = eval_params.get('coordinate_tolerance_for_verification', 1e-3)
    
    # --- Define keys for accessing data from NPZ files ---
    # GT NPZ structure (assuming from generate_and_save_point_labels_for_scene_npz)
    gt_sweep_tokens_key = 'sweep_lidar_sd_tokens'
    gt_labels_key = 'all_gt_point_labels' # Fields: 'x','y','z','velocity_x','velocity_y',...
    gt_indices_key = 'gt_point_labels_indices'
    
    # M-Detector NPZ structure (ASSUMING REFACOR FOR INDEX MATCHING)
    mdet_sweep_tokens_key = 'sweep_lidar_sd_tokens' # Must match GT's sweep token key
    mdet_points_key = 'all_points_predictions' # Structured array: 'x','y','z', 'mdet_label', 'mdet_score'
    mdet_indices_key = 'points_predictions_indices'
    
    # Mapping M-Detector's 'mdet_label' field to a binary 'is_dynamic' prediction
    # This depends on how OcclusionResult enum (or other labels) are stored and what means "dynamic"
    # Example: if OcclusionResult.OCCLUDING_IMAGE (value 0) means dynamic
    mdet_dynamic_label_value = eval_params.get('mdet_dynamic_label_value', 0) # Value in 'mdet_label' that means dynamic
    mdet_label_field_name = eval_params.get('mdet_label_field_name', 'mdet_label') # Actual field name in NPZ

    # For ROC/PR curves, using scores:
    mdet_score_field_name = eval_params.get('mdet_score_field_name', 'mdet_score')
    # The `pred_is_dynamic` below will be overridden if `current_score_threshold` is passed in eval_params (for ROC)

    # Validate NPZ contents
    required_gt_fields = [gt_sweep_tokens_key, gt_labels_key, gt_indices_key]
    if not all(k in gt_data for k in required_gt_fields):
        print(f"GT NPZ {gt_scene_npz_path} missing required fields. Expected: {required_gt_fields}")
        return None
    required_mdet_fields = [mdet_sweep_tokens_key, mdet_points_key, mdet_indices_key]
    if not all(k in mdet_data for k in required_mdet_fields):
        print(f"MDet NPZ {mdet_scene_npz_path} missing required fields (for index matching). Expected: {required_mdet_fields}")
        return None
    if mdet_label_field_name not in mdet_data[mdet_points_key].dtype.names and \
       eval_params.get("current_score_threshold") is None : # Check label field only if not using score threshold
         print(f"MDet NPZ {mdet_scene_npz_path} '{mdet_points_key}' missing MDet label field: '{mdet_label_field_name}'")
         return None

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_gt_points_scene, total_pred_points_scene = 0,0
    total_gt_dynamic_scene, total_pred_dynamic_scene = 0,0
    processed_sweeps_count = 0
    
    gt_sweep_tokens = gt_data[gt_sweep_tokens_key].astype(str)
    mdet_sweep_tokens_map = {token: i for i, token in enumerate(mdet_data[mdet_sweep_tokens_key].astype(str))}

    for i_gt_sweep, sweep_token in enumerate(gt_sweep_tokens):
        if sweep_token not in mdet_sweep_tokens_map:
            # print(f"Sweep {sweep_token} in GT but not in MDet results. Skipping.")
            continue 
        
        i_mdet_sweep = mdet_sweep_tokens_map[sweep_token]
        
        gt_start, gt_end = gt_data[gt_indices_key][i_gt_sweep], gt_data[gt_indices_key][i_gt_sweep+1]
        gt_sweep_labels_structured = gt_data[gt_labels_key][gt_start:gt_end]

        mdet_start, mdet_end = mdet_data[mdet_indices_key][i_mdet_sweep], mdet_data[mdet_indices_key][i_mdet_sweep+1]
        mdet_sweep_preds_structured = mdet_data[mdet_points_key][mdet_start:mdet_end]

        if gt_sweep_labels_structured.size == 0 and mdet_sweep_preds_structured.size == 0:
            processed_sweeps_count += 1
            continue
        
        gt_pts_xyz = np.stack((gt_sweep_labels_structured['x'], gt_sweep_labels_structured['y'], gt_sweep_labels_structured['z']), axis=-1)
        gt_vels = np.stack((gt_sweep_labels_structured['velocity_x'], gt_sweep_labels_structured['velocity_y']), axis=-1)
        gt_is_dyn = get_gt_dynamic_status(gt_vels, gt_vel_thresh)

        pred_pts_xyz = np.stack((mdet_sweep_preds_structured['x'], mdet_sweep_preds_structured['y'], mdet_sweep_preds_structured['z']), axis=-1)
        
        # Determine pred_is_dyn:
        # If a score threshold is provided (for ROC/PR), use it. Otherwise, use the binary label.
        current_score_threshold_for_roc = eval_params.get("current_score_threshold")
        if current_score_threshold_for_roc is not None:
            if mdet_score_field_name not in mdet_sweep_preds_structured.dtype.names:
                print(f"    Error: Score field '{mdet_score_field_name}' not found in MDet NPZ for ROC. Skipping sweep.")
                continue
            pred_scores = mdet_sweep_preds_structured[mdet_score_field_name]
            pred_is_dyn = pred_scores >= current_score_threshold_for_roc
        else: # Use the stored binary mdet_label
            if mdet_label_field_name not in mdet_sweep_preds_structured.dtype.names:
                print(f"    Error: Label field '{mdet_label_field_name}' not found in MDet NPZ. Skipping sweep.")
                continue
            pred_mdet_labels = mdet_sweep_preds_structured[mdet_label_field_name]
            pred_is_dyn = (pred_mdet_labels == mdet_dynamic_label_value)
        
        sweep_metrics = calculate_point_metrics_for_sweep(
            gt_pts_xyz, gt_is_dyn, pred_pts_xyz, pred_is_dyn, coord_tol
        )
        
        if "Error_Msg" in sweep_metrics:
            print(f"  Skipping metrics for sweep {sweep_token} due to: {sweep_metrics['Error_Msg']}")
            continue # Skip this sweep's contribution if fundamental mismatch

        processed_sweeps_count += 1
        total_tp += sweep_metrics["TP"]
        total_fp += sweep_metrics["FP"]
        total_fn += sweep_metrics["FN"]
        total_tn += sweep_metrics["TN"]
        total_gt_points_scene += sweep_metrics["N_GT_Points"]
        total_pred_points_scene += sweep_metrics["N_Pred_Points"]
        total_gt_dynamic_scene += sweep_metrics["N_GT_Dynamic"]
        total_pred_dynamic_scene += sweep_metrics["N_Pred_Dynamic"]

    if processed_sweeps_count == 0: 
        print(f"No common sweeps successfully processed for scene from {mdet_scene_npz_path}")
        return None

    scene_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else (1.0 if total_tp == 0 else 0.0)
    scene_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else (1.0 if total_tp == 0 else 0.0)
    scene_f1 = 2*(scene_precision * scene_recall) / (scene_precision + scene_recall) if (scene_precision + scene_recall) > 0 else 0.0
    scene_accuracy = (total_tp + total_tn) / total_gt_points_scene if total_gt_points_scene > 0 else 1.0
    
    scene_name_from_path = os.path.basename(gt_scene_npz_path).replace("gt_point_labels_scene_", "").replace(".npz","")
    return {
        "scene_name": scene_name_from_path,
        "mdet_run_config_path": mdet_scene_npz_path,
        "mdet_run_config": mdet_run_config,
        "evaluation_params": eval_params,
        "TP": total_tp, "FP": total_fp, "FN": total_fn, "TN": total_tn,
        "Precision": scene_precision, "Recall": scene_recall, "F1": scene_f1, "Accuracy": scene_accuracy,
        "N_GT_Points_Total": total_gt_points_scene,
        "N_GT_Dynamic_Total": total_gt_dynamic_scene,
        "N_Pred_Dynamic_Total": total_pred_dynamic_scene,
        "num_common_sweeps_processed": processed_sweeps_count
    }

# (calculate_metrics_for_experiment remains largely the same, but will sum 'TN' instead of 'TN_approx')
def calculate_metrics_for_experiment(
    mdet_experiment_dir: str,
    gt_labels_base_dir: str,
    eval_params: Dict
) -> Optional[Dict[str, Any]]:
    """Calculates aggregated metrics for an experiment (a set of scenes under one MDet config)."""
    
    experiment_tp, experiment_fp, experiment_fn, experiment_tn = 0,0,0,0
    total_gt_points_exp = 0
    scenes_processed_count = 0
    first_mdet_run_config = None
    experiment_scene_results = []

    # Adjust filename pattern to match your script's output if necessary
    # Current script: mdet_results_{scene_name}.npz
    # Validation util was: mdetector_results_scene_{scene_name_key}.npz
    # Let's assume the script's naming:
    mdet_scene_files = [f for f in os.listdir(mdet_experiment_dir) if f.startswith("mdet_results_") and f.endswith(".npz")]

    if not mdet_scene_files:
        print(f"No MDet result files (e.g., mdet_results_scene-xxxx.npz) found in {mdet_experiment_dir}")
        return None

    print(f"Processing experiment in: {os.path.basename(mdet_experiment_dir)} ({len(mdet_scene_files)} scenes)")
    for mdet_file_name in tqdm(mdet_scene_files, desc="  Scenes"):
        mdet_scene_npz_path = os.path.join(mdet_experiment_dir, mdet_file_name)
        # Extract scene_name from mdet_file_name (e.g., "scene-0103" from "mdet_results_scene-0103.npz")
        scene_name_key = mdet_file_name.replace("mdet_results_", "").replace(".npz","")
        gt_scene_npz_path = os.path.join(gt_labels_base_dir, f"gt_point_labels_scene_{scene_name_key}.npz")

        if not os.path.exists(gt_scene_npz_path):
            # print(f"  Skipping {scene_name_key}: GT file not found at {gt_scene_npz_path}")
            continue
        
        scene_metrics = calculate_metrics_for_scene(gt_scene_npz_path, mdet_scene_npz_path, eval_params)
        if scene_metrics:
            experiment_scene_results.append(scene_metrics)
            experiment_tp += scene_metrics["TP"]
            experiment_fp += scene_metrics["FP"]
            experiment_fn += scene_metrics["FN"]
            experiment_tn += scene_metrics["TN"]
            total_gt_points_exp += scene_metrics["N_GT_Points_Total"]
            scenes_processed_count += 1
            if first_mdet_run_config is None:
                first_mdet_run_config = scene_metrics["mdet_run_config"]
    
    if scenes_processed_count == 0: return None

    exp_precision = experiment_tp / (experiment_tp + experiment_fp) if (experiment_tp + experiment_fp) > 0 else (1.0 if experiment_tp == 0 else 0.0)
    exp_recall = experiment_tp / (experiment_tp + experiment_fn) if (experiment_tp + experiment_fn) > 0 else (1.0 if experiment_tp == 0 else 0.0)
    exp_f1 = 2*(exp_precision * exp_recall) / (exp_precision + exp_recall) if (exp_precision + exp_recall) > 0 else 0.0
    exp_accuracy = (experiment_tp + experiment_tn) / total_gt_points_exp if total_gt_points_exp > 0 else 1.0

    return {
        "experiment_id": os.path.basename(mdet_experiment_dir),
        "mdet_run_config": first_mdet_run_config,
        "evaluation_params": eval_params,
        "TP": experiment_tp, "FP": experiment_fp, "FN": experiment_fn, "TN": experiment_tn,
        "Precision": exp_precision, "Recall": exp_recall, "F1": exp_f1, "Accuracy": exp_accuracy,
        "N_GT_Points_Experiment": total_gt_points_exp,
        "num_scenes_total_in_dir": len(mdet_scene_files),
        "num_scenes_successfully_evaluated": scenes_processed_count,
        "per_scene_details": experiment_scene_results
    }

# (load_and_evaluate_all_experiments remains the same)
def load_and_evaluate_all_experiments(
    mdet_results_root_dir: str,
    gt_labels_base_dir: str,
    eval_params: Dict
) -> List[Dict[str, Any]]:
    """Loads and evaluates all experiments, assuming each subdir in mdet_results_root_dir is one experiment."""
    all_experiment_metrics = []
    if not os.path.isdir(mdet_results_root_dir):
        print(f"Error: M-Detector results root directory not found: {mdet_results_root_dir}")
        return []
        
    for experiment_id in os.listdir(mdet_results_root_dir):
        mdet_experiment_dir = os.path.join(mdet_results_root_dir, experiment_id)
        if os.path.isdir(mdet_experiment_dir):
            exp_summary = calculate_metrics_for_experiment(mdet_experiment_dir, gt_labels_base_dir, eval_params)
            if exp_summary:
                all_experiment_metrics.append(exp_summary)
    return all_experiment_metrics

# --- ROC / PR Curve Data Generation (adapting to index-based matching logic) ---
def generate_roc_pr_data(
    mdet_experiment_dir: str,
    gt_labels_base_dir: str,
    eval_params: Dict, # Contains GT vel thresh, MDet score field, etc.
    score_thresholds_for_roc: Union[List[float], np.ndarray] # Renamed for clarity
) -> Optional[Dict[str, List[float]]]:
    """
    Generates data for ROC and PR curves by varying the threshold on M-Detector's output scores
    for a single experiment (set of scenes), using index-based matching.
    """
    mdet_score_field = eval_params.get('mdet_score_field_name', 'mdet_score')
    if not mdet_score_field:
        print("Error: 'mdet_score_field_name' not specified in eval_params for ROC/PR generation.")
        return None

    # Check if score field exists in a sample MDet NPZ
    mdet_scene_files = [f for f in os.listdir(mdet_experiment_dir) if f.startswith("mdet_results_") and f.endswith(".npz")]
    if not mdet_scene_files: 
        print(f"No MDet result files found in {mdet_experiment_dir} for ROC/PR.")
        return None
    
    try:
        with np.load(os.path.join(mdet_experiment_dir, mdet_scene_files[0]), allow_pickle=True) as temp_data:
            mdet_points_key = 'all_points_predictions' # Must match your NPZ structure
            if mdet_points_key not in temp_data or \
               mdet_score_field not in temp_data[mdet_points_key].dtype.names:
                print(f"Error: MDet score field '{mdet_score_field}' not in '{mdet_points_key}' dtype. Cannot generate ROC/PR.")
                return None
    except Exception as e:
        print(f"Error checking MDet NPZ for score field: {e}")
        return None

    tpr_values, fpr_values, precision_values, recall_values = [], [], [], []
    
    print(f"Generating ROC/PR data for {os.path.basename(mdet_experiment_dir)} using score field '{mdet_score_field}'...")
    for score_thresh_val in tqdm(sorted(list(score_thresholds_for_roc), reverse=True), desc="  Score Thresh"):
        current_eval_params_for_roc = eval_params.copy()
        current_eval_params_for_roc["current_score_threshold"] = score_thresh_val # Pass to calculate_metrics_for_scene

        # This will call calculate_metrics_for_scene for each scene with the current score threshold
        # The result is an aggregated TP, FP, FN, TN for the whole experiment at this score_thresh_val
        exp_metrics_at_thresh = calculate_metrics_for_experiment(
            mdet_experiment_dir, gt_labels_base_dir, current_eval_params_for_roc
        )

        if not exp_metrics_at_thresh:
            print(f"Warning: Could not get experiment metrics for score threshold {score_thresh_val}. Skipping this threshold.")
            # Append NaN or handle as appropriate if a threshold fails, to maintain list lengths
            tpr_values.append(np.nan); fpr_values.append(np.nan)
            precision_values.append(np.nan); recall_values.append(np.nan)
            continue

        tp_at_thresh = exp_metrics_at_thresh["TP"]
        fp_at_thresh = exp_metrics_at_thresh["FP"]
        fn_at_thresh = exp_metrics_at_thresh["FN"]
        tn_at_thresh = exp_metrics_at_thresh["TN"]

        # Total actual positives = TP + FN
        total_actual_positives = tp_at_thresh + fn_at_thresh
        # Total actual negatives = FP + TN
        total_actual_negatives = fp_at_thresh + tn_at_thresh
        
        tpr = tp_at_thresh / total_actual_positives if total_actual_positives > 0 else 0.0
        fpr = fp_at_thresh / total_actual_negatives if total_actual_negatives > 0 else \
              (1.0 if fp_at_thresh > 0 else 0.0) # FPR is 1 if any FP and no actual negatives
        
        prec = tp_at_thresh / (tp_at_thresh + fp_at_thresh) if (tp_at_thresh + fp_at_thresh) > 0 else \
               (1.0 if tp_at_thresh == 0 else 0.0)
        rec = tpr # Recall is TPR

        tpr_values.append(tpr)
        fpr_values.append(fpr)
        precision_values.append(prec)
        recall_values.append(rec)
        
    return {
        "fpr": fpr_values, "tpr": tpr_values, 
        "precision": precision_values, "recall": recall_values, 
        "score_thresholds": list(score_thresholds_for_roc) # Store the thresholds used
    }