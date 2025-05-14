# src/utils/validation_utils.py
import os
import numpy as np
from nuscenes.nuscenes import NuScenes # For type hinting
from tqdm import tqdm
import json
from scipy.spatial import KDTree # For efficient point matching
from typing import Dict, List, Tuple, Any, Optional, Union
import h5py

from ..core.constants import POINT_LABEL_DTYPE 

def get_gt_dynamic_points_for_sweep(
    nusc: NuScenes,
    sweep_data_dict: Dict,
    all_points_global: np.ndarray,
    gt_labels_scene_hdf5_path: str,
    velocity_threshold: float
) -> Dict[str, np.ndarray]:
    lidar_sd_token = sweep_data_dict['lidar_sd_token']
    # print(f"    DEBUG (get_gt_dynamic): Getting GT for token: {lidar_sd_token}, HDF5: {gt_labels_scene_hdf5_path}") # DEBUG
    empty_result = {'dynamic': np.empty((0, 3)), 'static': np.empty((0,3)), 'unlabeled': all_points_global if all_points_global is not None else np.empty((0,3))}

    if not os.path.exists(gt_labels_scene_hdf5_path):
        tqdm.write(f"    DEBUG (get_gt_dynamic): GT Label HDF5 file NOT FOUND: {gt_labels_scene_hdf5_path}") # DEBUG
        return empty_result

    point_labels_for_sweep: Optional[np.ndarray] = None
    try:
        with h5py.File(gt_labels_scene_hdf5_path, 'r') as hf:
            # print(f"    DEBUG (get_gt_dynamic): Opened GT HDF5: {gt_labels_scene_hdf5_path}") # DEBUG
            all_tokens_bytes = hf['sweep_lidar_sd_tokens'][:]
            target_token_bytes = lidar_sd_token.encode('utf-8') if isinstance(lidar_sd_token, str) else lidar_sd_token
            
            matches = np.array([]) # Initialize
            try:
                matches = np.where(all_tokens_bytes == target_token_bytes)[0]
                if not matches.size > 0:
                    all_tokens_str_decoded = [t.decode('utf-8', 'ignore') for t in all_tokens_bytes]
                    matches = np.where(np.array(all_tokens_str_decoded) == (target_token_bytes.decode('utf-8', 'ignore') if isinstance(target_token_bytes, bytes) else target_token_bytes) )[0]
            except TypeError:
                 matches = np.where(all_tokens_bytes == (target_token_bytes.decode('utf-8', 'ignore') if isinstance(target_token_bytes, bytes) else target_token_bytes) )[0]


            if not matches.size > 0:
                # print(f"    DEBUG (get_gt_dynamic): Sweep token {lidar_sd_token} NOT FOUND in GT HDF5.") # DEBUG
                return empty_result
            idx_in_token_list = matches[0]
            # print(f"    DEBUG (get_gt_dynamic): Token {lidar_sd_token} found at GT HDF5 index {idx_in_token_list}.") # DEBUG


            indices_array = hf['gt_point_labels_indices'][:]
            start_idx = indices_array[idx_in_token_list]
            if idx_in_token_list + 1 >= len(indices_array):
                tqdm.write(f"    DEBUG (get_gt_dynamic): Index issue for token {lidar_sd_token} in GT HDF5 indices.") # DEBUG
                return empty_result
            end_idx = indices_array[idx_in_token_list + 1]
            # print(f"    DEBUG (get_gt_dynamic): For token {lidar_sd_token}, GT point indices: {start_idx} to {end_idx}") # DEBUG

            point_labels_for_sweep = hf['all_gt_point_labels'][start_idx:end_idx]
            # print(f"    DEBUG (get_gt_dynamic): Loaded 'all_gt_point_labels' for token {lidar_sd_token}, shape: {point_labels_for_sweep.shape}, dtype: {point_labels_for_sweep.dtype}") # DEBUG


            if point_labels_for_sweep.dtype != POINT_LABEL_DTYPE:
                # print(f"    DEBUG (get_gt_dynamic): Dtype mismatch for GT labels. Expected {POINT_LABEL_DTYPE}, got {point_labels_for_sweep.dtype}. Attempting cast.") # DEBUG
                try:
                    point_labels_for_sweep = point_labels_for_sweep.astype(POINT_LABEL_DTYPE)
                except ValueError as ve:
                    tqdm.write(f"    DEBUG (get_gt_dynamic): CRITICAL DTYPE cast failed for GT labels. Error: {ve}.") # DEBUG
                    return empty_result
                    
    except Exception as e:
        tqdm.write(f"    DEBUG (get_gt_dynamic): Error loading GT labels from HDF5 {gt_labels_scene_hdf5_path} for sweep {lidar_sd_token}: {e}") # DEBUG
        import traceback
        traceback.print_exc()
        return empty_result

    if point_labels_for_sweep is None or point_labels_for_sweep.shape[0] == 0 :
        # print(f"    DEBUG (get_gt_dynamic): No GT point labels loaded for sweep {lidar_sd_token} (empty slice or None).") # DEBUG
        return empty_result
        
    points_with_gt_labels_global = np.stack((
        point_labels_for_sweep['x'], point_labels_for_sweep['y'], point_labels_for_sweep['z']
    ), axis=-1)

    gt_speed_sq = point_labels_for_sweep['velocity_x']**2 + point_labels_for_sweep['velocity_y']**2
    is_valid_instance_mask = (point_labels_for_sweep['instance_token'] != b'')
    dynamic_mask = (gt_speed_sq >= velocity_threshold**2) & is_valid_instance_mask
    static_mask = (~dynamic_mask) & is_valid_instance_mask
    unlabeled_mask = ~is_valid_instance_mask
    
    # print(f"    DEBUG (get_gt_dynamic): Token {lidar_sd_token} - GT Dynamic pts: {np.sum(dynamic_mask)}, Static pts: {np.sum(static_mask)}, Unlabeled: {np.sum(unlabeled_mask)}") # DEBUG

    return {
        'dynamic': points_with_gt_labels_global[dynamic_mask],
        'static': points_with_gt_labels_global[static_mask],
        'unlabeled': points_with_gt_labels_global[unlabeled_mask]
    }

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

def get_pred_dynamic_status(pred_labels: np.ndarray, mdet_label_val_positive_class: int) -> np.ndarray:
    if pred_labels.shape[0] == 0:
        return np.array([], dtype=bool)
    return pred_labels == mdet_label_val_positive_class

def get_gt_dynamic_status(gt_velocities_xy: np.ndarray, gt_vel_thresh: float) -> np.ndarray:
    """gt_velocities_xy is shape (N, 2) for vx, vy."""
    if gt_velocities_xy.shape[0] == 0:
        return np.array([], dtype=bool)
    # Calculates magnitude squared and compares with threshold squared
    gt_speed_sq = np.sum(gt_velocities_xy**2, axis=1) 
    return gt_speed_sq > (gt_vel_thresh**2)


# --- Ground Truth Processing ---
# def get_gt_dynamic_status(
#     gt_velocities_xyz: np.ndarray, 
#     gt_velocity_threshold: float
# ) -> np.ndarray:
#     """Determines dynamic status of GT points based on velocity magnitude."""
#     if gt_velocities_xyz.ndim != 2 or gt_velocities_xyz.shape[0] == 0 or gt_velocities_xyz.shape[1] < 2:
#         return np.zeros(gt_velocities_xyz.shape[0], dtype=bool)
#     gt_speeds_sq = np.sum(gt_velocities_xyz[:, :2]**2, axis=1) 
#     return gt_speeds_sq > gt_velocity_threshold**2

# --- Core Sweep Metrics Calculation ---
def calculate_point_metrics_for_sweep(
    gt_sweep_labels: np.ndarray,
    pred_sweep_labels: np.ndarray,
    eval_params: Dict
) -> Dict:
    gt_vel_thresh = eval_params.get('gt_vel_thresh', 0.5) # Renamed from gt_velocity_threshold for consistency
    mdet_label_val_positive_class = eval_params.get('mdet_label_val', 0) 
    coordinate_tolerance = eval_params.get('coordinate_tolerance', 0.1)

    if gt_sweep_labels.shape[0] == 0 and pred_sweep_labels.shape[0] == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'num_gt_points': 0, 'num_pred_points': 0, 'gt_is_dyn_sum':0, 'pred_is_dyn_sum':0, 'iou_dynamic': 0.0}

    # Ensure point counts match *after* GT filtering (which happens before this function is called)
    if gt_sweep_labels.shape[0] != pred_sweep_labels.shape[0]:
        return {"Error_Msg": f"Point count mismatch. GT_filtered: {gt_sweep_labels.shape[0]}, Pred: {pred_sweep_labels.shape[0]}"}

    num_points = gt_sweep_labels.shape[0]
    if num_points == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'num_gt_points': 0, 'num_pred_points': 0, 'gt_is_dyn_sum':0, 'pred_is_dyn_sum':0, 'iou_dynamic': 0.0}

    # Coordinate Sanity Check
    try:
        gt_points_xyz = np.stack((gt_sweep_labels['x'], gt_sweep_labels['y'], gt_sweep_labels['z']), axis=-1)
        pred_points_xyz = np.stack((pred_sweep_labels['x'], pred_sweep_labels['y'], pred_sweep_labels['z']), axis=-1)
    except ValueError as e_coord_stack:
        return {"Error_Msg": f"Failed to stack x,y,z coordinates for comparison. Check dtype. {e_coord_stack}"}

    coord_diff = np.linalg.norm(gt_points_xyz - pred_points_xyz, axis=1)
    if np.any(coord_diff > coordinate_tolerance):
        max_diff = np.max(coord_diff) if coord_diff.size > 0 else 0
        num_mismatched_coords = np.sum(coord_diff > coordinate_tolerance)
        return {"Error_Msg": (f"Coordinate mismatch. Max_diff: {max_diff:.3f}m ({num_mismatched_coords}/{num_points} pts > {coordinate_tolerance}m).")}

    # Get GT dynamic status
    try:
        # Ensure 'velocity_x' and 'velocity_y' are present in the filtered gt_sweep_labels
        gt_vels_for_dyn_status = np.stack((gt_sweep_labels['velocity_x'], gt_sweep_labels['velocity_y']), axis=-1)
        gt_is_dyn = get_gt_dynamic_status(gt_vels_for_dyn_status, gt_vel_thresh)
    except ValueError as e_gt_dyn: # This can happen if keys are missing or due to stacking issues
        return {"Error_Msg": f"Failed to get GT dynamic status. Check 'velocity_x/y' fields in GT. {e_gt_dyn}"}

    # Get Pred dynamic status
    try:
        # Ensure 'mdet_label' is present in pred_sweep_labels
        pred_is_dyn = get_pred_dynamic_status(pred_sweep_labels['mdet_label'], mdet_label_val_positive_class)
    except ValueError as e_pred_dyn:
        return {"Error_Msg": f"Failed to get Pred dynamic status. Check 'mdet_label' field in Pred. {e_pred_dyn}"}

    tp = np.sum(gt_is_dyn & pred_is_dyn)
    fp = np.sum(~gt_is_dyn & pred_is_dyn)
    fn = np.sum(gt_is_dyn & ~pred_is_dyn)
    tn = np.sum(~gt_is_dyn & ~pred_is_dyn)

    denominator_iou = tp + fp + fn
    iou_dynamic = tp / denominator_iou if denominator_iou > 0 else 0.0
    
    gt_is_dyn_sum = np.sum(gt_is_dyn)
    pred_is_dyn_sum = np.sum(pred_is_dyn)

    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'num_gt_points': num_points, 'num_pred_points': num_points,
            'gt_is_dyn_sum': int(gt_is_dyn_sum), 'pred_is_dyn_sum': int(pred_is_dyn_sum),
            'iou_dynamic': float(iou_dynamic)}



# --- Scene and Experiment Level Metrics ---
def calculate_metrics_for_scene(
    gt_scene_npz_path: str,
    mdet_scene_npz_path: str,
    eval_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    scene_name_for_log = os.path.basename(mdet_scene_npz_path)
    print(f"\nDEBUG SCENE START: {os.path.basename(gt_scene_npz_path)} vs {scene_name_for_log}")

    gt_data_npz: Optional[np.NpzFile] = None
    mdet_data_npz: Optional[np.NpzFile] = None

    try:
        gt_data_npz = np.load(gt_scene_npz_path, allow_pickle=False)
        mdet_data_npz = np.load(mdet_scene_npz_path, allow_pickle=False)

        # --- Pre-load all necessary arrays from NPZ files ---
        # GT arrays
        gt_sweep_lidar_sd_tokens_arr = gt_data_npz['sweep_lidar_sd_tokens']
        gt_point_labels_indices_arr = gt_data_npz['gt_point_labels_indices']
        all_gt_point_labels_arr = gt_data_npz['all_gt_point_labels']
        # Add other GT arrays if needed, e.g., gt_data_npz['scene_token'] if used

        # MDet arrays
        mdet_sweep_lidar_sd_tokens_arr = mdet_data_npz['sweep_lidar_sd_tokens']
        points_predictions_indices_arr = mdet_data_npz['points_predictions_indices']
        all_points_predictions_arr = mdet_data_npz['all_points_predictions']
        # Add other MDet arrays if needed, e.g., mdet_data_npz['scene_token'] if used

        mdet_config_json_str_arr = mdet_data_npz.get('_config_json_str') # Use .get for optional presence
        # --- End pre-loading arrays ---

        # --- Determine M-Detector's range parameters ---
        default_min_range = eval_params.get('mdet_min_point_range_meters', 1.0)
        default_max_range = eval_params.get('mdet_max_point_range_meters', 80.0)
        mdet_min_range = default_min_range
        mdet_max_range = default_max_range
        loaded_from_npz_successfully = False

        if mdet_config_json_str_arr is not None:
            try:
                config_str = mdet_config_json_str_arr.item() if mdet_config_json_str_arr.ndim == 0 else str(mdet_config_json_str_arr)
                mdet_config_loaded = json.loads(config_str)
                filtering_config = mdet_config_loaded.get('filtering')
                if filtering_config and isinstance(filtering_config, dict):
                    min_r_from_cfg = filtering_config.get('min_point_range_meters')
                    max_r_from_cfg = filtering_config.get('max_point_range_meters')
                    if min_r_from_cfg is not None and max_r_from_cfg is not None:
                        mdet_min_range = float(min_r_from_cfg)
                        mdet_max_range = float(max_r_from_cfg)
                        print(f"  Successfully loaded MDet range params from NPZ: min={mdet_min_range:.2f}m, max={mdet_max_range:.2f}m")
                        loaded_from_npz_successfully = True
                    else:
                        print(f"  MDet config in NPZ found, but 'min_point_range_meters' or 'max_point_range_meters' missing in 'filtering' dict.")
                else:
                    print(f"  MDet config in NPZ found, but 'filtering' key missing or not a dictionary.")
            except json.JSONDecodeError as e_json:
                print(f"  Warning: Failed to parse MDet _config_json_str from NPZ for '{scene_name_for_log}'. Error: {e_json}")
            except Exception as e_cfg:
                print(f"  Warning: Error processing MDet config from NPZ for '{scene_name_for_log}'. Error: {e_cfg}")
        else:
            print(f"  Info: MDet NPZ for '{scene_name_for_log}' does not contain '_config_json_str'.")

        if not loaded_from_npz_successfully:
            mdet_min_range = default_min_range
            mdet_max_range = default_max_range
            print(f"  INFO: Using fallback/default MDet range params from eval_params: min={mdet_min_range:.2f}m, max={mdet_max_range:.2f}m")
        # --- End range parameter determination ---

        # --- Find common sweeps based on lidar_sd_token ---
        gt_token_to_idx: Dict[bytes, int] = {token: i for i, token in enumerate(gt_sweep_lidar_sd_tokens_arr)}
        mdet_token_to_idx: Dict[bytes, int] = {token: i for i, token in enumerate(mdet_sweep_lidar_sd_tokens_arr)}
        common_sd_tokens_bytes = set(gt_token_to_idx.keys()) & set(mdet_token_to_idx.keys())

        if not common_sd_tokens_bytes:
            print(f"  No common sweep tokens found between GT and MDet for scene '{scene_name_for_log}'. Skipping.")
            return None

        common_sweep_indices: List[Tuple[int, int]] = []
        for token_bytes in common_sd_tokens_bytes:
            common_sweep_indices.append((gt_token_to_idx[token_bytes], mdet_token_to_idx[token_bytes]))
        common_sweep_indices.sort(key=lambda x: x[0])
        print(f"  Found {len(common_sweep_indices)} common sweeps for scene '{scene_name_for_log}'.")

        scene_tp, scene_fp, scene_fn, scene_tn = 0, 0, 0, 0
        scene_total_gt_points_in_range = 0
        scene_total_mdet_points = 0
        processed_sweeps_count_scene = 0

        for sweep_idx_gt, sweep_idx_mdet in common_sweep_indices:
            gt_start, gt_end = gt_point_labels_indices_arr[sweep_idx_gt], gt_point_labels_indices_arr[sweep_idx_gt + 1]
            mdet_start, mdet_end = points_predictions_indices_arr[sweep_idx_mdet], points_predictions_indices_arr[sweep_idx_mdet + 1]

            gt_sweep_labels_structured_full = all_gt_point_labels_arr[gt_start:gt_end]
            mdet_sweep_preds_structured = all_points_predictions_arr[mdet_start:mdet_end]
            
            current_sweep_sd_token_str = gt_sweep_lidar_sd_tokens_arr[sweep_idx_gt].decode('utf-8', 'ignore')

            if gt_sweep_labels_structured_full.shape[0] > 0:
                try:
                    gt_points_sensor_for_filter = np.stack((
                        gt_sweep_labels_structured_full['x_sensor'],
                        gt_sweep_labels_structured_full['y_sensor'],
                        gt_sweep_labels_structured_full['z_sensor']
                    ), axis=-1)
                except ValueError as e_stack:
                    print(f"      ERROR for sweep {current_sweep_sd_token_str}: Missing 'x_sensor'/'y_sensor'/'z_sensor' in GT labels. Cannot filter. {e_stack}")
                    continue
                gt_ranges = np.linalg.norm(gt_points_sensor_for_filter, axis=1)
                gt_range_mask = (gt_ranges >= mdet_min_range) & (gt_ranges <= mdet_max_range)
                filtered_gt_labels_for_comparison = gt_sweep_labels_structured_full[gt_range_mask]
            else:
                filtered_gt_labels_for_comparison = gt_sweep_labels_structured_full

            sweep_metrics_results = calculate_point_metrics_for_sweep(
                gt_sweep_labels=filtered_gt_labels_for_comparison,
                pred_sweep_labels=mdet_sweep_preds_structured,
                eval_params=eval_params
            )

            if sweep_metrics_results and not sweep_metrics_results.get("Error_Msg"):
                scene_tp += sweep_metrics_results.get('tp', 0)
                scene_fp += sweep_metrics_results.get('fp', 0)
                scene_fn += sweep_metrics_results.get('fn', 0)
                scene_tn += sweep_metrics_results.get('tn', 0)
                scene_total_gt_points_in_range += sweep_metrics_results.get('num_gt_points', 0)
                scene_total_mdet_points += sweep_metrics_results.get('num_pred_points', 0)
                processed_sweeps_count_scene += 1
            else:
                error_msg = sweep_metrics_results.get("Error_Msg", "Unknown error") if sweep_metrics_results else "calculate_point_metrics_for_sweep returned None"
                print(f"      Skipping metrics accumulation for sweep {current_sweep_sd_token_str}: {error_msg}")
                if "Point count mismatch" in error_msg:
                    print(f"        CRITICAL: Point count mismatch persists for {current_sweep_sd_token_str} AFTER GT filtering. "
                          f"Filtered GT: {filtered_gt_labels_for_comparison.shape[0]}, "
                          f"MDet: {mdet_sweep_preds_structured.shape[0]}. "
                          f"Check MDet range params (min={mdet_min_range}, max={mdet_max_range}) & GT sensor coord usage.")
                elif "Coordinate mismatch" in error_msg:
                    print(f"        CRITICAL: Coordinate mismatch for {current_sweep_sd_token_str}. Ensure alignment.")

        if processed_sweeps_count_scene == 0:
            print(f"  No sweeps successfully processed for scene '{scene_name_for_log}'. Cannot calculate scene metrics.")
            return {
                "scene_name": scene_name_for_log, "processed_sweeps": 0,
                "error": "No sweeps processed due to errors or mismatches."}

        precision = scene_tp / (scene_tp + scene_fp) if (scene_tp + scene_fp) > 0 else 0.0
        recall = scene_tp / (scene_tp + scene_fn) if (scene_tp + scene_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (scene_tp + scene_tn) / (scene_tp + scene_fp + scene_fn + scene_tn) if (scene_tp + scene_fp + scene_fn + scene_tn) > 0 else 0.0
        scene_denominator_iou = scene_tp + scene_fp + scene_fn
        scene_iou_dynamic = scene_tp / scene_denominator_iou if scene_denominator_iou > 0 else 0.0

        if scene_total_gt_points_in_range != scene_total_mdet_points and scene_total_gt_points_in_range > 0:
            print(f"  Warning for scene '{scene_name_for_log}': Total filtered GT points ({scene_total_gt_points_in_range}) "
                  f"does not match total MDet points ({scene_total_mdet_points}) across processed sweeps. This may indicate an issue.")

        scene_summary_stats = {
            "scene_name": scene_name_for_log,
            "tp": scene_tp, "fp": scene_fp, "fn": scene_fn, "tn": scene_tn,
            "precision": precision, "recall": recall, "f1_score": f1_score, "accuracy": accuracy,
            "iou_dynamic": scene_iou_dynamic,
            "total_gt_points_in_range": scene_total_gt_points_in_range,
            "total_mdet_points_processed": scene_total_mdet_points,
            "processed_sweeps": processed_sweeps_count_scene,
            "mdet_min_range_used": mdet_min_range,
            "mdet_max_range_used": mdet_max_range
        }
        print(f"  Scene Summary for '{scene_name_for_log}': P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, IoU_dyn={scene_iou_dynamic:.3f} ({processed_sweeps_count_scene} sweeps)")
        return scene_summary_stats

    except FileNotFoundError:
        print(f"  Error: One or both NPZ files not found. GT: '{gt_scene_npz_path}', MDet: '{mdet_scene_npz_path}'")
        return None
    except KeyError as e: # Catch KeyErrors from accessing NPZ files if keys are missing
        print(f"  Error: Missing key in NPZ file for scene '{scene_name_for_log}': {e}")
        return None
    except Exception as e:
        print(f"  Error processing scene '{scene_name_for_log}': {e}")
        return None
    finally:
        if gt_data_npz:
            gt_data_npz.close()
        if mdet_data_npz:
            mdet_data_npz.close()


def calculate_metrics_for_experiment(
    mdet_experiment_dir: str,
    gt_labels_base_dir: str,
    eval_params: Dict
) -> Optional[Dict[str, Any]]:
    experiment_tp, experiment_fp, experiment_fn, experiment_tn = 0,0,0,0
    total_gt_points_exp = 0
    scenes_processed_count = 0
    experiment_scene_results = []

    if not os.path.isdir(mdet_experiment_dir):
        print(f"MDet experiment directory not found: {mdet_experiment_dir}")
        return None
    if not os.path.isdir(gt_labels_base_dir):
        print(f"GT labels base directory not found: {gt_labels_base_dir}")
        return None

    mdet_scene_files = [f for f in os.listdir(mdet_experiment_dir) if f.startswith("mdet_results_") and f.endswith(".npz")]

    if not mdet_scene_files:
        print(f"No MDet result files (e.g., mdet_results_scene-xxxx.npz) found in {mdet_experiment_dir}")
        return None

    print(f"Processing experiment in: {os.path.basename(mdet_experiment_dir)} ({len(mdet_scene_files)} scenes)")
    for mdet_file_name in tqdm(mdet_scene_files, desc="  Scenes"):
        mdet_scene_npz_path = os.path.join(mdet_experiment_dir, mdet_file_name)
        scene_name_key = mdet_file_name.replace("mdet_results_", "").replace(".npz","")
        gt_scene_npz_path = os.path.join(gt_labels_base_dir, f"gt_point_labels_{scene_name_key}.npz")
        
        # print(f"  Attempting to process MDet file: {mdet_file_name}") # Debug
        # print(f"    Derived scene_name_key: {scene_name_key}") # Debug
        # print(f"    Checking for GT file at: {gt_scene_npz_path}") # Debug
        
        if not os.path.exists(gt_scene_npz_path):
            # print(f"    GT file exists? False. Skipping.") # Debug
            print(f"  Skipping {scene_name_key}: GT file not found at {gt_scene_npz_path}")
            continue
        # print(f"    GT file exists? True.") # Debug
        
        scene_metrics = calculate_metrics_for_scene(gt_scene_npz_path, mdet_scene_npz_path, eval_params)
        if scene_metrics and not scene_metrics.get("error"): # Check for the error key from no processed sweeps
            experiment_scene_results.append(scene_metrics)
            experiment_tp += scene_metrics.get("tp", 0)
            experiment_fp += scene_metrics.get("fp", 0)
            experiment_fn += scene_metrics.get("fn", 0)
            experiment_tn += scene_metrics.get("tn", 0)
            total_gt_points_exp += scene_metrics.get("total_gt_points_in_range", 0)
            scenes_processed_count += 1
        elif scene_metrics and scene_metrics.get("error"):
            print(f"  Scene {scene_name_key} processed with error: {scene_metrics.get('error')}")
        # else: scene_metrics was None, error already printed by calculate_metrics_for_scene

    if scenes_processed_count == 0:
        print(f"No scenes were successfully processed for experiment {os.path.basename(mdet_experiment_dir)}.")
        return None

    exp_precision = experiment_tp / (experiment_tp + experiment_fp) if (experiment_tp + experiment_fp) > 0 else 0.0
    exp_recall = experiment_tp / (experiment_tp + experiment_fn) if (experiment_tp + experiment_fn) > 0 else 0.0
    exp_f1 = 2 * (exp_precision * exp_recall) / (exp_precision + exp_recall) if (exp_precision + exp_recall) > 0 else 0.0
    exp_accuracy = (experiment_tp + experiment_tn) / (experiment_tp + experiment_fp + experiment_fn + experiment_tn) if (experiment_tp + experiment_fp + experiment_fn + experiment_tn) > 0 else 0.0
    exp_denominator_iou = experiment_tp + experiment_fp + experiment_fn
    exp_iou_dynamic = experiment_tp / exp_denominator_iou if exp_denominator_iou > 0 else 0.0

    return {
        "experiment_id": os.path.basename(mdet_experiment_dir),
        "evaluation_params": eval_params,
        "TP": experiment_tp, "FP": experiment_fp, "FN": experiment_fn, "TN": experiment_tn,
        "Precision": exp_precision, "Recall": exp_recall, "F1": exp_f1, "Accuracy": exp_accuracy,
        "N_GT_Points_Experiment": total_gt_points_exp,
        "num_scenes_total_in_dir": len(mdet_scene_files),
        "num_scenes_successfully_evaluated": scenes_processed_count,
        "overall_iou_dynamic": exp_iou_dynamic,
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

def generate_roc_pr_data_gt_velocity_variant(
    mdet_experiment_dir: str,
    gt_labels_base_dir: str,
    eval_params_base: Dict, # Base eval params (MDet label interpretation, etc.)
    gt_velocity_thresholds_for_roc: Union[List[float], np.ndarray]
) -> Optional[Dict[str, List[float]]]:
    """
    Generates data for ROC-like and PR-like curves by varying the 
    GROUND TRUTH velocity threshold to define "actual positives".
    M-Detector's predictions are considered fixed (binary based on its output labels).
    """
    tpr_values, fpr_values, precision_values, recall_values = [], [], [], []
    
    # Ensure that the base eval_params does not contain 'current_score_threshold'
    # as we want calculate_metrics_for_scene to use the M-Detector's binary labels.
    eval_params_base.pop("current_score_threshold", None) 
    if "mdet_label_field_name" not in eval_params_base or \
       "mdet_dynamic_label_value" not in eval_params_base:
        print("Error: eval_params_base must contain 'mdet_label_field_name' and 'mdet_dynamic_label_value' for fixed MDet predictions.")
        return None

    print(f"Generating ROC-like data for {os.path.basename(mdet_experiment_dir)} by varying GT velocity threshold...")
    
    # Sort thresholds: typically ROC curves are plotted with increasing FPR.
    # Varying GT velocity threshold has a less direct relationship with FPR monotonicity,
    # but sorting the input thresholds can make resulting plots easier to interpret if there's a trend.
    # Smallest GT velocity threshold = most GT points are dynamic.
    # Largest GT velocity threshold = fewest GT points are dynamic.
    sorted_gt_velocity_thresholds = sorted(list(gt_velocity_thresholds_for_roc))

    for gt_vel_thresh_roc_val in tqdm(sorted_gt_velocity_thresholds, desc="  GT Vel Thresh"):
        current_eval_params_for_roc = eval_params_base.copy()
        # Key change: Update the GT velocity threshold in eval_params for this iteration
        current_eval_params_for_roc["gt_velocity_threshold"] = gt_vel_thresh_roc_val

        exp_metrics_at_thresh = calculate_metrics_for_experiment(
            mdet_experiment_dir, gt_labels_base_dir, current_eval_params_for_roc
        )

        if not exp_metrics_at_thresh:
            print(f"Warning: Could not get experiment metrics for GT velocity threshold {gt_vel_thresh_roc_val:.3f}. Skipping.")
            tpr_values.append(np.nan); fpr_values.append(np.nan)
            precision_values.append(np.nan); recall_values.append(np.nan)
            continue

        tp_at_thresh = exp_metrics_at_thresh["TP"]
        fp_at_thresh = exp_metrics_at_thresh["FP"]
        fn_at_thresh = exp_metrics_at_thresh["FN"]
        tn_at_thresh = exp_metrics_at_thresh["TN"]

        total_actual_positives = tp_at_thresh + fn_at_thresh
        total_actual_negatives = fp_at_thresh + tn_at_thresh
        
        tpr = tp_at_thresh / total_actual_positives if total_actual_positives > 0 else 0.0
        # FPR definition: FP / (FP + TN)
        fpr = fp_at_thresh / total_actual_negatives if total_actual_negatives > 0 else \
              (1.0 if fp_at_thresh > 0 and total_actual_negatives == 0 else 0.0) 
        
        prec = tp_at_thresh / (tp_at_thresh + fp_at_thresh) if (tp_at_thresh + fp_at_thresh) > 0 else \
               (1.0 if tp_at_thresh == 0 and fp_at_thresh == 0 else 0.0) # Precision is 1 if detector makes no positive predictions
        rec = tpr # Recall is the same as TPR

        tpr_values.append(tpr)
        fpr_values.append(fpr)
        precision_values.append(prec)
        recall_values.append(rec)
        
    return {
        "fpr": fpr_values, "tpr": tpr_values, 
        "precision": precision_values, "recall": recall_values, 
        "gt_velocity_thresholds": sorted_gt_velocity_thresholds # Store the thresholds used, in sorted order
    }

# def _calculate_tfpn_for_sweep_from_preloaded(
#     filtered_gt_labels_for_comparison: np.ndarray, # GT data already range-filtered
#     mdet_sweep_preds_structured: np.ndarray,     # MDet data for the sweep
#     current_gt_velocity_threshold: float,
#     eval_params_base: Dict # For mdet_label_val, etc.
# ) -> Dict[str, int]:
#     """
#     Core logic for TP/FP/FN/TN from pre-loaded, range-filtered sweep data.
#     Assumes coordinate checks have passed or are handled at a higher level if needed.
#     """
#     if filtered_gt_labels_for_comparison.shape[0] != mdet_sweep_preds_structured.shape[0]:
#         # This check should ideally pass if pre-loading and range filtering were correct
#         # print(f"    Internal Error: Point count mismatch in _calculate_tfpn_for_sweep_from_preloaded. "
#         #       f"GT: {filtered_gt_labels_for_comparison.shape[0]}, MDet: {mdet_sweep_preds_structured.shape[0]}")
#         return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, "error": 1}

#     if filtered_gt_labels_for_comparison.shape[0] == 0:
#         return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

#     # Get GT dynamic status using the CURRENT GT velocity threshold for ROC
#     try:
#         gt_vels_for_dyn_status = np.stack((
#             filtered_gt_labels_for_comparison['velocity_x'], 
#             filtered_gt_labels_for_comparison['velocity_y']
#         ), axis=-1)
#         gt_is_dyn = get_gt_dynamic_status(gt_vels_for_dyn_status, current_gt_velocity_threshold)
#     except ValueError: # Should not happen if data is correct
#         return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, "error": 1}


#     # Get Pred dynamic status (MDet predictions are fixed)
#     mdet_label_val_positive_class = eval_params_base.get('mdet_label_val', 0)
#     try:
#         pred_is_dyn = get_pred_dynamic_status(mdet_sweep_preds_structured['mdet_label'], mdet_label_val_positive_class)
#     except ValueError: # Should not happen
#         return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, "error": 1}

#     tp = np.sum(gt_is_dyn & pred_is_dyn)
#     fp = np.sum(~gt_is_dyn & pred_is_dyn)
#     fn = np.sum(gt_is_dyn & ~pred_is_dyn)
#     tn = np.sum(~gt_is_dyn & ~pred_is_dyn)
    
#     return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}

def generate_roc_pr_data_gt_velocity_variant(
    mdet_experiment_dir: str,
    gt_labels_base_dir: str,
    eval_params_base: Dict, # Base eval params (MDet label interpretation, range defaults, etc.)
    gt_velocity_thresholds_for_roc: Union[List[float], np.ndarray]
) -> Optional[Dict[str, List[float]]]:
    """
    Generates data for ROC-like and PR-like curves by varying the 
    GROUND TRUTH velocity threshold. Data is loaded once.
    """
    print(f"Pre-loading all experiment data for ROC generation from: {os.path.basename(mdet_experiment_dir)}...")
    
    # --- Phase 1: Pre-load ALL necessary data for the experiment ---
    all_sweeps_data_for_roc: List[Dict[str, np.ndarray]] = []
    
    mdet_scene_files = [f for f in os.listdir(mdet_experiment_dir) if f.startswith("mdet_results_") and f.endswith(".npz")]
    if not mdet_scene_files:
        print(f"No MDet result files found in {mdet_experiment_dir}")
        return None

    for mdet_file_name in tqdm(mdet_scene_files, desc="  Loading Scenes"):
        mdet_scene_npz_path = os.path.join(mdet_experiment_dir, mdet_file_name)
        scene_name_key = mdet_file_name.replace("mdet_results_", "").replace(".npz","")
        gt_scene_npz_path = os.path.join(gt_labels_base_dir, f"gt_point_labels_{scene_name_key}.npz")

        if not os.path.exists(gt_scene_npz_path):
            # print(f"    Skipping {scene_name_key}: GT file not found at {gt_scene_npz_path}")
            continue
        
        try:
            gt_data_scene = np.load(gt_scene_npz_path, allow_pickle=False)
            mdet_data_scene = np.load(mdet_scene_npz_path, allow_pickle=False)
        except Exception as e:
            print(f"    Error loading NPZ for scene {scene_name_key} during pre-load: {e}")
            continue

        # Determine M-Detector's range parameters for THIS scene (as in calculate_metrics_for_scene)
        default_min_range = eval_params_base.get('mdet_min_point_range_meters', 1.0)
        default_max_range = eval_params_base.get('mdet_max_point_range_meters', 80.0)
        mdet_min_r, mdet_max_r = default_min_range, default_max_range
        # (Include your robust logic here to load from mdet_data_scene['_config_json_str'] 
        #  or fall back to defaults, similar to calculate_metrics_for_scene)
        # For brevity, I'll assume mdet_min_r, mdet_max_r are determined here.
        # Example simplified logic:
        if '_config_json_str' in mdet_data_scene:
            try:
                config_str = mdet_data_scene['_config_json_str'].item() if mdet_data_scene['_config_json_str'].ndim == 0 else str(mdet_data_scene['_config_json_str'])
                mdet_config_loaded = json.loads(config_str)
                filtering_config = mdet_config_loaded.get('filtering')
                if filtering_config and isinstance(filtering_config, dict):
                    min_r_cfg = filtering_config.get('min_point_range_meters')
                    max_r_cfg = filtering_config.get('max_point_range_meters')
                    if min_r_cfg is not None: mdet_min_r = float(min_r_cfg)
                    if max_r_cfg is not None: mdet_max_r = float(max_r_cfg)
            except Exception: # Simplified error handling for this snippet
                pass # Defaults will be used

        # Find common sweeps for this scene
        gt_tokens_bytes = gt_data_scene['sweep_lidar_sd_tokens']
        mdet_tokens_bytes = mdet_data_scene['sweep_lidar_sd_tokens']
        gt_token_to_idx = {token: i for i, token in enumerate(gt_tokens_bytes)}
        mdet_token_to_idx = {token: i for i, token in enumerate(mdet_tokens_bytes)}
        common_sd_tokens_bytes = set(gt_token_to_idx.keys()) & set(mdet_token_to_idx.keys())
        
        scene_common_sweep_indices = []
        for token_bytes in common_sd_tokens_bytes:
            scene_common_sweep_indices.append((gt_token_to_idx[token_bytes], mdet_token_to_idx[token_bytes]))
        scene_common_sweep_indices.sort(key=lambda x: x[0])

        # For each common sweep in this scene, extract and filter data
        for sweep_idx_gt, sweep_idx_mdet in scene_common_sweep_indices:
            gt_start, gt_end = gt_data_scene['gt_point_labels_indices'][sweep_idx_gt], gt_data_scene['gt_point_labels_indices'][sweep_idx_gt + 1]
            mdet_start, mdet_end = mdet_data_scene['points_predictions_indices'][sweep_idx_mdet], mdet_data_scene['points_predictions_indices'][sweep_idx_mdet + 1]

            gt_sweep_labels_full = gt_data_scene['all_gt_point_labels'][gt_start:gt_end]
            mdet_sweep_preds = mdet_data_scene['all_points_predictions'][mdet_start:mdet_end]

            if gt_sweep_labels_full.shape[0] == 0: # Empty GT sweep
                filtered_gt_for_comp = gt_sweep_labels_full
            else:
                try:
                    gt_points_sensor = np.stack((
                        gt_sweep_labels_full['x_sensor'], gt_sweep_labels_full['y_sensor'], gt_sweep_labels_full['z_sensor']
                    ), axis=-1)
                    gt_ranges = np.linalg.norm(gt_points_sensor, axis=1)
                    gt_range_mask = (gt_ranges >= mdet_min_r) & (gt_ranges <= mdet_max_r)
                    filtered_gt_for_comp = gt_sweep_labels_full[gt_range_mask]
                except ValueError: # Missing sensor coords
                    # print(f"    Skipping sweep due to missing sensor coords in GT during pre-load.")
                    continue # Skip this sweep

            # Coordinate sanity check (optional here, but good for consistency)
            # If counts don't match after range filtering, or coord mismatch, skip adding this sweep
            if filtered_gt_for_comp.shape[0] != mdet_sweep_preds.shape[0]:
                # print(f"    Skipping sweep: Point count mismatch after range filtering during pre-load. "
                #       f"GT_filt: {filtered_gt_for_comp.shape[0]}, MDet: {mdet_sweep_preds.shape[0]}")
                continue
            
            if filtered_gt_for_comp.shape[0] > 0: # Only if there are points to compare
                try:
                    gt_points_xyz = np.stack((filtered_gt_for_comp['x'], filtered_gt_for_comp['y'], filtered_gt_for_comp['z']), axis=-1)
                    pred_points_xyz = np.stack((mdet_sweep_preds['x'], mdet_sweep_preds['y'], mdet_sweep_preds['z']), axis=-1)
                    coord_diff = np.linalg.norm(gt_points_xyz - pred_points_xyz, axis=1)
                    coordinate_tolerance = eval_params_base.get('coordinate_tolerance', 0.1)
                    if np.any(coord_diff > coordinate_tolerance):
                        # print(f"    Skipping sweep: Coordinate mismatch during pre-load.")
                        continue
                except ValueError: # Missing global coords
                    # print(f"    Skipping sweep due to missing global coords during pre-load.")
                    continue

            # Add the pre-processed, range-filtered GT and MDet data for this sweep
            all_sweeps_data_for_roc.append({
                'gt_labels_in_range': filtered_gt_for_comp,
                'mdet_preds': mdet_sweep_preds
            })
            
    if not all_sweeps_data_for_roc:
        print("No sweep data successfully pre-loaded for ROC generation. Aborting.")
        return None
    print(f"Finished pre-loading. Total comparable sweeps in memory: {len(all_sweeps_data_for_roc)}")

    # --- Phase 2: Iterate through thresholds using in-memory data ---
    tpr_values, fpr_values, precision_values, recall_values = [], [], [], []
    sorted_gt_velocity_thresholds = sorted(list(gt_velocity_thresholds_for_roc))

    print(f"Calculating ROC/PR points by varying GT velocity threshold...")
    for current_gt_vel_thresh in tqdm(sorted_gt_velocity_thresholds, desc="  GT Vel Thresh"):
        exp_tp_at_thresh, exp_fp_at_thresh, exp_fn_at_thresh, exp_tn_at_thresh = 0, 0, 0, 0
        
        for sweep_data in all_sweeps_data_for_roc:
            tfpn_results = _calculate_tfpn_for_sweep_from_preloaded(
                sweep_data['gt_labels_in_range'],
                sweep_data['mdet_preds'],
                current_gt_vel_thresh,
                eval_params_base 
            )
            if not tfpn_results.get("error"):
                exp_tp_at_thresh += tfpn_results['tp']
                exp_fp_at_thresh += tfpn_results['fp']
                exp_fn_at_thresh += tfpn_results['fn']
                exp_tn_at_thresh += tfpn_results['tn']

        # Calculate overall TPR, FPR, Precision, Recall for this GT velocity threshold
        total_actual_positives = exp_tp_at_thresh + exp_fn_at_thresh
        total_actual_negatives = exp_fp_at_thresh + exp_tn_at_thresh
        
        tpr = exp_tp_at_thresh / total_actual_positives if total_actual_positives > 0 else 0.0
        fpr = exp_fp_at_thresh / total_actual_negatives if total_actual_negatives > 0 else \
              (1.0 if exp_fp_at_thresh > 0 and total_actual_negatives == 0 else 0.0)
        prec = exp_tp_at_thresh / (exp_tp_at_thresh + exp_fp_at_thresh) if (exp_tp_at_thresh + exp_fp_at_thresh) > 0 else \
               (1.0 if exp_tp_at_thresh == 0 and exp_fp_at_thresh == 0 else 0.0)
        rec = tpr

        tpr_values.append(tpr)
        fpr_values.append(fpr)
        precision_values.append(prec)
        recall_values.append(rec)
        
    return {
        "fpr": fpr_values, "tpr": tpr_values, 
        "precision": precision_values, "recall": recall_values, 
        "gt_velocity_thresholds": sorted_gt_velocity_thresholds
    }

def calculate_metrics_for_scene_hdf5( # Renamed
    gt_scene_hdf5_path: str,         # Changed
    mdet_scene_hdf5_path: str,       # Changed
    eval_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    scene_name_for_log = os.path.basename(mdet_scene_hdf5_path) # Use hdf5 path for logging
    # print(f"\nDEBUG SCENE START (HDF5): {os.path.basename(gt_scene_hdf5_path)} vs {scene_name_for_log}")

    gt_data_h5_file: Optional[h5py.File] = None
    mdet_data_h5_file: Optional[h5py.File] = None

    try:
        gt_data_h5_file = h5py.File(gt_scene_hdf5_path, 'r')
        mdet_data_h5_file = h5py.File(mdet_scene_hdf5_path, 'r')

        # --- Pre-load all necessary arrays from HDF5 files ---
        # Slicing with [:] loads the data into a NumPy array in memory.
        # For scalar datasets (like scene_token or _config_json_str if saved as scalar), use [()]
        
        # GT arrays
        gt_sweep_lidar_sd_tokens_arr = gt_data_h5_file['sweep_lidar_sd_tokens'][:]
        gt_point_labels_indices_arr = gt_data_h5_file['gt_point_labels_indices'][:]
        all_gt_point_labels_arr = gt_data_h5_file['all_gt_point_labels'][:]
        # scene_token_gt = gt_data_h5_file['scene_token'][()] # Example if needed

        # MDet arrays
        mdet_sweep_lidar_sd_tokens_arr = mdet_data_h5_file['sweep_lidar_sd_tokens'][:]
        points_predictions_indices_arr = mdet_data_h5_file['points_predictions_indices'][:]
        all_points_predictions_arr = mdet_data_h5_file['all_points_predictions'][:]
        # scene_token_mdet = mdet_data_h5_file['scene_token'][()] # Example if needed

        mdet_config_loaded_dict = None
        if '_config_json_str' in mdet_data_h5_file:
            config_str_data = mdet_data_h5_file['_config_json_str'][()]
            if isinstance(config_str_data, bytes):
                config_str = config_str_data.decode('utf-8')
            else: # Assumed to be str
                config_str = str(config_str_data)
            mdet_config_loaded_dict = json.loads(config_str)
        # --- End pre-loading arrays ---

        # --- Determine M-Detector's range parameters ---
        default_min_range = eval_params.get('mdet_min_point_range_meters', 1.0)
        default_max_range = eval_params.get('mdet_max_point_range_meters', 80.0)
        mdet_min_range = default_min_range
        mdet_max_range = default_max_range
        loaded_from_config_successfully = False # Renamed for clarity

        if mdet_config_loaded_dict:
            try:
                filtering_config = mdet_config_loaded_dict.get('filtering')
                if filtering_config and isinstance(filtering_config, dict):
                    min_r_from_cfg = filtering_config.get('min_point_range_meters')
                    max_r_from_cfg = filtering_config.get('max_point_range_meters')
                    if min_r_from_cfg is not None and max_r_from_cfg is not None:
                        mdet_min_range = float(min_r_from_cfg)
                        mdet_max_range = float(max_r_from_cfg)
                        # print(f"  Successfully loaded MDet range params from HDF5 config: min={mdet_min_range:.2f}m, max={mdet_max_range:.2f}m")
                        loaded_from_config_successfully = True
                    # else:
                        # print(f"  MDet config in HDF5 found, but 'min_point_range_meters' or 'max_point_range_meters' missing in 'filtering' dict.")
                # else:
                    # print(f"  MDet config in HDF5 found, but 'filtering' key missing or not a dictionary.")
            except Exception as e_cfg: # Catch any error during parsing of the loaded dict
                print(f"  Warning: Error processing MDet config dict from HDF5 for '{scene_name_for_log}'. Error: {e_cfg}")
        # else:
            # print(f"  Info: MDet HDF5 for '{scene_name_for_log}' does not contain '_config_json_str' or failed to parse.")

        if not loaded_from_config_successfully:
            mdet_min_range = default_min_range
            mdet_max_range = default_max_range
            # print(f"  INFO: Using fallback/default MDet range params from eval_params: min={mdet_min_range:.2f}m, max={mdet_max_range:.2f}m")
        # --- End range parameter determination ---

        # --- Find common sweeps based on lidar_sd_token ---
        # Ensure tokens are bytes for set operations if they were loaded as bytes
        gt_token_to_idx: Dict[bytes, int] = {token: i for i, token in enumerate(gt_sweep_lidar_sd_tokens_arr)}
        mdet_token_to_idx: Dict[bytes, int] = {token: i for i, token in enumerate(mdet_sweep_lidar_sd_tokens_arr)}
        common_sd_tokens_bytes = set(gt_token_to_idx.keys()) & set(mdet_token_to_idx.keys())

        if not common_sd_tokens_bytes:
            # print(f"  No common sweep tokens found between GT and MDet for scene '{scene_name_for_log}'. Skipping.")
            return None # Or an error dict

        common_sweep_indices: List[Tuple[int, int]] = []
        for token_bytes in common_sd_tokens_bytes:
            common_sweep_indices.append((gt_token_to_idx[token_bytes], mdet_token_to_idx[token_bytes]))
        common_sweep_indices.sort(key=lambda x: x[0]) # Sort by GT index
        # print(f"  Found {len(common_sweep_indices)} common sweeps for scene '{scene_name_for_log}'.")

        # --- The rest of the logic is identical to your NPZ version ---
        # as it operates on the pre-loaded NumPy arrays.
        scene_tp, scene_fp, scene_fn, scene_tn = 0, 0, 0, 0
        scene_total_gt_points_in_range = 0
        scene_total_mdet_points = 0 # Renamed for clarity
        processed_sweeps_count_scene = 0

        for sweep_idx_gt, sweep_idx_mdet in common_sweep_indices:
            gt_start, gt_end = gt_point_labels_indices_arr[sweep_idx_gt], gt_point_labels_indices_arr[sweep_idx_gt + 1]
            mdet_start, mdet_end = points_predictions_indices_arr[sweep_idx_mdet], points_predictions_indices_arr[sweep_idx_mdet + 1]

            gt_sweep_labels_structured_full = all_gt_point_labels_arr[gt_start:gt_end]
            mdet_sweep_preds_structured = all_points_predictions_arr[mdet_start:mdet_end]
            
            # current_sweep_sd_token_str = gt_sweep_lidar_sd_tokens_arr[sweep_idx_gt].decode('utf-8', 'ignore')

            if gt_sweep_labels_structured_full.shape[0] > 0:
                try:
                    gt_points_sensor_for_filter = np.stack((
                        gt_sweep_labels_structured_full['x_sensor'],
                        gt_sweep_labels_structured_full['y_sensor'],
                        gt_sweep_labels_structured_full['z_sensor']
                    ), axis=-1)
                except ValueError: # Missing keys
                    # print(f"      ERROR for sweep {current_sweep_sd_token_str}: Missing 'x_sensor'/'y_sensor'/'z_sensor' in GT labels. Cannot filter.")
                    continue
                gt_ranges = np.linalg.norm(gt_points_sensor_for_filter, axis=1)
                gt_range_mask = (gt_ranges >= mdet_min_range) & (gt_ranges <= mdet_max_range)
                filtered_gt_labels_for_comparison = gt_sweep_labels_structured_full[gt_range_mask]
            else:
                filtered_gt_labels_for_comparison = gt_sweep_labels_structured_full

            sweep_metrics_results = calculate_point_metrics_for_sweep(
                gt_sweep_labels=filtered_gt_labels_for_comparison,
                pred_sweep_labels=mdet_sweep_preds_structured,
                eval_params=eval_params
            )

            if sweep_metrics_results and not sweep_metrics_results.get("Error_Msg"):
                scene_tp += sweep_metrics_results.get('tp', 0)
                scene_fp += sweep_metrics_results.get('fp', 0)
                scene_fn += sweep_metrics_results.get('fn', 0)
                scene_tn += sweep_metrics_results.get('tn', 0)
                scene_total_gt_points_in_range += sweep_metrics_results.get('num_gt_points', 0)
                scene_total_mdet_points += sweep_metrics_results.get('num_pred_points', 0)
                processed_sweeps_count_scene += 1
            # else: # Error logging already in your original, can be kept or simplified
                # error_msg = sweep_metrics_results.get("Error_Msg", "Unknown error") if sweep_metrics_results else "calc_point_metrics returned None"
                # print(f"      Skipping metrics for sweep {current_sweep_sd_token_str}: {error_msg}")


        if processed_sweeps_count_scene == 0:
            # print(f"  No sweeps successfully processed for scene '{scene_name_for_log}'. Cannot calculate scene metrics.")
            return {"scene_name": scene_name_for_log, "processed_sweeps": 0, "error": "No sweeps processed"}

        precision = scene_tp / (scene_tp + scene_fp) if (scene_tp + scene_fp) > 0 else 0.0
        recall = scene_tp / (scene_tp + scene_fn) if (scene_tp + scene_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (scene_tp + scene_tn) / (scene_tp + scene_fp + scene_fn + scene_tn) if (scene_tp + scene_fp + scene_fn + scene_tn) > 0 else 0.0
        scene_denominator_iou = scene_tp + scene_fp + scene_fn
        scene_iou_dynamic = scene_tp / scene_denominator_iou if scene_denominator_iou > 0 else 0.0

        # if scene_total_gt_points_in_range != scene_total_mdet_points and scene_total_gt_points_in_range > 0 :
        #     print(f"  Warning for scene '{scene_name_for_log}': Total filtered GT points ({scene_total_gt_points_in_range}) "
        #           f"does not match total MDet points ({scene_total_mdet_points}) across processed sweeps.")

        scene_summary_stats = {
            "scene_name": scene_name_for_log,
            "tp": scene_tp, "fp": scene_fp, "fn": scene_fn, "tn": scene_tn,
            "precision": precision, "recall": recall, "f1_score": f1_score, "accuracy": accuracy,
            "iou_dynamic": scene_iou_dynamic,
            "total_gt_points_in_range": scene_total_gt_points_in_range,
            "total_mdet_points_processed": scene_total_mdet_points,
            "processed_sweeps": processed_sweeps_count_scene,
            "mdet_min_range_used": mdet_min_range,
            "mdet_max_range_used": mdet_max_range
        }
        # print(f"  Scene Summary (HDF5) for '{scene_name_for_log}': P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}, IoU_dyn={scene_iou_dynamic:.3f} ({processed_sweeps_count_scene} sweeps)")
        return scene_summary_stats

    except FileNotFoundError:
        print(f"  Error: One or both HDF5 files not found. GT: '{gt_scene_hdf5_path}', MDet: '{mdet_scene_hdf5_path}'")
        return None
    except KeyError as e:
        print(f"  Error: Missing key in HDF5 file for scene '{scene_name_for_log}': {e}")
        return None
    except Exception as e:
        print(f"  Error processing scene '{scene_name_for_log}' with HDF5: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed errors during debugging
        return None
    finally:
        if gt_data_h5_file:
            gt_data_h5_file.close()
        if mdet_data_h5_file:
            mdet_data_h5_file.close()


def calculate_metrics_for_experiment_hdf5( # Renamed
    mdet_experiment_dir: str, # This dir now contains HDF5 mdet files
    gt_labels_base_dir: str,  # This dir now contains HDF5 gt files
    eval_params: Dict
) -> Optional[Dict[str, Any]]:
    experiment_tp, experiment_fp, experiment_fn, experiment_tn = 0,0,0,0
    total_gt_points_exp = 0
    scenes_processed_count = 0
    experiment_scene_results = []

    if not os.path.isdir(mdet_experiment_dir):
        print(f"MDet experiment directory not found: {mdet_experiment_dir}")
        return None
    if not os.path.isdir(gt_labels_base_dir):
        print(f"GT labels base directory not found: {gt_labels_base_dir}")
        return None

    # Look for .h5 files now
    mdet_scene_files = sorted([f for f in os.listdir(mdet_experiment_dir) if f.startswith("mdet_results_") and f.endswith(".h5")])

    if not mdet_scene_files:
        print(f"No MDet result files (e.g., mdet_results_scene-xxxx.h5) found in {mdet_experiment_dir}")
        return None

    # print(f"Processing experiment (HDF5) in: {os.path.basename(mdet_experiment_dir)} ({len(mdet_scene_files)} scenes)")
    for mdet_file_name in tqdm(mdet_scene_files, desc="  Scenes (HDF5)"):
        mdet_scene_hdf5_path = os.path.join(mdet_experiment_dir, mdet_file_name)
        scene_name_key = mdet_file_name.replace("mdet_results_", "").replace(".h5","")
        gt_scene_hdf5_path = os.path.join(gt_labels_base_dir, f"gt_point_labels_{scene_name_key}.h5") # Look for .h5 GT file
        
        if not os.path.exists(gt_scene_hdf5_path):
            # print(f"  Skipping {scene_name_key}: GT HDF5 file not found at {gt_scene_hdf5_path}")
            continue
        
        scene_metrics = calculate_metrics_for_scene_hdf5( # Call the HDF5 version
            gt_scene_hdf5_path, mdet_scene_hdf5_path, eval_params
        )
        if scene_metrics and not scene_metrics.get("error"):
            experiment_scene_results.append(scene_metrics)
            experiment_tp += scene_metrics.get("tp", 0)
            experiment_fp += scene_metrics.get("fp", 0)
            experiment_fn += scene_metrics.get("fn", 0)
            experiment_tn += scene_metrics.get("tn", 0)
            total_gt_points_exp += scene_metrics.get("total_gt_points_in_range", 0)
            scenes_processed_count += 1
        # elif scene_metrics and scene_metrics.get("error"):
            # print(f"  Scene {scene_name_key} (HDF5) processed with error: {scene_metrics.get('error')}")

    if scenes_processed_count == 0:
        print(f"No scenes were successfully processed for experiment {os.path.basename(mdet_experiment_dir)} using HDF5.")
        return None

    exp_precision = experiment_tp / (experiment_tp + experiment_fp) if (experiment_tp + experiment_fp) > 0 else 0.0
    exp_recall = experiment_tp / (experiment_tp + experiment_fn) if (experiment_tp + experiment_fn) > 0 else 0.0
    exp_f1 = 2 * (exp_precision * exp_recall) / (exp_precision + exp_recall) if (exp_precision + exp_recall) > 0 else 0.0
    exp_accuracy = (experiment_tp + experiment_tn) / (experiment_tp + experiment_fp + experiment_fn + experiment_tn) if (experiment_tp + experiment_fp + experiment_fn + experiment_tn) > 0 else 0.0
    exp_denominator_iou = experiment_tp + experiment_fp + experiment_fn
    exp_iou_dynamic = experiment_tp / exp_denominator_iou if exp_denominator_iou > 0 else 0.0

    return {
        "experiment_id": os.path.basename(mdet_experiment_dir),
        "evaluation_params": eval_params,
        "TP": experiment_tp, "FP": experiment_fp, "FN": experiment_fn, "TN": experiment_tn,
        "Precision": exp_precision, "Recall": exp_recall, "F1": exp_f1, "Accuracy": exp_accuracy,
        "N_GT_Points_Experiment": total_gt_points_exp,
        "num_scenes_total_in_dir": len(mdet_scene_files),
        "num_scenes_successfully_evaluated": scenes_processed_count,
        "overall_iou_dynamic": exp_iou_dynamic,
        "per_scene_details": experiment_scene_results,
        "file_format_used": "hdf5" # Added for clarity
    }

# --- Update the ROC generation function to use HDF5 ---
# The _calculate_tfpn_for_sweep_from_preloaded remains the same.
# The main change is in the pre-loading phase of generate_roc_pr_data_gt_velocity_variant_hdf5

def _calculate_tfpn_for_sweep_from_preloaded( # This helper is fine
    filtered_gt_labels_for_comparison: np.ndarray,
    mdet_sweep_preds_structured: np.ndarray,
    current_gt_velocity_threshold: float,
    eval_params_base: Dict
) -> Dict[str, int]:
    if filtered_gt_labels_for_comparison.shape[0] != mdet_sweep_preds_structured.shape[0]:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, "error": 1}
    if filtered_gt_labels_for_comparison.shape[0] == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    try:
        gt_vels_for_dyn_status = np.stack((
            filtered_gt_labels_for_comparison['velocity_x'],
            filtered_gt_labels_for_comparison['velocity_y']
        ), axis=-1)
        gt_is_dyn = get_gt_dynamic_status(gt_vels_for_dyn_status, current_gt_velocity_threshold)
    except ValueError: return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, "error": 1}

    mdet_label_val = eval_params_base.get('mdet_label_val', eval_params_base.get('mdet_dynamic_label_value',0))
    try:
        pred_is_dyn = get_pred_dynamic_status(mdet_sweep_preds_structured['mdet_label'], mdet_label_val)
    except ValueError: return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, "error": 1}

    tp = np.sum(gt_is_dyn & pred_is_dyn)
    fp = np.sum(~gt_is_dyn & pred_is_dyn)
    fn = np.sum(gt_is_dyn & ~pred_is_dyn)
    tn = np.sum(~gt_is_dyn & ~pred_is_dyn)
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}


def load_config_from_hdf5(hdf5_path: str) -> Optional[Dict]:
    """Loads a configuration dictionary saved as a JSON string from an HDF5 file."""
    try:
        with h5py.File(hdf5_path, 'r') as hf:
            if '_config_json_str' in hf:
                # h5py reads scalar string datasets often as bytes or np.string_
                config_str_data = hf['_config_json_str'][()] # Use [()] to get scalar
                if isinstance(config_str_data, bytes):
                    config_str = config_str_data.decode('utf-8')
                elif isinstance(config_str_data, str):
                    config_str = config_str_data
                else: # Should ideally be bytes or str if saved correctly
                    print(f"Warning: _config_json_str in {hdf5_path} is of unexpected type: {type(config_str_data)}. Attempting to convert.")
                    config_str = str(config_str_data)
                return json.loads(config_str)
            # Fallback for older format if config was saved as a pickled object array (less likely in HDF5)
            # If you had a 'config_object' in NPZ that was a pickled dict,
            # saving it to HDF5 might have been as a compound type or as a string representation.
            # For now, we'll assume it's primarily '_config_json_str'.
            # If you have a specific way 'config_object' was saved to HDF5, this part might need adjustment.
            print(f"Warning: Configuration key '_config_json_str' not found in HDF5: {hdf5_path}")
            return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading config from HDF5 {hdf5_path}: {e}")
        return None

def generate_roc_pr_data_gt_velocity_variant_hdf5( # Renamed
    mdet_experiment_dir: str, # Expects HDF5 files here
    gt_labels_base_dir: str,  # Expects HDF5 files here
    eval_params_base: Dict,
    gt_velocity_thresholds_for_roc: Union[List[float], np.ndarray]
) -> Optional[Dict[str, List[float]]]:
    """
    Generates data for ROC-like and PR-like curves by varying the
    GROUND TRUTH velocity threshold. Data is loaded once from HDF5 files.
    """
    # print(f"Pre-loading all experiment data for ROC (HDF5) from: {os.path.basename(mdet_experiment_dir)}...")
    all_sweeps_data_for_roc: List[Dict[str, np.ndarray]] = []
    mdet_scene_files = sorted([f for f in os.listdir(mdet_experiment_dir) if f.startswith("mdet_results_") and f.endswith(".h5")])

    if not mdet_scene_files:
        print(f"No MDet HDF5 result files found in {mdet_experiment_dir}")
        return None

    for mdet_file_name in tqdm(mdet_scene_files, desc="  Loading Scenes (HDF5) for ROC"):
        mdet_scene_hdf5_path = os.path.join(mdet_experiment_dir, mdet_file_name)
        scene_name_key = mdet_file_name.replace("mdet_results_", "").replace(".h5","")
        gt_scene_hdf5_path = os.path.join(gt_labels_base_dir, f"gt_point_labels_{scene_name_key}.h5")

        if not os.path.exists(gt_scene_hdf5_path):
            continue
        
        gt_h5_file: Optional[h5py.File] = None
        mdet_h5_file: Optional[h5py.File] = None
        try:
            gt_h5_file = h5py.File(gt_scene_hdf5_path, 'r')
            mdet_h5_file = h5py.File(mdet_scene_hdf5_path, 'r')

            # Determine M-Detector's range parameters for THIS scene
            default_min_range = eval_params_base.get('mdet_min_point_range_meters', 1.0)
            default_max_range = eval_params_base.get('mdet_max_point_range_meters', 80.0)
            mdet_min_r, mdet_max_r = default_min_range, default_max_range
            
            if '_config_json_str' in mdet_h5_file:
                try:
                    config_str_data = mdet_h5_file['_config_json_str'][()]
                    config_str = config_str_data.decode('utf-8') if isinstance(config_str_data, bytes) else str(config_str_data)
                    mdet_config_loaded = json.loads(config_str)
                    filtering_config = mdet_config_loaded.get('filtering')
                    if filtering_config and isinstance(filtering_config, dict):
                        min_r_cfg = filtering_config.get('min_point_range_meters')
                        max_r_cfg = filtering_config.get('max_point_range_meters')
                        if min_r_cfg is not None: mdet_min_r = float(min_r_cfg)
                        if max_r_cfg is not None: mdet_max_r = float(max_r_cfg)
                except Exception: pass

            gt_tokens_arr = gt_h5_file['sweep_lidar_sd_tokens'][:]
            mdet_tokens_arr = mdet_h5_file['sweep_lidar_sd_tokens'][:]
            gt_indices_arr = gt_h5_file['gt_point_labels_indices'][:]
            mdet_indices_arr = mdet_h5_file['points_predictions_indices'][:]
            all_gt_labels_arr = gt_h5_file['all_gt_point_labels'][:]
            all_mdet_preds_arr = mdet_h5_file['all_points_predictions'][:]

            gt_token_to_idx = {token: i for i, token in enumerate(gt_tokens_arr)}
            mdet_token_to_idx = {token: i for i, token in enumerate(mdet_tokens_arr)}
            common_sd_tokens_bytes = set(gt_token_to_idx.keys()) & set(mdet_token_to_idx.keys())
            
            scene_common_sweep_indices = []
            for token_bytes in common_sd_tokens_bytes:
                scene_common_sweep_indices.append((gt_token_to_idx[token_bytes], mdet_token_to_idx[token_bytes]))
            scene_common_sweep_indices.sort(key=lambda x: x[0])

            for sweep_idx_gt, sweep_idx_mdet in scene_common_sweep_indices:
                gt_start, gt_end = gt_indices_arr[sweep_idx_gt], gt_indices_arr[sweep_idx_gt + 1]
                mdet_start, mdet_end = mdet_indices_arr[sweep_idx_mdet], mdet_indices_arr[sweep_idx_mdet + 1]

                gt_sweep_labels_full = all_gt_labels_arr[gt_start:gt_end]
                mdet_sweep_preds = all_mdet_preds_arr[mdet_start:mdet_end]

                if gt_sweep_labels_full.shape[0] == 0:
                    filtered_gt_for_comp = gt_sweep_labels_full
                else:
                    try:
                        gt_points_sensor = np.stack((
                            gt_sweep_labels_full['x_sensor'], gt_sweep_labels_full['y_sensor'], gt_sweep_labels_full['z_sensor']
                        ), axis=-1)
                        gt_ranges = np.linalg.norm(gt_points_sensor, axis=1)
                        gt_range_mask = (gt_ranges >= mdet_min_r) & (gt_ranges <= mdet_max_r)
                        filtered_gt_for_comp = gt_sweep_labels_full[gt_range_mask]
                    except ValueError: continue

                if filtered_gt_for_comp.shape[0] != mdet_sweep_preds.shape[0]: continue
                if filtered_gt_for_comp.shape[0] > 0:
                    try:
                        gt_xyz = np.stack((filtered_gt_for_comp['x'], filtered_gt_for_comp['y'], filtered_gt_for_comp['z']), axis=-1)
                        pred_xyz = np.stack((mdet_sweep_preds['x'], mdet_sweep_preds['y'], mdet_sweep_preds['z']), axis=-1)
                        coord_tol = eval_params_base.get('coordinate_tolerance', eval_params_base.get('coordinate_tolerance_for_verification', 0.1))
                        if np.any(np.linalg.norm(gt_xyz - pred_xyz, axis=1) > coord_tol): continue
                    except ValueError: continue
                
                all_sweeps_data_for_roc.append({
                    'gt_labels_in_range': filtered_gt_for_comp,
                    'mdet_preds': mdet_sweep_preds
                })
        except Exception as e:
            # print(f"    Error processing HDF5 scene {scene_name_key} during ROC pre-load: {e}")
            pass # Continue to next file
        finally:
            if gt_h5_file: gt_h5_file.close()
            if mdet_h5_file: mdet_h5_file.close()
            
    if not all_sweeps_data_for_roc:
        print("No sweep data successfully pre-loaded for HDF5 ROC generation. Aborting.")
        return None
    # print(f"Finished HDF5 pre-loading for ROC. Total comparable sweeps: {len(all_sweeps_data_for_roc)}")

    tpr_values, fpr_values, precision_values, recall_values = [], [], [], []
    sorted_gt_velocity_thresholds = sorted(list(gt_velocity_thresholds_for_roc))

    # print(f"Calculating ROC/PR points (HDF5) by varying GT velocity threshold...")
    for current_gt_vel_thresh in tqdm(sorted_gt_velocity_thresholds, desc="  GT Vel Thresh (HDF5)"):
        exp_tp_at_thresh, exp_fp_at_thresh, exp_fn_at_thresh, exp_tn_at_thresh = 0, 0, 0, 0
        for sweep_data in all_sweeps_data_for_roc:
            tfpn_results = _calculate_tfpn_for_sweep_from_preloaded(
                sweep_data['gt_labels_in_range'], sweep_data['mdet_preds'],
                current_gt_vel_thresh, eval_params_base
            )
            if not tfpn_results.get("error"):
                exp_tp_at_thresh += tfpn_results['tp']
                exp_fp_at_thresh += tfpn_results['fp']
                exp_fn_at_thresh += tfpn_results['fn']
                exp_tn_at_thresh += tfpn_results['tn']

        total_actual_positives = exp_tp_at_thresh + exp_fn_at_thresh
        total_actual_negatives = exp_fp_at_thresh + exp_tn_at_thresh
        tpr = exp_tp_at_thresh / total_actual_positives if total_actual_positives > 0 else 0.0
        fpr = exp_fp_at_thresh / total_actual_negatives if total_actual_negatives > 0 else (1.0 if exp_fp_at_thresh > 0 else 0.0)
        prec = exp_tp_at_thresh / (exp_tp_at_thresh + exp_fp_at_thresh) if (exp_tp_at_thresh + exp_fp_at_thresh) > 0 else (1.0 if exp_tp_at_thresh == 0 and exp_fp_at_thresh == 0 else 0.0)
        rec = tpr
        tpr_values.append(tpr); fpr_values.append(fpr); precision_values.append(prec); recall_values.append(rec)
        
    return {"fpr": fpr_values, "tpr": tpr_values, "precision": precision_values, "recall": recall_values, "gt_velocity_thresholds": sorted_gt_velocity_thresholds}