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
    nusc: NuScenes, # Not strictly used if only relying on HDF5 and sweep_data_dict for token
    sweep_data_dict: Dict, # Primarily for lidar_sd_token
    all_points_global_mdetector: np.ndarray, # M-Detector's input points for this sweep
    gt_labels_scene_hdf5_path: str,
    velocity_threshold: float,
    mdetector_min_range: float, # NEW: M-Detector's min_range for filtering GT
    mdetector_max_range: float  # NEW: M-Detector's max_range for filtering GT
) -> Dict[str, np.ndarray]: # Will return INDICES relative to all_points_global_mdetector

    lidar_sd_token = sweep_data_dict['lidar_sd_token']
    
    # Initialize result for indices
    result_dict = {
        'gt_dynamic_indices': np.array([], dtype=int),
        'gt_static_indices': np.array([], dtype=int),
        'unlabeled_indices': np.array([], dtype=int),
        'error_msg': None,
        'debug_gt_hdf5_raw_count': 0,
        'debug_gt_hdf5_filtered_count': 0,
        'debug_mdet_input_count': all_points_global_mdetector.shape[0]
    }

    if not os.path.exists(gt_labels_scene_hdf5_path):
        result_dict['error_msg'] = f"GT HDF5 file NOT FOUND: {gt_labels_scene_hdf5_path}"
        return result_dict

    point_labels_for_sweep_raw_hdf5: Optional[np.ndarray] = None
    try:
        with h5py.File(gt_labels_scene_hdf5_path, 'r') as hf:
            all_tokens_bytes = hf['sweep_lidar_sd_tokens'][:]
            target_token_bytes = lidar_sd_token.encode('utf-8') if isinstance(lidar_sd_token, str) else lidar_sd_token
            matches = np.array([])
            try:
                matches = np.where(all_tokens_bytes == target_token_bytes)[0]
                if not matches.size > 0: # Try decoding if direct byte match failed
                    all_tokens_str_decoded = [t.decode('utf-8', 'ignore') for t in all_tokens_bytes]
                    matches = np.where(np.array(all_tokens_str_decoded) == (target_token_bytes.decode('utf-8', 'ignore') if isinstance(target_token_bytes, bytes) else target_token_bytes))[0]
            except TypeError: # Fallback for certain h5py/numpy versions with mixed types
                 matches = np.where(all_tokens_bytes == (target_token_bytes.decode('utf-8', 'ignore') if isinstance(target_token_bytes, bytes) else target_token_bytes) )[0]

            if not matches.size > 0:
                result_dict['error_msg'] = f"Sweep token {lidar_sd_token} NOT FOUND in GT HDF5."
                return result_dict
            idx_in_token_list = matches[0]

            indices_array = hf['gt_point_labels_indices'][:]
            start_idx = indices_array[idx_in_token_list]
            if idx_in_token_list + 1 >= len(indices_array):
                result_dict['error_msg'] = f"Index issue for token {lidar_sd_token} in GT HDF5 indices."
                return result_dict
            end_idx = indices_array[idx_in_token_list + 1]
            
            point_labels_for_sweep_raw_hdf5 = hf['all_gt_point_labels'][start_idx:end_idx]
            result_dict['debug_gt_hdf5_raw_count'] = point_labels_for_sweep_raw_hdf5.shape[0]

            if point_labels_for_sweep_raw_hdf5.dtype != POINT_LABEL_DTYPE:
                try:
                    point_labels_for_sweep_raw_hdf5 = point_labels_for_sweep_raw_hdf5.astype(POINT_LABEL_DTYPE)
                except ValueError as ve:
                    result_dict['error_msg'] = f"CRITICAL DTYPE cast failed for GT labels. Error: {ve}."
                    return result_dict
                    
    except Exception as e:
        result_dict['error_msg'] = f"Error loading GT labels from HDF5 {gt_labels_scene_hdf5_path} for sweep {lidar_sd_token}: {e}"
        # import traceback; traceback.print_exc() # Uncomment for full traceback during dev
        return result_dict

    if point_labels_for_sweep_raw_hdf5 is None or point_labels_for_sweep_raw_hdf5.shape[0] == 0:
        result_dict['error_msg'] = "No GT point labels loaded from HDF5 for sweep (empty slice or None)."
        return result_dict
        
    # --- Apply M-Detector's range filtering to the GT points from HDF5 ---
    # This assumes 'x_sensor', 'y_sensor', 'z_sensor' are present in POINT_LABEL_DTYPE
    try:
        gt_points_sensor_frame_from_hdf5 = np.stack((
            point_labels_for_sweep_raw_hdf5['x_sensor'],
            point_labels_for_sweep_raw_hdf5['y_sensor'],
            point_labels_for_sweep_raw_hdf5['z_sensor']
        ), axis=-1)
    except ValueError as e: # Handles missing sensor coordinate fields
        result_dict['error_msg'] = f"Missing sensor coordinates (x_sensor, etc.) in GT HDF5 data for range filtering. Error: {e}"
        return result_dict

    gt_ranges_from_hdf5 = np.linalg.norm(gt_points_sensor_frame_from_hdf5, axis=1)
    gt_range_filter_mask = (gt_ranges_from_hdf5 >= mdetector_min_range) & \
                           (gt_ranges_from_hdf5 <= mdetector_max_range)
    
    point_labels_for_sweep_filtered = point_labels_for_sweep_raw_hdf5[gt_range_filter_mask]
    result_dict['debug_gt_hdf5_filtered_count'] = point_labels_for_sweep_filtered.shape[0]

    # --- CRITICAL CHECK: Point counts must match for index-based association ---
    if point_labels_for_sweep_filtered.shape[0] != all_points_global_mdetector.shape[0]:
        msg = (f"Point count mismatch AFTER range filtering GT HDF5. "
               f"Filtered GT HDF5: {point_labels_for_sweep_filtered.shape[0]}, "
               f"M-Detector input: {all_points_global_mdetector.shape[0]}. "
               f"Index-based matching cannot proceed.")
        print(f"ERROR [get_gt_dynamic_points_for_sweep]: {msg}") # Print error prominently
        result_dict['error_msg'] = msg
        return result_dict # Return with error and empty indices

    # --- If point counts match, proceed with index-based labeling ---
    # The masks are now the same length as all_points_global_mdetector and assumed to be in the same order.
    
    # Optional: Coordinate verification for a few points if counts match
    # num_verify = min(5, all_points_global_mdetector.shape[0])
    # if num_verify > 0:
    #     mdet_sample_coords = all_points_global_mdetector[:num_verify, :3] # Assuming MDet points are Nx3
    #     gt_filtered_sample_coords = np.stack((
    #         point_labels_for_sweep_filtered['x'][:num_verify],
    #         point_labels_for_sweep_filtered['y'][:num_verify],
    #         point_labels_for_sweep_filtered['z'][:num_verify]
    #     ), axis=-1)
    #     coord_diffs = np.linalg.norm(mdet_sample_coords - gt_filtered_sample_coords, axis=1)
    #     print(f"[DEBUG get_gt_dynamic_points_for_sweep] Sample coord diffs after filtering (max 5 pts): {coord_diffs}")
    #     # Add a warning if diffs are too large, as it might indicate ordering is still not right despite count match.

    gt_speed_sq = point_labels_for_sweep_filtered['velocity_x']**2 + point_labels_for_sweep_filtered['velocity_y']**2
    is_valid_instance_mask = (point_labels_for_sweep_filtered['instance_token'] != b'')
    
    dynamic_mask = (gt_speed_sq >= velocity_threshold**2) & is_valid_instance_mask
    static_mask = (~dynamic_mask) & is_valid_instance_mask
    unlabeled_mask = ~is_valid_instance_mask

    # store the points for review
    result_dict['all_gt_point_labels_filtered'] = point_labels_for_sweep_filtered 
    # store the labels
    result_dict['gt_dynamic_indices'] = np.where(dynamic_mask)[0]
    result_dict['gt_static_indices'] = np.where(static_mask)[0]
    result_dict['unlabeled_indices'] = np.where(unlabeled_mask)[0]
    
    return result_dict

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


# --- Core Sweep Metrics Calculation ---
def calculate_point_metrics_for_sweep(
    gt_sweep_labels: np.ndarray,
    pred_sweep_labels: np.ndarray,
    eval_params: Dict
) -> Dict:
    gt_vel_thresh = eval_params.get('gt_velocity_threshold', 0.5) # Renamed from gt_velocity_threshold for consistency
    mdet_label_val_positive_class = eval_params.get('mdet_dynamic_label_value', 0) 
    coordinate_tolerance = eval_params.get('coordinate_tolerance_for_verification', 0.01)

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




def calculate_metrics_for_scene_hdf5(
    gt_scene_hdf5_path: str,    
    mdet_scene_hdf5_path: str,    
    eval_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    scene_name_for_log = os.path.basename(mdet_scene_hdf5_path)
    gt_data_h5_file: Optional[h5py.File] = None
    mdet_data_h5_file: Optional[h5py.File] = None

    # Get the keyframe evaluation flag from eval_params 
    evaluate_only_keyframes = eval_params.get('evaluate_only_keyframes')

    try:
        mdet_config_loaded_dict = load_config_from_hdf5(mdet_scene_hdf5_path)
        gt_data_h5_file = h5py.File(gt_scene_hdf5_path, 'r')
        mdet_data_h5_file = h5py.File(mdet_scene_hdf5_path, 'r')

        # GT arrays
        gt_sweep_lidar_sd_tokens_arr = gt_data_h5_file['sweep_lidar_sd_tokens'][:]
        gt_point_labels_indices_arr = gt_data_h5_file['gt_point_labels_indices'][:]
        all_gt_point_labels_arr = gt_data_h5_file['all_gt_point_labels'][:]
        
        # Load keyframe flags from GT HDF5 
        gt_sweep_is_key_frame_arr: Optional[np.ndarray] = None
        if 'sweep_is_key_frame' in gt_data_h5_file:
            gt_sweep_is_key_frame_arr = gt_data_h5_file['sweep_is_key_frame'][:]
        elif evaluate_only_keyframes:
            # If we need to evaluate only keyframes but the flag isn't in HDF5, we can't proceed.
            print(f"  Error: 'evaluate_only_keyframes' is True, but 'sweep_is_key_frame' dataset "
                  f"not found in GT HDF5: {gt_scene_hdf5_path}")
            # Close files before returning
            if gt_data_h5_file: gt_data_h5_file.close()
            if mdet_data_h5_file: mdet_data_h5_file.close()
            return {"scene_name": scene_name_for_log, "processed_sweeps": 0, "error": "Missing sweep_is_key_frame in GT HDF5"}

        # MDet arrays
        mdet_sweep_lidar_sd_tokens_arr = mdet_data_h5_file['sweep_lidar_sd_tokens'][:]
        points_predictions_indices_arr = mdet_data_h5_file['points_predictions_indices'][:]
        all_points_predictions_arr = mdet_data_h5_file['all_points_predictions'][:]

        # ... (Determine M-Detector's range parameters ...
        default_min_range = eval_params.get('mdet_min_point_range_meters')
        default_max_range = eval_params.get('mdet_max_point_range_meters')
        mdet_min_range = default_min_range
        mdet_max_range = default_max_range
        loaded_from_config_successfully = False
        if mdet_config_loaded_dict:
            if not mdet_config_loaded_dict.get("error_while_saving_config"):
                try:
                    m_detector_cfg = mdet_config_loaded_dict.get('m_detector', {})
                    filtering_config = m_detector_cfg.get('point_pre_filtering')
                    if filtering_config and isinstance(filtering_config, dict):
                        min_r_from_cfg = filtering_config.get('min_range_meters')
                        max_r_from_cfg = filtering_config.get('max_range_meters')
                        if min_r_from_cfg is not None and max_r_from_cfg is not None:
                            mdet_min_range = float(min_r_from_cfg)
                            mdet_max_range = float(max_r_from_cfg)
                            loaded_from_config_successfully = True
                except Exception as e_cfg:
                    print(f"  Warning: Error processing MDet config dict from HDF5 for '{scene_name_for_log}'. Error: {e_cfg}")
        if not loaded_from_config_successfully:
            # print(f"  INFO: Using fallback/default MDet range params from eval_params: min={mdet_min_range:.2f}m, max={mdet_max_range:.2f}m for {scene_name_for_log}")
            pass


        # --- Find common sweeps based on lidar_sd_token ---
        gt_token_to_idx: Dict[bytes, int] = {token: i for i, token in enumerate(gt_sweep_lidar_sd_tokens_arr)}
        mdet_token_to_idx: Dict[bytes, int] = {token: i for i, token in enumerate(mdet_sweep_lidar_sd_tokens_arr)}
        common_sd_tokens_bytes = set(gt_token_to_idx.keys()) & set(mdet_token_to_idx.keys())

        if not common_sd_tokens_bytes:
            if gt_data_h5_file: gt_data_h5_file.close()
            if mdet_data_h5_file: mdet_data_h5_file.close()
            return None

        common_sweep_indices: List[Tuple[int, int]] = []
        for token_bytes in common_sd_tokens_bytes:
            gt_idx = gt_token_to_idx[token_bytes]
            mdet_idx = mdet_token_to_idx[token_bytes]

            # Filter for keyframes if requested 
            if evaluate_only_keyframes:
                if gt_sweep_is_key_frame_arr is not None and gt_idx < len(gt_sweep_is_key_frame_arr):
                    if not gt_sweep_is_key_frame_arr[gt_idx]:
                        continue # Skip this sweep if it's not a keyframe
                else: # Should have been caught if gt_sweep_is_key_frame_arr was None and evaluate_only_keyframes was True
                    print(f"  Warning: Keyframe flag missing for GT sweep index {gt_idx} but evaluate_only_keyframes is True. Skipping sweep.")
                    continue
            common_sweep_indices.append((gt_idx, mdet_idx))

        common_sweep_indices.sort(key=lambda x: x[0])

        scene_tp, scene_fp, scene_fn, scene_tn = 0, 0, 0, 0
        scene_total_gt_points_in_range = 0
        scene_total_mdet_points = 0
        processed_sweeps_count_scene = 0

        desc_suffix = " (Keyframes Only)" if evaluate_only_keyframes else " (All Sweeps)"
        for sweep_idx_gt, sweep_idx_mdet in tqdm(
            common_sweep_indices,
            desc=f"  Sweeps ({scene_name_for_log}{desc_suffix})",
            position=1, leave=True, unit="sweep"
        ):
            gt_start, gt_end = gt_point_labels_indices_arr[sweep_idx_gt], gt_point_labels_indices_arr[sweep_idx_gt + 1]
            mdet_start, mdet_end = points_predictions_indices_arr[sweep_idx_mdet], points_predictions_indices_arr[sweep_idx_mdet + 1]
            gt_sweep_labels_structured_full = all_gt_point_labels_arr[gt_start:gt_end]
            mdet_sweep_preds_structured = all_points_predictions_arr[mdet_start:mdet_end]

            if gt_sweep_labels_structured_full.shape[0] > 0:
                try:
                    gt_points_sensor_for_filter = np.stack((
                        gt_sweep_labels_structured_full['x_sensor'],
                        gt_sweep_labels_structured_full['y_sensor'],
                        gt_sweep_labels_structured_full['z_sensor']
                    ), axis=-1)
                except ValueError:
                    continue # Skip sweep if sensor coords are missing
                gt_ranges = np.linalg.norm(gt_points_sensor_for_filter, axis=1)
                gt_range_mask = (gt_ranges >= mdet_min_range) & (gt_ranges <= mdet_max_range)
                filtered_gt_labels_for_comparison = gt_sweep_labels_structured_full[gt_range_mask]
            else:
                filtered_gt_labels_for_comparison = gt_sweep_labels_structured_full

            sweep_metrics_results = calculate_point_metrics_for_sweep(
                gt_sweep_labels=filtered_gt_labels_for_comparison,
                pred_sweep_labels=mdet_sweep_preds_structured,
                eval_params=eval_params # Pass the full eval_params
            )

            if sweep_metrics_results and not sweep_metrics_results.get("Error_Msg"):
                scene_tp += sweep_metrics_results.get('tp', 0)
                scene_fp += sweep_metrics_results.get('fp', 0)
                scene_fn += sweep_metrics_results.get('fn', 0)
                scene_tn += sweep_metrics_results.get('tn', 0)
                scene_total_gt_points_in_range += sweep_metrics_results.get('num_gt_points', 0)
                scene_total_mdet_points += sweep_metrics_results.get('num_pred_points', 0)
                processed_sweeps_count_scene += 1

        if processed_sweeps_count_scene == 0:
            if gt_data_h5_file: gt_data_h5_file.close()
            if mdet_data_h5_file: mdet_data_h5_file.close()
            return {"scene_name": scene_name_for_log, "processed_sweeps": 0, "error": "No sweeps processed (check keyframe filter or data)"}

        precision = scene_tp / (scene_tp + scene_fp) if (scene_tp + scene_fp) > 0 else 0.0
        recall = scene_tp / (scene_tp + scene_fn) if (scene_tp + scene_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (scene_tp + scene_tn) / (scene_tp + scene_fp + scene_fn + scene_tn) if (scene_tp + scene_fp + scene_fn + scene_tn) > 0 else 0.0
        scene_denominator_iou = scene_tp + scene_fp + scene_fn
        scene_iou_dynamic = scene_tp / scene_denominator_iou if scene_denominator_iou > 0 else 0.0
        
        scene_summary_stats = {
            "scene_name": scene_name_for_log,
            "tp": scene_tp, "fp": scene_fp, "fn": scene_fn, "tn": scene_tn,
            "precision": precision, "recall": recall, "f1_score": f1_score, "accuracy": accuracy,
            "iou_dynamic": scene_iou_dynamic,
            "total_gt_points_in_range": scene_total_gt_points_in_range,
            "total_mdet_points_processed": scene_total_mdet_points,
            "processed_sweeps": processed_sweeps_count_scene,
            "mdet_min_range_used": mdet_min_range,
            "mdet_max_range_used": mdet_max_range,
            "evaluated_only_keyframes": evaluate_only_keyframes 
        }
        return scene_summary_stats

    except FileNotFoundError:
        print(f"  Error: One or both HDF5 files not found. GT: '{gt_scene_hdf5_path}', MDet: '{mdet_scene_hdf5_path}'")
        return None
    except KeyError as e:
        print(f"  Error: Missing key in HDF5 file for scene '{scene_name_for_log}': {e}")
        return None
    except Exception as e:
        print(f"  Error processing scene '{scene_name_for_log}' with HDF5: {e}")
        return None
    finally:
        if gt_data_h5_file: gt_data_h5_file.close()
        if mdet_data_h5_file: mdet_data_h5_file.close()


def calculate_metrics_for_experiment_hdf5(
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