# src/data_utils/validation_utils_torch.py

import os
import numpy as np
import torch
from typing import Dict, Any, Optional
import re
import sys
from nuscenes.nuscenes import NuScenes

# --- Custom Imports ---
from ..config_loader import MDetectorConfigAccessor
from ..core.constants import POINT_LABEL_DTYPE

# --- Helper Function (Unchanged) ---
def structured_np_to_dict_of_tensors(
    structured_array: np.ndarray,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Converts a NumPy structured array to a dictionary of PyTorch tensors on a specific device."""
    fields_to_convert = [name for name, dtype in structured_array.dtype.fields.items() if dtype[0].kind not in ('S', 'U')]
    return {
        name: torch.from_numpy(structured_array[name].copy()).to(device)
        for name in fields_to_convert
    }



def get_predictions_from_dict(mdet_results_dict: dict, num_expected_points: int, 
                              eval_params: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    Decodes M-Detector results from a dictionary and returns a boolean tensor of dynamic predictions.
    This is the in-memory version of get_predictions_from_file.
    """
    pred_is_dyn = torch.zeros(num_expected_points, dtype=torch.bool, device=device)
    mdet_label_val = eval_params["mdet_dynamic_label_value"]

    if 'all_points_labels' in mdet_results_dict:
        all_mdet_labels_np = mdet_results_dict['all_points_labels']
        if num_expected_points != all_mdet_labels_np.shape[0]:
            raise ValueError(f"Point count mismatch. Expected: {num_expected_points}, MDet dict has: {all_mdet_labels_np.shape[0]}")
        pred_is_dyn = (torch.from_numpy(all_mdet_labels_np).to(device) == mdet_label_val)
    elif 'sparse_point_indices' in mdet_results_dict:
        sparse_indices = mdet_results_dict['sparse_point_indices']
        if sparse_indices.size > 0:
            pred_is_dyn[torch.from_numpy(sparse_indices).long().to(device)] = True
    else:
        raise ValueError("Invalid M-Detector result dict format. Missing 'all_points_labels' or 'sparse_point_indices'.")
        
    return pred_is_dyn

def calculate_metrics_for_scene_from_memory(
    gt_scene_pt_path: str,
    mdet_results_dict: dict, # Takes the result dict directly
    eval_params: Dict[str, Any],
    device: torch.device,
    skip_frames: int,
    max_frames: int,
    num_sweeps_for_initial_map: int
) -> Dict[str, Any]:
    """
    Orchestrates loading GT data from a file and calculating metrics against predictions passed in memory.
    """
    try:
        gt_data = torch.load(gt_scene_pt_path, map_location='cpu', weights_only=False)
        gt_labels_all_sweeps_np = gt_data['point_labels']
        gt_sweep_indices = gt_data['sweep_indices']

        # This slicing logic must remain to correctly align with the GT file
        num_sweeps_in_gt_file = len(gt_sweep_indices) - 1
        processing_start_sweep_idx = min(skip_frames, num_sweeps_in_gt_file)
        processing_end_sweep_idx = num_sweeps_in_gt_file
        if max_frames is not None and max_frames >= 0:
            processing_end_sweep_idx = min(processing_start_sweep_idx + max_frames, num_sweeps_in_gt_file)
        first_output_sweep_idx_relative = num_sweeps_for_initial_map - 1
        if first_output_sweep_idx_relative < 0: first_output_sweep_idx_relative = 0
        first_output_sweep_idx_absolute = processing_start_sweep_idx + first_output_sweep_idx_relative
        output_end_sweep_idx_absolute = processing_end_sweep_idx
        
        if first_output_sweep_idx_absolute >= output_end_sweep_idx_absolute:
            start_point_idx, end_point_idx = 0, 0
        else:
            start_point_idx = gt_sweep_indices[first_output_sweep_idx_absolute]
            end_point_idx = gt_sweep_indices[output_end_sweep_idx_absolute]

        gt_labels_eval_np = gt_labels_all_sweeps_np[start_point_idx:end_point_idx]
        
        pred_is_dyn_eval = get_predictions_from_dict(
            mdet_results_dict, gt_labels_eval_np.shape[0], eval_params, device
        )
        
        metrics = calculate_metrics(gt_labels_eval_np, pred_is_dyn_eval, eval_params, device)
        metrics["device_used"] = str(device)
        return metrics

    except Exception as e:
        import traceback
        return {"error": f"Exception in calculate_metrics_for_scene_from_memory: {e}\n{traceback.format_exc()}"}

def calculate_metrics_for_optuna_trial_in_memory(
    mdet_results_dict: dict,
    eval_params: Dict[str, Any],
    nusc: NuScenes, # It now receives the handle directly
    scene_idx: int
) -> Dict[str, Any]:
    """
    Calculates metrics from a dictionary that already contains both
    predictions and ground truth. The nusc object is required to get scene metadata.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # This function no longer needs to find GT files.
        # It just does the final calculation.
        pred_labels_np = mdet_results_dict['predictions']
        gt_labels_np = mdet_results_dict['ground_truth']

        mdet_label_val = eval_params["mdet_dynamic_label_value"]
        pred_is_dyn = torch.from_numpy(pred_labels_np).to(device) == mdet_label_val
        
        metrics = calculate_metrics(gt_labels_np, pred_is_dyn, eval_params, device)
        
        # Add metadata for context
        metrics['scene_name'] = nusc.scene[scene_idx]['name']
        return metrics
    except Exception as e:
        import traceback
        return {"error": f"Exception in calculate_metrics: {e}\n{traceback.format_exc()}"}

def get_predictions_from_file(mdet_scene_pt_path: str, num_expected_points: int, 
                              eval_params: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    Loads M-Detector results from a .pt file and returns a boolean tensor of dynamic predictions.
    Handles both 'full' and 'sparse_dynamic' formats.
    """
    mdet_results_dict = torch.load(mdet_scene_pt_path, map_location='cpu', weights_only=False)
    pred_is_dyn = torch.zeros(num_expected_points, dtype=torch.bool, device=device)
    mdet_label_val = eval_params["mdet_dynamic_label_value"]

    if 'all_points_labels' in mdet_results_dict:
        all_mdet_labels_np = mdet_results_dict['all_points_labels']
        if num_expected_points != all_mdet_labels_np.shape[0]:
            raise ValueError(f"Point count mismatch. Expected: {num_expected_points}, MDet file has: {all_mdet_labels_np.shape[0]}")
        pred_is_dyn = (torch.from_numpy(all_mdet_labels_np).to(device) == mdet_label_val)
    elif 'sparse_point_indices' in mdet_results_dict:
        sparse_indices = mdet_results_dict['sparse_point_indices']
        if sparse_indices.size > 0:
            pred_is_dyn[torch.from_numpy(sparse_indices).long().to(device)] = True
    else:
        raise ValueError("Invalid M-Detector result file format. Missing 'all_points_labels' or 'sparse_point_indices'.")
        
    return pred_is_dyn

def calculate_metrics(
    gt_labels_np: np.ndarray,
    pred_is_dyn: torch.Tensor,
    eval_params: Dict[str, Any],
    device: torch.device,
    batch_size: int = 2_000_000
) -> Dict[str, Any]:
    """
    Core metric calculation logic. Takes GT and prediction tensors and returns metrics.
    """
    num_eval_points = gt_labels_np.shape[0]
    if num_eval_points == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    if num_eval_points != pred_is_dyn.shape[0]:
        raise ValueError(f"GT and Prediction tensor sizes do not match: {num_eval_points} vs {pred_is_dyn.shape[0]}")

    gt_vel_thresh = eval_params['gt_velocity_threshold']
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for i in range(0, num_eval_points, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, num_eval_points)
        
        gt_batch_np = gt_labels_np[start_idx:end_idx]
        pred_batch_tensor = pred_is_dyn[start_idx:end_idx]

        gt_is_valid_instance_batch = torch.from_numpy(gt_batch_np['instance_token'] != b'').to(device)
        gt_labels_batch_tensor = structured_np_to_dict_of_tensors(gt_batch_np, device)

        gt_speed_sq = gt_labels_batch_tensor['velocity_x']**2 + gt_labels_batch_tensor['velocity_y']**2
        gt_is_dyn_batch = (gt_speed_sq >= gt_vel_thresh**2) & gt_is_valid_instance_batch
        
        total_tp += torch.sum(gt_is_dyn_batch & pred_batch_tensor).item()
        total_fp += torch.sum(~gt_is_dyn_batch & pred_batch_tensor).item()
        total_fn += torch.sum(gt_is_dyn_batch & ~pred_batch_tensor).item()
        total_tn += torch.sum(~gt_is_dyn_batch & ~pred_batch_tensor).item()

    return {'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn}

def calculate_metrics_for_scene_from_file(
    gt_scene_pt_path: str,
    mdet_scene_pt_path: str,
    eval_params: Dict[str, Any],
    device: torch.device,
    skip_frames: int,
    max_frames: int,
    num_sweeps_for_initial_map: int
) -> Dict[str, Any]:
    """
    Orchestrates loading data from files and calculating metrics.
    This is the primary function for the metrics notebook.
    """
    print(f"[DEBUG] Slicing with: skip={skip_frames}, max={max_frames}, init_sweeps={num_sweeps_for_initial_map}")
    try:
        gt_data = torch.load(gt_scene_pt_path, map_location='cpu', weights_only=False)
        gt_labels_all_sweeps_np = gt_data['point_labels']
        gt_sweep_indices = gt_data['sweep_indices']

        # 1. Determine the window of sweeps that M-Detector PROCESSED.
        # This logic must be identical to NuScenesProcessor.process_scene.
        num_sweeps_in_gt_file = len(gt_sweep_indices) - 1
        
        processing_start_sweep_idx = min(skip_frames, num_sweeps_in_gt_file)
        processing_end_sweep_idx = num_sweeps_in_gt_file
        if max_frames is not None and max_frames >= 0:
            processing_end_sweep_idx = min(processing_start_sweep_idx + max_frames, num_sweeps_in_gt_file)

        # 2. Determine the window of sweeps that M-Detector produced OUTPUT for.
        # The first output is produced after the initialization phase is complete.
        first_output_sweep_idx_relative = num_sweeps_for_initial_map - 1
        if first_output_sweep_idx_relative < 0:
             first_output_sweep_idx_relative = 0 # Handle edge case

        first_output_sweep_idx_absolute = processing_start_sweep_idx + first_output_sweep_idx_relative
        
        output_end_sweep_idx_absolute = processing_end_sweep_idx
        
        # 3. Get the corresponding point indices from the GT sweep index array.
        if first_output_sweep_idx_absolute >= output_end_sweep_idx_absolute:
            start_point_idx, end_point_idx = 0, 0
        else:
            start_point_idx = gt_sweep_indices[first_output_sweep_idx_absolute]
            end_point_idx = gt_sweep_indices[output_end_sweep_idx_absolute]

        # 4. Slice the GT numpy array to get the exact points that should be in the M-Det output.
        gt_labels_eval_np = gt_labels_all_sweeps_np[start_point_idx:end_point_idx]
        
        pred_is_dyn_eval = get_predictions_from_file(
            mdet_scene_pt_path, gt_labels_eval_np.shape[0], eval_params, device
        )
        
        metrics = calculate_metrics(gt_labels_eval_np, pred_is_dyn_eval, eval_params, device)
        metrics["device_used"] = str(device)
        return metrics

    except Exception as e:
        import traceback
        return {"scene_name": os.path.basename(mdet_scene_pt_path), "error": f"Exception: {e}\n{traceback.format_exc()}"}

# --- Worker Function (Unchanged, it correctly calls the top-level function) ---
def scene_metrics_worker(args_tuple):
    # ... (This function remains unchanged as its inputs/outputs are correct)
    worker_id, gpu_id_for_worker, tuning_name, mdet_scene_pt_path, \
        gt_labels_dir, eval_params, project_root, experiment_config_path, nusc_obj = args_tuple
    
    device = torch.device(f"cuda:{gpu_id_for_worker}") if gpu_id_for_worker is not None and torch.cuda.is_available() else torch.device("cpu")

    if project_root not in sys.path:
         sys.path.append(project_root)

    match = re.search(r'_scene_(\d+)\.pt$', os.path.basename(mdet_scene_pt_path))
    if not match:
        return {"tuning_name": tuning_name, "scene_name": os.path.basename(mdet_scene_pt_path), "error": "Could not parse scene index from .pt file."}
    
    scene_idx = int(match.group(1))
    
    try:
        scene_rec = nusc_obj.scene[scene_idx]
        scene_name_key = scene_rec['name']
        accessor = MDetectorConfigAccessor(experiment_config_path)
        filt_cfg = accessor.get_point_pre_filtering_params()
        proc_cfg = accessor.get_processing_settings()
        init_cfg = accessor.get_initialization_phase_params()
        
        min_r, max_r = filt_cfg['min_range_meters'], filt_cfg['max_range_meters']
        skip_f, max_f = proc_cfg['skip_frames'], proc_cfg['max_frames']
        init_sweeps = init_cfg['num_sweeps_for_initial_map']

        gt_filename = f"gt_point_labels_{scene_name_key}_r{min_r}-{max_r}.pt"
        gt_scene_pt_path = os.path.join(gt_labels_dir, gt_filename)

        if not os.path.exists(gt_scene_pt_path):
            return {"tuning_name": tuning_name, "scene_name": os.path.basename(mdet_scene_pt_path), "error": f"Required GT file not found: {gt_filename}"}
    except Exception as e:
        return {"tuning_name": tuning_name, "scene_name": os.path.basename(mdet_scene_pt_path), "error": f"File lookup failed: {e}"}

    scene_metrics = calculate_metrics_for_scene_from_file(
        gt_scene_pt_path=gt_scene_pt_path,
        mdet_scene_pt_path=mdet_scene_pt_path,
        eval_params=eval_params,
        device=device,
        skip_frames=skip_f,
        max_frames=max_f,
        num_sweeps_for_initial_map=init_sweeps
    )
    
    scene_metrics['tuning_name'] = tuning_name
    return scene_metrics

def calculate_metrics_for_optuna_trial(
    tuning_name: str,
    mdet_scene_pt_path: str,
    gt_labels_dir: str,
    eval_params: Dict[str, Any],
    project_root: str,
    experiment_config_path: str,
    nusc: NuScenes
) -> Dict[str, Any]:
    """
    A direct-callable wrapper for the scene_metrics_worker logic, designed
    to be used within the Optuna Ray task. It bypasses the need for
    multiprocessing.Pool.
    """
    # The worker expects a tuple of arguments, the first two of which are worker/GPU IDs.
    # We can pass dummy values for these as they are not used in the core logic.
    worker_id = 0
    gpu_id_for_worker = 0 
    
    args_tuple = (
        worker_id, gpu_id_for_worker, tuning_name, mdet_scene_pt_path,
        gt_labels_dir, eval_params, project_root, experiment_config_path, nusc
    )
    
    # Call the exact same worker function the notebook uses.
    return scene_metrics_worker(args_tuple)