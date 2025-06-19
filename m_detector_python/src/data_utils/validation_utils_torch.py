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


def calculate_metrics_for_optuna_trial_in_memory(
    mdet_results_dict: dict,
    eval_params: Dict[str, Any],
    nusc: NuScenes,
    scene_idx: int
) -> Dict[str, Any]:
    """
    Calculates metrics from a dictionary containing sparse GT and predictions.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    try:
        validation_data_list = mdet_results_dict.get('validation_data')
        mdet_label_val = eval_params["mdet_dynamic_label_value"]

        for sweep_data in validation_data_list:
            # 1. Get predictions, the index map, and sparse GT for the sweep
            pred_labels_np = sweep_data['predictions']
            original_indices_map = sweep_data['original_indices_map']
            gt_sparse_indices = sweep_data['gt_sparse_indices']
            
            num_filtered_points = len(pred_labels_np)
            if num_filtered_points == 0:
                continue

            # 2. Create the dense prediction mask for filtered points
            pred_is_dyn = torch.from_numpy(pred_labels_np).to(device) == mdet_label_val

            # 3. Reconstruct the dense GT mask for filtered points
            gt_is_dyn = torch.zeros(num_filtered_points, dtype=torch.bool, device=device)
            
            # Create a reverse lookup from original index to filtered index
            # This is the most efficient way to do the mapping
            map_tensor = torch.from_numpy(original_indices_map.copy()).long().to(device) 
            gt_sparse_tensor = torch.from_numpy(gt_sparse_indices.copy()).long().to(device)
            
            # Find which of our filtered points appear in the sparse GT list
            # `torch.isin` is highly optimized for this
            is_in_gt_mask = torch.isin(map_tensor, gt_sparse_tensor)
            
            # Set the corresponding positions in our dense GT tensor to True
            gt_is_dyn[is_in_gt_mask] = True

            # 4. Calculate metrics for the sweep and accumulate
            sweep_metrics = calculate_metrics(pred_is_dyn, gt_is_dyn)
            total_tp += sweep_metrics['tp']
            total_fp += sweep_metrics['fp']
            total_fn += sweep_metrics['fn']
            total_tn += sweep_metrics['tn']

        final_metrics = {'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn}
        final_metrics['scene_name'] = nusc.scene[scene_idx]['name']
        return final_metrics

    except Exception as e:
        import traceback
        return {"error": f"Exception in calculate_metrics: {e}\n{traceback.format_exc()}"}


def calculate_metrics(
    pred_is_dyn: torch.Tensor,
    gt_is_dyn: torch.Tensor,
) -> Dict[str, int]:
    """
    Core metric calculation logic. Takes two aligned boolean tensors and returns metrics.
    """
    tp = torch.sum(gt_is_dyn & pred_is_dyn).item()
    fp = torch.sum(~gt_is_dyn & pred_is_dyn).item()
    fn = torch.sum(gt_is_dyn & ~pred_is_dyn).item()
    tn = torch.sum(~gt_is_dyn & ~pred_is_dyn).item()
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
