# src/core/m_detector/processing.py

import torch
import numpy as np
from typing import Dict
import logging

from ..depth_image import DepthImage
from ..constants import OcclusionResult

def process_and_label_di(self,  # self is MDetector instance
                         current_di: DepthImage,
                         current_di_idx_in_lib: int
                         ) -> Dict:
    device = self.device
    num_points = current_di.total_points_added_to_di_arrays
    if num_points == 0:
        return {'success': True, 'reason': 'current_di has no points', 'processed_di': current_di}

    all_points_global = current_di.original_points_global_coords
    prelabeled_mask = (current_di.mdet_labels_for_points == OcclusionResult.PRELABELED_STATIC_GROUND.value)
    active_mask = ~prelabeled_mask
    
    final_labels = torch.full((num_points,), OcclusionResult.UNDETERMINED.value, dtype=torch.int8, device=device)
    final_labels[prelabeled_mask] = OcclusionResult.PRELABELED_STATIC_GROUND.value

    if not torch.any(active_mask):
        current_di.mdet_labels_for_points = final_labels
        return {'success': True, 'processed_di': current_di}

    raw_occlusion_results = torch.full(
        (num_points, self.test1_N_historical_DIs),
        OcclusionResult.UNDETERMINED.value, dtype=torch.int8, device=device
    )
    for k in range(self.test1_N_historical_DIs):
        hist_di_idx = current_di_idx_in_lib - 1 - k
        if hist_di_idx < 0: break
        historical_di = self.depth_image_library.get_image_by_index(hist_di_idx)
        if historical_di and historical_di.is_prepared_for_projection():
            raw_occlusion_results[:, k] = self.check_occlusion_batch(all_points_global, historical_di)

    preliminary_labels = raw_occlusion_results[:, 0].clone()
    occluding_counts = torch.sum(raw_occlusion_results == OcclusionResult.OCCLUDING_IMAGE.value, dim=1)
    test4_passed_mask = (occluding_counts >= self.test1_M4_threshold)
    preliminary_labels[test4_passed_mask] = OcclusionResult.OCCLUDING_IMAGE.value
    
    dynamic_after_t1_t4_mask = (preliminary_labels == OcclusionResult.OCCLUDING_IMAGE.value) & active_mask
    if torch.any(dynamic_after_t1_t4_mask):
        points_for_mcc = all_points_global[dynamic_after_t1_t4_mask]
        mcc_results = self.is_map_consistent(points_for_mcc, current_di, current_di.timestamp)
        indices_to_flip = torch.where(dynamic_after_t1_t4_mask)[0][mcc_results]
        preliminary_labels[indices_to_flip] = OcclusionResult.OCCLUDED_BY_IMAGE.value

    # --- Event Test Sequence ---
    labels_after_test2 = self.execute_test2_parallel_motion(preliminary_labels, current_di_idx_in_lib)
    
    # --- NEW: Test 3 Logic ---
    # Identify candidates for Test 3: points that are still UNDETERMINED after Test 2
    test3_candidate_mask = (labels_after_test2 == OcclusionResult.UNDETERMINED.value) & active_mask
    labels_after_test3 = labels_after_test2.clone()

    if torch.any(test3_candidate_mask):
        # Run MCC on these candidates. Only those NOT consistent with the map can be dynamic.
        points_for_t3_mcc = all_points_global[test3_candidate_mask]
        t3_mcc_results = self.is_map_consistent(points_for_t3_mcc, current_di, current_di.timestamp)
        
        # Get the original indices of candidates that were rejected by MCC (i.e., are potentially dynamic)
        t3_mcc_rejected_indices = torch.where(test3_candidate_mask)[0][~t3_mcc_results]

        if t3_mcc_rejected_indices.numel() > 0:
            # Create a temporary label tensor for Test 3, marking only the valid candidates
            test3_input_labels = torch.full_like(labels_after_test2, -1) # Use -1 for non-candidates
            test3_input_labels[t3_mcc_rejected_indices] = OcclusionResult.UNDETERMINED.value
            
            # Run Test 3
            test3_output_labels = self.execute_test3_perpendicular_motion(test3_input_labels, current_di_idx_in_lib)
            
            # Merge the results back into our main label tensor
            newly_dynamic_mask = (test3_output_labels == OcclusionResult.OCCLUDING_IMAGE.value)
            labels_after_test3[newly_dynamic_mask] = OcclusionResult.OCCLUDING_IMAGE.value

    final_labels[active_mask] = labels_after_test3[active_mask]
    current_di.mdet_labels_for_points = final_labels

    label_counts = torch.bincount(final_labels.cpu(), minlength=len(OcclusionResult)).numpy()
    label_counts_dict = {OcclusionResult(i).name: count for i, count in enumerate(label_counts) if count > 0}
    
    self.logger.info(f"Processed DI (TS: {current_di.timestamp}, Idx: {current_di_idx_in_lib}): "
                      f"Labeled {num_points} points. Counts: {label_counts_dict}")

    return {
        'points_labeled': num_points,
        'label_counts': dict(enumerate(label_counts)),
        'success': True,
        'timestamp': current_di.timestamp,
        'processed_di': current_di
    }