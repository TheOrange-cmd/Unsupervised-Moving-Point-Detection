# src/core/m_detector/processing.py

import torch
from typing import Dict
import logging

from ..depth_image import DepthImage
from ..constants import OcclusionResult

# This file is bound to the MDetector class, so `self` refers to an MDetector instance.

def _perform_initial_occlusion_pass(self, current_di: DepthImage, current_di_idx_in_lib: int) -> torch.Tensor:
    """
    Performs the initial occlusion check against historical depth images (Test 1)
    and applies the perpendicular motion heuristic (Test 4).

    Args:
        self: The MDetector instance.
        current_di: The DepthImage object to process.
        current_di_idx_in_lib: The index of the current DI in the library.

    Returns:
        torch.Tensor: A tensor of preliminary labels after Test 1 and Test 4.
    """
    num_points = current_di.total_points_added_to_di_arrays
    all_points_global = current_di.original_points_global_coords

    # Store raw occlusion results against N historical DIs
    raw_occlusion_results = torch.full(
        (num_points, self.test1_N_historical_DIs),
        OcclusionResult.UNDETERMINED.value, dtype=torch.int8, device=self.device
    )
    for k in range(self.test1_N_historical_DIs):
        hist_di_idx = current_di_idx_in_lib - 1 - k
        if hist_di_idx < 0: break
        historical_di = self.depth_image_library.get_image_by_index(hist_di_idx)
        if historical_di and historical_di.is_prepared_for_projection():
            raw_occlusion_results[:, k] = self.check_occlusion_batch(all_points_global, historical_di)

    # Apply Test 4: A point is preliminarily dynamic if it occludes M4 or more of the last N frames.
    preliminary_labels = raw_occlusion_results[:, 0].clone() # Start with the most recent result
    occluding_counts = torch.sum(raw_occlusion_results == OcclusionResult.OCCLUDING_IMAGE.value, dim=1)
    test4_passed_mask = (occluding_counts >= self.test1_M4_threshold)
    preliminary_labels[test4_passed_mask] = OcclusionResult.OCCLUDING_IMAGE.value
    
    return preliminary_labels


def _apply_map_consistency_check(self, labels: torch.Tensor, active_mask: torch.Tensor, current_di: DepthImage) -> torch.Tensor:
    """
    Applies the Map Consistency Check (MCC) to filter out points that are likely
    part of the static map, despite appearing dynamic.

    Args:
        self: The MDetector instance.
        labels: The current tensor of point labels.
        active_mask: A boolean mask of points that are not pre-labeled as ground.
        current_di: The current DepthImage being processed.

    Returns:
        torch.Tensor: The updated label tensor after MCC has been applied.
    """
    # Identify points that are currently labeled as dynamic and are not ground
    dynamic_candidates_mask = (labels == OcclusionResult.OCCLUDING_IMAGE.value) & active_mask
    
    if torch.any(dynamic_candidates_mask):
        points_for_mcc = current_di.original_points_global_coords[dynamic_candidates_mask]
        
        # is_map_consistent returns True for points that ARE consistent with the map (i.e., static)
        mcc_static_results = self.is_map_consistent(points_for_mcc, current_di, current_di.timestamp)
        
        # Find the original indices of the candidates that MCC identified as static
        indices_to_revert = torch.where(dynamic_candidates_mask)[0][mcc_static_results]
        
        # Revert these points to OCCLUDED_BY_IMAGE, as they are likely static map points
        labels[indices_to_revert] = OcclusionResult.OCCLUDED_BY_IMAGE.value
        
    return labels


def _run_event_test_sequence(self, labels: torch.Tensor, active_mask: torch.Tensor, current_di: DepthImage, current_di_idx_in_lib: int) -> torch.Tensor:
    """
    Runs the sequence of event-based tests (Test 2 and Test 3) to find additional dynamic points.

    Args:
        self: The MDetector instance.
        labels: The current tensor of point labels.
        active_mask: A boolean mask of points that are not pre-labeled as ground.
        current_di: The current DepthImage being processed.
        current_di_idx_in_lib: The index of the current DI in the library.

    Returns:
        torch.Tensor: The updated label tensor after running event tests.
    """
    # Run Test 2 (Parallel Motion Away)
    labels_after_test2 = self.execute_test2_parallel_motion(labels, current_di_idx_in_lib)
    
    # Identify candidates for Test 3: points that are still UNDETERMINED and not ground
    test3_candidate_mask = (labels_after_test2 == OcclusionResult.UNDETERMINED.value) & active_mask
    labels_after_test3 = labels_after_test2.clone()

    if torch.any(test3_candidate_mask):
        # Run MCC on these candidates. Only those NOT consistent with the map can be dynamic.
        points_for_t3_mcc = current_di.original_points_global_coords[test3_candidate_mask]
        t3_mcc_is_static = self.is_map_consistent(points_for_t3_mcc, current_di, current_di.timestamp)
        
        # Get the original indices of candidates that were rejected by MCC (potentially dynamic)
        t3_mcc_rejected_indices = torch.where(test3_candidate_mask)[0][~t3_mcc_is_static]

        if t3_mcc_rejected_indices.numel() > 0:
            # Create a temporary label tensor for Test 3, marking only the valid candidates
            test3_input_labels = torch.full_like(labels_after_test2, -1) # Use -1 for non-candidates
            test3_input_labels[t3_mcc_rejected_indices] = OcclusionResult.UNDETERMINED.value
            
            # Run Test 3 (Perpendicular Motion)
            test3_output_labels = self.execute_test3_perpendicular_motion(test3_input_labels, current_di_idx_in_lib)
            
            # Merge the results back into our main label tensor
            newly_dynamic_mask = (test3_output_labels == OcclusionResult.OCCLUDING_IMAGE.value)
            labels_after_test3[newly_dynamic_mask] = OcclusionResult.OCCLUDING_IMAGE.value
            
    return labels_after_test3


def forward(self, current_di: DepthImage, current_di_idx_in_lib: int) -> Dict:
    """
    This is the main forward pass for the M-Detector on a single sweep.
    It orchestrates the sequence of tests to label points as dynamic or static.

    Args:
        self: The MDetector instance.
        current_di: The DepthImage object for the current sweep to be processed.
        current_di_idx_in_lib: The index of the current DI in the library.

    Returns:
        A dictionary containing the results of the processing.
    """
    num_points = current_di.total_points_added_to_di_arrays
    if num_points == 0:
        return {'success': True, 'reason': 'current_di has no points', 'processed_di': current_di}

    # Initialize final labels, respecting the pre-labeled ground points
    final_labels = current_di.mdet_labels_for_points.clone()
    active_mask = (final_labels != OcclusionResult.PRELABELED_STATIC_GROUND.value)

    if not torch.any(active_mask):
        # All points were pre-labeled as ground, nothing to do.
        return {'success': True, 'processed_di': current_di}

    # --- Algorithm Stages ---
    
    # 1. Perform initial occlusion checks (Test 1 & 4)
    preliminary_labels = self._perform_initial_occlusion_pass(current_di, current_di_idx_in_lib)
    
    # 2. Apply Map Consistency Check to filter false positives from the static map
    labels_after_mcc = self._apply_map_consistency_check(preliminary_labels, active_mask, current_di)
    
    # 3. Run event-based tests (Test 2 & 3) to find additional dynamic points
    labels_after_events = self._run_event_test_sequence(labels_after_mcc, active_mask, current_di, current_di_idx_in_lib)
    
    # 4. Combine results into the final label tensor
    final_labels[active_mask] = labels_after_events[active_mask]
    current_di.mdet_labels_for_points = final_labels

    # --- Logging and Return ---
    label_counts = torch.bincount(final_labels.cpu(), minlength=len(OcclusionResult)).numpy()
    label_counts_dict = {OcclusionResult(i).name: count for i, count in enumerate(label_counts) if count > 0}
    logging.info(f"Processed DI (TS: {current_di.timestamp}, Idx: {current_di_idx_in_lib}): "
                 f"Labeled {num_points} points. Counts: {label_counts_dict}")

    return {
        'points_labeled': num_points,
        'label_counts': dict(enumerate(label_counts)),
        'success': True,
        'timestamp': current_di.timestamp,
        'processed_di': current_di
    }