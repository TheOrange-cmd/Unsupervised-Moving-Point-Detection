# src/core/m_detector/processing.py

import torch
from typing import Dict

from ..depth_image import DepthImage
from ..constants import OcclusionResult

from .refinement_algorithms import apply_clustering_and_refinement

# This file is bound to the MDetector class, so `self` refers to an MDetector instance.

def _perform_initial_occlusion_pass(self, current_di: DepthImage, current_di_idx_in_lib: int) -> torch.Tensor:
    """
    Performs the initial occlusion check against historical depth images (Test 1)
    and applies the perpendicular motion heuristic (Test 4).
    """
    num_points = current_di.total_points_added_to_di_arrays
    all_points_global = current_di.original_points_global_coords

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

    preliminary_labels = raw_occlusion_results[:, 0].clone()
    occluding_counts = torch.sum(raw_occlusion_results == OcclusionResult.OCCLUDING_IMAGE.value, dim=1)
    test4_passed_mask = (occluding_counts >= self.test1_M4_threshold)
    preliminary_labels[test4_passed_mask] = OcclusionResult.OCCLUDING_IMAGE.value
    
    return preliminary_labels


def _apply_map_consistency_check(self, labels: torch.Tensor, active_mask: torch.Tensor, current_di: DepthImage) -> torch.Tensor:
    """
    Applies the Map Consistency Check (MCC) to filter out points that are likely
    part of the static map, despite appearing dynamic from the initial pass.
    """
    dynamic_candidates_mask = (labels == OcclusionResult.OCCLUDING_IMAGE.value) & active_mask
    
    if torch.any(dynamic_candidates_mask):
        points_for_mcc = current_di.original_points_global_coords[dynamic_candidates_mask]
        # 1. Get the confidence scores (a float tensor) from the MCC function.
        mcc_confidence_scores = self.is_map_consistent(points_for_mcc, current_di, current_di.timestamp, caller_id="_apply_mcc")
        
        # 2. Create the boolean mask by comparing the scores to our new threshold.
        mcc_is_static_mask = mcc_confidence_scores >= self.mc_static_confidence_threshold
        indices_to_revert = torch.where(dynamic_candidates_mask)[0][mcc_is_static_mask]
        labels[indices_to_revert] = OcclusionResult.OCCLUDED_BY_IMAGE.value
        
    return labels


def _apply_frame_refinement(self, labels: torch.Tensor, current_di: DepthImage) -> torch.Tensor:
    """
    This function is now a simple wrapper that calls the modular refinement logic.
    """
    refinement_params = self.config.get_frame_refinement_params()
    # The new function handles the 'enabled' check internally.
    return apply_clustering_and_refinement(labels, current_di, refinement_params)


def _run_event_test_sequence(self, labels: torch.Tensor, active_mask: torch.Tensor, current_di: DepthImage, current_di_idx_in_lib: int) -> torch.Tensor:
    """
    Runs the sequence of event-based tests (Test 2 and Test 3) with OPTIMIZED logic.
    For Test 3, it runs the cheap event test first to generate a small number of candidates,
    then runs the expensive MCC only on those candidates.
    """
    # Run Test 2 (Parallel Motion Away) - this test is fine as is.
    labels_after_test2 = self.execute_test2_parallel_motion(labels, current_di_idx_in_lib)
    
    # --- OPTIMIZED Test 3 Logic ---
    
    # 1. Identify all points that are eligible for Test 3 (still undetermined).
    test3_eligible_mask = (labels_after_test2 == OcclusionResult.UNDETERMINED.value) & active_mask
    labels_after_test3 = labels_after_test2.clone()

    if torch.any(test3_eligible_mask):
        # 2. Create a temporary label tensor for Test 3 to operate on.
        test3_input_labels = torch.full_like(labels_after_test2, -1) # -1 means not a candidate
        test3_input_labels[test3_eligible_mask] = OcclusionResult.UNDETERMINED.value
        
        # 3. Run the CHEAP event test FIRST on the large pool of eligible points.
        # This will find a SMALL number of points that appear to be occluding something.
        test3_output_labels = self.execute_test3_perpendicular_motion(test3_input_labels, current_di_idx_in_lib)
        
        # 4. Identify the small set of candidates that Test 3 flagged as potentially dynamic.
        test3_dynamic_candidates_mask = (test3_output_labels == OcclusionResult.OCCLUDING_IMAGE.value)
        
        # 5. NOW, run the EXPENSIVE MCC only on these few, highly-qualified candidates.
        if torch.any(test3_dynamic_candidates_mask):
            points_for_t3_mcc = current_di.original_points_global_coords[test3_dynamic_candidates_mask]

            if self.map_consistency_enabled:
                # 1. Get the confidence scores.
                t3_mcc_scores = self.is_map_consistent(points_for_t3_mcc, current_di, current_di.timestamp, caller_id="_run_event_tests_optimized")
                
                # 2. Compare scores to the threshold to get the boolean mask.
                t3_mcc_is_static = t3_mcc_scores >= self.mc_static_confidence_threshold
            else:
                t3_mcc_is_static = torch.zeros(points_for_t3_mcc.shape[0], dtype=torch.bool, device=self.device)

            # 6. The final dynamic points are those that Test 3 found AND MCC did NOT reject.
            final_test3_dynamic_indices = torch.where(test3_dynamic_candidates_mask)[0][~t3_mcc_is_static]
            
            # 7. Update our main label tensor with the confirmed dynamic points.
            labels_after_test3[final_test3_dynamic_indices] = OcclusionResult.OCCLUDING_IMAGE.value
            
    return labels_after_test3


def forward(self, current_di: DepthImage, current_di_idx_in_lib: int) -> Dict:
    """
    Main forward pass for the M-Detector, orchestrating the sequence of tests.
    """
    # Call the geometric pass and then apply refinement.
    labels_before_refinement = self._forward_geometric_only(current_di, current_di_idx_in_lib)
    
    # Apply the final refinement step
    final_labels = self._apply_frame_refinement(labels_before_refinement, current_di)

    # Assign the fully-processed, final, refined labels to the DI object.
    current_di.mdet_labels_for_points = final_labels
    
    # --- Logging and Return ---
    num_points = current_di.total_points_added_to_di_arrays
    label_counts = torch.bincount(final_labels.cpu(), minlength=len(OcclusionResult)).numpy()
    label_counts_dict = {OcclusionResult(i).name: count for i, count in enumerate(label_counts) if count > 0}
    self.logger.debug(f"Processed DI (TS: {current_di.timestamp}, Idx: {current_di_idx_in_lib}): "
                 f"Labeled {num_points} points. Counts: {label_counts_dict}")

    return {
        'points_labeled': num_points,
        'label_counts': dict(enumerate(label_counts)),
        'success': True,
        'timestamp': current_di.timestamp,
        'processed_di': current_di
    }

def _forward_geometric_only(self, current_di: DepthImage, current_di_idx_in_lib: int) -> torch.Tensor:
    """
    A special version of the forward pass that returns the labels
    right before the frame refinement step.
    """
    num_points = current_di.total_points_added_to_di_arrays
    if num_points == 0:
        return current_di.mdet_labels_for_points.clone()

    final_labels = current_di.mdet_labels_for_points.clone()
    active_mask = (final_labels != OcclusionResult.PRELABELED_STATIC_GROUND.value)

    if not torch.any(active_mask):
        return final_labels

    # --- Run all geometric stages ---
    preliminary_labels = self._perform_initial_occlusion_pass(current_di, current_di_idx_in_lib)
    labels_after_mcc = self._apply_map_consistency_check(preliminary_labels, active_mask, current_di)
    labels_after_events = self._run_event_test_sequence(labels_after_mcc, active_mask, current_di, current_di_idx_in_lib)
    
    # Combine results into the final label tensor
    final_labels[active_mask] = labels_after_events[active_mask]
    
    # --- RETURN BEFORE REFINEMENT ---
    return final_labels