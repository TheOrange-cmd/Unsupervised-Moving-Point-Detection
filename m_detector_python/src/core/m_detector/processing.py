# src/core/m_detector/processing.py

import torch
from typing import Dict, Callable

from ..depth_image import DepthImage
from ..constants import OcclusionResult

from .refinement_algorithms import apply_clustering_and_refinement

# This file is bound to the MDetector class, so `self` refers to an MDetector instance.

def _perform_initial_occlusion_pass(self, current_di: DepthImage, current_di_idx_in_lib: int) -> torch.Tensor:
    """
    Performs the initial occlusion check against historical depth images (Test 1)
    and applies the perpendicular motion heuristic (Test 4).
    """
    # Collect the relevant undetermined points
    num_points = current_di.num_points
    all_points_global = current_di.original_points_global_coords

    raw_occlusion_results = torch.full(
        (num_points, self.initial_pass_history_length),
        OcclusionResult.UNDETERMINED.value, dtype=torch.int8, device=self.device
    )
    # Go through the relevant historical depth images and call the course grained occlusion check
    for k in range(self.initial_pass_history_length):
        hist_di_idx = current_di_idx_in_lib - 1 - k
        if hist_di_idx < 0: break
        historical_di = self.depth_image_library.get_image_by_index(hist_di_idx)
        if historical_di and historical_di.is_prepared_for_projection():
            raw_occlusion_results[:, k] = self.check_occlusion_batch(all_points_global, historical_di)

    # Mark the points that are occluding consistently: X/M total relevant historical depth images show occlusion
    preliminary_labels = raw_occlusion_results[:, 0].clone()
    occluding_counts = torch.sum(raw_occlusion_results == OcclusionResult.OCCLUDING_IMAGE.value, dim=1)
    test4_passed_mask = (occluding_counts >= self.initial_pass_min_occlusion_count)
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
        mcc_confidence_scores = self.is_map_consistent(points_for_mcc, current_di)
        
        # 2. Create the boolean mask by comparing the scores to the threshold.
        mcc_is_static_mask = mcc_confidence_scores >= self.mc_static_confidence_threshold
        indices_to_revert = torch.where(dynamic_candidates_mask)[0][mcc_is_static_mask]
        labels[indices_to_revert] = OcclusionResult.OCCLUDED_BY_IMAGE.value
        
    return labels


def _apply_frame_refinement(self, labels: torch.Tensor, current_di: DepthImage) -> torch.Tensor:
    """
    This function is a simple wrapper that calls the modular refinement logic.
    """
    refinement_params = self.config.get_frame_refinement_params()
    return apply_clustering_and_refinement(labels, current_di, refinement_params)


def _run_single_event_test_with_mcc(
    self,
    labels: torch.Tensor,
    active_mask: torch.Tensor,
    current_di: DepthImage,
    current_di_idx_in_lib: int,
    event_test_func: Callable,
    target_label: OcclusionResult,
    apply_mcc: bool,
    test_name: str
) -> torch.Tensor:
    """
    Generic helper to run an event test and optionally filter its results with MCC.

    This function encapsulates the pattern of:
    1. Finding eligible points with a specific label.
    2. Running a given event test function on them.
    3. Taking the dynamic candidates produced by the test.
    4. If configured, running the Map Consistency Check on those candidates.
    5. Updating the main labels tensor with the final, high-confidence dynamic points.

    Args:
        labels (torch.Tensor): The main labels tensor to be modified.
        active_mask (torch.Tensor): Mask of points eligible for any processing.
        current_di (DepthImage): The current DI being processed.
        current_di_idx_in_lib (int): The index of the current DI.
        event_test_func (Callable): The specific event test function to execute.
        target_label (OcclusionResult): The label to select as input for the test.
        apply_mcc (bool): If True, run MCC on the test's output.
        test_name (str): A name for logging purposes.

    Returns:
        torch.Tensor: The modified labels tensor.
    """
    # 1. Find points eligible for this specific test.
    eligible_mask = (labels == target_label.value) & active_mask
    if not torch.any(eligible_mask):
        return labels # No candidates, no work to do.

    # 2. Create a temporary input tensor for the event test function.
    test_input_labels = torch.full_like(labels, -1) # -1 means not a candidate
    test_input_labels[eligible_mask] = target_label.value

    # 3. Run the provided event test to generate initial candidates.
    test_output_labels = event_test_func(test_input_labels, current_di_idx_in_lib)
    
    # 4. Identify the candidates that the test flagged as potentially dynamic.
    dynamic_candidates_mask = (test_output_labels == OcclusionResult.OCCLUDING_IMAGE.value)
    if not torch.any(dynamic_candidates_mask):
        return labels # The test found no dynamic points.

    # 5. Optionally filter these candidates with the Map Consistency Check.
    if apply_mcc and self.map_consistency_enabled:
        points_for_mcc = current_di.original_points_global_coords[dynamic_candidates_mask]
        mcc_scores = self.is_map_consistent(points_for_mcc, current_di)
        is_static_mask = mcc_scores >= self.mc_static_confidence_threshold
        
        # The final dynamic points are those that were NOT filtered out by MCC.
        final_dynamic_indices = torch.where(dynamic_candidates_mask)[0][~is_static_mask]
    else:
        # If MCC is disabled, all candidates from the test are considered final.
        final_dynamic_indices = torch.where(dynamic_candidates_mask)[0]

    # 6. Update the main label tensor with the confirmed dynamic points.
    if final_dynamic_indices.numel() > 0:
        labels[final_dynamic_indices] = OcclusionResult.OCCLUDING_IMAGE.value
        self.logger.debug(f"{test_name}: Confirmed {final_dynamic_indices.numel()} new dynamic points.")
        
    return labels

def _run_event_test_sequence(self, labels: torch.Tensor, active_mask: torch.Tensor, current_di: DepthImage, current_di_idx_in_lib: int) -> torch.Tensor:
    """
    Runs the sequence of specialized event tests.
    
    Each test identifies new dynamic point candidates
    from different initial states ('OCCLUDED_BY_IMAGE' or 'UNDETERMINED') and
    subjects them to an optional, final quality check with the Map Consistency Check.
    """
    # Run test for parallel motion away
    labels_after_parallel = self._run_single_event_test_with_mcc(
        labels,
        active_mask,
        current_di,
        current_di_idx_in_lib,
        event_test_func=self.execute_parallel_motion_away_test,
        target_label=OcclusionResult.OCCLUDED_BY_IMAGE,
        apply_mcc=self.parallel_motion_apply_mcc, 
        test_name="Parallel Motion Away"
    )
    
    # Run test for perpendicular motion
    labels_after_perpendicular = self._run_single_event_test_with_mcc(
        labels_after_parallel,
        active_mask,
        current_di,
        current_di_idx_in_lib,
        event_test_func=self.execute_perpendicular_motion_test, 
        target_label=OcclusionResult.UNDETERMINED,
        apply_mcc=self.perpendicular_motion_apply_mcc, 
        test_name="Perpendicular Motion"
    )
            
    return labels_after_perpendicular


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
    num_points = current_di.num_points
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
    num_points = current_di.num_points
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