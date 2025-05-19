# src/core/m_detector/processing.py
# This file is imported into MDetector class

import numpy as np
from typing import Dict, Optional, Any
from ..depth_image import DepthImage
from ..constants import OcclusionResult
from tqdm import tqdm

def extract_mdetector_points(depth_image_output_from_mdetector: Optional[Any]) -> Dict[str, np.ndarray]:
    """
    Extracts points by MDetector's label from its processed depth_image representation.
    NOTE: This function is based on the OLD DepthImage structure (pixel_points).
    It will not work correctly with the refactored DepthImage unless adapted.
    """
    mdet_points = {
        'dynamic': [],
        'occluded_by_mdet': [],
        'undetermined_by_mdet': []
    }
    if depth_image_output_from_mdetector is not None and \
       hasattr(depth_image_output_from_mdetector, 'pixel_points') and \
       isinstance(depth_image_output_from_mdetector.pixel_points, dict):
        # This part uses the old structure
        for _, points_list_in_pixel in depth_image_output_from_mdetector.pixel_points.items():
            for pt_info in points_list_in_pixel:
                label = pt_info.get('label')
                point_global_coords = pt_info.get('global_pt')
                if point_global_coords is None: continue
                if label == OcclusionResult.OCCLUDING_IMAGE:
                    mdet_points['dynamic'].append(point_global_coords)
                elif label == OcclusionResult.OCCLUDED_BY_IMAGE:
                    mdet_points['occluded_by_mdet'].append(point_global_coords)
                elif label == OcclusionResult.UNDETERMINED:
                    mdet_points['undetermined_by_mdet'].append(point_global_coords)
    elif depth_image_output_from_mdetector is not None:
        # This warning might trigger if a new DI object is passed.
        # tqdm.write("Warning: extract_mdetector_points received an object not matching old DI.pixel_points structure.")
        pass # Avoid excessive warnings if this function is called with new DI

    return {k: (np.array(v) if v else np.empty((0,3))) for k, v in mdet_points.items()}

def process_and_label_di(self, # self is MDetector instance
                        current_di: DepthImage,
                        current_di_idx_in_lib: int # Index of current_di in MDetector's library
                        ) -> Dict:
    """
    Process points in current_di using causal logic (Tests 1, 2, 3 from paper).
    Populates current_di.raw_occlusion_results_vs_history.
    Updates current_di.mdet_labels_for_points with the final point-out label.
    """
    if not isinstance(current_di, DepthImage):
        raise TypeError("current_di must be a DepthImage object.")

    label_counts = {label: 0 for label in OcclusionResult}
    points_labeled_count = 0

    if current_di.original_points_global_coords is None or \
       current_di.original_points_global_coords.shape[0] == 0:
        self.logger.debug(f"Causal Process: current_di (TS: {current_di.timestamp}) has no points. Skipping.")
        return {'points_labeled': 0, 'label_counts': label_counts, 'success': True,
                'timestamp': current_di.timestamp, 'reason': 'current_di has no points',
                'processed_di': current_di} # Return current_di even if no points

    num_points_in_current_di = current_di.original_points_global_coords.shape[0]
    points_global_batch_current = current_di.original_points_global_coords

    # Initialize final labels for current_di points to UNDETERMINED
    if current_di.mdet_labels_for_points is None or \
       current_di.mdet_labels_for_points.shape[0] != num_points_in_current_di:
        current_di.mdet_labels_for_points = np.full(num_points_in_current_di,
                                                    OcclusionResult.UNDETERMINED.value,
                                                    dtype=np.int8)

    # --- Parameters for Test 3 (Event Test) from config ---
    event_test_cfg = self.config.get('event_tests', {})
    test1_N_historical_DIs = event_test_cfg.get('test1_N_depth_images', 5) # N for Test 3
    test1_M1_threshold = event_test_cfg.get('test1_M1_threshold', 2)   # M1 for Test 3

    # --- 1. Populate current_di.raw_occlusion_results_vs_history ---
    # Initialize the history array for current_di
    current_di.raw_occlusion_results_vs_history = np.full(
        (num_points_in_current_di, test1_N_historical_DIs),
        OcclusionResult.UNDETERMINED.value, # Default if no historical DI or error
        dtype=np.int8
    )

    for k_hist_idx in range(test1_N_historical_DIs):
        # Get the (k_hist_idx)-th older DI (0 is immediate predecessor, 1 is before that, etc.)
        actual_historical_di_index_in_lib = current_di_idx_in_lib - 1 - k_hist_idx
        if actual_historical_di_index_in_lib < 0:
            break # No more historical DIs available

        historical_di_k = self.depth_image_library.get_image_by_index(actual_historical_di_index_in_lib)

        if historical_di_k and \
           historical_di_k.original_points_global_coords is not None and \
           historical_di_k.original_points_global_coords.shape[0] > 0:
            # Perform batch occlusion check of all points in current_di against historical_di_k
            raw_occlusion_enums_vs_hist_k = self.check_occlusion_batch(
                points_global_batch_current,
                historical_di_k
            )
            current_di.raw_occlusion_results_vs_history[:, k_hist_idx] = [
                res.value for res in raw_occlusion_enums_vs_hist_k
            ]
        # Else: leave as UNDETERMINED in current_di.raw_occlusion_results_vs_history[:, k_hist_idx]

    # --- Iterate through each point in the current_di to apply full causal logic ---
    for pt_idx in range(num_points_in_current_di):
        point_global_P = points_global_batch_current[pt_idx]
        # final_label_for_P = OcclusionResult.UNDETERMINED # Default

        # --- Test 1 (Occlusion vs. Immediate Predecessor) & Test 2 (Map Consistency) ---
        occlusion_result_after_MC = OcclusionResult.UNDETERMINED
        map_consistency_check_performed = False
        map_consistent_result_bool = None # To store boolean result of MCC if performed

        if test1_N_historical_DIs > 0:
            raw_occlusion_P_vs_immediate_hist_val = current_di.raw_occlusion_results_vs_history[pt_idx, 0]
            raw_occlusion_P_vs_immediate_hist = OcclusionResult(raw_occlusion_P_vs_immediate_hist_val)
        else:
            raw_occlusion_P_vs_immediate_hist = OcclusionResult.UNDETERMINED

        occlusion_result_after_MC = raw_occlusion_P_vs_immediate_hist

        if raw_occlusion_P_vs_immediate_hist == OcclusionResult.OCCLUDING_IMAGE and self.map_consistency_enabled:
            map_consistency_check_performed = True
            is_consistent_bool = self.is_map_consistent(point_global_P, current_di.timestamp, check_direction='past')
            map_consistent_result_bool = is_consistent_bool # Store for tracing
            if is_consistent_bool: # True means map consistent
                occlusion_result_after_MC = OcclusionResult.OCCLUDED_BY_IMAGE
            # Else (map inconsistent), occlusion_result_after_MC remains OCCLUDING_IMAGE

        # --- Test 3 (Event Test) using the populated history ---
        passed_event_test = False
        occluding_count_for_P = 0 # Initialize
        if test1_N_historical_DIs > 0 and test1_M1_threshold > 0:
            occluding_count_for_P = np.sum(
                current_di.raw_occlusion_results_vs_history[pt_idx, :] == OcclusionResult.OCCLUDING_IMAGE.value
            )
            if occluding_count_for_P >= test1_M1_threshold:
                passed_event_test = True

        # --- Trace intermediate data if collector is active for this point ---
        if self.debug_collector and self.debug_collector.is_tracing(pt_idx):
            trace_info = {
                "1_raw_occ_vs_imm": raw_occlusion_P_vs_immediate_hist.name,
                "2_mcc_performed": map_consistency_check_performed,
                "2a_mcc_is_consistent_result": map_consistent_result_bool if map_consistency_check_performed else "N/A",
                "3_occ_after_mcc": occlusion_result_after_MC.name,
                "4_event_test_occluding_count": occluding_count_for_P,
                "5_event_test_passed": passed_event_test,
            }
            self.debug_collector.trace(pt_idx, **trace_info)

        # --- Final Labeling Decision for Point P (Causal) ---
        final_label_for_P = OcclusionResult.UNDETERMINED # Default
        decision_path_trace = "Fallback_to_UNDETERMINED"

        if passed_event_test: # Primary condition for dynamic based on longer history
            # Further check if MCC contradicts this for the immediate past
            if occlusion_result_after_MC == OcclusionResult.OCCLUDED_BY_IMAGE: # MCC said it's static
                final_label_for_P = OcclusionResult.UNDETERMINED 
                # Or maybe even UNDETERMINED if event test and MCC conflict strongly
            else:
                final_label_for_P = OcclusionResult.OCCLUDING_IMAGE # Event test implies dynamic
        elif occlusion_result_after_MC == OcclusionResult.OCCLUDED_BY_IMAGE:
            final_label_for_P = OcclusionResult.OCCLUDED_BY_IMAGE
            decision_path_trace = "Static: OccAfterMCC_is_OCCLUDED_BY_IMAGE (map_consistent or raw_occluded)"
        elif occlusion_result_after_MC == OcclusionResult.EMPTY_IN_IMAGE:
            final_label_for_P = OcclusionResult.EMPTY_IN_IMAGE
            decision_path_trace = "Static: OccAfterMCC_is_EMPTY_IN_IMAGE"
        # else: final_label_for_P remains UNDETERMINED (already set as default)

        current_di.mdet_labels_for_points[pt_idx] = final_label_for_P.value
        label_counts[final_label_for_P] += 1
        points_labeled_count += 1

        if self.debug_collector and self.debug_collector.is_tracing(pt_idx):
            self.debug_collector.trace(pt_idx, 
                                        final_label=final_label_for_P.name,
                                        decision_logic=decision_path_trace)

    self.logger.debug(f"Causal Processed DI (TS: {current_di.timestamp}, Idx: {current_di_idx_in_lib}): "
                      f"Labeled {points_labeled_count}/{num_points_in_current_di} points. "
                      f"Counts: { {k.name:v for k,v in label_counts.items() if v > 0} }")

    return {
        'points_labeled': points_labeled_count, 'label_counts': label_counts,
        'success': True, 'timestamp': current_di.timestamp,
        'processed_di': current_di
    }


def _process_causal_di(self, di_to_process_idx: int) -> Dict:
    current_di = self.depth_image_library._images[di_to_process_idx]
    historical_di = None
    if di_to_process_idx > 0:
        historical_di = self.depth_image_library._images[di_to_process_idx - 1]
    
    # Calls the core logic of your original process_and_label_di
    # This is just a sketch of refactoring
    result = self.actual_causal_processing_logic(current_di, historical_di) 
    result['processed_frame_timestamp'] = current_di.timestamp
    result['frame_index'] = di_to_process_idx
    return result

def _process_bidirectional_di(self, di_to_process_idx: int) -> Dict:
    # This is essentially your existing process_and_label_di_bidirectional
    # It already takes center_index, which is di_to_process_idx here.
    result = self.process_and_label_di_bidirectional(di_to_process_idx) # Call existing func
    # Ensure it populates 'processed_frame_timestamp' and 'frame_index'
    if result.get('success'):
        result['processed_frame_timestamp'] = self.depth_image_library._images[di_to_process_idx].timestamp
        result['frame_index'] = di_to_process_idx # Already in your func
    return result