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
                        current_di_idx_in_lib: int
                        ) -> Dict:
    if not isinstance(current_di, DepthImage):
        raise TypeError("current_di must be a DepthImage object.")

    # Initialize label_counts with OcclusionResult enum members as keys for clarity, convert to value for storage if needed
    label_counts_enum_keys = {label: 0 for label in OcclusionResult}
    points_labeled_count = 0

    if current_di.original_points_global_coords is None or \
       current_di.original_points_global_coords.shape[0] == 0:
        self.logger.debug(f"ProcessDI: current_di (TS: {current_di.timestamp}) has no points.")
        # Convert enum keys to values for the return dict if that's the expected format
        label_counts_val_keys = {k.value: v for k,v in label_counts_enum_keys.items()}
        return {'points_labeled': 0, 'label_counts': label_counts_val_keys, 'success': True,
                'timestamp': current_di.timestamp, 'reason': 'current_di has no points',
                'processed_di': current_di}

    num_points_in_current_di = current_di.original_points_global_coords.shape[0]
    points_global_batch_current = current_di.original_points_global_coords

    if current_di.mdet_labels_for_points is None or \
       current_di.mdet_labels_for_points.shape[0] != num_points_in_current_di:
        current_di.mdet_labels_for_points = np.full(num_points_in_current_di,
                                                    OcclusionResult.UNDETERMINED.value,
                                                    dtype=np.int8)

    # Populate raw_occlusion_results_vs_history (for Test 1/4)
    current_di.raw_occlusion_results_vs_history = np.full(
        (num_points_in_current_di, self.test1_N_historical_DIs), # Uses self.test1_N_historical_DIs
        OcclusionResult.UNDETERMINED.value, dtype=np.int8
    )
    for k_hist_idx in range(self.test1_N_historical_DIs):
        actual_historical_di_index_in_lib = current_di_idx_in_lib - 1 - k_hist_idx
        if actual_historical_di_index_in_lib < 0: break
        historical_di_k = self.depth_image_library.get_image_by_index(actual_historical_di_index_in_lib)
        if historical_di_k and historical_di_k.is_prepared_for_projection():
            raw_occlusion_enums_vs_hist_k = self.check_occlusion_batch(
                points_global_batch_current, historical_di_k
            )
            current_di.raw_occlusion_results_vs_history[:, k_hist_idx] = [
                res.value for res in raw_occlusion_enums_vs_hist_k
            ]

    # Iterate through each point for final labeling
    # Use disable=not self.config_accessor.get_processing_settings().get('show_progress_bar_per_di', False)
    # For now, disable it to keep logs cleaner during this complex addition.
    for pt_idx in tqdm(range(num_points_in_current_di), 
                       desc=f"Processing DI {current_di.timestamp:.2f}", 
                       leave=False, 
                       disable=True): # Temporarily disable for cleaner logs
        point_global_P = points_global_batch_current[pt_idx]

        # --- Check if point was pre-labeled ---
        current_label_val = current_di.mdet_labels_for_points[pt_idx]
        if current_label_val == OcclusionResult.PRELABELED_STATIC_GROUND.value:
            # This point is already confidently static ground.
            # Do not run Test 1/2/3/4 to re-label it.
            # It will still be used by other points for their occlusion checks (as part of historical DIs).
            final_label_for_P = OcclusionResult.PRELABELED_STATIC_GROUND # Keep its pre-label
            
            # Optionally log or trace this skip
            if self.debug_collector and self.debug_collector.is_tracing(pt_idx):
                self.debug_collector.trace(pt_idx, final_label="PRELABELED_STATIC_GROUND (skipped M-Det tests)")
            
            label_counts_enum_keys[final_label_for_P] += 1
            points_labeled_count +=1 # Still counts as "processed" by the DI labeling stage
            continue # Move to the next point
        
        # Test 1 (Occlusion vs. Imm. Past) & MCC (Implicitly part of Test 4 logic)
        outcome_test1_mcc = OcclusionResult.UNDETERMINED # Result after immediate occlusion and MCC
        raw_occ_P_vs_imm_hist = OcclusionResult.UNDETERMINED
        if self.test1_N_historical_DIs > 0: # Ensure there's history to check against
            raw_occ_P_vs_imm_hist = OcclusionResult(current_di.raw_occlusion_results_vs_history[pt_idx, 0])
        
        outcome_test1_mcc = raw_occ_P_vs_imm_hist
        mcc1_performed = False
        mcc1_result_bool = None
        if raw_occ_P_vs_imm_hist == OcclusionResult.OCCLUDING_IMAGE and self.map_consistency_enabled:
            mcc1_performed = True
            mcc1_result_bool = self.is_map_consistent(point_global_P, current_di.timestamp, check_direction='past')
            if mcc1_result_bool: 
                outcome_test1_mcc = OcclusionResult.OCCLUDED_BY_IMAGE 
        
        # Test 4 (Perpendicular Event Test based on history counts)
        passed_perpendicular_event_test = False
        occluding_count_for_test1_N = 0
        if self.test1_N_historical_DIs > 0 and self.test1_M1_threshold > 0:
            occluding_count_for_test1_N = np.sum(
                current_di.raw_occlusion_results_vs_history[pt_idx, :self.test1_N_historical_DIs] == OcclusionResult.OCCLUDING_IMAGE.value
            )
            if occluding_count_for_test1_N >= self.test1_M1_threshold:
                passed_perpendicular_event_test = True

        # Combined outcome of Test 1 (immediate + MCC) and Test 4 (perpendicular event)
        # This represents the "dynamic" signal from the original paper's simpler tests.
        final_outcome_tests1_and_4 = OcclusionResult.UNDETERMINED
        if passed_perpendicular_event_test: # If the longer history perpendicular test says dynamic
            # Even if MCC on immediate past said static, the longer history might be more robust
            final_outcome_tests1_and_4 = OcclusionResult.OCCLUDING_IMAGE
        else: # Perpendicular event test did not pass, rely on immediate past + MCC
            final_outcome_tests1_and_4 = outcome_test1_mcc
        
        # --- Test 2 (Parallel Motion - Event by Disappearance) ---
        outcome_test2 = OcclusionResult.UNDETERMINED
        if self.execute_test2_parallel_motion(point_global_P, current_di.timestamp, current_di_idx_in_lib):
            outcome_test2 = OcclusionResult.OCCLUDING_IMAGE 

        # --- Test 3 (Perpendicular Motion - Event by Appearance/Revealing) ---
        outcome_test3 = OcclusionResult.UNDETERMINED
        if self.execute_test3_perpendicular_motion(point_global_P, current_di.timestamp, current_di_idx_in_lib):
            outcome_test3 = OcclusionResult.OCCLUDING_IMAGE

        # --- Final Label Aggregation ---
        final_label_for_P = OcclusionResult.UNDETERMINED
        
        if final_outcome_tests1_and_4 == OcclusionResult.OCCLUDING_IMAGE or \
           outcome_test2 == OcclusionResult.OCCLUDING_IMAGE or \
           outcome_test3 == OcclusionResult.OCCLUDING_IMAGE:
            final_label_for_P = OcclusionResult.OCCLUDING_IMAGE
        elif final_outcome_tests1_and_4 == OcclusionResult.OCCLUDED_BY_IMAGE:
            final_label_for_P = OcclusionResult.OCCLUDED_BY_IMAGE
        elif final_outcome_tests1_and_4 == OcclusionResult.EMPTY_IN_IMAGE:
            final_label_for_P = OcclusionResult.EMPTY_IN_IMAGE
        
        current_di.mdet_labels_for_points[pt_idx] = final_label_for_P.value
        label_counts_enum_keys[final_label_for_P] += 1 # Use enum member as key
        points_labeled_count += 1

        if self.debug_collector and self.debug_collector.is_tracing(pt_idx):
            trace_info = {
                "T1_raw_occ_vs_imm": raw_occ_P_vs_imm_hist.name,
                "T1_mcc_performed": mcc1_performed,
                "T1_mcc_result": mcc1_result_bool if mcc1_performed else "N/A",
                "T1_outcome_after_mcc": outcome_test1_mcc.name,
                "T4_perp_event_test_count": occluding_count_for_test1_N,
                "T4_perp_event_test_passed": passed_perpendicular_event_test,
                "OUTCOME_Tests1_and_4": final_outcome_tests1_and_4.name,
                "OUTCOME_Test2_parallel": outcome_test2.name,
                "OUTCOME_Test3_perpendicular": outcome_test3.name,
                "final_label": final_label_for_P.name,
            }
            self.debug_collector.trace(pt_idx, **trace_info)
            
    # Convert enum keys to values for the return dict
    label_counts_val_keys = {k.value: v for k,v in label_counts_enum_keys.items()}
    self.logger.info(f"Processed DI (TS: {current_di.timestamp}, Idx: {current_di_idx_in_lib}): "
                      f"Labeled {points_labeled_count}/{num_points_in_current_di} points. "
                      f"Counts: { {OcclusionResult(k).name:v for k,v in label_counts_val_keys.items() if v > 0} }")

    return {
        'points_labeled': points_labeled_count, 'label_counts': label_counts_val_keys,
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