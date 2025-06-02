# src/core/m_detector/processing.py
# This file is imported into MDetector class

import numpy as np
from typing import Dict, Optional, Any
from ..depth_image import DepthImage
from ..constants import OcclusionResult
from tqdm import tqdm
import logging

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
        if self.logger.isEnabledFor(logging.DEBUG):
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
    # This section remains the same
    if current_di.raw_occlusion_results_vs_history is None or \
       current_di.raw_occlusion_results_vs_history.shape[0] != num_points_in_current_di or \
       current_di.raw_occlusion_results_vs_history.shape[1] != self.test1_N_historical_DIs: # Check shape
        current_di.raw_occlusion_results_vs_history = np.full(
            (num_points_in_current_di, self.test1_N_historical_DIs), 
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
    for pt_idx in tqdm(range(num_points_in_current_di), 
                       desc=f"Processing DI {current_di.timestamp:.2f}", 
                       leave=False, 
                       disable=True): 
        point_global_P = points_global_batch_current[pt_idx]

        # --- Check if point was pre-labeled ---
        # This section remains the same
        current_label_val = current_di.mdet_labels_for_points[pt_idx]
        if current_label_val == OcclusionResult.PRELABELED_STATIC_GROUND.value:
            final_label_for_P = OcclusionResult.PRELABELED_STATIC_GROUND 
            if self.debug_collector and self.debug_collector.is_tracing(pt_idx):
                # Simplified trace for pre-labeled, as it won't have other test outcomes
                self.debug_collector.trace(pt_idx, final_label="PRELABELED_STATIC_GROUND (skipped M-Det tests)",
                                           config_interp_enabled_during_call = self.mc_interp_enabled)
            label_counts_enum_keys[final_label_for_P] += 1
            points_labeled_count +=1 
            continue 
        
        # --- Initialize trace_info for this point if being traced ---
        # This section remains the same
        current_trace_info = {}
        is_tracing_this_point = self.debug_collector and self.debug_collector.is_tracing(pt_idx)

        # --- Test 1 (Occlusion vs. Imm. Past) & MCC ---
        # This section remains the same
        outcome_test1_mcc = OcclusionResult.UNDETERMINED
        raw_occ_P_vs_imm_hist = OcclusionResult.UNDETERMINED
        if self.test1_N_historical_DIs > 0 and current_di.raw_occlusion_results_vs_history is not None:
            if pt_idx < current_di.raw_occlusion_results_vs_history.shape[0]: 
                raw_occ_P_vs_imm_hist = OcclusionResult(current_di.raw_occlusion_results_vs_history[pt_idx, 0])
        # current_trace_info["T1_raw_occ_vs_imm"] is added later if is_tracing_this_point

        outcome_test1_mcc = raw_occ_P_vs_imm_hist # Default if MCC not performed or doesn't change it
        mcc1_performed = False
        mcc1_result_bool = None 

        if raw_occ_P_vs_imm_hist == OcclusionResult.OCCLUDING_IMAGE and self.map_consistency_enabled:
            mcc1_performed = True
            mcc_result_package_t1 = self.is_map_consistent(
                point_global_P,
                current_di,                             # origin_di_of_point_global
                pt_idx,                                 # original_idx_of_point_global_in_origin_di
                current_di.timestamp,                   # current_timestamp (of origin_di)
                check_direction='past',
                return_debug_info=is_tracing_this_point 
            )
            if is_tracing_this_point:
                mcc1_result_bool, mcc1_full_debug_data = mcc_result_package_t1
                current_trace_info["T1_mcc_full_debug"] = mcc1_full_debug_data 
            else:
                mcc1_result_bool = mcc_result_package_t1

            if mcc1_result_bool: # If MCC says P IS consistent with static map
                outcome_test1_mcc = OcclusionResult.OCCLUDED_BY_IMAGE
        
        # --- Test 4 (Perpendicular Event Test) ---
        # This section remains the same (using test1_M4_threshold from config)
        passed_perpendicular_event_test = False
        occluding_count_for_test1_N = 0
        # Assuming self.test1_M4_threshold is the correct parameter from your config for Test 4
        if self.test1_N_historical_DIs > 0 and self.test1_M4_threshold > 0 and \
           current_di.raw_occlusion_results_vs_history is not None and \
           pt_idx < current_di.raw_occlusion_results_vs_history.shape[0]: 
            occluding_count_for_test1_N = np.sum(
                current_di.raw_occlusion_results_vs_history[pt_idx, :self.test1_N_historical_DIs] == OcclusionResult.OCCLUDING_IMAGE.value
            )
            if occluding_count_for_test1_N >= self.test1_M4_threshold: 
                passed_perpendicular_event_test = True
        
        # --- Combined outcome of Test 1 (immediate + MCC) and Test 4 ---
        final_outcome_tests1_and_4 = OcclusionResult.UNDETERMINED
        if passed_perpendicular_event_test:
            final_outcome_tests1_and_4 = OcclusionResult.OCCLUDING_IMAGE
        else:
            final_outcome_tests1_and_4 = outcome_test1_mcc
        
        # --- Test 2 (Parallel Motion) ---
        # UPDATED CALL to execute_test2_parallel_motion
        test2_passed_package = self.execute_test2_parallel_motion(
            point_global_P,
            current_di,  # Pass current_di as origin of P
            pt_idx,      # Pass original index of P in current_di
            current_di.timestamp,
            current_di_idx_in_lib,
            return_debug_info=is_tracing_this_point
        )
        
        test2_passed: bool
        outcome_test2 = OcclusionResult.UNDETERMINED
        if is_tracing_this_point:
            if isinstance(test2_passed_package, tuple):
                test2_passed, test2_full_debug = test2_passed_package
                current_trace_info["Test2_full_debug"] = test2_full_debug
            else: # Should not happen if return_debug_info is True, but handle defensively
                test2_passed = test2_passed_package 
        else:
            test2_passed = test2_passed_package

        if test2_passed: outcome_test2 = OcclusionResult.OCCLUDING_IMAGE


        # --- Test 3 (Perpendicular Motion - Appearance) ---
        # This section remains the same
        outcome_test3 = OcclusionResult.UNDETERMINED
        mcc_rejects_P_for_test3 = False
        mcc_for_test3_result_bool = None

        if self.map_consistency_enabled:
            mcc_result_package_t3 = self.is_map_consistent(
                point_global_P,
                current_di,                             # origin_di_of_point_global
                pt_idx,                                 # original_idx_of_point_global_in_origin_di
                current_di.timestamp,                   # current_timestamp (of origin_di)
                check_direction='past',
                return_debug_info=is_tracing_this_point
            )
            if is_tracing_this_point:
                mcc_for_test3_result_bool, mcc_t3_full_debug_data = mcc_result_package_t3
                current_trace_info["T3_mcc_on_P_full_debug"] = mcc_t3_full_debug_data
            else:
                mcc_for_test3_result_bool = mcc_result_package_t3
            
            if mcc_for_test3_result_bool: # If P is map consistent
                mcc_rejects_P_for_test3 = True
        
        test3_chain_passed = False 
        if not mcc_rejects_P_for_test3:
            # adaptive_config_for_test3_from_main_config = self.config_accessor.get_event_detection_logic_params().get('test3_adaptive_epsilon_params', None)

            test3_result_package = self.execute_test3_perpendicular_motion(
                point_global_P,
                current_di,                             
                pt_idx,                                 
                current_di.timestamp,
                current_di_idx_in_lib,
                return_debug_info=is_tracing_this_point
            )
            if is_tracing_this_point:
                if isinstance(test3_result_package, tuple): # (bool, debug_dict)
                    test3_chain_passed, test3_full_debug = test3_result_package
                    current_trace_info["Test3_full_debug"] = test3_full_debug
                else: # just bool
                    test3_chain_passed = test3_result_package
            else: # not tracing
                test3_chain_passed = test3_result_package

            if test3_chain_passed: outcome_test3 = OcclusionResult.OCCLUDING_IMAGE
        
        # ========================================================================
        # START OF MODIFIED FINAL LABEL AGGREGATION WITH MCC REFINEMENT
        # ========================================================================
        
        # --- Determine Preliminary Dynamic Status ---
        # This checks if ANY of the main tests (T1/4, T2, T3) suggest the point is dynamic.
        preliminary_is_dynamic = False
        if final_outcome_tests1_and_4 == OcclusionResult.OCCLUDING_IMAGE or \
           outcome_test2 == OcclusionResult.OCCLUDING_IMAGE or \
           outcome_test3 == OcclusionResult.OCCLUDING_IMAGE:
            preliminary_is_dynamic = True

        # --- Initialize final_label_for_P before refinement ---
        if preliminary_is_dynamic:
            final_label_for_P = OcclusionResult.OCCLUDING_IMAGE
        elif final_outcome_tests1_and_4 == OcclusionResult.OCCLUDED_BY_IMAGE:
            final_label_for_P = OcclusionResult.OCCLUDED_BY_IMAGE
        elif final_outcome_tests1_and_4 == OcclusionResult.EMPTY_IN_IMAGE:
            final_label_for_P = OcclusionResult.EMPTY_IN_IMAGE
        else: # All tests resulted in UNDETERMINED or were not decisive for static/empty
            final_label_for_P = OcclusionResult.UNDETERMINED


        # --- MCC Refinement Step ---
        # If the point was preliminarily labeled as dynamic (e.g., by Test 4),
        # give MCC a chance to override this if P is actually consistent with the map.
        mcc_refinement_performed = False
        mcc_refinement_result_bool = None 
        mcc_refinement_overrode_to_static = False

        if preliminary_is_dynamic and self.map_consistency_enabled:
            mcc_refinement_performed = True
            # CORRECTED CALL for MCC Refinement
            mcc_package_refinement = self.is_map_consistent(
                point_global_P,
                current_di,                             # origin_di_of_point_global
                pt_idx,                                 # original_idx_of_point_global_in_origin_di
                current_di.timestamp,                   # current_timestamp (of origin_di)
                check_direction='past', # Or 'both' if you implement that for refinement
                return_debug_info=is_tracing_this_point
            )
            if is_tracing_this_point:
                mcc_refinement_result_bool, mcc_refinement_full_debug = mcc_package_refinement
                current_trace_info["MCC_Refinement_full_debug"] = mcc_refinement_full_debug
            else:
                mcc_refinement_result_bool = mcc_package_refinement

            if mcc_refinement_result_bool: 
                final_label_for_P = OcclusionResult.OCCLUDED_BY_IMAGE 
                mcc_refinement_overrode_to_static = True
            # else: final_label_for_P remains OCCLUDING_IMAGE (as set by preliminary_is_dynamic)
        
        # ========================================================================
        # END OF MODIFIED FINAL LABEL AGGREGATION
        # ========================================================================

        current_di.mdet_labels_for_points[pt_idx] = final_label_for_P.value
        label_counts_enum_keys[final_label_for_P] += 1
        points_labeled_count += 1

        # --- Update Debug Trace Information ---
        if is_tracing_this_point:
            # Populate all standard trace fields
            current_trace_info["T1_raw_occ_vs_imm"] = raw_occ_P_vs_imm_hist.name
            current_trace_info["T1_mcc_performed"] = mcc1_performed
            current_trace_info["T1_mcc_result_bool"] = mcc1_result_bool # Already captured if T1_mcc_full_debug exists
            current_trace_info["T1_outcome_after_mcc"] = outcome_test1_mcc.name
            
            current_trace_info["T4_perp_event_count"] = occluding_count_for_test1_N
            current_trace_info["T4_perp_event_passed"] = passed_perpendicular_event_test
            current_trace_info["OUTCOME_T1_T4"] = final_outcome_tests1_and_4.name # Before refinement
            
            current_trace_info["Test2_passed"] = test2_passed # Boolean result
            current_trace_info["OUTCOME_T2"] = outcome_test2.name # Before refinement

            current_trace_info["T3_mcc_on_P_result_bool"] = mcc_for_test3_result_bool
            current_trace_info["T3_mcc_rejects_P"] = mcc_rejects_P_for_test3
            current_trace_info["Test3_chain_passed"] = test3_chain_passed # Boolean result
            current_trace_info["OUTCOME_T3"] = outcome_test3.name # Before refinement
            
            current_trace_info["preliminary_is_dynamic_before_refinement"] = preliminary_is_dynamic
            current_trace_info["MCC_Refinement_performed"] = mcc_refinement_performed
            current_trace_info["MCC_Refinement_result_bool"] = mcc_refinement_result_bool
            current_trace_info["MCC_Refinement_overrode_to_static"] = mcc_refinement_overrode_to_static
            
            current_trace_info["final_label"] = final_label_for_P.name # The actual final label
            current_trace_info["config_interp_enabled_during_call"] = self.mc_interp_enabled
            
            self.debug_collector.trace(pt_idx, **current_trace_info)
            
    # Convert enum keys to values for the return dict
    # This section remains the same
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