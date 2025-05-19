import numpy as np
from typing import Tuple, Optional, Dict, List, Any

# Assuming these are in src.core...
from ..core.depth_image import DepthImage
from ..core.m_detector.base import MDetector # For MDetector type hint
from ..core.constants import OcclusionResult
from ..core.debug_collector import PointDebugCollector
# Assuming this is in src.utils... (if not, adjust path)
# from ..utils.transformations import transform_points_numpy # Not directly needed if points are passed correctly

# --- Helper for Step 1 Recalculation ---
def _debug_recalculate_step1_occlusion_details(
    debug_point_global_coord: np.ndarray,
    current_di_timestamp: float, # For logging
    current_di_idx_in_lib: int,
    immediate_past_di: DepthImage,
    detector_instance: MDetector
) -> Tuple[OcclusionResult, float, float, float, Optional[Tuple[int, int]], Dict[str, Any]]:
    """
    Performs the detailed recalculation for Step 1 (Occlusion vs. Immediate Past).
    Returns:
        - OcclusionResult enum
        - recalc_d_curr
        - recalc_min_depth_in_region
        - recalc_max_depth_in_region
        - proj_pixel_indices_in_past_di
        - debug_info dictionary for this step
    """
    step1_recalc_debug_info = {}
    print(f"    RECALC_STEP1_DETAILS: Recalculating for point (Current DI TS {current_di_timestamp}, Lib Idx {current_di_idx_in_lib}):")
    print(f"      RECALC_STEP1_DETAILS: Debug Point Global Coords: {np.round(debug_point_global_coord,6).tolist()}")
    print(f"      RECALC_STEP1_DETAILS: Using Immediate Past DI TS: {immediate_past_di.timestamp}") # Assuming idx_of_imm_past_di is already determined by caller
    # print(f"      RECALC_STEP1_DETAILS: Imm. Past DI T_global_lidar:\n{immediate_past_di.image_pose_global}") # Can be verbose

    recalc_d_curr = float('nan')
    recalc_min_depth_in_region = float('inf')
    recalc_max_depth_in_region = float('-inf')
    raw_occlusion_enum_recalc = OcclusionResult.UNDETERMINED
    proj_pixel_indices_in_past_di_for_output = None

    # Check if past DI is prepared (simplified check here, assuming it should be)
    fully_prepared_past_di = True
    if immediate_past_di.local_sph_coords_for_points is None or not immediate_past_di.pixel_original_indices:
        print("      RECALC_STEP1_DETAILS: WARNING - Immediate past DI seems unprepared (missing local_sph_coords or pixel_original_indices).")
        fully_prepared_past_di = False # Context search will likely fail

    if not fully_prepared_past_di:
        print("      RECALC_STEP1_DETAILS: Aborting due to unprepared past DI.")
        step1_recalc_debug_info['reason'] = "Past DI unprepared"
    else:
        _point_in_past_sensor_frame_cartesian, sph_coords_in_past_frame, proj_pixel_indices_in_past_di = \
            immediate_past_di.project_point_to_pixel_indices(debug_point_global_coord)
        
        proj_pixel_indices_in_past_di_for_output = proj_pixel_indices_in_past_di

        if _point_in_past_sensor_frame_cartesian is not None:
            print(f"      RECALC_STEP1_DETAILS: Debug Pt in Past DI Sensor Frame: {np.round(_point_in_past_sensor_frame_cartesian.flatten(),3).tolist()}")
        
        if sph_coords_in_past_frame is not None:
            recalc_d_curr = sph_coords_in_past_frame[2]
            print(f"      RECALC_STEP1_DETAILS: Recalculated d_curr: {recalc_d_curr:.3f}")
            # print(f"      RECALC_STEP1_DETAILS: Full Spherical Coords in Past DI: {np.round(sph_coords_in_past_frame,3).tolist()}")
        else:
            print(f"      RECALC_STEP1_DETAILS: Could not get spherical coordinates for debug point in past DI. d_curr is NaN.")

        print(f"      RECALC_STEP1_DETAILS: Projected (v,h) in past DI: {proj_pixel_indices_in_past_di}")

        if proj_pixel_indices_in_past_di is not None and not np.isnan(recalc_d_curr):
            v_proj, h_proj = proj_pixel_indices_in_past_di
            neigh_v = detector_instance.neighbor_search_pixels_v
            neigh_h = detector_instance.neighbor_search_pixels_h
            print(f"      RECALC_STEP1_DETAILS: Neighbor search window: v +/- {neigh_v}, h +/- {neigh_h}")
            
            all_depths_in_region = []
            for v_offset in range(-neigh_v, neigh_v + 1):
                for h_offset in range(-neigh_h, neigh_h + 1):
                    check_v, check_h = v_proj + v_offset, h_proj + h_offset
                    if 0 <= check_v < immediate_past_di.num_pixels_v and \
                       0 <= check_h < immediate_past_di.num_pixels_h:
                        pixel_info = immediate_past_di.get_pixel_info(check_v, check_h)
                        original_indices_in_px = pixel_info.get('original_indices_in_pixel', [])
                        if original_indices_in_px:
                            for original_idx in original_indices_in_px:
                                if immediate_past_di.local_sph_coords_for_points is not None and \
                                   original_idx < len(immediate_past_di.local_sph_coords_for_points):
                                    depth_val = immediate_past_di.local_sph_coords_for_points[original_idx, 2]
                                    all_depths_in_region.append(depth_val)
            
            if all_depths_in_region:
                recalc_min_depth_in_region = np.min(all_depths_in_region)
                recalc_max_depth_in_region = np.max(all_depths_in_region)
                print(f"      RECALC_STEP1_DETAILS: Collected {len(all_depths_in_region)} depths from context region.")
                print(f"      RECALC_STEP1_DETAILS: Recalculated min_depth_in_region: {recalc_min_depth_in_region:.3f}")
                print(f"      RECALC_STEP1_DETAILS: Recalculated max_depth_in_region: {recalc_max_depth_in_region:.3f}")
                
                epsilon_depth_occlusion = detector_instance.epsilon_depth_occlusion
                raw_occlusion_str_recalc = "UNDETERMINED"
                if not np.isnan(recalc_d_curr) and not np.isinf(recalc_min_depth_in_region) and \
                   recalc_d_curr < recalc_min_depth_in_region - epsilon_depth_occlusion:
                    raw_occlusion_str_recalc = "OCCLUDING_IMAGE"
                    raw_occlusion_enum_recalc = OcclusionResult.OCCLUDING_IMAGE
                elif not np.isnan(recalc_d_curr) and not np.isinf(recalc_max_depth_in_region) and \
                     recalc_d_curr > recalc_max_depth_in_region + epsilon_depth_occlusion:
                     raw_occlusion_str_recalc = "OCCLUDED_BY_IMAGE"
                     raw_occlusion_enum_recalc = OcclusionResult.OCCLUDED_BY_IMAGE
                
                step1_recalc_debug_info['reason'] = f"d_curr ({recalc_d_curr:.3f}) vs min_depth ({recalc_min_depth_in_region:.3f}) / max_depth ({recalc_max_depth_in_region:.3f}) with eps ({epsilon_depth_occlusion:.3f})"
                print(f"      RECALC_STEP1_DETAILS: Result: {raw_occlusion_str_recalc}, Reason: {step1_recalc_debug_info['reason']}")
                if not np.isinf(recalc_min_depth_in_region) and not np.isnan(recalc_d_curr) and raw_occlusion_enum_recalc == OcclusionResult.OCCLUDING_IMAGE:
                     print(f"              Condition: {recalc_d_curr:.3f} < {recalc_min_depth_in_region - epsilon_depth_occlusion:.3f} -> {recalc_d_curr < recalc_min_depth_in_region - epsilon_depth_occlusion}")

            else:
                step1_recalc_debug_info['reason'] = "No context points found in region"
                print(f"      RECALC_STEP1_DETAILS: Result: UNDETERMINED (no context)")
        elif np.isnan(recalc_d_curr):
            step1_recalc_debug_info['reason'] = "Projection failed - d_curr NaN"
            print(f"      RECALC_STEP1_DETAILS: Result: UNDETERMINED (projection failed - d_curr NaN)")
        else: # proj_pixel_indices_in_past_di was None
            step1_recalc_debug_info['reason'] = "Projection to past DI pixel failed (e.g. out of FoV)"
            print(f"      RECALC_STEP1_DETAILS: Result: UNDETERMINED (projection to past DI pixel failed)")
            
    return raw_occlusion_enum_recalc, recalc_d_curr, recalc_min_depth_in_region, recalc_max_depth_in_region, proj_pixel_indices_in_past_di_for_output, step1_recalc_debug_info


# --- Main Orchestrator ---
def debug_point_m_detector_logic(
    pt_idx: int,
    current_di: DepthImage,
    current_di_idx_in_lib: int,
    detector_instance: MDetector,
    gt_indices_dict: Optional[Dict[str, np.ndarray]] = None,
    debug_collector_instance: Optional[PointDebugCollector] = None
):
    """Provides a step-by-step debug trace for a single point through M-Detector's causal logic."""
    print(f"\n--- Debugging Point Index: {pt_idx} (TS: {current_di.timestamp}) ---")

    if current_di.original_points_global_coords is None or \
       pt_idx >= len(current_di.original_points_global_coords):
        print(f"ERROR: Point index {pt_idx} is out of bounds for current_di or points not loaded.")
        return
    
    point_global = current_di.original_points_global_coords[pt_idx, :3] # Use the :3 to ensure it's just XYZ
    print(f"  Global Coords: {np.round(point_global, 3).tolist()}")

    if gt_indices_dict:
        gt_status = "GT Unknown"
        # Ensure gt_indices_dict keys are present and are sets for efficient lookup
        gt_dyn_set = set(gt_indices_dict.get('gt_dynamic_indices', []))
        gt_static_set = set(gt_indices_dict.get('gt_static_indices', []))
        gt_unlabeled_set = set(gt_indices_dict.get('unlabeled_indices', []))
        if pt_idx in gt_dyn_set: gt_status = "GT Dynamic"
        elif pt_idx in gt_static_set: gt_status = "GT Static"
        elif pt_idx in gt_unlabeled_set: gt_status = "GT Unlabeled (Background)"
        print(f"  Ground Truth Status: {gt_status}")

    trace_data = None
    if debug_collector_instance and debug_collector_instance.is_tracing(pt_idx):
        trace_data = debug_collector_instance.get_trace_data(pt_idx)
        if trace_data:
            print("  Debug Collector Trace (from MDetector run):")
            for key, value in trace_data.items():
                val_to_print = value
                if isinstance(value, str) and value in OcclusionResult.__members__: pass
                elif isinstance(value, int):
                    try: val_to_print = f"{OcclusionResult(value).name} ({value})"
                    except ValueError: pass
                print(f"    {key}: {val_to_print}")
        else:
            print("  No trace data found in collector for this point.")

    # --- Step 1: Raw Occlusion vs. Immediate Past DI ---
    print("\n  Step 1: Raw Occlusion vs. Immediate Past DI")
    raw_occ_vs_imm_val_from_di_attr = OcclusionResult.UNDETERMINED.value
    if current_di.raw_occlusion_results_vs_history is not None and \
       current_di.raw_occlusion_results_vs_history.shape[0] > pt_idx and \
       current_di.raw_occlusion_results_vs_history.shape[1] > 0:
        raw_occ_vs_imm_val_from_di_attr = current_di.raw_occlusion_results_vs_history[pt_idx, 0]
        print(f"    Stored Raw Result (from current_di.raw_occlusion_results_vs_history[:,0]): {OcclusionResult(raw_occ_vs_imm_val_from_di_attr).name}")
    else:
        print("    Stored Raw Result not available in current_di.raw_occlusion_results_vs_history.")

    # Initial verification using _perform_detailed_pixel_level_occlusion_check
    idx_of_imm_past_di_for_initial_check = current_di_idx_in_lib - 1
    recheck_result_step1_initial = OcclusionResult(raw_occ_vs_imm_val_from_di_attr) # Default to stored

    if 0 <= idx_of_imm_past_di_for_initial_check < len(detector_instance.depth_image_library.get_all_images()):
        immediate_past_di_for_initial_check = detector_instance.depth_image_library.get_image_by_index(idx_of_imm_past_di_for_initial_check)
        if immediate_past_di_for_initial_check:
            print(f"    Detailed Check against Immediate Past DI (TS: {immediate_past_di_for_initial_check.timestamp}):")
            recheck_result_step1_initial, step1_initial_debug_info = _perform_detailed_pixel_level_occlusion_check(
                point_global, immediate_past_di_for_initial_check, detector_instance
            )
            print(f"      Re-calculated Detailed Check Result (Initial): {recheck_result_step1_initial.name}")
            print(f"      Reason: {step1_initial_debug_info.get('reason', 'N/A')}")
            if recheck_result_step1_initial.value != raw_occ_vs_imm_val_from_di_attr:
                print(f"      WARNING: Initial re-calculated result ({recheck_result_step1_initial.name}) MISMATCHES stored ({OcclusionResult(raw_occ_vs_imm_val_from_di_attr).name})!")
        else:
            print("    Could not retrieve immediate past DI for initial detailed re-check.")
    else:
        print("    No immediate past DI available for initial detailed re-check (current DI is likely first or history too short).")
    
    # This is the recheck_result_step1 that subsequent steps should use
    recheck_result_step1_final_enum = recheck_result_step1_initial


    # --- Full Recalculation for Step 1 (with more prints) ---
    # This part now calls the new helper _debug_recalculate_step1_occlusion_details
    imm_past_di_for_recalc = None
    if current_di_idx_in_lib > 0:
        idx_of_imm_past_di_for_recalc = current_di_idx_in_lib - 1
        if 0 <= idx_of_imm_past_di_for_recalc < len(detector_instance.depth_image_library.get_all_images()):
            imm_past_di_for_recalc = detector_instance.depth_image_library.get_image_by_index(idx_of_imm_past_di_for_recalc)

    if imm_past_di_for_recalc:
        # Call the new helper function
        recalc_step1_enum, recalc_d_curr, recalc_min_depth, recalc_max_depth, recalc_proj_pixels, _ = \
            _debug_recalculate_step1_occlusion_details(
                point_global, # Use the definitive point_global from current_di
                current_di.timestamp,
                current_di_idx_in_lib,
                imm_past_di_for_recalc,
                detector_instance
            )
        # The prints are now inside _debug_recalculate_step1_occlusion_details
        # We can use the returned values if needed for further comparison here,
        # but recheck_result_step1_final_enum (from the initial detailed check) is what's used downstream.
        # For strictness, we could assert that recalc_step1_enum matches recheck_result_step1_final_enum
        if recalc_step1_enum != recheck_result_step1_final_enum:
            print(f"    INTERNAL DEBUG WARNING: Result from _debug_recalculate_step1_occlusion_details ({recalc_step1_enum.name}) "
                  f"differs from initial detailed check ({recheck_result_step1_final_enum.name}). This might indicate subtle differences in logic or state.")
            # Decide which one to trust for subsequent steps. For now, using the initial detailed check's result.
    else:
        print("    Skipping full Step 1 recalculation details as immediate past DI is not available.")


    # --- Step 2: Map Consistency Check (MCC) ---
    print("\n  Step 2: Map Consistency Check (MCC)")
    mcc_config = detector_instance.config.get('map_consistency_check', {})
    mcc_enabled = mcc_config.get('enabled', False)
    
    # Use recheck_result_step1_final_enum for this logic
    should_mcc_run_calculated = mcc_enabled and (recheck_result_step1_final_enum == OcclusionResult.OCCLUDING_IMAGE)
    print(f"    MCC Enabled in Config: {mcc_enabled}")
    print(f"    Re-checked Raw Occ vs Imm was OCCLUDING_IMAGE: {recheck_result_step1_final_enum == OcclusionResult.OCCLUDING_IMAGE}")
    print(f"    MCC Expected to Run (based on re-check): {should_mcc_run_calculated}")

    mcc_performed_trace = trace_data.get('2_mcc_performed', 'N/A') if trace_data else 'N/A'
    mcc_is_consistent_trace_val = trace_data.get('2a_mcc_is_consistent_result', 'N/A') if trace_data else 'N/A'
    print(f"    MCC Performed (from trace): {mcc_performed_trace}")
    if mcc_performed_trace is True: # Note: trace might store bool
        print(f"    MCC IsConsistent (from trace): {mcc_is_consistent_trace_val}")

    recheck_is_consistent_mcc = None
    if should_mcc_run_calculated:
        print("    Performing direct MCC call for verification:")
        recheck_is_consistent_mcc, mcc_direct_debug_info = detector_instance.is_map_consistent(
            point_global, current_di.timestamp, check_direction='past', return_debug_info=True
        )
        print(f"      Direct MCC Call Result: {'MAP CONSISTENT (Static)' if recheck_is_consistent_mcc else 'MAP INCONSISTENT (Dynamic)'}")
        if mcc_direct_debug_info:
            print(f"        MCC Debug: Relevant DIs={mcc_direct_debug_info.get('relevant_dis_count')}, Consistent DIs={mcc_direct_debug_info.get('consistent_count_final')}, Reason='{mcc_direct_debug_info.get('reason_for_result')}'")
        if mcc_performed_trace is True and mcc_is_consistent_trace_val != recheck_is_consistent_mcc:
            print(f"      WARNING: Direct MCC call ({recheck_is_consistent_mcc}) MISMATCHES trace ({mcc_is_consistent_trace_val})!")
    
    # --- Step 3: Occlusion Result after MCC ---
    print("\n  Step 3: Occlusion Result after MCC (combining re-checked Step 1 & 2)")
    occ_after_mcc_recalculated = recheck_result_step1_final_enum
    if should_mcc_run_calculated and recheck_is_consistent_mcc is not None:
        if recheck_is_consistent_mcc is True: 
            occ_after_mcc_recalculated = OcclusionResult.OCCLUDED_BY_IMAGE
    
    print(f"    Re-calculated Result after MCC: {occ_after_mcc_recalculated.name}")
    occ_after_mcc_trace_str = trace_data.get('3_occ_after_mcc', 'N/A') if trace_data else 'N/A'
    print(f"    Result after MCC (from trace): {occ_after_mcc_trace_str}")
    if trace_data and occ_after_mcc_trace_str != 'N/A' and occ_after_mcc_recalculated.name != occ_after_mcc_trace_str:
         print(f"    WARNING: Re-calculated occ_after_mcc ({occ_after_mcc_recalculated.name}) MISMATCHES trace ({occ_after_mcc_trace_str})!")
    
    # --- Step 4: Event Test ---
    print("\n  Step 4: Event Test (Perpendicular Test)")
    event_cfg = detector_instance.config.get('event_tests', {})
    n_hist_for_event_test = event_cfg.get('test1_N_depth_images', 3) 
    m1_threshold = event_cfg.get('test1_M1_threshold', 2)
    
    occluding_count_recalculated = 0
    passed_event_test_recalculated = False
    if current_di.raw_occlusion_results_vs_history is not None and \
       current_di.raw_occlusion_results_vs_history.shape[0] > pt_idx:
        history_for_point = current_di.raw_occlusion_results_vs_history[pt_idx, :n_hist_for_event_test]
        occluding_count_recalculated = np.sum(history_for_point == OcclusionResult.OCCLUDING_IMAGE.value)
        passed_event_test_recalculated = (occluding_count_recalculated >= m1_threshold)
        print(f"    Raw History (first {min(n_hist_for_event_test, history_for_point.shape[0])} DIs from DI attr): {[OcclusionResult(v).name for v in history_for_point]}")
    else:
        print("    Raw occlusion history not available in current_di for event test re-calculation.")

    print(f"    Re-calculated Occluding Count = {occluding_count_recalculated} (N_hist_cfg={n_hist_for_event_test}, M1_thresh_cfg={m1_threshold})")
    print(f"    Re-calculated Passed Event Test: {passed_event_test_recalculated}")
    
    if trace_data:
        trace_event_count = trace_data.get('4_event_test_occluding_count', 'N/A')
        trace_event_passed = trace_data.get('5_event_test_passed', 'N/A')
        print(f"    Event Test Occluding Count (from trace): {trace_event_count}")
        print(f"    Event Test Passed (from trace): {trace_event_passed}")
        if isinstance(trace_event_count, (int, float)) and trace_event_count != occluding_count_recalculated:
            print(f"    WARNING: Event count mismatch! Re-calculated={occluding_count_recalculated}, Trace={trace_event_count}")
        if isinstance(trace_event_passed, bool) and trace_event_passed != passed_event_test_recalculated:
            print(f"    WARNING: Event passed mismatch! Re-calculated={passed_event_test_recalculated}, Trace={trace_event_passed}")

    # --- Step 5: Final Label ---
    print("\n  Step 5: Final Label Decision Logic")
    final_label_val_from_di_attr = current_di.mdet_labels_for_points[pt_idx]
    print(f"    Final M-Detector Label (from DI attribute): {OcclusionResult(final_label_val_from_di_attr).name}")

    final_label_recalculated = OcclusionResult.UNDETERMINED
    if passed_event_test_recalculated:
        if occ_after_mcc_recalculated == OcclusionResult.OCCLUDED_BY_IMAGE:
            final_label_recalculated = OcclusionResult.UNDETERMINED
        else:
            final_label_recalculated = OcclusionResult.OCCLUDING_IMAGE
    elif occ_after_mcc_recalculated != OcclusionResult.UNDETERMINED:
        final_label_recalculated = occ_after_mcc_recalculated
    
    print(f"    Re-calculated Final Label (based on re-derived logic): {final_label_recalculated.name}")
    if final_label_val_from_di_attr != final_label_recalculated.value:
        print(f"    WARNING: Re-calculated final label ({final_label_recalculated.name}) MISMATCHES stored DI label ({OcclusionResult(final_label_val_from_di_attr).name})!")

    if trace_data:
        final_label_trace_str = trace_data.get('final_label', 'N/A')
        print(f"    Final Label (from trace): {final_label_trace_str}")
        # decision_logic_trace = trace_data.get('decision_logic', 'N/A') # Not used here but good for completeness
        # print(f"    Decision Logic (from trace): {decision_logic_trace}")
    
    print(f"--- End Debugging Point Index: {pt_idx} ---")


# --- Original Helper Functions (to be included in the same file) ---

def get_misclassified_points(
    current_di_with_labels: DepthImage,
    gt_indices_dict: Dict[str, np.ndarray], 
    mdet_dynamic_label: OcclusionResult = OcclusionResult.OCCLUDING_IMAGE
) -> Dict[str, List[int]]:
    """
    Identifies misclassified points by comparing M-Detector labels with GT labels.
    gt_indices_dict should be the output of src.utils.validation_utils.get_gt_dynamic_points_for_sweep
    which contains indices relative to current_di_with_labels.original_points_global_coords.
    """
    misclassified = {
        "fp_dynamic": [],
        "fn_dynamic": [],
        "other_static_as_occluded": [],
        "other_dynamic_as_occluded": [],
    }
    if current_di_with_labels.mdet_labels_for_points is None:
        print("ERROR (get_misclassified_points): M-Detector labels not available in current_di_with_labels.")
        return misclassified
    if not gt_indices_dict:
        print("ERROR (get_misclassified_points): gt_indices_dict not provided.")
        return misclassified

    mdet_labels = current_di_with_labels.mdet_labels_for_points
    gt_dynamic_indices = set(gt_indices_dict.get('gt_dynamic_indices', np.array([], dtype=int)))
    gt_static_indices = set(gt_indices_dict.get('gt_static_indices', np.array([], dtype=int)))
    gt_unlabeled_indices = set(gt_indices_dict.get('unlabeled_indices', np.array([], dtype=int)))

    for i in range(len(mdet_labels)):
        mdet_label_enum = OcclusionResult(mdet_labels[i])
        is_gt_dynamic = i in gt_dynamic_indices
        is_gt_static_or_unlabeled = i in gt_static_indices or i in gt_unlabeled_indices

        if mdet_label_enum == mdet_dynamic_label:
            if is_gt_static_or_unlabeled:
                misclassified["fp_dynamic"].append(i)
        elif mdet_label_enum == OcclusionResult.OCCLUDED_BY_IMAGE:
            if is_gt_static_or_unlabeled:
                 misclassified["other_static_as_occluded"].append(i)
            elif is_gt_dynamic:
                misclassified["other_dynamic_as_occluded"].append(i)
        else: # MDet is UNDETERMINED or EMPTY_IN_IMAGE
            if is_gt_dynamic:
                misclassified["fn_dynamic"].append(i)
                 
    # print(f"Misclassification Summary (MDet Dynamic = {mdet_dynamic_label.name}):")
    # print(f"  False Positives (Static/Unlabeled -> MDet Dynamic): {len(misclassified['fp_dynamic'])}")
    # print(f"  False Negatives (Dynamic -> MDet Not Dynamic): {len(misclassified['fn_dynamic'])}")
    # print(f"  Other: Static/Unlabeled labeled by MDet as Occluded: {len(misclassified['other_static_as_occluded'])}")
    # print(f"  Other: Dynamic labeled by MDet as Occluded: {len(misclassified['other_dynamic_as_occluded'])}")
    return misclassified

def _perform_detailed_pixel_level_occlusion_check(
    point_global: np.ndarray, 
    historical_di: DepthImage, 
    detector_instance: MDetector 
) -> Tuple[OcclusionResult, Dict[str, Any]]:
    """
    Performs a detailed pixel-level occlusion check for a point against a historical DI.
    Returns the OcclusionResult and a dictionary with intermediate values for debugging.
    """
    debug_info = {}
    
    neighbor_v = detector_instance.neighbor_search_pixels_v
    neighbor_h = detector_instance.neighbor_search_pixels_h
    epsilon_occ = detector_instance.epsilon_depth_occlusion

    point_in_hist_di_frame, sph_coords_curr, pixel_indices_in_hist_di = \
        historical_di.project_point_to_pixel_indices(point_global)

    debug_info['point_in_hist_di_frame'] = point_in_hist_di_frame.tolist() if point_in_hist_di_frame is not None else None
    debug_info['sph_coords_curr'] = sph_coords_curr.tolist() if sph_coords_curr is not None else None
    debug_info['pixel_indices_in_hist_di'] = pixel_indices_in_hist_di

    if pixel_indices_in_hist_di is None or sph_coords_curr is None:
        debug_info['reason'] = "Point projects outside historical DI's FoV or invalid projection"
        return OcclusionResult.UNDETERMINED, debug_info

    v_idx_curr_proj, h_idx_curr_proj = pixel_indices_in_hist_di
    d_curr = sph_coords_curr[2] 
    debug_info['d_curr'] = d_curr

    v_start = max(0, v_idx_curr_proj - neighbor_v)
    v_end = min(historical_di.num_pixels_v, v_idx_curr_proj + neighbor_v + 1)
    h_start = max(0, h_idx_curr_proj - neighbor_h)
    h_end = min(historical_di.num_pixels_h, h_idx_curr_proj + neighbor_h + 1)
    
    if historical_di.pixel_min_depth is None or \
       historical_di.pixel_max_depth is None or \
       historical_di.pixel_count is None:
        debug_info['reason'] = "Historical DI pixel data (min_depth, max_depth, or count) is None."
        return OcclusionResult.UNDETERMINED, debug_info

    region_min_depths_slice = historical_di.pixel_min_depth[v_start:v_end, h_start:h_end]
    region_max_depths_slice = historical_di.pixel_max_depth[v_start:v_end, h_start:h_end]
    region_counts_slice = historical_di.pixel_count[v_start:v_end, h_start:h_end]
    
    has_data_mask = region_counts_slice > 0
    found_data_in_region = np.any(has_data_mask)
    
    if not found_data_in_region:
        debug_info['reason'] = "No points in the historical DI's pixel region"
        return OcclusionResult.EMPTY_IN_IMAGE, debug_info

    min_depth_in_region = np.min(region_min_depths_slice[has_data_mask])
    max_depth_in_region = np.max(region_max_depths_slice[has_data_mask])
    debug_info['min_depth_in_region'] = min_depth_in_region
    debug_info['max_depth_in_region'] = max_depth_in_region
    debug_info['epsilon_depth_occlusion'] = epsilon_occ

    if d_curr > max_depth_in_region + epsilon_occ:
        debug_info['reason'] = f"d_curr ({d_curr:.3f}) > max_depth_in_region ({max_depth_in_region:.3f}) + eps ({epsilon_occ:.3f})"
        return OcclusionResult.OCCLUDED_BY_IMAGE, debug_info
    
    if d_curr < min_depth_in_region - epsilon_occ:
        debug_info['reason'] = f"d_curr ({d_curr:.3f}) < min_depth_in_region ({min_depth_in_region:.3f}) - eps ({epsilon_occ:.3f})"
        return OcclusionResult.OCCLUDING_IMAGE, debug_info
    
    debug_info['reason'] = "d_curr is within or too close to the historical depth range"
    return OcclusionResult.UNDETERMINED, debug_info