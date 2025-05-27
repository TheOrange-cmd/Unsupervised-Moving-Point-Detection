# src/core/m_detector/event_tests.py
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Any, Optional 
import logging
from .adaptive_epsilon_utils import calculate_adaptive_epsilon

if TYPE_CHECKING:
    from .base import MDetector
    from ..depth_image import DepthImage 

logger_et = logging.getLogger(__name__) # Logger for this module

def execute_test2_parallel_motion(
    mdetector_instance: 'MDetector',
    point_global_P: np.ndarray,          # The initial point from current_di being evaluated for Test 2
    current_di_of_P: 'DepthImage',       # The DI from which point_global_P originates
    original_pt_idx_of_P_in_current_di: int, # Index of point_global_P in current_di_of_P
    current_di_timestamp: float,         # Timestamp of current_di_of_P
    current_di_idx_in_lib: int,          # Library index of current_di_of_P
    return_debug_info: bool = False
) -> Any: # Returns bool or Tuple[bool, List[Dict[str, Any]]]
    """
    Implements Test 2 (Parallel Motion / Event by Disappearance).
    P is occluded by P1 (in H1), P1 by P2 (in H2), ..., P(M2-1) by PM2 (in HM2).
    MCC is on P1, P2, ..., PM2 (the occluders in historical frames).
    Adaptive epsilon is used for the detailed occlusion check.
    """
    logger_et.debug(f"Executing Test 2 for P idx {original_pt_idx_of_P_in_current_di} ({point_global_P.tolist()}) "
                    f"from DI @ ts {current_di_timestamp:.2f} (lib_idx {current_di_idx_in_lib})")
    
    # --- Initialize Final Debug Info ---
    test2_final_debug_info: Dict[str, Any] = {
        "point_global_P_initial": point_global_P.tolist(),
        "original_pt_idx_of_P": original_pt_idx_of_P_in_current_di,
        "current_timestamp_of_P": current_di_timestamp,
        "current_di_idx_in_lib_of_P": current_di_idx_in_lib,
        "test_passed": False, 
        "reason_for_fail_or_pass": "Test did not meet all conditions.", 
        "num_historical_DIs_M2": mdetector_instance.test2_M2_depth_images,
        "steps_debug": [] 
    }
    debug_steps_list = test2_final_debug_info["steps_debug"]

    if mdetector_instance.test2_M2_depth_images <= 0:
        test2_final_debug_info["reason_for_fail_or_pass"] = "Test disabled (M2 <= 0)"
        debug_steps_list.append({"step": 0, "status": "Test disabled (M2 <= 0)", "details": "N/A"})
        if return_debug_info: return False, test2_final_debug_info
        return False

    # --- Initialize variables for the chain ---
    # point_to_be_occluded_in_step: The point that needs to be occluded in the current step of the chain.
    # Starts as the initial point_global_P.
    point_to_be_occluded_in_step = point_global_P
    
    # d_anchor_for_pto: The depth of point_to_be_occluded_in_step in its *originating* DI's sensor frame.
    d_anchor_for_pto: Optional[float] = None
    if current_di_of_P.local_sph_coords_for_points is not None and \
       original_pt_idx_of_P_in_current_di < current_di_of_P.local_sph_coords_for_points.shape[0]:
        d_anchor_for_pto = current_di_of_P.local_sph_coords_for_points[original_pt_idx_of_P_in_current_di, 2]
    
    # timestamp_of_origin_di_for_pto: Timestamp of the DI from which point_to_be_occluded_in_step originated.
    timestamp_of_origin_di_for_pto = current_di_timestamp
    
    # original_idx_of_pto_in_origin_di: Original index of point_to_be_occluded_in_step in its originating DI.
    # This is used for detailed tracing if the point being evaluated is the initial P.
    original_idx_of_pto_in_origin_di = original_pt_idx_of_P_in_current_di

    test2_final_debug_info["initial_d_anchor_of_P"] = d_anchor_for_pto

    for k_step in range(mdetector_instance.test2_M2_depth_images): # k_step from 0 to M2-1
        step_debug_info: Dict[str, Any] = {
            "step": k_step,
            "point_to_be_occluded_global": point_to_be_occluded_in_step.tolist(),
            "d_anchor_of_pto": d_anchor_for_pto,
            "origin_di_timestamp_of_pto": timestamp_of_origin_di_for_pto,
            "status": "pending",
            "historical_di_checked_idx_in_lib": None,
            "historical_di_checked_timestamp": None,
            "projection_in_hist_successful": False,
            "occlusion_check_details": [] # List to store details for each candidate in the pixel
        }

        historical_di_to_check_idx_in_lib = current_di_idx_in_lib - 1 - k_step
        step_debug_info["historical_di_checked_idx_in_lib"] = historical_di_to_check_idx_in_lib

        if historical_di_to_check_idx_in_lib < 0:
            step_debug_info["status"] = f"Failed: Not enough historical DIs (needed idx {historical_di_to_check_idx_in_lib})."
            test2_final_debug_info["reason_for_fail_or_pass"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test2_final_debug_info
            return False
        
        historical_di_k = mdetector_instance.depth_image_library.get_image_by_index(historical_di_to_check_idx_in_lib)
        if not historical_di_k or not historical_di_k.is_prepared_for_projection():
            ts_val_str = str(historical_di_k.timestamp) if historical_di_k and hasattr(historical_di_k, 'timestamp') else "N/A"
            step_debug_info["historical_di_checked_timestamp"] = ts_val_str
            step_debug_info["status"] = f"Failed: Historical DI {historical_di_to_check_idx_in_lib} (TS: {ts_val_str}) not found or not prepared."
            test2_final_debug_info["reason_for_fail_or_pass"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test2_final_debug_info
            return False
        step_debug_info["historical_di_checked_timestamp"] = historical_di_k.timestamp

        # Project point_to_be_occluded_in_step into historical_di_k
        _, _, px_indices_pto_in_hist_k = historical_di_k.project_point_to_pixel_indices(point_to_be_occluded_in_step)
        if px_indices_pto_in_hist_k is None:
            step_debug_info["status"] = f"Failed: Point to be occluded did not project into historical DI {historical_di_to_check_idx_in_lib}."
            test2_final_debug_info["reason_for_fail_or_pass"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test2_final_debug_info
            return False
        step_debug_info["projection_in_hist_successful"] = True
        step_debug_info["projected_pixel_of_pto_in_hist"] = list(px_indices_pto_in_hist_k)

        v_proj, h_proj = px_indices_pto_in_hist_k
        found_valid_occluder_for_this_step = False
        next_point_for_chain_global: Optional[np.ndarray] = None 
        d_anchor_for_next_pto_step: Optional[float] = None
        timestamp_of_origin_di_for_next_pto: Optional[float] = None
        original_idx_of_next_pto_in_origin_di: Optional[int] = None


        pixel_info_hist_k = historical_di_k.get_pixel_info(v_proj, h_proj)
        if pixel_info_hist_k and pixel_info_hist_k.get('original_indices_in_pixel'):
            step_debug_info["num_candidates_in_projected_pixel"] = len(pixel_info_hist_k['original_indices_in_pixel'])
            
            for original_idx_hist_cand in pixel_info_hist_k['original_indices_in_pixel']:
                occlusion_cand_debug: Dict[str, Any] = {
                    "hist_candidate_original_idx": original_idx_hist_cand,
                    "detailed_occlusion_check_inputs": {
                        "d_anchor_of_point_eval": d_anchor_for_pto, # Use current d_anchor of PTO
                        "origin_di_of_point_eval_ts": timestamp_of_origin_di_for_pto
                    },
                    "detailed_occlusion_check_results": {} # To be filled by the call
                }

                if original_idx_hist_cand >= len(historical_di_k.original_points_global_coords):
                    occlusion_cand_debug["error"] = "Index out of bounds for hist_candidate"
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    continue
                
                point_hist_cand_global = historical_di_k.original_points_global_coords[original_idx_hist_cand]
                occlusion_cand_debug["hist_candidate_global"] = point_hist_cand_global.tolist()

                # Determine if point_to_be_occluded_in_step is being traced for detailed check debug
                pt_idx_for_tracing_detailed_check = None
                if mdetector_instance.debug_collector and \
                   timestamp_of_origin_di_for_pto == current_di_timestamp: # Only trace if PTO is the original P
                    if mdetector_instance.debug_collector.is_tracing(original_idx_of_pto_in_origin_di):
                        pt_idx_for_tracing_detailed_check = original_idx_of_pto_in_origin_di

                is_pto_occluded_by_hist_cand = mdetector_instance.check_occlusion_point_level_detailed(
                    point_eval_global=point_to_be_occluded_in_step,
                    d_anchor_of_point_eval=d_anchor_for_pto,
                    origin_di_of_point_eval_ts=timestamp_of_origin_di_for_pto,
                    point_hist_cand_global=point_hist_cand_global,
                    historical_di=historical_di_k,
                    occlusion_type_to_check="OCCLUDED_BY",
                    pt_idx_eval_if_tracing=pt_idx_for_tracing_detailed_check,
                    debug_dict_for_caller=occlusion_cand_debug["detailed_occlusion_check_results"]
                )
                occlusion_cand_debug["is_pto_occluded_by_hist_cand"] = is_pto_occluded_by_hist_cand

                mcc_rejects_this_occluder = False
                if is_pto_occluded_by_hist_cand: 
                    occlusion_cand_debug["mcc_applied_on_point"] = "historical_candidate_occluder"
                    occlusion_cand_debug["mcc_point_coords"] = point_hist_cand_global.tolist()
                    occlusion_cand_debug["mcc_timestamp"] = historical_di_k.timestamp
                    
                    if mdetector_instance.map_consistency_enabled:
                        request_mcc_debug = return_debug_info # Simplified: if outer wants debug, get MCC debug
                        mcc_result_package_test2 = mdetector_instance.is_map_consistent(
                            point_hist_cand_global, historical_di_k, original_idx_hist_cand,
                            historical_di_k.timestamp, check_direction='past',
                            return_debug_info=request_mcc_debug
                        )
                        mcc_is_consistent_bool: bool
                        if request_mcc_debug:
                            mcc_is_consistent_bool, mcc_t2_link_debug_data = mcc_result_package_test2
                            occlusion_cand_debug["mcc_on_hist_cand_full_debug"] = mcc_t2_link_debug_data
                        else:
                            mcc_is_consistent_bool = mcc_result_package_test2
                        if mcc_is_consistent_bool: mcc_rejects_this_occluder = True
                            
                    occlusion_cand_debug["mcc_result_is_consistent_with_map"] = mcc_rejects_this_occluder
                    occlusion_cand_debug["mcc_rejects_this_link"] = mcc_rejects_this_occluder
                
                if is_pto_occluded_by_hist_cand and not mcc_rejects_this_occluder:
                    found_valid_occluder_for_this_step = True
                    next_point_for_chain_global = point_hist_cand_global
                    
                    # Prepare d_anchor and origin info for the next step's PTO
                    if historical_di_k.local_sph_coords_for_points is not None and \
                       original_idx_hist_cand < historical_di_k.local_sph_coords_for_points.shape[0]:
                        d_anchor_for_next_pto_step = historical_di_k.local_sph_coords_for_points[original_idx_hist_cand, 2]
                    else:
                        d_anchor_for_next_pto_step = None 
                        logger_et.warning(f"Test2: Could not get d_anchor for next link: hist_cand_idx {original_idx_hist_cand} in DI TS {historical_di_k.timestamp}")
                    timestamp_of_origin_di_for_next_pto = historical_di_k.timestamp
                    original_idx_of_next_pto_in_origin_di = original_idx_hist_cand

                    occlusion_cand_debug["forms_next_link_in_chain"] = True
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    break # Found a valid occluder for this step
                else:
                    occlusion_cand_debug["forms_next_link_in_chain"] = False
                step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
            
            if found_valid_occluder_for_this_step and next_point_for_chain_global is not None:
                point_to_be_occluded_in_step = next_point_for_chain_global
                d_anchor_for_pto = d_anchor_for_next_pto_step
                timestamp_of_origin_di_for_pto = timestamp_of_origin_di_for_next_pto
                original_idx_of_pto_in_origin_di = original_idx_of_next_pto_in_origin_di # type: ignore
                
                step_debug_info["status"] = "Success: Valid occluder found, chain continues."
                step_debug_info["selected_occluder_for_next_step"] = next_point_for_chain_global.tolist()
                step_debug_info["d_anchor_for_next_pto_step"] = d_anchor_for_pto # Log the d_anchor that will be used
            else:
                step_debug_info["status"] = "Failed: No valid (non-MCC-rejected) occluder found in historical DI pixel."
                test2_final_debug_info["reason_for_fail_or_pass"] = step_debug_info["status"]
                debug_steps_list.append(step_debug_info)
                if return_debug_info: return False, test2_final_debug_info
                return False
        else: # No points in projected pixel of historical_di_k
            step_debug_info["num_candidates_in_projected_pixel"] = 0
            step_debug_info["status"] = "Failed: Projected pixel (of PTO) in historical DI is empty."
            test2_final_debug_info["reason_for_fail_or_pass"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test2_final_debug_info
            return False
        debug_steps_list.append(step_debug_info)

    # If loop completes, all M2 steps were successful
    test2_final_debug_info["test_passed"] = True
    test2_final_debug_info["reason_for_fail_or_pass"] = "All Test 2 chain conditions met."
    if return_debug_info: return True, test2_final_debug_info
    return True

def execute_test3_perpendicular_motion(
    mdetector_instance: 'MDetector',
    point_global_P: np.ndarray,
    current_di: 'DepthImage',                # PASSED IN: The DI from which P originates
    original_pt_idx_of_P_in_current_di: int, # PASSED IN: Index of P in current_di
    current_di_timestamp: float,
    current_di_idx_in_lib: int,
    # adaptive_epsilon_config: Optional[Dict[str, Any]] = None, # NOT ADDING THIS YET
    return_debug_info: bool = False
) -> Any:
    """
    Implements Test 3 (Perpendicular Motion / Event by Appearance/Revealing).
    P (current) occludes P1 (in H_t-1), P1 occludes P2 (in H_t-2), ..., P(M3-1) occludes PM3 (in H_t-M3).
    Initial MCC is on point_global_P.
    This version adds more debug info to help analyze adaptive epsilon post-hoc.
    """
    logger_et.debug(f"Executing Test 3 for P idx {original_pt_idx_of_P_in_current_di} ({point_global_P.tolist()}) "
                    f"from DI @ ts {current_di_timestamp:.0f} (lib_idx {current_di_idx_in_lib})")

    # --- Initialize Final Debug Info ---
    test3_final_debug_info: Dict[str, Any] = {
        "point_global_P_initial": point_global_P.tolist(),
        "original_pt_idx_of_P": original_pt_idx_of_P_in_current_di, # For clarity
        "current_timestamp": current_di_timestamp,
        "current_di_idx_in_lib": current_di_idx_in_lib,
        "test_passed": False, 
        "reason_for_fail": "Test did not meet all conditions.", 
        "num_historical_DIs_M3": mdetector_instance.test3_M3_depth_images,
        "fixed_epsilon_depth_occlusion_used_in_logic": mdetector_instance.epsilon_depth_occlusion, # Log the epsilon actually used
        "steps_debug": [] 
    }
    debug_steps_list = test3_final_debug_info["steps_debug"]

    if mdetector_instance.test3_M3_depth_images <= 0:
        test3_final_debug_info["reason_for_fail"] = "Test disabled (M3 <= 0)"
        debug_steps_list.append({"step": 0, "status": "Test disabled (M3 <= 0)", "details": "N/A"})
        if return_debug_info: return False, test3_final_debug_info
        return False

    point_that_occludes_in_step = point_global_P
    
    # --- Determine d_anchor for the *initial* point_global_P ---
    # This is the depth of P in its own (current_di) sensor frame.
    d_anchor_initial_P: Optional[float] = None
    if current_di.local_sph_coords_for_points is not None and \
       original_pt_idx_of_P_in_current_di < current_di.local_sph_coords_for_points.shape[0]:
        d_anchor_initial_P = current_di.local_sph_coords_for_points[original_pt_idx_of_P_in_current_di, 2]
    test3_final_debug_info["d_anchor_of_initial_P_in_current_di"] = d_anchor_initial_P

    # These variables will track the properties of the point currently acting as the occluder in the chain
    d_anchor_for_current_occluder_point = d_anchor_initial_P
    # original_idx_of_current_occluder_in_its_di = original_pt_idx_of_P_in_current_di # Not strictly needed if d_anchor is passed
    # origin_di_of_current_occluder = current_di # Not strictly needed if d_anchor is passed

    for k_step in range(mdetector_instance.test3_M3_depth_images):
        step_debug_info: Dict[str, Any] = {
            "step": k_step,
            "point_that_occludes_global": point_that_occludes_in_step.tolist(),
            # ADDED: d_anchor of the point_that_occludes_in_step (its depth in its own originating DI)
            "d_anchor_of_occluder_for_this_step": d_anchor_for_current_occluder_point,
            "status": "pending",
            "historical_di_checked_idx_in_lib": None,
            "historical_di_checked_timestamp": None,
            "projection_in_hist_successful": False,
            "occlusion_check_details": [] 
        }

        historical_di_to_check_idx_in_lib = current_di_idx_in_lib - 1 - k_step
        step_debug_info["historical_di_checked_idx_in_lib"] = historical_di_to_check_idx_in_lib
        
        if historical_di_to_check_idx_in_lib < 0:
            step_debug_info["status"] = f"Failed: Not enough historical DIs (needed idx {historical_di_to_check_idx_in_lib})."
            test3_final_debug_info["reason_for_fail"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test3_final_debug_info
            return False
        
        historical_di_k = mdetector_instance.depth_image_library.get_image_by_index(historical_di_to_check_idx_in_lib)
        if not historical_di_k or not historical_di_k.is_prepared_for_projection():
            ts_val_str = str(historical_di_k.timestamp) if historical_di_k and hasattr(historical_di_k, 'timestamp') else "N/A"
            step_debug_info["historical_di_checked_timestamp"] = ts_val_str
            step_debug_info["status"] = f"Failed: Historical DI {historical_di_to_check_idx_in_lib} (TS: {ts_val_str}) not found or not prepared."
            test3_final_debug_info["reason_for_fail"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test3_final_debug_info
            return False
        step_debug_info["historical_di_checked_timestamp"] = historical_di_k.timestamp

        _, sph_coords_occluder_in_hist_frame, px_indices_occluder_in_hist_k = \
            historical_di_k.project_point_to_pixel_indices(point_that_occludes_in_step)
            
        if px_indices_occluder_in_hist_k is None or sph_coords_occluder_in_hist_frame is None:
            step_debug_info["status"] = f"Failed: Current chain occluder did not project into historical DI {historical_di_to_check_idx_in_lib}."
            test3_final_debug_info["reason_for_fail"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test3_final_debug_info
            return False
        step_debug_info["projection_in_hist_successful"] = True
        step_debug_info["projected_pixel_of_occluder_in_hist"] = list(px_indices_occluder_in_hist_k)
        step_debug_info["sph_coords_of_occluder_in_hist_frame"] = sph_coords_occluder_in_hist_frame.tolist()

        v_proj, h_proj = px_indices_occluder_in_hist_k
        found_valid_occluded_point_for_this_step = False
        next_point_for_chain = None 
        # For the next iteration, if successful, these will be updated:
        d_anchor_for_next_occluder = None

        pixel_info_hist_k = historical_di_k.get_pixel_info(v_proj, h_proj)
        if pixel_info_hist_k and pixel_info_hist_k.get('original_indices_in_pixel'):
            step_debug_info["num_candidates_in_projected_pixel"] = len(pixel_info_hist_k['original_indices_in_pixel'])
            
            for original_idx_hist_cand in pixel_info_hist_k['original_indices_in_pixel']:
                occlusion_cand_debug: Dict[str, Any] = {"hist_candidate_original_idx": original_idx_hist_cand}
                if original_idx_hist_cand >= len(historical_di_k.original_points_global_coords):
                    occlusion_cand_debug["error"] = "Index out of bounds for hist_candidate"
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    continue

                point_hist_cand_being_occluded_global = historical_di_k.original_points_global_coords[original_idx_hist_cand]
                occlusion_cand_debug["hist_candidate_being_occluded_global"] = point_hist_cand_being_occluded_global.tolist()

                is_occluding_cand = False 
                depth_occluder_in_hist_frame = sph_coords_occluder_in_hist_frame[2]
                
                sph_coords_hist_cand_native_to_hist: Optional[np.ndarray] = None
                if historical_di_k.local_sph_coords_for_points is not None and \
                   original_idx_hist_cand < historical_di_k.local_sph_coords_for_points.shape[0]:
                    sph_coords_hist_cand_native_to_hist = historical_di_k.local_sph_coords_for_points[original_idx_hist_cand]
                
                occlusion_cand_debug["sph_coords_hist_cand_native_to_hist_frame"] = sph_coords_hist_cand_native_to_hist.tolist() if sph_coords_hist_cand_native_to_hist is not None else None

                if sph_coords_hist_cand_native_to_hist is not None:
                    depth_hist_cand_native_to_hist = sph_coords_hist_cand_native_to_hist[2]
                    occlusion_cand_debug["depth_occluder_in_hist_frame"] = depth_occluder_in_hist_frame
                    occlusion_cand_debug["depth_hist_cand_native_to_hist_frame"] = depth_hist_cand_native_to_hist
                    
                    current_test3_epsilon = calculate_adaptive_epsilon(
                        d_anchor_for_current_occluder_point, # d_anchor of the point *doing the occluding*
                        mdetector_instance.epsilon_depth_occlusion, # di_base
                        mdetector_instance.adaptive_eps_config_occ_depth # The config for general occlusion
                    )
                    occlusion_cand_debug["epsilon_used_for_check"] = current_test3_epsilon # Log it
                    occlusion_cand_debug["d_anchor_for_this_check"] = d_anchor_for_current_occluder_point
                    occlusion_cand_debug["base_epsilon_for_this_check"] = mdetector_instance.epsilon_depth_occlusion


                    if depth_occluder_in_hist_frame < depth_hist_cand_native_to_hist - current_test3_epsilon:
                        is_occluding_cand = True
                else:
                    occlusion_cand_debug["error_sph_hist_cand"] = "Could not get native sph_coords for hist_cand in its own frame."

                occlusion_cand_debug["is_current_occluder_in_front_of_hist_cand_fixed_eps"] = is_occluding_cand
                                
                if is_occluding_cand:
                    found_valid_occluded_point_for_this_step = True
                    next_point_for_chain = point_hist_cand_being_occluded_global
                    # The point that WAS occluded (hist_cand) becomes the occluder for the next step.
                    # Its d_anchor is its native depth in its originating DI (which is historical_di_k for this hist_cand).
                    if sph_coords_hist_cand_native_to_hist is not None:
                         d_anchor_for_next_occluder = sph_coords_hist_cand_native_to_hist[2]

                    occlusion_cand_debug["forms_next_link_in_chain"] = True
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    break 
                else:
                    occlusion_cand_debug["forms_next_link_in_chain"] = False
                step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)

            if found_valid_occluded_point_for_this_step:
                point_that_occludes_in_step = next_point_for_chain
                d_anchor_for_current_occluder_point = d_anchor_for_next_occluder # Update for next iteration's d_anchor
                
                step_debug_info["status"] = "Success: Valid occluded point found, chain continues."
                step_debug_info["selected_occluded_point_for_next_step"] = next_point_for_chain.tolist()
            else:
                step_debug_info["status"] = "Failed: No point found in historical DI pixel that was occluded by current chain point."
                test3_final_debug_info["reason_for_fail"] = step_debug_info["status"]
                debug_steps_list.append(step_debug_info)
                if return_debug_info: return False, test3_final_debug_info
                return False
        else:
            step_debug_info["num_candidates_in_projected_pixel"] = 0
            step_debug_info["status"] = "Failed: Projected pixel (of current chain occluder) in historical DI is empty."
            test3_final_debug_info["reason_for_fail"] = step_debug_info["status"]
            debug_steps_list.append(step_debug_info)
            if return_debug_info: return False, test3_final_debug_info
            return False
        debug_steps_list.append(step_debug_info)

    test3_final_debug_info["test_passed"] = True
    test3_final_debug_info["reason_for_fail"] = "All Test 3 conditions met."
    if return_debug_info: return True, test3_final_debug_info
    return True