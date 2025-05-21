# src/core/m_detector/event_tests.py
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Any, Optional 
# from ..depth_image import DepthImage # If needed directly
# from ..constants import OcclusionResult # If needed directly
import logging

if TYPE_CHECKING:
    from .base import MDetector # Import MDetector only for type hinting

logger_et = logging.getLogger(__name__) # Logger for this module

def execute_test2_parallel_motion(
    mdetector_instance: 'MDetector',
    point_global_P: np.ndarray,
    current_di_timestamp: float, # Timestamp of the DI containing point_global_P
    current_di_idx_in_lib: int,
    return_debug_info: bool = False
) -> Any: # Returns bool or Tuple[bool, List[Dict[str, Any]]]
    """
    Implements Test 2 (Parallel Motion / Event by Disappearance).
    P is occluded by P1 (in H1), P1 by P2 (in H2), ..., P(M2-1) by PM2 (in HM2).
    MCC is on P1, P2, ..., PM2 (the occluders in historical frames).
    """
    # Ensure logger level is set to DEBUG in the notebook if you want to see these logs
    logger_et.debug(f"Executing Test 2 for P: {point_global_P.tolist()} from DI @ ts {current_di_timestamp:.2f} (lib_idx {current_di_idx_in_lib})")
    
    debug_steps = []
    if mdetector_instance.test2_M2_depth_images <= 0:
        if return_debug_info:
            return False, [{"step": 0, "status": "Test disabled (M2 <= 0)", "details": "N/A"}]
        return False

    # This point is the one that needs to be occluded in the current step of the chain
    point_to_be_occluded_in_step = point_global_P
    # Timestamp of the DI from which point_to_be_occluded_in_step originated
    timestamp_of_origin_di_for_point_to_be_occluded = current_di_timestamp

    for k_step in range(mdetector_instance.test2_M2_depth_images): # k_step from 0 to M2-1
        step_debug_info: Dict[str, Any] = {
            "step": k_step,
            "point_to_be_occluded_global": point_to_be_occluded_in_step.tolist(),
            "origin_di_timestamp_of_pto": timestamp_of_origin_di_for_point_to_be_occluded,
            "status": "pending",
            "historical_di_checked_idx_in_lib": None,
            "historical_di_checked_timestamp": None,
            "projection_in_hist_successful": False,
            "occlusion_check_details": []
        }

        # Determine the historical DI to check against.
        
        historical_di_to_check_idx_in_lib = current_di_idx_in_lib - 1 - k_step
        step_debug_info["historical_di_checked_idx_in_lib"] = historical_di_to_check_idx_in_lib

        if historical_di_to_check_idx_in_lib < 0:
            step_debug_info["status"] = f"Failed: Not enough historical DIs (needed idx {historical_di_to_check_idx_in_lib})."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        
        historical_di_k = mdetector_instance.depth_image_library.get_image_by_index(historical_di_to_check_idx_in_lib)
        if not historical_di_k or not historical_di_k.is_prepared_for_projection():
            ts_val = historical_di_k.timestamp if historical_di_k else "N/A"
            step_debug_info["historical_di_checked_timestamp"] = ts_val
            step_debug_info["status"] = f"Failed: Historical DI {historical_di_to_check_idx_in_lib} (TS: {ts_val}) not found or not prepared."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        step_debug_info["historical_di_checked_timestamp"] = historical_di_k.timestamp

        _, _, px_indices_in_hist_k = historical_di_k.project_point_to_pixel_indices(point_to_be_occluded_in_step)
        if px_indices_in_hist_k is None:
            step_debug_info["status"] = f"Failed: Point to be occluded did not project into historical DI {historical_di_to_check_idx_in_lib}."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        step_debug_info["projection_in_hist_successful"] = True
        step_debug_info["projected_pixel_in_hist"] = list(px_indices_in_hist_k)

        v_proj, h_proj = px_indices_in_hist_k
        found_valid_occluder_for_this_step = False
        next_point_for_chain = None # This will be the occluder from historical_di_k

        # Simplified neighborhood: just the projected pixel. Expand if needed.
        pixel_info_hist_k = historical_di_k.get_pixel_info(v_proj, h_proj)
        if pixel_info_hist_k and pixel_info_hist_k.get('original_indices_in_pixel'):
            step_debug_info["num_candidates_in_projected_pixel"] = len(pixel_info_hist_k['original_indices_in_pixel'])
            for original_idx_hist_cand in pixel_info_hist_k['original_indices_in_pixel']:
                occlusion_cand_debug: Dict[str, Any] = {"hist_candidate_original_idx": original_idx_hist_cand}
                if original_idx_hist_cand >= len(historical_di_k.original_points_global_coords):
                    occlusion_cand_debug["error"] = "Index out of bounds"
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    continue
                
                point_hist_cand_global = historical_di_k.original_points_global_coords[original_idx_hist_cand]
                occlusion_cand_debug["hist_candidate_global"] = point_hist_cand_global.tolist()

                is_occluded_by_cand = mdetector_instance.check_occlusion_point_level_detailed(
                    point_eval_global=point_to_be_occluded_in_step,
                    point_hist_cand_global=point_hist_cand_global,
                    historical_di=historical_di_k,
                    occlusion_type_to_check="OCCLUDED_BY"
                )
                occlusion_cand_debug["detailed_occlusion_result"] = is_occluded_by_cand

                mcc_rejects_this_occluder = False
                if is_occluded_by_cand: # If P is occluded by H_cand
                    occlusion_cand_debug["mcc_applied_on_point"] = "historical_candidate_occluder"
                    occlusion_cand_debug["mcc_point_coords"] = point_hist_cand_global.tolist()
                    occlusion_cand_debug["mcc_timestamp"] = historical_di_k.timestamp
                    if mdetector_instance.map_consistency_enabled:
                        # MCC on the occluder (point_hist_cand_global). If it's static, this link is invalid for Test 2.
                        if mdetector_instance.is_map_consistent(point_hist_cand_global, historical_di_k.timestamp, check_direction='past'):
                            mcc_rejects_this_occluder = True
                    occlusion_cand_debug["mcc_result_is_consistent_with_map"] = mcc_rejects_this_occluder # True if static
                    occlusion_cand_debug["mcc_rejects_this_link"] = mcc_rejects_this_occluder
                
                if is_occluded_by_cand and not mcc_rejects_this_occluder:
                    found_valid_occluder_for_this_step = True
                    next_point_for_chain = point_hist_cand_global
                    occlusion_cand_debug["forms_next_link_in_chain"] = True
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    break # Found a valid occluder for this step
                else:
                    occlusion_cand_debug["forms_next_link_in_chain"] = False
                step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
            
            if found_valid_occluder_for_this_step:
                point_to_be_occluded_in_step = next_point_for_chain # This occluder becomes the point to be occluded in the next step
                timestamp_of_origin_di_for_point_to_be_occluded = historical_di_k.timestamp
                step_debug_info["status"] = "Success: Valid occluder found, chain continues."
                step_debug_info["selected_occluder_for_next_step"] = next_point_for_chain.tolist()
            else:
                step_debug_info["status"] = "Failed: No valid (non-MCC-rejected) occluder found in historical DI pixel."
                debug_steps.append(step_debug_info)
                if return_debug_info: return False, debug_steps
                return False
        else: # No points in projected pixel of historical_di_k
            step_debug_info["num_candidates_in_projected_pixel"] = 0
            step_debug_info["status"] = "Failed: Projected pixel in historical DI is empty."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        debug_steps.append(step_debug_info)

    # If loop completes, all M2 steps were successful
    if return_debug_info: return True, debug_steps
    return True


def execute_test3_perpendicular_motion(
    mdetector_instance: 'MDetector',
    point_global_P: np.ndarray, # The point from current_di being evaluated
    current_di_timestamp: float,
    current_di_idx_in_lib: int,
    return_debug_info: bool = False
) -> Any:
    """
    Implements Test 3 (Perpendicular Motion / Event by Appearance/Revealing).
    P occludes P1 (in H1), P1 occludes P2 (in H2), ..., P(M3-1) occludes PM3 (in HM3).
    MCC is on P (the original point from current_di).
    """
    logger_et.debug(f"Executing Test 3 for P: {point_global_P.tolist()} from DI @ ts {current_di_timestamp:.2f} (lib_idx {current_di_idx_in_lib})")
    debug_steps = []

    if mdetector_instance.test3_M3_depth_images <= 0:
        if return_debug_info:
            return False, [{"step": 0, "status": "Test disabled (M3 <= 0)", "details": "N/A"}]
        return False

    # First, check MCC for point_global_P. If it's map consistent, Test 3 fails upfront.
    mcc_rejects_P_as_dynamic_occluder = False
    mcc_check_on_P_details = {
        "mcc_applied_on_point": "point_global_P (initial occluder)",
        "mcc_point_coords": point_global_P.tolist(),
        "mcc_timestamp": current_di_timestamp,
        "mcc_result_is_consistent_with_map": None # bool
    }
    if mdetector_instance.map_consistency_enabled:
        if mdetector_instance.is_map_consistent(point_global_P, current_di_timestamp, check_direction='past'):
            mcc_rejects_P_as_dynamic_occluder = True
        mcc_check_on_P_details["mcc_result_is_consistent_with_map"] = mcc_rejects_P_as_dynamic_occluder
    
    if mcc_rejects_P_as_dynamic_occluder:
        step_debug_info = {"step": -1, "status": "Failed: Initial point_global_P is map consistent.", "mcc_details_on_P": mcc_check_on_P_details}
        debug_steps.append(step_debug_info)
        if return_debug_info: return False, debug_steps
        return False

    # This point is the one that must occlude something in the current step of the chain
    point_that_occludes_in_step = point_global_P

    for k_step in range(mdetector_instance.test3_M3_depth_images): # k_step from 0 to M3-1
        step_debug_info: Dict[str, Any] = {
            "step": k_step,
            "point_that_occludes_global": point_that_occludes_in_step.tolist(),
            "status": "pending",
            "historical_di_checked_idx_in_lib": None,
            "historical_di_checked_timestamp": None,
            "projection_in_hist_successful": False,
            "occlusion_check_details": []
        }
        if k_step == 0: # Include initial MCC check info for the first step's debug
            step_debug_info["initial_mcc_on_point_P_details"] = mcc_check_on_P_details


        historical_di_to_check_idx_in_lib = current_di_idx_in_lib - 1 - k_step
        step_debug_info["historical_di_checked_idx_in_lib"] = historical_di_to_check_idx_in_lib
        
        if historical_di_to_check_idx_in_lib < 0:
            step_debug_info["status"] = f"Failed: Not enough historical DIs."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        
        historical_di_k = mdetector_instance.depth_image_library.get_image_by_index(historical_di_to_check_idx_in_lib)
        if not historical_di_k or not historical_di_k.is_prepared_for_projection():
            ts_val = historical_di_k.timestamp if historical_di_k else "N/A"
            step_debug_info["historical_di_checked_timestamp"] = ts_val
            step_debug_info["status"] = f"Failed: Historical DI {historical_di_to_check_idx_in_lib} (TS: {ts_val}) not found or not prepared."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        step_debug_info["historical_di_checked_timestamp"] = historical_di_k.timestamp

        _, _, px_indices_in_hist_k = historical_di_k.project_point_to_pixel_indices(point_that_occludes_in_step)
        if px_indices_in_hist_k is None:
            step_debug_info["status"] = f"Failed: Point that occludes did not project into historical DI {historical_di_to_check_idx_in_lib}."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        step_debug_info["projection_in_hist_successful"] = True
        step_debug_info["projected_pixel_in_hist"] = list(px_indices_in_hist_k)

        v_proj, h_proj = px_indices_in_hist_k
        found_valid_occluded_point_for_this_step = False
        next_point_for_chain = None # This will be the point from historical_di_k that was occluded

        pixel_info_hist_k = historical_di_k.get_pixel_info(v_proj, h_proj)
        if pixel_info_hist_k and pixel_info_hist_k.get('original_indices_in_pixel'):
            step_debug_info["num_candidates_in_projected_pixel"] = len(pixel_info_hist_k['original_indices_in_pixel'])
            for original_idx_hist_cand in pixel_info_hist_k['original_indices_in_pixel']:
                occlusion_cand_debug: Dict[str, Any] = {"hist_candidate_original_idx": original_idx_hist_cand}
                if original_idx_hist_cand >= len(historical_di_k.original_points_global_coords):
                    occlusion_cand_debug["error"] = "Index out of bounds"
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    continue

                point_hist_cand_being_occluded_global = historical_di_k.original_points_global_coords[original_idx_hist_cand]
                occlusion_cand_debug["hist_candidate_being_occluded_global"] = point_hist_cand_being_occluded_global.tolist()

                is_occluding_cand = mdetector_instance.check_occlusion_point_level_detailed(
                    point_eval_global=point_that_occludes_in_step, # This is the occluder
                    point_hist_cand_global=point_hist_cand_being_occluded_global, # This is being occluded
                    historical_di=historical_di_k,
                    occlusion_type_to_check="OCCLUDING"
                )
                occlusion_cand_debug["detailed_occlusion_result"] = is_occluding_cand
                
                # For Test 3, MCC was already applied to point_global_P. No further MCC per link.
                if is_occluding_cand:
                    found_valid_occluded_point_for_this_step = True
                    next_point_for_chain = point_hist_cand_being_occluded_global # This point becomes the occluder in the next step
                    occlusion_cand_debug["forms_next_link_in_chain"] = True
                    step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)
                    break 
                else:
                    occlusion_cand_debug["forms_next_link_in_chain"] = False
                step_debug_info["occlusion_check_details"].append(occlusion_cand_debug)

            if found_valid_occluded_point_for_this_step:
                point_that_occludes_in_step = next_point_for_chain
                step_debug_info["status"] = "Success: Valid occluded point found, chain continues."
                step_debug_info["selected_occluded_point_for_next_step"] = next_point_for_chain.tolist()
            else:
                step_debug_info["status"] = "Failed: No point found in historical DI pixel that was occluded by current chain point."
                debug_steps.append(step_debug_info)
                if return_debug_info: return False, debug_steps
                return False
        else:
            step_debug_info["num_candidates_in_projected_pixel"] = 0
            step_debug_info["status"] = "Failed: Projected pixel in historical DI is empty."
            debug_steps.append(step_debug_info)
            if return_debug_info: return False, debug_steps
            return False
        debug_steps.append(step_debug_info)

    if return_debug_info: return True, debug_steps
    return True