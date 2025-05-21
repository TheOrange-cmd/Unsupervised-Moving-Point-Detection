# src/core/m_detector/map_consistency.py
# This file is imported into MDetector class

import numpy as np
from typing import Any, Dict, Tuple, List, Optional # Added for type hinting
from ..constants import OcclusionResult
import logging # Import logging


logger = logging.getLogger(__name__) # This will use 'src.core.m_detector.map_consistency'


def is_map_consistent(self, # 'self' implies this is a method of MDetector class
                      point_global: np.ndarray, 
                      current_timestamp: float, 
                      check_direction: str = 'past',
                      return_debug_info: bool = False) -> Any:
    """
    Checks if a given global point is consistent with static points in relevant past or future DIs.
    ADAPTED FOR NEW DepthImage STRUCTURE.
    """
    
    # --- Initial setup and early exit for disabled MCC ---
    if not self.map_consistency_enabled:
        if return_debug_info:
            debug_data = {
                'point_global': point_global.tolist(), 'current_timestamp': current_timestamp,
                'check_direction': check_direction, 'map_consistency_enabled': False,
                'map_consistent_result': False, 'reason_for_result': 'Map consistency check disabled in config.',
                'relevant_dis_count': 0, 'relevant_dis_details': []
            }
            return False, debug_data
        return False

    # --- Initialize debug_data if returning ---
    debug_data: Dict[str, Any] = {} 
    if return_debug_info:
        debug_data = {
            'point_global': point_global.tolist(), 'current_timestamp': current_timestamp,
            'check_direction': check_direction, 'map_consistency_enabled': True,
            'map_consistent_result': False, 'reason_for_result': 'No consistent static points found in relevant DIs.',
            'relevant_dis_count': 0, 'relevant_dis_details': [],
            'map_consistency_time_window_past_s': self.map_consistency_time_window_past_s,
            'consistency_threshold_ratio': self.mc_threshold_value_ratio,
            'consistency_threshold_count': self.mc_threshold_value_count,
            'epsilon_phi_map_rad': self.epsilon_phi_map_rad,
            'epsilon_theta_map_rad': self.epsilon_theta_map_rad,
            'epsilon_depth_forward_map': self.epsilon_depth_forward_map,
            'epsilon_depth_backward_map': self.epsilon_depth_backward_map,
            'static_labels_for_map_check': [label for label in self.static_labels_for_map_check] # Show names
        }

    # --- Get relevant historical/future DIs ---
    relevant_dis: List[Tuple[int, Any]] = []
    if check_direction == 'past':
        relevant_dis = self.depth_image_library.get_relevant_past_images(
            current_timestamp, self.map_consistency_time_window_past_s
        )
    # ... (elif for 'future' ...)
    else:
        raise ValueError("check_direction must be 'past' or 'future'")

    if return_debug_info:
        debug_data['relevant_dis_count'] = len(relevant_dis)
        debug_data['relevant_dis_timestamps'] = [di.timestamp for _, di in relevant_dis]

    if not relevant_dis:
        if return_debug_info:
            debug_data['reason_for_result'] = 'No relevant DIs found in the time window.'
            return False, debug_data
        return False 
    
    # --- Loop through relevant DIs to check for consistency ---
    found_enough_consistent_matches = False # Flag to track if overall consistency is met
    consistent_matches_across_dis = 0 # How many DIs had at least one consistent static point
    total_dis_where_projection_and_pixel_content_ok = 0 # DIs where we could actually check points
    
    logger.debug(f"MCC_TRACE: Checking point_global {point_global.tolist()} against {len(relevant_dis)} DIs.") # General trace

    for di_original_idx_in_deque, di_hist_future in relevant_dis: # di_hist_future is the historical or future DepthImage
        di_debug_details: Dict[str, Any] = {}
        if return_debug_info:
            # ... (initialize di_debug_details as before) ...
            di_debug_details = {
                'di_timestamp': di_hist_future.timestamp, 'di_original_idx_in_deque': di_original_idx_in_deque,
                'projection_successful': False, 'target_sph_coords_in_di': None,
                'target_pixel_indices_in_di': None, 'pixel_content_found_in_di': False,
                'static_points_in_pixel': [] 
            }

        # Project the target point (from current_di) into this historical/future 'di_hist_future'
        _point_in_di_frame, sph_coords_target, pixel_indices_target = di_hist_future.project_point_to_pixel_indices(point_global)
        
        if pixel_indices_target is None or sph_coords_target is None:
            # ... (handle no projection, update di_debug_details, continue) ...
            if return_debug_info:
                di_debug_details['reason_skipped_di'] = 'Target point did not project into this DI FoV.'
                debug_data['relevant_dis_details'].append(di_debug_details)
            continue

        logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp}: Target projected to pixel {pixel_indices_target}, sph {np.round(sph_coords_target,3)}")

        if return_debug_info:
            # ... (update di_debug_details for successful projection) ...
            di_debug_details['projection_successful'] = True
            di_debug_details['target_sph_coords_in_di'] = sph_coords_target.tolist()
            di_debug_details['target_pixel_indices_in_di'] = list(pixel_indices_target)
            
        v_idx, h_idx = pixel_indices_target
        pixel_info_in_di_hist_future = di_hist_future.get_pixel_info(v_idx, h_idx) 
        
        if not pixel_info_in_di_hist_future or not pixel_info_in_di_hist_future.get('original_indices_in_pixel'):
            # ... (handle empty pixel, update di_debug_details, continue) ...
            if return_debug_info:
                di_debug_details['reason_skipped_di'] = 'Target pixel in this DI is empty.'
                debug_data['relevant_dis_details'].append(di_debug_details)
            continue
        
        logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp}: Pixel has {len(pixel_info_in_di_hist_future.get('original_indices_in_pixel',[]))} candidate points.")

        if return_debug_info:
            # ... (update di_debug_details for pixel content found) ...
            di_debug_details['pixel_content_found_in_di'] = True
            di_debug_details['pixel_stats_in_di'] = {
                'min_depth': pixel_info_in_di_hist_future['min_depth'], 'max_depth': pixel_info_in_di_hist_future['max_depth'],
                'count': pixel_info_in_di_hist_future['count'], 
                'num_original_indices': len(pixel_info_in_di_hist_future['original_indices_in_pixel'])
            }

        phi_target_in_hist_di, theta_target_in_hist_di, depth_target_in_hist_di = sph_coords_target[0], sph_coords_target[1], sph_coords_target[2]
        
        total_dis_where_projection_and_pixel_content_ok += 1 
        found_consistent_static_point_in_this_di = False 
        
        for static_candidate_original_idx in pixel_info_in_di_hist_future['original_indices_in_pixel']:
            static_pt_debug: Dict[str, Any] = {}
            if return_debug_info:
                # ... (initialize static_pt_debug) ...
                static_pt_debug = {
                    'original_index_in_di': static_candidate_original_idx, 'label_in_di': None,
                    'is_static_label': False, 'sph_coords_in_di': None,
                    'phi_consistent': None, 'theta_consistent': None, 'depth_consistent': None,
                    'match_found': False
                }

            static_candidate_label_val = di_hist_future.mdet_labels_for_points[static_candidate_original_idx]
            static_candidate_label_enum = OcclusionResult(static_candidate_label_val)

            if return_debug_info: static_pt_debug['label_in_di'] = static_candidate_label_enum.name

            if static_candidate_label_enum not in self.static_labels_for_map_check:
                logger.debug(f"MCC_TRACE:    StaticCand (idx {static_candidate_original_idx}, label {static_candidate_label_enum.name}) REJECTED (not static label).")
                if return_debug_info:
                    static_pt_debug['reason_skipped_point'] = 'Not a static label.'
                    di_debug_details['static_points_in_pixel'].append(static_pt_debug)
                continue
            
            if return_debug_info: static_pt_debug['is_static_label'] = True
            
            static_sph_coords_in_di_hist_future = di_hist_future.local_sph_coords_for_points[static_candidate_original_idx]

            if static_sph_coords_in_di_hist_future is None:
                # ... (handle missing sph_coords for static candidate) ...
                if return_debug_info:
                    static_pt_debug['reason_skipped_point'] = 'Missing local spherical coords for this static point in DI.'
                    di_debug_details['static_points_in_pixel'].append(static_pt_debug)
                continue
            
            if return_debug_info: static_pt_debug['sph_coords_in_di'] = static_sph_coords_in_di_hist_future.tolist()
                
            phi_static_cand, theta_static_cand, depth_static_cand = static_sph_coords_in_di_hist_future[0], static_sph_coords_in_di_hist_future[1], static_sph_coords_in_di_hist_future[2]
            
            # --- DETAILED GEOMETRIC COMPARISON ---
            phi_diff = abs(phi_target_in_hist_di - phi_static_cand)
            theta_diff = abs(theta_target_in_hist_di - theta_static_cand)
            
            phi_consistent = phi_diff <= self.epsilon_phi_map_rad
            theta_consistent = theta_diff <= self.epsilon_theta_map_rad
            
            # Depth consistency: d_target_proj must be within [d_static - eps_back, d_static + eps_fwd]
            depth_consistent = (depth_target_in_hist_di >= depth_static_cand - self.epsilon_depth_backward_map) and \
                               (depth_target_in_hist_di <= depth_static_cand + self.epsilon_depth_forward_map)

            logger.debug(f"MCC_TRACE:    StaticCand (idx {static_candidate_original_idx}, label {static_candidate_label_enum.name}, sph {np.round(static_sph_coords_in_di_hist_future,3)}):")
            logger.debug(f"MCC_TRACE:      Phi_target={phi_target_in_hist_di:.3f}, Phi_static={phi_static_cand:.3f}, Diff={phi_diff:.3f}, Eps_phi={self.epsilon_phi_map_rad:.3f}, Consistent={phi_consistent}")
            logger.debug(f"MCC_TRACE:      Theta_target={theta_target_in_hist_di:.3f}, Theta_static={theta_static_cand:.3f}, Diff={theta_diff:.3f}, Eps_theta={self.epsilon_theta_map_rad:.3f}, Consistent={theta_consistent}")
            logger.debug(f"MCC_TRACE:      Depth_target={depth_target_in_hist_di:.3f}, Depth_static={depth_static_cand:.3f}, Eps_fwd={self.epsilon_depth_forward_map:.3f}, Eps_back={self.epsilon_depth_backward_map:.3f}, Consistent={depth_consistent}")

            if return_debug_info:
                static_pt_debug.update({
                    'phi_target': phi_target_in_hist_di, 'phi_static_cand': phi_static_cand, 'phi_diff': phi_diff, 'phi_consistent': phi_consistent,
                    'theta_target': theta_target_in_hist_di, 'theta_static_cand': theta_static_cand, 'theta_diff': theta_diff, 'theta_consistent': theta_consistent,
                    'depth_target': depth_target_in_hist_di, 'depth_static_cand': depth_static_cand, 'depth_consistent': depth_consistent
                })

            if phi_consistent and theta_consistent and depth_consistent:
                logger.debug(f"MCC_TRACE:      MATCH FOUND for StaticCand idx {static_candidate_original_idx} in DI TS {di_hist_future.timestamp}!")
                consistent_matches_across_dis += 1
                found_consistent_static_point_in_this_di = True
                if return_debug_info: static_pt_debug['match_found'] = True
                break # Found a match in this DI's pixel, move to next DI
            else:
                logger.debug(f"MCC_TRACE:      NO MATCH for StaticCand idx {static_candidate_original_idx}.")
            
            if return_debug_info: di_debug_details['static_points_in_pixel'].append(static_pt_debug)
        # --- End loop for static_candidate_original_idx ---
        
        if return_debug_info: debug_data['relevant_dis_details'].append(di_debug_details) 
    # --- End loop for di_hist_future in relevant_dis ---
            
    # --- Final consistency decision based on counts/ratios ---
    final_map_consistent_result = False
    if total_dis_where_projection_and_pixel_content_ok > 0: # Only if we actually had a chance to check points in some DIs
        if self.mc_threshold_mode == 'count':
            if self.mc_threshold_value_count > 0 and consistent_matches_across_dis >= self.mc_threshold_value_count:
                final_map_consistent_result = True
        elif self.mc_threshold_mode == 'ratio':
            # Ratio should be based on DIs where we could actually check points, not just len(relevant_dis)
            ratio_val = consistent_matches_across_dis / total_dis_where_projection_and_pixel_content_ok
            if self.mc_threshold_value_ratio > 0 and ratio_val >= self.mc_threshold_value_ratio:
                final_map_consistent_result = True
    
    if return_debug_info:
        debug_data['map_consistent_result'] = final_map_consistent_result
        debug_data['consistent_count_final'] = consistent_matches_across_dis # This is now 'consistent_matches_across_dis'
        debug_data['total_checks_attempted_on_dis_final'] = total_dis_where_projection_and_pixel_content_ok # Use the more accurate count

        reason = f"Consistent static points found in {consistent_matches_across_dis}/{total_dis_where_projection_and_pixel_content_ok} DIs where checks were possible. "
        if self.mc_threshold_mode == 'count':
            reason += f"Mode: Count, Threshold: {self.mc_threshold_value_count}. "
        elif self.mc_threshold_mode == 'ratio':
            ratio_calc = consistent_matches_across_dis / total_dis_where_projection_and_pixel_content_ok if total_dis_where_projection_and_pixel_content_ok > 0 else 0
            reason += f"Mode: Ratio, Calc Ratio: {ratio_calc:.2f}, Threshold: {self.mc_threshold_value_ratio}. "
        
        if final_map_consistent_result:
            debug_data['reason_for_result'] = reason + "Met thresholds."
        else:
            debug_data['reason_for_result'] = reason + "Did NOT meet thresholds."
        
        return final_map_consistent_result, debug_data

    return final_map_consistent_result