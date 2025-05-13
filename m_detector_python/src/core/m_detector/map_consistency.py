# src/core/m_detector/map_consistency.py
# This file is imported into MDetector class

import numpy as np
from typing import Any, Dict, Tuple, Optional # Added for type hinting

def is_map_consistent(self, # 'self' implies this is a method of MDetector class
                      point_global: np.ndarray, 
                      current_timestamp: float, 
                      check_direction: str = 'past',
                      return_debug_info: bool = False) -> Any:
    """
    Checks if a given global point is consistent with static points in relevant past or future DIs.
    ADAPTED FOR NEW DepthImage STRUCTURE.
    """
    
    if not self.map_consistency_enabled:
        if return_debug_info:
            debug_data = {
                'point_global': point_global.tolist(),
                'current_timestamp': current_timestamp,
                'check_direction': check_direction,
                'map_consistency_enabled': False,
                'map_consistent_result': False, # Default for disabled
                'reason_for_result': 'Map consistency check disabled in config.',
                'relevant_dis_count': 0,
                'relevant_dis_details': []
            }
            return False, debug_data
        return False

    relevant_dis: List[Tuple[int, Any]] = [] # List of (original_index_in_deque, DepthImage_instance)
    
    debug_data: Dict[str, Any] = {} # Initialize here
    if return_debug_info:
        debug_data = {
            'point_global': point_global.tolist(),
            'current_timestamp': current_timestamp,
            'check_direction': check_direction,
            'map_consistency_enabled': True,
            'map_consistent_result': False, # Default
            'reason_for_result': 'No consistent static points found in relevant DIs.',
            'relevant_dis_count': 0,
            'relevant_dis_details': [],
            'consistency_threshold_count': self.map_consistency_threshold_count,
            'consistency_threshold_ratio': self.map_consistency_threshold_ratio,
            'epsilon_phi_map_rad': self.epsilon_phi_map_rad,
            'epsilon_theta_map_rad': self.epsilon_theta_map_rad,
            'epsilon_depth_forward_map': self.epsilon_depth_forward_map,
            'epsilon_depth_backward_map': self.epsilon_depth_backward_map,
        }

    if check_direction == 'past':
        relevant_dis = self.depth_image_library.get_relevant_past_images(
            current_timestamp, self.map_consistency_time_window_past_s
        )
    elif check_direction == 'future':
        relevant_dis = self.depth_image_library.get_relevant_future_images(
            current_timestamp, self.map_consistency_time_window_future_s
        )
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
    
    consistent_count = 0
    total_checks_attempted_on_dis = 0 # DIs where projection was successful
    
    for di_original_idx_in_deque, di in relevant_dis: # di is the historical or future DepthImage
        di_debug_details: Dict[str, Any] = {}
        if return_debug_info:
            di_debug_details = {
                'di_timestamp': di.timestamp,
                'di_original_idx_in_deque': di_original_idx_in_deque,
                'projection_successful': False,
                'target_sph_coords_in_di': None,
                'target_pixel_indices_in_di': None,
                'pixel_content_found_in_di': False,
                'static_points_in_pixel': [] # List of dicts for each static point
            }

        # Project the target point (from current_di) into this historical/future 'di'
        # project_point_to_pixel_indices returns: point_in_di_frame, sph_coords, pixel_indices
        _point_in_di_frame, sph_coords_target, pixel_indices_target = di.project_point_to_pixel_indices(point_global)
        
        if pixel_indices_target is None or sph_coords_target is None:
            if return_debug_info:
                di_debug_details['reason_skipped_di'] = 'Target point did not project into this DI FoV.'
                debug_data['relevant_dis_details'].append(di_debug_details)
            continue

        if return_debug_info:
            di_debug_details['projection_successful'] = True
            di_debug_details['target_sph_coords_in_di'] = sph_coords_target.tolist()
            di_debug_details['target_pixel_indices_in_di'] = list(pixel_indices_target) # tuple to list
            
        v_idx, h_idx = pixel_indices_target
        
        # Get info for the pixel in 'di' using the new get_pixel_info structure
        pixel_info_in_di = di.get_pixel_info(v_idx, h_idx) 
        
        # Check if there are any original_indices stored for this pixel in 'di'
        if not pixel_info_in_di or not pixel_info_in_di.get('original_indices_in_pixel'):
            if return_debug_info:
                di_debug_details['reason_skipped_di'] = 'Target pixel in this DI is empty.'
                debug_data['relevant_dis_details'].append(di_debug_details)
            continue
        
        if return_debug_info:
            di_debug_details['pixel_content_found_in_di'] = True
            # Add overall pixel stats from di to debug
            di_debug_details['pixel_stats_in_di'] = {
                'min_depth': pixel_info_in_di['min_depth'],
                'max_depth': pixel_info_in_di['max_depth'],
                'count': pixel_info_in_di['count'],
                'num_original_indices': len(pixel_info_in_di['original_indices_in_pixel'])
            }


        phi_target, theta_target, depth_target = sph_coords_target[0], sph_coords_target[1], sph_coords_target[2]
        
        total_checks_attempted_on_dis += 1 
        
        found_match_in_this_di_pixel = False # Renamed for clarity
        
        # Iterate through the original_indices of points that fall into this pixel in 'di'
        for static_candidate_original_idx in pixel_info_in_di['original_indices_in_pixel']:
            static_pt_debug: Dict[str, Any] = {}
            if return_debug_info:
                static_pt_debug = {
                    'original_index_in_di': static_candidate_original_idx,
                    'label_in_di': None,
                    'is_static_label': False,
                    'sph_coords_in_di': None,
                    'phi_consistent': None,
                    'theta_consistent': None,
                    'depth_consistent': None,
                    'match_found': False
                }

            # Get the label of this static candidate point from 'di's main label array
            static_candidate_label_val = di.mdet_labels_for_points[static_candidate_original_idx]
            static_candidate_label_enum = OcclusionResult(static_candidate_label_val)

            if return_debug_info:
                static_pt_debug['label_in_di'] = static_candidate_label_enum.name

            if static_candidate_label_enum not in self.static_labels_for_map_check:
                if return_debug_info:
                    static_pt_debug['reason_skipped_point'] = 'Not a static label.'
                    di_debug_details['static_points_in_pixel'].append(static_pt_debug)
                continue
            
            if return_debug_info:
                static_pt_debug['is_static_label'] = True
            
            # Get the pre-calculated local spherical coordinates of this static candidate *within 'di'*
            # This is di.local_sph_coords_for_points[original_idx_of_point_in_di]
            static_sph_coords_in_di = di.local_sph_coords_for_points[static_candidate_original_idx]
            # static_sph_coords_in_di is [phi, theta, depth]

            if static_sph_coords_in_di is None: # Should not happen if local_sph_coords_for_points is always populated
                if return_debug_info:
                    static_pt_debug['reason_skipped_point'] = 'Missing local spherical coords for this static point in DI.'
                    di_debug_details['static_points_in_pixel'].append(static_pt_debug)
                # logger.error(f"Missing local_sph_coords for original_idx {static_candidate_original_idx} in DI {di.timestamp}")
                continue
            
            if return_debug_info:
                 static_pt_debug['sph_coords_in_di'] = static_sph_coords_in_di.tolist()
                
            phi_static_in_di, theta_static_in_di, depth_static_in_di = static_sph_coords_in_di[0], static_sph_coords_in_di[1], static_sph_coords_in_di[2]
            
            phi_consistent = abs(phi_target - phi_static_in_di) <= self.epsilon_phi_map_rad
            theta_consistent = abs(theta_target - theta_static_in_di) <= self.epsilon_theta_map_rad
            
            depth_forward_ok = depth_target <= depth_static_in_di + self.epsilon_depth_forward_map
            depth_backward_ok = depth_target >= depth_static_in_di - self.epsilon_depth_backward_map
            depth_consistent = depth_forward_ok and depth_backward_ok
            
            if return_debug_info:
                static_pt_debug['phi_consistent'] = phi_consistent
                static_pt_debug['theta_consistent'] = theta_consistent
                static_pt_debug['depth_consistent'] = depth_consistent
                static_pt_debug['depth_target'] = depth_target # Add for comparison
                static_pt_debug['depth_static_in_di'] = depth_static_in_di # Add for comparison

            if phi_consistent and theta_consistent and depth_consistent:
                consistent_count += 1
                found_match_in_this_di_pixel = True
                if return_debug_info:
                    static_pt_debug['match_found'] = True
                    di_debug_details['static_points_in_pixel'].append(static_pt_debug)
                    # Add overall DI debug details now that we've processed its points
                    debug_data['relevant_dis_details'].append(di_debug_details) 
                break # Found a consistent static point in this DI's pixel, move to next DI
            
            if return_debug_info: # If no match for this static point
                di_debug_details['static_points_in_pixel'].append(static_pt_debug)
        
        if return_debug_info and not found_match_in_this_di_pixel:
            # If loop finished for this DI and no match was found, add its debug details
            debug_data['relevant_dis_details'].append(di_debug_details)
            
    # Final consistency decision based on counts/ratios
    map_consistent_result = False
    if total_checks_attempted_on_dis > 0: # Only if we actually checked against some DIs
        if self.map_consistency_threshold_count > 0:
            if consistent_count >= self.map_consistency_threshold_count:
                map_consistent_result = True
        # Ratio check can be an alternative or additional condition
        if self.map_consistency_threshold_ratio > 0 and not map_consistent_result: # only if count check failed or not used
            ratio_consistent = consistent_count / total_checks_attempted_on_dis
            if ratio_consistent >= self.map_consistency_threshold_ratio:
                map_consistent_result = True
    
    if return_debug_info:
        debug_data['map_consistent_result'] = map_consistent_result
        debug_data['consistent_count_final'] = consistent_count
        debug_data['total_checks_attempted_on_dis_final'] = total_checks_attempted_on_dis
        if not map_consistent_result and total_checks_attempted_on_dis > 0 :
            reason = f"Consistent static points found in {consistent_count}/{total_checks_attempted_on_dis} DIs. "
            if self.map_consistency_threshold_count > 0:
                reason += f"Count threshold: {self.map_consistency_threshold_count}. "
            if self.map_consistency_threshold_ratio > 0:
                ratio = consistent_count / total_checks_attempted_on_dis if total_checks_attempted_on_dis > 0 else 0
                reason += f"Ratio: {ratio:.2f}, Ratio threshold: {self.map_consistency_threshold_ratio}."
            debug_data['reason_for_result'] = reason
        elif map_consistent_result:
             debug_data['reason_for_result'] = f"Consistent static points found in {consistent_count}/{total_checks_attempted_on_dis} DIs, meeting thresholds."
        
        return map_consistent_result, debug_data

    return map_consistent_result