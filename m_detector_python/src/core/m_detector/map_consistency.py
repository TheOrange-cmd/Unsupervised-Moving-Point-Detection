# src/core/m_detector/map_consistency.py
# This file is imported into MDetector class

import numpy as np
from typing import Any, Dict, Tuple, Optional # Added for type hinting

def is_map_consistent(self, 
                      point_global: np.ndarray, 
                      current_timestamp: float, 
                      check_direction: str = 'past',
                      return_debug_info: bool = False) -> Any: # Return type can be bool or Tuple[bool, Dict]
    
    if not self.map_consistency_enabled:
        if return_debug_info:
            return False, {"reason": "Map consistency disabled in config", "relevant_dis_details": [], "consistent_count": 0, "total_checks": 0, "final_result": False}
        return False

    relevant_dis = []
    debug_data = None

    if return_debug_info:
        debug_data = {
            'point_global_checked': point_global.tolist(),
            'current_timestamp_of_point': current_timestamp,
            'check_direction': check_direction,
            'relevant_dis_details': [],
            'consistent_count': 0,
            'total_checks': 0,
            'final_result': False,
            'config_params': {
                'num_historical_di_for_map_check': self.num_historical_di_for_map_check,
                'num_future_di_for_map_check': self.num_future_di_for_map_check,
                'epsilon_phi_map_rad': self.epsilon_phi_map_rad,
                'epsilon_theta_map_rad': self.epsilon_theta_map_rad,
                'epsilon_depth_forward_map': self.epsilon_depth_forward_map,
                'epsilon_depth_backward_map': self.epsilon_depth_backward_map,
                'static_labels_for_map_check': [str(label) for label in self.static_labels_for_map_check], # Convert enums to strings for JSON serializability if needed
                'consistency_threshold': self.config.get('map_consistency_check', {}).get('consistency_threshold', 0.5)
            }
        }

    if check_direction == 'past':
        num_dis_to_check = self.num_historical_di_for_map_check
        # Iterate all DIs in the library to find relevant past ones
        # Sorting is done after collecting to ensure we consider all valid past DIs before limiting by num_dis_to_check
        temp_relevant_dis = []
        for di in self.depth_image_library._images: 
            if di.timestamp < current_timestamp:
                temp_relevant_dis.append(di)
        temp_relevant_dis.sort(key=lambda di: di.timestamp, reverse=True) # Most recent past first
        relevant_dis = temp_relevant_dis[:num_dis_to_check]

    elif check_direction == 'future':
        num_dis_to_check = self.num_future_di_for_map_check
        temp_relevant_dis = []
        for di in self.depth_image_library._images:
            if di.timestamp > current_timestamp:
                temp_relevant_dis.append(di)
        temp_relevant_dis.sort(key=lambda di: di.timestamp, reverse=False) # Earliest future first
        relevant_dis = temp_relevant_dis[:num_dis_to_check]
    
    # No need to re-assign relevant_dis = relevant_dis[:num_dis_to_check] as it's done above

    if not relevant_dis:
        if return_debug_info:
            debug_data['reason'] = "No relevant DIs found for map consistency check."
            return False, debug_data
        return False 
    
    consistent_count = 0
    total_checks_attempted_on_dis = 0 # Renamed to avoid confusion with successful checks
    
    for di_idx, di in enumerate(relevant_dis):
        di_debug_details = None
        if return_debug_info:
            di_debug_details = {
                'di_object_id': id(di), # To uniquely identify the DI object if needed
                'di_timestamp': di.timestamp,
                'di_pose_global_translation': di.image_pose_global[:3,3].tolist(),
                'projection_successful': False,
                'projected_sph_coords_of_target': None,
                'projected_pixel_indices_of_target': None,
                'pixel_content_found': False,
                'checked_static_points_in_di': [],
                'match_found_in_this_di': False
            }

        point_in_di_frame, sph_coords_target, pixel_indices_target = di.project_point_to_pixel_indices(point_global)
        
        if pixel_indices_target is None or sph_coords_target is None:
            if return_debug_info:
                debug_data['relevant_dis_details'].append(di_debug_details)
            continue  # Point outside FoV of this DI

        if return_debug_info:
            di_debug_details['projection_successful'] = True
            di_debug_details['projected_sph_coords_of_target'] = sph_coords_target.tolist()
            di_debug_details['projected_pixel_indices_of_target'] = list(pixel_indices_target) # Convert tuple to list
            
        v_idx, h_idx = pixel_indices_target
        # Ensure get_pixel_info is robust and handles out-of-bounds if project_point_to_pixel_indices doesn't fully prevent it
        pixel_content = di.get_pixel_info(v_idx, h_idx) 
        
        if not pixel_content or not pixel_content.get('points'): # Check if 'points' key exists and is not empty
            if return_debug_info:
                debug_data['relevant_dis_details'].append(di_debug_details)
            continue  # No points in this pixel of the reference DI
        
        if return_debug_info:
            di_debug_details['pixel_content_found'] = True

        phi_target, theta_target, depth_target = sph_coords_target
        
        # A check was made against this DI
        total_checks_attempted_on_dis += 1 
        
        # Iterate through points in the DI's pixel
        # (and potentially neighbors, though current code only uses the direct pixel)
        # To implement neighbor search, you'd expand (v_idx, h_idx) here.
        # For now, sticking to the provided code's logic (only current pixel).
        
        found_match_in_current_di = False
        for pt_info_static in pixel_content['points']:
            # Skip if point label doesn't match static labels
            # Ensure self.static_labels_for_map_check contains OcclusionResult enums or correct string representations
            if pt_info_static.get('label') not in self.static_labels_for_map_check:
                if return_debug_info:
                    static_pt_debug = {
                        'static_pt_global': pt_info_static['global_pt'].tolist(),
                        'label': str(pt_info_static.get('label')),
                        'reason_skipped': 'Not a static label'
                    }
                    di_debug_details['checked_static_points_in_di'].append(static_pt_debug)
                continue
            
            # Get spherical coordinates of the static point *within the current reference DI (di)*
            # The pt_info_static['sph_coords'] should already be relative to 'di' if stored correctly
            # Or, re-project if 'sph_coords' isn't guaranteed to be relative to 'di'
            # Assuming pt_info_static['sph_coords'] are correct for 'di'
            if 'sph_coords' not in pt_info_static or pt_info_static['sph_coords'] is None:
                 # Fallback: re-project if necessary, though ideally 'sph_coords' is pre-calculated and stored
                _, static_sph_coords_in_di, _ = di.project_point_to_pixel_indices(pt_info_static['global_pt'])
            else:
                static_sph_coords_in_di = pt_info_static['sph_coords']

            if static_sph_coords_in_di is None:
                if return_debug_info:
                    static_pt_debug = {
                        'static_pt_global': pt_info_static['global_pt'].tolist(),
                        'label': str(pt_info_static.get('label')),
                        'reason_skipped': 'Static point projection failed'
                    }
                    di_debug_details['checked_static_points_in_di'].append(static_pt_debug)
                continue
                
            phi_static_in_di, theta_static_in_di, depth_static_in_di = static_sph_coords_in_di
            
            # Check angular consistency
            phi_consistent = abs(phi_target - phi_static_in_di) <= self.epsilon_phi_map_rad
            theta_consistent = abs(theta_target - theta_static_in_di) <= self.epsilon_theta_map_rad
            
            # Check depth consistency
            depth_forward_ok = depth_target <= depth_static_in_di + self.epsilon_depth_forward_map
            depth_backward_ok = depth_target >= depth_static_in_di - self.epsilon_depth_backward_map
            depth_consistent = depth_forward_ok and depth_backward_ok
            
            if return_debug_info:
                static_pt_debug = {
                    'static_pt_global': pt_info_static['global_pt'].tolist(),
                    'label': str(pt_info_static.get('label')),
                    'static_pt_sph_coords_in_di': static_sph_coords_in_di.tolist(),
                    'phi_diff': abs(phi_target - phi_static_in_di),
                    'theta_diff': abs(theta_target - theta_static_in_di),
                    'depth_target': depth_target,
                    'depth_static_in_di': depth_static_in_di,
                    'phi_consistent': phi_consistent,
                    'theta_consistent': theta_consistent,
                    'depth_consistent': depth_consistent,
                    'is_match': False
                }
                di_debug_details['checked_static_points_in_di'].append(static_pt_debug)

            if phi_consistent and theta_consistent and depth_consistent:
                consistent_count += 1
                found_match_in_current_di = True
                if return_debug_info:
                    static_pt_debug['is_match'] = True # Update the last added static_pt_debug
                    di_debug_details['match_found_in_this_di'] = True
                break  # One matching point per DI is enough for this DI's check
        
        if return_debug_info:
             debug_data['relevant_dis_details'].append(di_debug_details)
    
    # Point is considered map-consistent if it matches points in a certain proportion of DIs where a check was possible
    # Using the configured consistency_threshold
    consistency_threshold = self.config.get('map_consistency_check', {}).get('consistency_threshold', 0.5)
    
    if total_checks_attempted_on_dis == 0: # No DIs where a check could even be performed (e.g., all out of FoV or empty pixels)
        map_consistent_result = False # Or True, depending on desired conservative behavior. Paper implies False.
    else:
        map_consistent_result = (consistent_count / total_checks_attempted_on_dis >= consistency_threshold)

    if return_debug_info:
        debug_data['consistent_count'] = consistent_count
        debug_data['total_checks_attempted_on_dis'] = total_checks_attempted_on_dis # How many DIs we actually compared against
        debug_data['final_result'] = map_consistent_result
        return map_consistent_result, debug_data
    else:
        return map_consistent_result