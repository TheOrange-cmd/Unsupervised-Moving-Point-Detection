# src/core/m_detector/map_consistency.py
# This file is imported into MDetector class

from __future__ import annotations # for safe type checking without writing as strings
import numpy as np
from typing import Any, Dict, Tuple, List, Optional, TYPE_CHECKING
from ..constants import OcclusionResult
from .interpolation_utils import interpolate_surface_depth_at_angle
import logging # Import logging
from .adaptive_epsilon_utils import calculate_adaptive_epsilon
import torch

# if TYPE_CHECKING:
#     from ..depth_image_legacy import DepthImage 

logger = logging.getLogger(__name__) # This will use 'src.core.m_detector.map_consistency'

def is_map_consistent(self,
                      points_global_batch: torch.Tensor,
                      origin_di: DepthImage, # The DI where the points originated
                      # We no longer need original_idx if we pass the whole batch
                      current_timestamp: float,
                      check_direction: str = 'past') -> torch.Tensor:
    """
    Checks if a BATCH of global points is consistent with static points in relevant DIs.
    This version processes all points simultaneously against each relevant historical DI.

    Args:
        points_global_batch (torch.Tensor): (N, 3) tensor of points to check.
        origin_di (DepthImage): The tensor-based DepthImage where the points originated.
        current_timestamp (float): The timestamp of the origin_di.
        check_direction (str): Currently only 'past' is supported.

    Returns:
        torch.Tensor: A boolean tensor of shape (N,) where True means the point is
                      consistent with the static map.
    """
    batch_size = points_global_batch.shape[0]
    device = self.device

    if not self.map_consistency_enabled or batch_size == 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 1. Get Relevant Historical DIs (this part remains a small Python loop)
    if check_direction == 'past':
        relevant_dis = self.depth_image_library.get_relevant_past_images(
            current_timestamp, self.map_consistency_time_window_past_s
        )
    else: # 'future' or 'both' not implemented for batch yet
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    if not relevant_dis:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 2. Prepare for aggregation
    # This tensor will store the count of consistent matches for each point.
    consistent_matches_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    # This tensor tracks how many DIs were actually checked for each point.
    dis_checked_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    # Get d_anchor for all points in the batch for adaptive epsilon
    d_anchors = origin_di.local_sph_coords_for_points[:, 2] # Assuming all points are from this DI

    # 3. Loop over relevant DIs and process points in a batch
    for _, di_hist in relevant_dis:
        # Project the entire batch of points into the historical DI
        _, sph_coords_target, _, valid_mask = di_hist.project_point_to_pixel_indices(points_global_batch)
        
        if not torch.any(valid_mask):
            continue # Skip this DI if no points project into it

        # --- Vectorized Consistency Check within this DI ---
        # We only work with the points that were validly projected
        active_indices = torch.where(valid_mask)[0]
        
        # Increment the checked count for these points
        dis_checked_count[active_indices] += 1
        
        active_sph_coords = sph_coords_target[active_indices]
        phi_target, theta_target, depth_target = active_sph_coords.T
        
        # Get all static points from the historical DI
        static_mask_hist = di_hist.get_static_points_mask(self.static_labels_for_map_check_values)
        if static_mask_hist is None or not torch.any(static_mask_hist):
            continue # No static points in this historical DI to compare against
            
        static_points_sph_hist = di_hist.local_sph_coords_for_points[static_mask_hist]
        phi_static, theta_static, depth_static = static_points_sph_hist.T
        
        # Use broadcasting to find neighbors for all active points at once
        # Shape: (num_active_points, num_static_points)
        phi_diff = torch.abs(phi_target.unsqueeze(1) - phi_static)
        theta_diff = torch.abs(theta_target.unsqueeze(1) - theta_static)
        
        # Find static points that are angular neighbors for each active point
        is_angular_neighbor = (phi_diff <= self.epsilon_phi_map_rad) & (theta_diff <= self.epsilon_theta_map_rad)
        
        # Calculate adaptive epsilons for the active points
        active_d_anchors = d_anchors[active_indices]
        eps_fwd = calculate_adaptive_epsilon(active_d_anchors, self.epsilon_depth_forward_map, ...) # Simplified
        eps_bwd = calculate_adaptive_epsilon(active_d_anchors, self.epsilon_depth_backward_map, ...) # Simplified

        # Check depth consistency against all angular neighbors
        # Shape: (num_active_points, num_static_points)
        depth_lower_bound = depth_static - eps_bwd.unsqueeze(1)
        depth_upper_bound = depth_static + eps_fwd.unsqueeze(1)
        
        is_depth_consistent = (depth_target.unsqueeze(1) >= depth_lower_bound) & \
                              (depth_target.unsqueeze(1) <= depth_upper_bound)
                              
        # A match is found if it's both an angular and depth neighbor
        is_consistent_match = is_angular_neighbor & is_depth_consistent
        
        # For each active point, check if ANY consistent match was found
        found_match_in_di = torch.any(is_consistent_match, dim=1)
        
        # Increment the match count for the points that found a match
        consistent_matches_count[active_indices[found_match_in_di]] += 1

    # 4. Final Decision based on aggregated counts
    final_map_consistent_result = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if self.mc_threshold_mode == 'count':
        # Check only where at least one DI was checked to avoid false positives
        can_be_consistent = dis_checked_count > 0
        passes_threshold = consistent_matches_count >= self.mc_threshold_value_count
        final_map_consistent_result = passes_threshold & can_be_consistent
    elif self.mc_threshold_mode == 'ratio':
        # Avoid division by zero
        ratio = torch.zeros_like(consistent_matches_count, dtype=torch.float32)
        valid_ratio_mask = dis_checked_count > 0
        ratio[valid_ratio_mask] = consistent_matches_count[valid_ratio_mask].float() / dis_checked_count[valid_ratio_mask].float()
        final_map_consistent_result = ratio >= self.mc_threshold_value_ratio
        
    return final_map_consistent_result

def is_map_consistent_legacy(self, # self is MDetector instance
                      point_global: np.ndarray,
                      # ADDED: Parameters to determine d_anchor for point_global
                      origin_di_of_point_global: 'DepthImage', 
                      original_idx_of_point_global_in_origin_di: int,
                      current_timestamp: float, # Timestamp of origin_di_of_point_global
                      check_direction: str = 'past',
                      return_debug_info: bool = False) -> Any:
    """
    Checks if a given global point is consistent with static points in relevant past or future DIs.
    Revised to correctly handle interpolation attempts even if the direct target pixel is empty.
    """

    # --- 1. Initial Setup & Early Exit for Disabled MCC ---
    if not self.map_consistency_enabled:
        if return_debug_info:
            debug_data_disabled = {
                'point_global': point_global.tolist(), 'current_timestamp': current_timestamp,
                'check_direction': check_direction, 'map_consistency_enabled': False,
                'map_consistent_result': False, 'reason_for_result': 'Map consistency check disabled in config.',
                'relevant_dis_count': 0, 'relevant_dis_details': []
            }
            return False, debug_data_disabled
        return False

    # --- 2. Initialize Debug Data (if requested) ---
    debug_data: Dict[str, Any] = {}
    if return_debug_info:
        debug_data = {
            'point_global': point_global.tolist(), 'current_timestamp': current_timestamp,
            'check_direction': check_direction, 'map_consistency_enabled': True,
            'map_consistent_result': False, 
            'reason_for_result': 'No consistent static points found meeting criteria.', 
            'relevant_dis_count': 0, 'relevant_dis_details': [],
            'config_map_consistency_time_window_past_s': self.map_consistency_time_window_past_s,
            'config_mc_threshold_mode': self.mc_threshold_mode,
            'config_mc_threshold_value_count': self.mc_threshold_value_count,
            'config_mc_threshold_value_ratio': self.mc_threshold_value_ratio,
            'config_epsilon_phi_map_rad': self.epsilon_phi_map_rad,
            'config_epsilon_theta_map_rad': self.epsilon_theta_map_rad,
            'config_epsilon_depth_forward_map': self.epsilon_depth_forward_map,
            'config_epsilon_depth_backward_map': self.epsilon_depth_backward_map,
            'config_static_labels_for_map_check': [label.name if isinstance(label, OcclusionResult) else str(label) for label in self.static_labels_for_map_check],
            'config_mc_interp_enabled': self.mc_interp_enabled,
            'config_mc_interp_min_depth': self.mc_interp_min_depth,
            'config_mc_interp_max_depth': self.mc_interp_max_depth,
            'config_mc_interp_fallback': self.mc_interp_fallback,
        }

    # --- Determine d_anchor for point_global ---
    d_anchor_point_global: Optional[float] = None
    if origin_di_of_point_global.local_sph_coords_for_points is not None and \
       original_idx_of_point_global_in_origin_di < origin_di_of_point_global.local_sph_coords_for_points.shape[0]:
        d_anchor_point_global = origin_di_of_point_global.local_sph_coords_for_points[original_idx_of_point_global_in_origin_di, 2]
    
    if return_debug_info:
        debug_data['d_anchor_of_point_global_for_mcc'] = d_anchor_point_global
        # Log the adaptive configs being used
        debug_data['adaptive_config_fwd_used'] = self.adaptive_eps_config_mc_fwd
        debug_data['adaptive_config_bwd_used'] = self.adaptive_eps_config_mc_bwd

    # --- 3. Get Relevant Historical/Future DIs ---
    relevant_dis: List[Tuple[int, Any]] = []
    if check_direction == 'past':
        relevant_dis = self.depth_image_library.get_relevant_past_images(
            current_timestamp, self.map_consistency_time_window_past_s
        )
    else:
        raise ValueError(f"check_direction must be 'past' (or 'future' if implemented), got {check_direction}")

    if return_debug_info:
        debug_data['relevant_dis_count'] = len(relevant_dis)
        debug_data['relevant_dis_timestamps'] = [di.timestamp for _, di in relevant_dis]

    if not relevant_dis:
        if return_debug_info:
            debug_data['reason_for_result'] = 'No relevant DIs found in the time window.'
            return False, debug_data
        return False

    # --- 4. Loop Through Relevant DIs to Check for Consistency ---
    consistent_matches_across_dis = 0
    num_dis_actually_checked = 0
    total_dis_where_projection_valid = 0
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"MCC_TRACE: Checking point_global {np.round(point_global[:3],3).tolist()} against {len(relevant_dis)} DIs.")

    for di_original_idx_in_deque, di_hist_future in relevant_dis:
        di_debug_details: Dict[str, Any] = {}
        if return_debug_info: # Initialize di_debug_details structure
            di_debug_details = {
                'di_timestamp': di_hist_future.timestamp,
                'di_original_idx_in_deque': di_original_idx_in_deque,
                'projection_successful': False,
                'attempted_interpolation': False,
                'interpolation_result_depth': None, # Will be updated
                # 'interp_depth_target_vs_surface' will be added conditionally later
                'attempted_direct_comparison': False,
                'direct_comparison_pixel_had_content': False,
                'match_found_in_di': False,
                'static_points_in_pixel_details': []
            }

        _point_in_di_frame, sph_coords_target, pixel_indices_target = \
            di_hist_future.project_point_to_pixel_indices(point_global)

        if sph_coords_target is None or pixel_indices_target is None: 
            if return_debug_info:
                reason_skipped = "Target point did not project into this DI FoV (sph_coords is None)." \
                                 if sph_coords_target is None else \
                                 "Target point projected to sph but not to valid pixel_indices."
                di_debug_details['reason_skipped_di_entirely'] = reason_skipped
                debug_data['relevant_dis_details'].append(di_debug_details)
            continue 

        total_dis_where_projection_valid += 1
        
        if return_debug_info:
            di_debug_details['projection_successful'] = True
            di_debug_details['target_sph_coords_in_di'] = sph_coords_target.tolist()
            di_debug_details['target_pixel_indices_in_di'] = list(pixel_indices_target)
        if self.logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: P_target projected to sph {np.round(sph_coords_target,3)}, pixel {pixel_indices_target}")
        
        phi_target_in_hist_di, theta_target_in_hist_di, depth_target_in_hist_di = sph_coords_target
        found_consistent_static_point_in_this_di = False
        interpolation_was_attempted_for_this_di = False
        direct_comparison_was_attempted_for_this_di = False
        
        # Initialize is_consistent_depth_wise for the interpolation path
        is_consistent_depth_wise_interp: Optional[bool] = None # Use Optional type

        if self.mc_interp_enabled and \
           self.mc_interp_min_depth <= depth_target_in_hist_di <= self.mc_interp_max_depth:
            
            interpolation_was_attempted_for_this_di = True
            if return_debug_info: di_debug_details['attempted_interpolation'] = True
            if self.logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: Attempting interpolation (d_proj={depth_target_in_hist_di:.2f}m).")
            interpolated_surface_depth = interpolate_surface_depth_at_angle(
                target_phi_rad=phi_target_in_hist_di,
                target_theta_rad=theta_target_in_hist_di,
                historical_di=di_hist_future,
                static_labels_for_map_check_values=self.static_labels_for_map_check_values,
                epsilon_phi_rad_search=self.epsilon_phi_map_rad,
                epsilon_theta_rad_search=self.epsilon_theta_map_rad,
                max_neighbors_to_consider=self.mc_interp_max_neighbors,
                max_triplets_to_try=self.mc_interp_max_triplets
            )
            if return_debug_info: # Store raw interpolation result
                di_debug_details['interpolation_result_depth'] = interpolated_surface_depth

            if interpolated_surface_depth is not None:
                current_epsilon_fwd_interp = calculate_adaptive_epsilon(
                    d_anchor_point_global,
                    self.epsilon_depth_forward_map, # di_base
                    self.adaptive_eps_config_mc_fwd.get('enabled'),
                    self.adaptive_eps_config_mc_fwd.get('dthr'),
                    self.adaptive_eps_config_mc_fwd.get('kthr'),
                    self.adaptive_eps_config_mc_fwd.get('dmax'),
                    self.adaptive_eps_config_mc_fwd.get('dmin')
                )

                adaptive_config_bwd = self.adaptive_eps_config_mc_bwd
                current_epsilon_bwd_interp = calculate_adaptive_epsilon(
                    d_anchor_point_global,
                    self.epsilon_depth_backward_map, # di_base
                    adaptive_config_bwd.get('enabled'),
                    adaptive_config_bwd.get('dthr'), 
                    adaptive_config_bwd.get('kthr'),
                    adaptive_config_bwd.get('dmax'),
                    adaptive_config_bwd.get('dmin')
                )
                if return_debug_info and di_debug_details.get('attempted_interpolation'):
                    di_debug_details.setdefault('interp_depth_target_vs_surface', {})
                    di_debug_details['interp_depth_target_vs_surface']['adaptive_epsilon_fwd_used'] = current_epsilon_fwd_interp
                    di_debug_details['interp_depth_target_vs_surface']['adaptive_epsilon_bwd_used'] = current_epsilon_bwd_interp
                    di_debug_details['interp_depth_target_vs_surface']['base_epsilon_fwd'] = self.epsilon_depth_forward_map
                    di_debug_details['interp_depth_target_vs_surface']['base_epsilon_bwd'] = self.epsilon_depth_backward_map


                is_consistent_depth_wise_interp = (
                    depth_target_in_hist_di >= interpolated_surface_depth - current_epsilon_bwd_interp and
                    depth_target_in_hist_di <= interpolated_surface_depth + current_epsilon_fwd_interp
                )
                if is_consistent_depth_wise_interp:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: Match FOUND via interpolation (interp_d={interpolated_surface_depth:.2f}, target_d={depth_target_in_hist_di:.2f}).")
                    found_consistent_static_point_in_this_di = True
                else:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: Interpolation successful (interp_d={interpolated_surface_depth:.2f}) but depth_target ({depth_target_in_hist_di:.2f}) NOT consistent.")
            else: # interpolated_surface_depth is None
                is_consistent_depth_wise_interp = False # Explicitly set to False if no surface
                if self.logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: Interpolation FAILED (returned None).")

            # Now, populate the debug details for interpolation if return_debug_info is True
            if return_debug_info and interpolation_was_attempted_for_this_di: # Check if interp was attempted
                interp_debug_details = {
                    'target_depth_in_hist_di': depth_target_in_hist_di,
                    'interpolated_surface_depth': interpolated_surface_depth, # Can be None
                    'was_consistent': is_consistent_depth_wise_interp # Now defined
                }
                if interpolated_surface_depth is not None: # Only add bounds if surface_depth is not None
                    interp_debug_details['lower_bound'] = interpolated_surface_depth - self.epsilon_depth_backward_map
                    interp_debug_details['upper_bound'] = interpolated_surface_depth + self.epsilon_depth_forward_map
                else:
                    interp_debug_details['lower_bound'] = None
                    interp_debug_details['upper_bound'] = None
                di_debug_details['interp_depth_target_vs_surface'] = interp_debug_details


        # --- Direct Comparison (if needed) ---
        if not found_consistent_static_point_in_this_di:
            should_try_direct_comparison = not interpolation_was_attempted_for_this_di or \
                                           (interpolation_was_attempted_for_this_di and self.mc_interp_fallback)

            if should_try_direct_comparison:
                direct_comparison_was_attempted_for_this_di = True
                if return_debug_info: di_debug_details['attempted_direct_comparison'] = True
                
                if pixel_indices_target is None: 
                     logger.error(f"MCC_ERROR: pixel_indices_target is None before direct comparison, DI TS {di_hist_future.timestamp:.0f}.")
                     if return_debug_info:
                         di_debug_details['reason_skipped_direct_comp'] = "pixel_indices_target became None unexpectedly."
                         debug_data['relevant_dis_details'].append(di_debug_details)
                     continue

                v_idx, h_idx = pixel_indices_target
                pixel_info = di_hist_future.get_pixel_info(v_idx, h_idx)

                if pixel_info and pixel_info.get('original_indices_in_pixel'):
                    if return_debug_info: 
                        di_debug_details['direct_comparison_pixel_had_content'] = True
                        di_debug_details['pixel_stats_in_di'] = {
                            'min_depth': pixel_info['min_depth'], 'max_depth': pixel_info['max_depth'],
                            'count': pixel_info['count'], 
                            'num_original_indices': len(pixel_info['original_indices_in_pixel'])
                        }
                    if self.logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: Attempting direct comparison (pixel has {len(pixel_info['original_indices_in_pixel'])} points).")
                    
                    for static_candidate_original_idx in pixel_info['original_indices_in_pixel']:
                        static_pt_debug_info: Dict[str, Any] = {}
                        if return_debug_info:
                            static_pt_debug_info = {
                                'original_index_in_di': static_candidate_original_idx, 'label_in_di': None,
                                'is_static_label': False, 'sph_coords_in_di': None,
                                'phi_consistent_direct': None, 'theta_consistent_direct': None, 'depth_consistent_direct': None,
                                'match_found': False
                            }
                        
                        static_candidate_label_val = di_hist_future.mdet_labels_for_points[static_candidate_original_idx]
                        static_candidate_label_enum = OcclusionResult(static_candidate_label_val)
                        if return_debug_info: static_pt_debug_info['label_in_di'] = static_candidate_label_enum.name

                        if static_candidate_label_enum not in self.static_labels_for_map_check:
                            if self.logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"MCC_TRACE:    StaticCand (idx {static_candidate_original_idx}, label {static_candidate_label_enum.name}) REJECTED (not static label).")
                            if return_debug_info:
                                static_pt_debug_info['reason_skipped_point'] = 'Not a static label.'
                                di_debug_details['static_points_in_pixel_details'].append(static_pt_debug_info)
                            continue
                        if return_debug_info: static_pt_debug_info['is_static_label'] = True
                        
                        static_sph_coords_in_di_hist_future = di_hist_future.local_sph_coords_for_points[static_candidate_original_idx]
                        if static_sph_coords_in_di_hist_future is None:
                            if return_debug_info:
                                static_pt_debug_info['reason_skipped_point'] = 'Missing local sph_coords for this static point in DI.'
                                di_debug_details['static_points_in_pixel_details'].append(static_pt_debug_info)
                            continue
                        if return_debug_info: static_pt_debug_info['sph_coords_in_di'] = static_sph_coords_in_di_hist_future.tolist()
                            
                        phi_static_cand, theta_static_cand, depth_static_cand = static_sph_coords_in_di_hist_future
                        
                        phi_diff = abs(phi_target_in_hist_di - phi_static_cand)
                        theta_diff = abs(theta_target_in_hist_di - theta_static_cand)
                        phi_consistent_direct = phi_diff <= self.epsilon_phi_map_rad
                        theta_consistent_direct = theta_diff <= self.epsilon_theta_map_rad 
                        
                        # Calculate adaptive epsilons for direct comparison
                        adaptive_config_fwd = self.adaptive_eps_config_mc_fwd 
                        current_epsilon_fwd_direct = calculate_adaptive_epsilon(
                            d_anchor_point_global,
                            self.epsilon_depth_forward_map,
                            self.adaptive_eps_config_mc_fwd .get('enabled'),
                            self.adaptive_eps_config_mc_fwd .get('dthr'),
                            self.adaptive_eps_config_mc_fwd .get('kthr'),
                            self.adaptive_eps_config_mc_fwd .get('dmax'),
                            self.adaptive_eps_config_mc_fwd .get('dmin')
                        )
                        current_epsilon_bwd_direct = calculate_adaptive_epsilon(
                            d_anchor_point_global,
                            self.epsilon_depth_backward_map,
                            self.adaptive_eps_config_mc_bwd.get('enabled'),
                            self.adaptive_eps_config_mc_bwd.get('dthr'),
                            self.adaptive_eps_config_mc_bwd.get('kthr'),
                            self.adaptive_eps_config_mc_bwd.get('dmax'),
                            self.adaptive_eps_config_mc_bwd.get('dmin')
                        )
                        if return_debug_info:
                            static_pt_debug_info['adaptive_epsilon_fwd_used'] = current_epsilon_fwd_direct
                            static_pt_debug_info['adaptive_epsilon_bwd_used'] = current_epsilon_bwd_direct
                            static_pt_debug_info['base_epsilon_fwd'] = self.epsilon_depth_forward_map
                            static_pt_debug_info['base_epsilon_bwd'] = self.epsilon_depth_backward_map


                        depth_consistent_direct = (
                            depth_target_in_hist_di >= depth_static_cand - current_epsilon_bwd_direct and
                            depth_target_in_hist_di <= depth_static_cand + current_epsilon_fwd_direct
                        )
                        if self.logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"MCC_TRACE:    StaticCand (idx {static_candidate_original_idx}, label {static_candidate_label_enum.name}, sph {np.round(static_sph_coords_in_di_hist_future,3)}):")
                            logger.debug(f"MCC_TRACE:      Phi_target={phi_target_in_hist_di:.3f}, Phi_static={phi_static_cand:.3f}, Diff={phi_diff:.3f}, Eps_phi={self.epsilon_phi_map_rad:.3f}, Consistent={phi_consistent_direct}")
                            logger.debug(f"MCC_TRACE:      Theta_target={theta_target_in_hist_di:.3f}, Theta_static={theta_static_cand:.3f}, Diff={theta_diff:.3f}, Eps_theta={self.epsilon_theta_map_rad:.3f}, Consistent={theta_consistent_direct}")
                            logger.debug(f"MCC_TRACE:      Depth_target={depth_target_in_hist_di:.3f}, Depth_static={depth_static_cand:.3f}, Eps_fwd={self.epsilon_depth_forward_map:.3f}, Eps_back={self.epsilon_depth_backward_map:.3f}, Consistent={depth_consistent_direct}")

                        if return_debug_info:
                            static_pt_debug_info.update({
                                'phi_target': phi_target_in_hist_di, 'phi_static_cand': phi_static_cand, 'phi_diff': phi_diff, 'phi_consistent': phi_consistent_direct,
                                'theta_target': theta_target_in_hist_di, 'theta_static_cand': theta_static_cand, 'theta_diff': theta_diff, 'theta_consistent': theta_consistent_direct,
                                'depth_target': depth_target_in_hist_di, 'depth_static_cand': depth_static_cand, 'depth_consistent': depth_consistent_direct
                            })

                        if phi_consistent_direct and theta_consistent_direct and depth_consistent_direct:
                            if self.logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: Match FOUND (direct) for StaticCand idx {static_candidate_original_idx}.")
                            found_consistent_static_point_in_this_di = True
                            if return_debug_info: static_pt_debug_info['match_found'] = True
                            if return_debug_info: di_debug_details['static_points_in_pixel_details'].append(static_pt_debug_info)
                            break 
                        else:
                            if self.logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"MCC_TRACE:    NO MATCH for StaticCand idx {static_candidate_original_idx}.")
                        
                        if return_debug_info: di_debug_details['static_points_in_pixel_details'].append(static_pt_debug_info)
                else: 
                    if self.logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"MCC_TRACE:  DI TS {di_hist_future.timestamp:.0f}: Direct comparison skipped (target pixel empty).")
                    if return_debug_info:
                        di_debug_details['direct_comparison_pixel_had_content'] = False
                        di_debug_details['reason_skipped_direct_comp'] = 'Direct pixel empty.'

        if found_consistent_static_point_in_this_di:
            consistent_matches_across_dis += 1
            if return_debug_info: di_debug_details['match_found_in_di'] = True
        
        # A DI was "actually checked" if interpolation was attempted OR direct comparison was attempted
        if interpolation_was_attempted_for_this_di or direct_comparison_was_attempted_for_this_di:
            # More precisely, direct_comparison_was_attempted AND pixel had content, but for simplicity:
            # If an attempt was made, we count it.
            num_dis_actually_checked += 1

        if return_debug_info:
            debug_data['relevant_dis_details'].append(di_debug_details)
    # --- End loop for di_hist_future in relevant_dis ---

    # --- 5. Final Consistency Decision ---
    final_map_consistent_result = False
    if num_dis_actually_checked > 0:
        if self.mc_threshold_mode == 'count':
            if self.mc_threshold_value_count > 0 and consistent_matches_across_dis >= self.mc_threshold_value_count:
                final_map_consistent_result = True
        elif self.mc_threshold_mode == 'ratio':
            ratio_val = consistent_matches_across_dis / num_dis_actually_checked
            if self.mc_threshold_value_ratio > 0 and ratio_val >= self.mc_threshold_value_ratio:
                final_map_consistent_result = True
    
    if return_debug_info:
        debug_data['interp_attempts_count'] = sum(1 for d_detail in debug_data.get('relevant_dis_details', []) if d_detail.get('attempted_interpolation'))
        debug_data['interp_surface_found_count'] = sum(1 for d_detail in debug_data.get('relevant_dis_details', []) if d_detail.get('attempted_interpolation') and d_detail.get('interpolation_result_depth') is not None)
        debug_data['direct_comp_attempts_count'] = sum(1 for d_detail in debug_data.get('relevant_dis_details', []) if d_detail.get('attempted_direct_comparison'))
        debug_data['direct_comp_pixel_had_content_count'] = sum(1 for d_detail in debug_data.get('relevant_dis_details', []) if d_detail.get('attempted_direct_comparison') and d_detail.get('direct_comparison_pixel_had_content'))
        debug_data['map_consistent_result'] = final_map_consistent_result
        debug_data['consistent_matches_final_count'] = consistent_matches_across_dis
        debug_data['num_dis_projection_valid'] = total_dis_where_projection_valid
        debug_data['num_dis_actually_checked_final'] = num_dis_actually_checked

        reason = f"Consistent static points found in {consistent_matches_across_dis}/{num_dis_actually_checked} DIs where checks were performed. "
        if self.mc_threshold_mode == 'count':
            reason += f"Mode: Count, Threshold: {self.mc_threshold_value_count}. "
        elif self.mc_threshold_mode == 'ratio' and num_dis_actually_checked > 0:
            ratio_calc = consistent_matches_across_dis / num_dis_actually_checked
            reason += f"Mode: Ratio, Calc Ratio: {ratio_calc:.2f}, Threshold: {self.mc_threshold_value_ratio}. "
        elif self.mc_threshold_mode == 'ratio' and num_dis_actually_checked == 0:
             reason += f"Mode: Ratio, Threshold: {self.mc_threshold_value_ratio}. (No DIs actually checked). "
        
        if final_map_consistent_result:
            debug_data['reason_for_result'] = reason + "Met consistency thresholds."
        elif num_dis_actually_checked == 0 and total_dis_where_projection_valid > 0 :
             debug_data['reason_for_result'] = "Point projected into DIs, but no checks could be performed (e.g., all target pixels empty and interpolation not triggered/failed without fallback, or no valid static points for direct comparison)."
        elif total_dis_where_projection_valid == 0 :
             debug_data['reason_for_result'] = "Point did not project into any relevant DIs, or no relevant DIs found."
        else:
            debug_data['reason_for_result'] = reason + "Did NOT meet consistency thresholds."
        
        return final_map_consistent_result, debug_data

    return final_map_consistent_result