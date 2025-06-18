# src/core/m_detector/occlusion_checks.py

import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING
from ..constants import OcclusionResult
from .adaptive_epsilon_utils import calculate_adaptive_epsilon
from ..depth_image import DepthImage

import logging
logger_oc = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .base import MDetector

def check_occlusion_batch(self,
                          points_global_batch: torch.Tensor,
                          historical_depth_image: DepthImage) -> torch.Tensor:
    """
    Batch occlusion check using pure PyTorch tensor operations.
    """
    batch_size = points_global_batch.shape[0]
    device = self.device

    if not isinstance(points_global_batch, torch.Tensor):
        points_global_batch = torch.from_numpy(points_global_batch).float()
    points_global_batch = points_global_batch.to(device)

    _, sph_coords, pixel_indices, valid_mask = historical_depth_image.project_points_batch(points_global_batch)
    results = torch.full((batch_size,), OcclusionResult.UNDETERMINED.value, dtype=torch.int32, device=device)
    valid_indices = torch.where(valid_mask)[0]

    if valid_indices.numel() == 0:
        return results

    valid_pixels = pixel_indices[valid_indices]
    valid_depths = sph_coords[valid_indices, 2]

    n_v, n_h = self.neighbor_search_pixels_v, self.neighbor_search_pixels_h
    dv = torch.arange(-n_v, n_v + 1, device=device)
    dh = torch.arange(-n_h, n_h + 1, device=device)
    grid_dv, grid_dh = torch.meshgrid(dv, dh, indexing='ij')
    offsets = torch.stack([grid_dv.flatten(), grid_dh.flatten()], dim=1)
    neighborhood_pixels = valid_pixels.unsqueeze(1) + offsets

    neighborhood_pixels[:, :, 0].clamp_(0, historical_depth_image.num_pixels_v - 1)
    neighborhood_pixels[:, :, 1].clamp_(0, historical_depth_image.num_pixels_h - 1)

    v_coords = neighborhood_pixels[:, :, 0]
    h_coords = neighborhood_pixels[:, :, 1]
    flat_indices = v_coords * historical_depth_image.num_pixels_h + h_coords

    hist_counts_flat = historical_depth_image.pixel_count.view(-1)
    hist_min_depth_flat = historical_depth_image.pixel_min_depth.view(-1)
    hist_max_depth_flat = historical_depth_image.pixel_max_depth.view(-1)

    neighborhood_counts = hist_counts_flat[flat_indices]
    neighborhood_min_depths = hist_min_depth_flat[flat_indices]
    neighborhood_max_depths = hist_max_depth_flat[flat_indices]

    has_data_mask = neighborhood_counts > 0
    any_data_in_neighborhood = torch.any(has_data_mask, dim=1)
    results[valid_indices[~any_data_in_neighborhood]] = OcclusionResult.EMPTY_IN_IMAGE.value

    if torch.any(any_data_in_neighborhood):
        points_with_data_mask = any_data_in_neighborhood
        min_depths_clean = torch.where(has_data_mask, neighborhood_min_depths, float('inf'))
        max_depths_clean = torch.where(has_data_mask, neighborhood_max_depths, float('-inf'))

        min_depth_in_region = torch.min(min_depths_clean[points_with_data_mask], dim=1).values
        max_depth_in_region = torch.max(max_depths_clean[points_with_data_mask], dim=1).values
        
        active_depths = valid_depths[points_with_data_mask]
        active_indices = valid_indices[points_with_data_mask]

        current_results = results[active_indices]
        
        occluded_mask = active_depths > max_depth_in_region + self.epsilon_depth_occlusion
        current_results = torch.where(occluded_mask, OcclusionResult.OCCLUDED_BY_IMAGE.value, current_results)

        occluding_mask = active_depths < min_depth_in_region - self.epsilon_depth_occlusion
        current_results = torch.where(occluding_mask, OcclusionResult.OCCLUDING_IMAGE.value, current_results)

        results[active_indices] = current_results

    return results

def check_occlusion_point_level_detailed_batch(
    self: 'MDetector',
    points_eval_global: torch.Tensor,
    d_anchors_of_points_eval: torch.Tensor,
    points_hist_cand_global: torch.Tensor,
    historical_di: 'DepthImage',
    occlusion_type_to_check: str,
) -> torch.Tensor:
    """
    Performs a detailed occlusion check between two BATCHES of global points.
    This is the fully vectorized version for high performance.
    """
    batch_size = points_eval_global.shape[0]
    final_results = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    if batch_size == 0:
        return final_results

    # Project both batches of points at once
    points_to_project = torch.cat([points_eval_global, points_hist_cand_global])
    _, sph_coords_batch, _, valid_mask = historical_di.project_points_batch(points_to_project)

    # Split the results back
    sph_coords_eval, sph_coords_hist = sph_coords_batch.chunk(2)
    valid_mask_eval, valid_mask_hist = valid_mask.chunk(2)

    # A check is only possible if BOTH points in a pair project successfully
    possible_check_mask = valid_mask_eval & valid_mask_hist
    if not torch.any(possible_check_mask):
        return final_results
    
    # Filter to only the pairs where the check is possible
    possible_indices = torch.where(possible_check_mask)[0]
    
    phi_eval = sph_coords_eval[possible_indices, 0]
    theta_eval = sph_coords_eval[possible_indices, 1]
    d_eval = sph_coords_eval[possible_indices, 2]

    phi_hist = sph_coords_hist[possible_indices, 0]
    theta_hist = sph_coords_hist[possible_indices, 1]
    d_hist = sph_coords_hist[possible_indices, 2]

    # Perform angular check on the batch
    angular_phi_match = torch.abs(phi_eval - phi_hist) <= self.detailed_check_angular_threshold_h_rad
    angular_theta_match = torch.abs(theta_eval - theta_hist) <= self.detailed_check_angular_threshold_v_rad
    angular_match_mask = angular_phi_match & angular_theta_match

    if not torch.any(angular_match_mask):
        return final_results

    # Filter again to pairs that passed the angular check
    passed_angular_indices = possible_indices[angular_match_mask]

    # Calculate adaptive epsilon for the remaining points
    d_anchors_for_calc = d_anchors_of_points_eval[passed_angular_indices]
    adaptive_config = self.adaptive_eps_config_occ_depth
    current_detailed_epsilon = calculate_adaptive_epsilon(
        d_anchors_for_calc,
        self.detailed_check_epsilon_depth,
        adaptive_config['enabled'],
        adaptive_config['dthr'],
        adaptive_config['kthr'],
        adaptive_config['dmax'],
        adaptive_config['dmin']
    )

    # Perform depth check on the remaining batch
    d_eval_final = d_eval[angular_match_mask]
    d_hist_final = d_hist[angular_match_mask]

    depth_condition_met = torch.zeros_like(d_eval_final, dtype=torch.bool)
    if occlusion_type_to_check == "OCCLUDING":
        depth_condition_met = d_eval_final < d_hist_final - current_detailed_epsilon
    elif occlusion_type_to_check == "OCCLUDED_BY":
        depth_condition_met = d_eval_final > d_hist_final + current_detailed_epsilon
    else:
        raise ValueError(f"Invalid occlusion_type_to_check: {occlusion_type_to_check}")

    # Update the final results tensor at the correct indices
    final_results[passed_angular_indices[depth_condition_met]] = True
        
    return final_results