# src/core/m_detector/occlusion_checks.py

import torch
from typing import TYPE_CHECKING

from ..constants import OcclusionResult
from .adaptive_epsilon_utils import calculate_adaptive_epsilon

if TYPE_CHECKING:
    from .base import MDetector
    from ..depth_image import DepthImage

def check_occlusion_batch(self: 'MDetector',
                          points_global_batch: torch.Tensor,
                          historical_depth_image: 'DepthImage') -> torch.Tensor:
    """
    Performs a fast, coarse-grained (broad-phase) occlusion check for a batch of points
    against an entire historical depth image.

    This function acts as a rapid filter, comparing each point to a neighborhood of
    pixels in the historical image rather than to individual points.

    Args:
        points_global_batch (torch.Tensor): Points to check, in global coordinates. Shape: (N, 3).
        historical_depth_image (DepthImage): The historical DI to check against.

    Returns:
        torch.Tensor: Integer labels from OcclusionResult for each point. Shape: (N,).
    """
    batch_size = points_global_batch.shape[0]
    device = self.device

    # Ensure input is a PyTorch tensor on the correct device
    if not isinstance(points_global_batch, torch.Tensor):
        points_global_batch = torch.from_numpy(points_global_batch).float()
    points_global_batch = points_global_batch.to(device)

    # --- 1. Project points onto the historical image ---
    _, sph_coords, pixel_coords, is_valid_projection_mask = historical_depth_image.project_points_batch(points_global_batch)
    
    # Initialize results, assuming points are occluded by default. This will be refined.
    results = torch.full((batch_size,), OcclusionResult.OCCLUDED_BY_IMAGE.value, dtype=torch.int32, device=device)
    
    # Get the indices of points that projected successfully into the image frame.
    valid_indices = torch.where(is_valid_projection_mask)[0]

    # If no points project into the image, we can return early.
    if valid_indices.numel() == 0:
        return results

    # --- 2. Define search neighborhoods for all valid points ---
    valid_pixels = pixel_coords[valid_indices]
    valid_depths = sph_coords[valid_indices, 2]

    # Create a 2D grid of offsets (e.g., a 3x3 grid for n_v=1, n_h=1)
    n_v, n_h = self.neighbor_search_pixels_v, self.neighbor_search_pixels_h
    dv = torch.arange(-n_v, n_v + 1, device=device)
    dh = torch.arange(-n_h, n_h + 1, device=device)
    grid_dv, grid_dh = torch.meshgrid(dv, dh, indexing='ij')
    offsets = torch.stack([grid_dv.flatten(), grid_dh.flatten()], dim=1)

    # Use broadcasting to create a neighborhood grid for each valid point.
    # Shape: (num_valid_points, num_neighborhood_pixels, 2)
    neighborhood_pixels = valid_pixels.unsqueeze(1) + offsets

    # Prevent neighborhood indices from going out of bounds of the image dimensions.
    neighborhood_pixels[:, :, 0].clamp_(0, historical_depth_image.num_pixels_v - 1)
    neighborhood_pixels[:, :, 1].clamp_(0, historical_depth_image.num_pixels_h - 1)

    # --- 3. Gather historical data from neighborhoods ---
    # Convert 2D neighborhood coordinates to 1D indices for efficient, vectorized lookup.
    v_coords = neighborhood_pixels[:, :, 0]
    h_coords = neighborhood_pixels[:, :, 1]
    flat_indices = v_coords * historical_depth_image.num_pixels_h + h_coords

    # Fetch pre-aggregated data from the historical image using the flat indices.
    hist_counts_flat = historical_depth_image.pixel_count.view(-1)
    hist_min_depth_flat = historical_depth_image.pixel_min_depth.view(-1)
    hist_max_depth_flat = historical_depth_image.pixel_max_depth.view(-1)

    neighborhood_counts = hist_counts_flat[flat_indices]
    neighborhood_min_depths = hist_min_depth_flat[flat_indices]
    neighborhood_max_depths = hist_max_depth_flat[flat_indices]

    # --- 4. Classify points based on neighborhood data ---
    # A neighborhood pixel is valid only if it contained points in the historical image.
    has_data_mask = neighborhood_counts > 0
    any_data_in_neighborhood = torch.any(has_data_mask, dim=1)

    # If a point's entire neighborhood was empty, classify it as such.
    results[valid_indices[~any_data_in_neighborhood]] = OcclusionResult.EMPTY_IN_IMAGE.value

    # Proceed only with points whose neighborhoods contained some data.
    if torch.any(any_data_in_neighborhood):
        points_with_data_mask = any_data_in_neighborhood

        # To find the true min/max depth, we must ignore empty pixels.
        # We achieve this by replacing their depths with +/- infinity.
        min_depths_clean = torch.where(has_data_mask, neighborhood_min_depths, float('inf'))
        max_depths_clean = torch.where(has_data_mask, neighborhood_max_depths, float('-inf'))

        # Aggregate the min/max depth across the entire neighborhood for each point.
        min_depth_in_region = torch.min(min_depths_clean[points_with_data_mask], dim=1).values
        max_depth_in_region = torch.max(max_depths_clean[points_with_data_mask], dim=1).values
        
        # Filter to the active points and depths we are currently classifying.
        active_depths = valid_depths[points_with_data_mask]
        active_indices = valid_indices[points_with_data_mask]
        current_results = results[active_indices]
        
        # If the point's depth is greater than the max depth of the region, it's occluded.
        occluded_mask = active_depths > max_depth_in_region + self.epsilon_depth_occlusion
        current_results = torch.where(occluded_mask, OcclusionResult.OCCLUDED_BY_IMAGE.value, current_results)

        # If the point's depth is less than the min depth of the region, it's an occluder.
        occluding_mask = active_depths < min_depth_in_region - self.epsilon_depth_occlusion
        current_results = torch.where(occluding_mask, OcclusionResult.OCCLUDING_IMAGE.value, current_results)

        # Update the main results tensor with these refined classifications.
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
    Performs a precise, fine-grained (narrow-phase) occlusion check between paired points.

    This high-precision step is used after a coarse check has identified candidates.
    It uses strict angular matching and an adaptive depth threshold.

    Args:
        points_eval_global (torch.Tensor): Points to evaluate. Shape: (N, 3).
        d_anchors_of_points_eval (torch.Tensor): Anchor depths for eval points. Shape: (N,).
        points_hist_cand_global (torch.Tensor): Paired historical candidate points. Shape: (N, 3).
        historical_di (DepthImage): The historical DI used for projection.
        occlusion_type_to_check (str): Relationship to verify ("OCCLUDING" or "OCCLUDED_BY").

    Returns:
        torch.Tensor: Boolean mask where True indicates the condition was met. Shape: (N,).
    """
    batch_size = points_eval_global.shape[0]
    final_results = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    # Guard clause for empty input
    if batch_size == 0:
        return final_results

    # --- 1. Project both sets of points ---
    # Concatenate for a single, efficient projection call to the GPU.
    points_to_project = torch.cat([points_eval_global, points_hist_cand_global])
    _, sph_coords_batch, _, valid_mask = historical_di.project_points_batch(points_to_project)

    # Split the results back into evaluation and historical sets.
    sph_coords_eval, sph_coords_hist = sph_coords_batch.chunk(2)
    valid_mask_eval, valid_mask_hist = valid_mask.chunk(2)

    # --- 2. Filter for valid pairs ---
    # A check is only meaningful if *both* points in a pair successfully project.
    possible_check_mask = valid_mask_eval & valid_mask_hist
    if not torch.any(possible_check_mask):
        return final_results
    
    # We now work only with the subset of pairs that can be checked.
    possible_indices = torch.where(possible_check_mask)[0]
    
    # Extract spherical coordinates for the valid pairs.
    phi_eval = sph_coords_eval[possible_indices, 0]
    theta_eval = sph_coords_eval[possible_indices, 1]
    d_eval = sph_coords_eval[possible_indices, 2]

    phi_hist = sph_coords_hist[possible_indices, 0]
    theta_hist = sph_coords_hist[possible_indices, 1]
    d_hist = sph_coords_hist[possible_indices, 2]

    # --- 3. Perform strict angular proximity check ---
    # This ensures both points are on almost the exact same line of sight.
    angular_phi_match = torch.abs(phi_eval - phi_hist) <= self.detailed_check_angular_threshold_h_rad
    angular_theta_match = torch.abs(theta_eval - theta_hist) <= self.detailed_check_angular_threshold_v_rad
    angular_match_mask = angular_phi_match & angular_theta_match

    if not torch.any(angular_match_mask):
        return final_results

    # Filter again to the subset of pairs that passed the angular check.
    passed_angular_indices = possible_indices[angular_match_mask]

    # --- 4. Calculate adaptive depth threshold ---
    # For points that are angularly close, calculate a more precise, distance-dependent
    # depth threshold to account for increasing sensor error over distance.
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

    # --- 5. Perform final depth comparison ---
    d_eval_final = d_eval[angular_match_mask]
    d_hist_final = d_hist[angular_match_mask]

    # Check the specific occlusion condition requested by the caller.
    depth_condition_met = torch.zeros_like(d_eval_final, dtype=torch.bool)
    if occlusion_type_to_check == "OCCLUDING":
        # Check if the evaluation point is significantly closer than the historical point.
        depth_condition_met = d_eval_final < d_hist_final - current_detailed_epsilon
    elif occlusion_type_to_check == "OCCLUDED_BY":
        # Check if the evaluation point is significantly farther than the historical point.
        depth_condition_met = d_eval_final > d_hist_final + current_detailed_epsilon
    else:
        # This should not happen with controlled inputs.
        raise ValueError(f"Invalid occlusion_type_to_check: {occlusion_type_to_check}")

    # --- 6. Update final results ---
    final_results[passed_angular_indices[depth_condition_met]] = True
        
    return final_results