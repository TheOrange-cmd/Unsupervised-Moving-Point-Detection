# src/core/m_detector/map_consistency.py

from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Optional, List
from ..constants import OcclusionResult
from .adaptive_epsilon_utils import calculate_adaptive_epsilon
import numpy as np
import logging
from itertools import combinations

logger_mcc_perf = logging.getLogger("MCC_PERF_DEBUG")

# Tolerance for floating point inaccuracies
BARYCENTRIC_WEIGHT_TOLERANCE = 1e-9

if TYPE_CHECKING:
    from ..depth_image import DepthImage
    from .base import MDetector

def is_map_consistent(self: 'MDetector',
                      points_global_batch: torch.Tensor,
                      origin_di: 'DepthImage') -> torch.Tensor:
    """
    Hybrid GPU-CPU map consistency check.
    """
    batch_size = points_global_batch.shape[0]
    device = self.device

    if not self.map_consistency_enabled or batch_size == 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    _, sph_coords_origin, _, valid_origin_mask = origin_di.project_points_batch(points_global_batch)

    # If none of the points are valid in their own frame (highly unlikely), we can exit.
    if not torch.any(valid_origin_mask):
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # This d_anchors tensor now correctly corresponds to the points_global_batch.
    d_anchors = sph_coords_origin[:, 2]
    # --- END OF FIX ---

    consistent_matches_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    dis_checked_count = torch.zeros(batch_size, dtype=torch.int32, device=device)

    relevant_dis = self.depth_image_library.get_relevant_past_images(num_sweeps=self.num_past_sweeps_for_mcc)
    if not relevant_dis:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    # --- Loop through historical DIs ---
    for _, di_hist in relevant_dis:
        if not di_hist.is_prepared_for_projection():
            continue

        _, sph_coords_target, pixel_indices_target, valid_mask = di_hist.project_points_batch(points_global_batch)
        
        if not torch.any(valid_mask):
            continue

        static_mask_hist = di_hist.get_static_points_mask(self.static_labels_for_map_check_values)
        if static_mask_hist is None or not torch.any(static_mask_hist):
            continue
            
        active_indices = torch.where(valid_mask)[0]
        dis_checked_count[active_indices] += 1
        
        v_coords = pixel_indices_target[active_indices, 0]
        h_coords = pixel_indices_target[active_indices, 1]
        pixel_has_content_mask = di_hist.pixel_count[v_coords, h_coords] > 0
        
        direct_comp_indices = active_indices[pixel_has_content_mask]
        if direct_comp_indices.numel() > 0:
            matches_gpu = _perform_direct_comparison_gpu(self,
                direct_comp_indices, sph_coords_target, d_anchors, di_hist, static_mask_hist
            )
            consistent_matches_count[direct_comp_indices[matches_gpu]] += 1

        interp_candidate_indices_gpu = active_indices[~pixel_has_content_mask]
        if self.mc_interp_enabled and interp_candidate_indices_gpu.numel() > 0:
            phi_target_cpu = sph_coords_target[interp_candidate_indices_gpu, 0].cpu().numpy()
            theta_target_cpu = sph_coords_target[interp_candidate_indices_gpu, 1].cpu().numpy()
            
            interp_results_mask_cpu = np.zeros(interp_candidate_indices_gpu.numel(), dtype=bool)
            for i in range(interp_candidate_indices_gpu.numel()):
                interpolated_depth = interpolate_surface_depth_at_angle(
                    target_phi_rad=phi_target_cpu[i],
                    target_theta_rad=theta_target_cpu[i],
                    historical_di=di_hist,
                    static_labels_for_map_check_values=self.static_labels_for_map_check_values,
                    epsilon_phi_rad_search=self.epsilon_phi_map_rad,
                    epsilon_theta_rad_search=self.epsilon_theta_map_rad,
                    max_neighbors_to_consider=self.mc_interp_max_neighbors,
                    max_triplets_to_try=self.mc_interp_max_triplets
                )
                if interpolated_depth is not None:
                    depth_target = sph_coords_target[interp_candidate_indices_gpu[i], 2].item()
                    d_anchor_target = d_anchors[interp_candidate_indices_gpu[i]].item()
                    
                    eps_fwd = calculate_adaptive_epsilon(torch.tensor([d_anchor_target]), self.epsilon_depth_forward_map, **self.adaptive_eps_config_mc_fwd).item()
                    eps_bwd = calculate_adaptive_epsilon(torch.tensor([d_anchor_target]), self.epsilon_depth_backward_map, **self.adaptive_eps_config_mc_bwd).item()

                    if (interpolated_depth - eps_bwd) <= depth_target <= (interpolated_depth + eps_fwd):
                        interp_results_mask_cpu[i] = True

            interp_results_mask_gpu = torch.from_numpy(interp_results_mask_cpu).to(device)
            consistent_matches_count[interp_candidate_indices_gpu[interp_results_mask_gpu]] += 1

    static_confidence_scores = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    # Create a mask to avoid division by zero
    checked_mask = dis_checked_count > 0
    
    # Calculate the ratio only for points that were checked at least once
    static_confidence_scores[checked_mask] = consistent_matches_count[checked_mask].float() / dis_checked_count[checked_mask].float()
    
    return static_confidence_scores

def interpolate_surface_depth_at_angle(
    target_phi_rad: float,
    target_theta_rad: float,
    historical_di: 'DepthImage',
    static_labels_for_map_check_values: List[int],
    epsilon_phi_rad_search: float,
    epsilon_theta_rad_search: float,
    max_neighbors_to_consider: int,
    max_triplets_to_try: int
) -> Optional[float]:
    """
    Interpolates the depth of a static surface at a given angle using barycentric coordinates.
    This function is designed to run on the CPU and is hardened to accept a DepthImage
    object that may contain GPU tensors.

    Returns:
        The interpolated depth as a float, or None if interpolation is not possible.
    """
    all_sph_coords_hist_cpu = historical_di.local_sph_coords_for_points.cpu().numpy()
    all_labels_hist_cpu = historical_di.mdet_labels_for_points.cpu().numpy()

    # 1. Find candidate static neighboring points using the CPU numpy arrays
    angular_mask_phi = np.abs(all_sph_coords_hist_cpu[:, 0] - target_phi_rad) <= epsilon_phi_rad_search
    angular_mask_theta = np.abs(all_sph_coords_hist_cpu[:, 1] - target_theta_rad) <= epsilon_theta_rad_search
    angular_mask = angular_mask_phi & angular_mask_theta
    
    static_values_tensor = np.array(static_labels_for_map_check_values, dtype=all_labels_hist_cpu.dtype)
    static_mask = np.isin(all_labels_hist_cpu, static_values_tensor)

    valid_neighbor_indices = np.where(angular_mask & static_mask)[0]

    if len(valid_neighbor_indices) < 3:
        return None

    # 2. Collect data for these neighbors
    # Use the CPU arrays for all subsequent operations
    neighbor_data_tuples = [
        (all_sph_coords_hist_cpu[idx, 0], all_sph_coords_hist_cpu[idx, 1], all_sph_coords_hist_cpu[idx, 2])
        for idx in valid_neighbor_indices
    ]
    
    # Sort neighbors by angular distance to the target
    neighbor_data_tuples.sort(key=lambda p_data: abs(target_phi_rad - p_data[0]) + abs(target_theta_rad - p_data[1]))

    # Limit the number of neighbors to reduce complexity
    if len(neighbor_data_tuples) > max_neighbors_to_consider:
        neighbor_data_tuples = neighbor_data_tuples[:max_neighbors_to_consider]
    
    if len(neighbor_data_tuples) < 3:
        return None

    # 3. Iterate through triplets of neighbors to find a valid plane
    triplet_count = 0
    for p1_data, p2_data, p3_data in combinations(neighbor_data_tuples, 3):
        if triplet_count >= max_triplets_to_try:
            break
        triplet_count += 1

        # Using float64 for linalg.solve for better precision
        A_matrix = np.array([
            [p1_data[0], p2_data[0], p3_data[0]],
            [p1_data[1], p2_data[1], p3_data[1]],
            [1,          1,          1         ]
        ], dtype=np.float64)
        b_vector = np.array([target_phi_rad, target_theta_rad, 1], dtype=np.float64)

        try:
            weights = np.linalg.solve(A_matrix, b_vector)
        except np.linalg.LinAlgError:
            # This happens if the three points are collinear in angular space
            continue 

        # If the target point is within the triangle formed by the neighbors, interpolate
        if np.all(weights >= -BARYCENTRIC_WEIGHT_TOLERANCE): # Allow for small floating point inaccuracies
            interpolated_depth = (weights[0] * p1_data[2] + 
                                  weights[1] * p2_data[2] + 
                                  weights[2] * p3_data[2])
            return float(interpolated_depth)
            
    return None


def _perform_direct_comparison_gpu(self: 'MDetector', indices, sph_coords_target, d_anchors, di_hist, static_mask_hist):
    """Helper to contain the GPU-based direct comparison logic."""
    device = self.device
    
    phi_target = sph_coords_target[indices, 0].unsqueeze(1)
    theta_target = sph_coords_target[indices, 1].unsqueeze(1)
    depth_target = sph_coords_target[indices, 2].unsqueeze(1)

    static_sph_coords_hist = di_hist.local_sph_coords_for_points[static_mask_hist]
    phi_static = static_sph_coords_hist[:, 0]
    theta_static = static_sph_coords_hist[:, 1]
    depth_static = static_sph_coords_hist[:, 2]

    is_angular_neighbor = (torch.abs(phi_target - phi_static) <= self.epsilon_phi_map_rad) & \
                          (torch.abs(theta_target - theta_static) <= self.epsilon_theta_map_rad)

    eps_fwd = calculate_adaptive_epsilon(d_anchors[indices], self.epsilon_depth_forward_map, **self.adaptive_eps_config_mc_fwd).unsqueeze(1)
    eps_bwd = calculate_adaptive_epsilon(d_anchors[indices], self.epsilon_depth_backward_map, **self.adaptive_eps_config_mc_bwd).unsqueeze(1)
    
    is_depth_consistent = (depth_target >= depth_static - eps_bwd) & (depth_target <= depth_static + eps_fwd)
    
    is_consistent_match = is_angular_neighbor & is_depth_consistent
    return torch.any(is_consistent_match, dim=1)