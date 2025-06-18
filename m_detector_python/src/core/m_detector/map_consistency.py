# src/core/m_detector/map_consistency.py

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from ..constants import OcclusionResult
from .adaptive_epsilon_utils import calculate_adaptive_epsilon
from .interpolation_utils import interpolate_surface_depth_at_angle
import numpy as np

if TYPE_CHECKING:
    from ..depth_image import DepthImage

def is_map_consistent(self,
                      points_global_batch: torch.Tensor,
                      origin_di: 'DepthImage',
                      current_timestamp: float,
                      check_direction: str = 'past') -> torch.Tensor:
    """
    Hybrid GPU-CPU map consistency check.
    - Performs fast, vectorized direct comparison on GPU.
    - For points projecting to empty pixels, it falls back to the precise,
      NumPy-based interpolation on the CPU for the small subset.
    """
    batch_size = points_global_batch.shape[0]
    device = self.device

    if not self.map_consistency_enabled or batch_size == 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    # --- Setup Tensors for Tracking Results ---
    consistent_matches_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    dis_checked_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    
    if origin_di.local_sph_coords_for_points is None or origin_di.local_sph_coords_for_points.shape[0] != batch_size:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    d_anchors = origin_di.local_sph_coords_for_points[:, 2]

    # --- Get relevant historical DIs ---
    relevant_dis = self.depth_image_library.get_relevant_past_images(
            num_sweeps=self.num_past_sweeps_for_mcc
    )
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
        
        # --- GPU Path: Direct Comparison for points projecting to NON-EMPTY pixels ---
        v_coords = pixel_indices_target[active_indices, 0]
        h_coords = pixel_indices_target[active_indices, 1]
        pixel_has_content_mask = di_hist.pixel_count[v_coords, h_coords] > 0
        
        direct_comp_indices = active_indices[pixel_has_content_mask]
        if direct_comp_indices.numel() > 0:
            # This logic is fast and runs entirely on the GPU
            matches_gpu = self._perform_direct_comparison_gpu(
                direct_comp_indices, sph_coords_target, d_anchors, di_hist, static_mask_hist
            )
            consistent_matches_count[direct_comp_indices[matches_gpu]] += 1

        # --- CPU Fallback Path: Interpolation for points projecting to EMPTY pixels ---
        interp_candidate_indices_gpu = active_indices[~pixel_has_content_mask]
        if self.mc_interp_enabled and interp_candidate_indices_gpu.numel() > 0:
            
            # 1. Gather data and move the small subset to CPU
            phi_target_cpu = sph_coords_target[interp_candidate_indices_gpu, 0].cpu().numpy()
            theta_target_cpu = sph_coords_target[interp_candidate_indices_gpu, 1].cpu().numpy()
            
            # 2. Run the NumPy function in a loop 
            interp_results_mask_cpu = np.zeros(interp_candidate_indices_gpu.numel(), dtype=bool)
            for i in range(interp_candidate_indices_gpu.numel()):
                interpolated_depth = interpolate_surface_depth_at_angle(
                    target_phi_rad=phi_target_cpu[i],
                    target_theta_rad=theta_target_cpu[i],
                    historical_di=di_hist, # Pass the torch object, the function will use .cpu().numpy()
                    static_labels_for_map_check_values=self.static_labels_for_map_check_values,
                    epsilon_phi_rad_search=self.epsilon_phi_map_rad,
                    epsilon_theta_rad_search=self.epsilon_theta_map_rad,
                    max_neighbors_to_consider=self.mc_interp_max_neighbors,
                    max_triplets_to_try=self.mc_interp_max_triplets
                )
                if interpolated_depth is not None:
                    # Check consistency
                    depth_target = sph_coords_target[interp_candidate_indices_gpu[i], 2].item()
                    d_anchor_target = d_anchors[interp_candidate_indices_gpu[i]].item()
                    
                    eps_fwd = calculate_adaptive_epsilon(torch.tensor([d_anchor_target]), self.epsilon_depth_forward_map, **self.adaptive_eps_config_mc_fwd).item()
                    eps_bwd = calculate_adaptive_epsilon(torch.tensor([d_anchor_target]), self.epsilon_depth_backward_map, **self.adaptive_eps_config_mc_bwd).item()

                    if (interpolated_depth - eps_bwd) <= depth_target <= (interpolated_depth + eps_fwd):
                        interp_results_mask_cpu[i] = True

            # 3. Move results back to GPU and update counts
            interp_results_mask_gpu = torch.from_numpy(interp_results_mask_cpu).to(device)
            consistent_matches_count[interp_candidate_indices_gpu[interp_results_mask_gpu]] += 1

    final_map_consistent_result = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if self.mc_threshold_mode == 'count':
        can_be_consistent = dis_checked_count > 0
        passes_threshold = consistent_matches_count >= self.mc_threshold_value_count
        final_map_consistent_result = passes_threshold & can_be_consistent
    elif self.mc_threshold_mode == 'ratio':
        ratio = torch.zeros_like(consistent_matches_count, dtype=torch.float32)
        valid_ratio_mask = dis_checked_count > 0
        ratio[valid_ratio_mask] = consistent_matches_count[valid_ratio_mask].float() / dis_checked_count[valid_ratio_mask].float()
        final_map_consistent_result = ratio >= self.mc_threshold_value_ratio
        
    return final_map_consistent_result

def _perform_direct_comparison_gpu(self, indices, sph_coords_target, d_anchors, di_hist, static_mask_hist):
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