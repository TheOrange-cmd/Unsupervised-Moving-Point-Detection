# src/core/m_detector/event_tests.py

import torch
from typing import TYPE_CHECKING
from ..constants import OcclusionResult

if TYPE_CHECKING:
    from .base import MDetector

def execute_test2_parallel_motion(mdetector_instance: 'MDetector',
                                  current_labels: torch.Tensor,
                                  di_to_process_idx: int) -> torch.Tensor:
    """
    Fully vectorized implementation of Test 2.
    Identifies points that were previously occluding but are now visible (moved away).
    """
    device = mdetector_instance.device
    current_di = mdetector_instance.depth_image_library.get_image_by_index(di_to_process_idx)
    
    candidate_indices = torch.where(current_labels == OcclusionResult.OCCLUDING_IMAGE.value)[0]
    if candidate_indices.numel() == 0:
        return current_labels

    points_to_check_global = current_di.get_points_global_by_idx(candidate_indices)
    d_anchors_of_points_to_check = current_di.get_depths_by_idx(candidate_indices)
    is_occluded_in_any_hist_di = torch.zeros(candidate_indices.shape[0], dtype=torch.bool, device=device)

    for k in range(mdetector_instance.test2_M2_depth_images):
        hist_di_idx = di_to_process_idx - 1 - k
        if hist_di_idx < 0: continue
        
        historical_di = mdetector_instance.depth_image_library.get_image_by_index(hist_di_idx)
        if not historical_di or not historical_di.is_prepared_for_projection(): continue

        _, _, px_indices, valid_mask = historical_di.project_points_batch(points_to_check_global)
        if not torch.any(valid_mask): continue
        
        valid_px_indices = px_indices[valid_mask]
        n_v, n_h = mdetector_instance.neighbor_search_pixels_v, mdetector_instance.neighbor_search_pixels_h
        dv = torch.arange(-n_v, n_v + 1, device=device)
        dh = torch.arange(-n_h, n_h + 1, device=device)
        grid_dv, grid_dh = torch.meshgrid(dv, dh, indexing='ij')
        offsets = torch.stack([grid_dv.flatten(), grid_dh.flatten()], dim=1)
        neighborhood_pixels = valid_px_indices.unsqueeze(1) + offsets
        neighborhood_pixels[:, :, 0].clamp_(0, historical_di.num_pixels_v - 1)
        neighborhood_pixels[:, :, 1].clamp_(0, historical_di.num_pixels_h - 1)
        
        flat_indices = neighborhood_pixels[..., 0] * historical_di.num_pixels_h + neighborhood_pixels[..., 1]
        neighborhood_max_depths = historical_di.pixel_max_depth.view(-1)[flat_indices]
        neighborhood_max_depths[historical_di.pixel_count.view(-1)[flat_indices] == 0] = -torch.inf
        
        _, max_depth_indices_in_neighborhood = torch.max(neighborhood_max_depths, dim=1)
        hist_cand_pixels = torch.gather(neighborhood_pixels, 1, max_depth_indices_in_neighborhood.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        hist_cands_global = historical_di.unproject_pixels_batch(hist_cand_pixels)

        points_eval_batch = points_to_check_global[valid_mask]
        d_anchors_batch = d_anchors_of_points_to_check[valid_mask]
        
        occluded_mask_batch = mdetector_instance.check_occlusion_point_level_detailed_batch(
            points_eval_global=points_eval_batch,
            d_anchors_of_points_eval=d_anchors_batch,
            points_hist_cand_global=hist_cands_global,
            historical_di=historical_di,
            occlusion_type_to_check="OCCLUDED_BY"
        )
        is_occluded_in_any_hist_di[valid_mask] |= occluded_mask_batch

    final_indices_to_update = candidate_indices[is_occluded_in_any_hist_di]
    new_labels = current_labels.clone()
    new_labels[final_indices_to_update] = OcclusionResult.OCCLUDING_IMAGE.value
    return new_labels

def execute_test3_perpendicular_motion(mdetector_instance: 'MDetector',
                                       current_labels: torch.Tensor,
                                       di_to_process_idx: int) -> torch.Tensor:
    """
    Fully vectorized implementation of Test 3.
    Identifies points that are occluding previously known static points.
    """
    device = mdetector_instance.device
    current_di = mdetector_instance.depth_image_library.get_image_by_index(di_to_process_idx)
    
    # Test 3 runs on points that are still undetermined and not rejected by MCC.
    # The filtering of MCC-rejected points is now handled in processing.py before this call.
    candidate_indices = torch.where(current_labels == OcclusionResult.UNDETERMINED.value)[0]
    if candidate_indices.numel() == 0:
        return current_labels

    points_to_check_global = current_di.get_points_global_by_idx(candidate_indices)
    d_anchors_of_points_to_check = current_di.get_depths_by_idx(candidate_indices)
    is_occluding_in_any_hist_di = torch.zeros(candidate_indices.shape[0], dtype=torch.bool, device=device)

    for k in range(mdetector_instance.test3_M3_depth_images):
        hist_di_idx = di_to_process_idx - 1 - k
        if hist_di_idx < 0: continue
        
        historical_di = mdetector_instance.depth_image_library.get_image_by_index(hist_di_idx)
        if not historical_di or not historical_di.is_prepared_for_projection(): continue

        # For Test 3, the historical "candidate" is any static point in the neighborhood.
        # This logic is more complex to vectorize perfectly without iterating DIs.
        # We will iterate DIs but vectorize the logic within each one.
        _, _, px_indices, valid_mask = historical_di.project_points_batch(points_to_check_global)
        if not torch.any(valid_mask): continue

        valid_px_indices = px_indices[valid_mask]
        
        # In Test 3, we check against ALL static points in the neighborhood, not just the max depth one.
        # This requires a different, more involved approach. For now, we will use a simplified
        # version analogous to Test 2 for performance comparison, checking against max depth point.
        # A full implementation would require a more complex scatter-gather operation.
        n_v, n_h = mdetector_instance.neighbor_search_pixels_v, mdetector_instance.neighbor_search_pixels_h
        dv = torch.arange(-n_v, n_v + 1, device=device)
        dh = torch.arange(-n_h, n_h + 1, device=device)
        grid_dv, grid_dh = torch.meshgrid(dv, dh, indexing='ij')
        offsets = torch.stack([grid_dv.flatten(), grid_dh.flatten()], dim=1)
        neighborhood_pixels = valid_px_indices.unsqueeze(1) + offsets
        neighborhood_pixels[:, :, 0].clamp_(0, historical_di.num_pixels_v - 1)
        neighborhood_pixels[:, :, 1].clamp_(0, historical_di.num_pixels_h - 1)
        
        flat_indices = neighborhood_pixels[..., 0] * historical_di.num_pixels_h + neighborhood_pixels[..., 1]
        
        # Check against min depth point, as we want to see if we are in front of the closest static surface
        neighborhood_min_depths = historical_di.pixel_min_depth.view(-1)[flat_indices]
        neighborhood_min_depths[historical_di.pixel_count.view(-1)[flat_indices] == 0] = torch.inf

        _, min_depth_indices_in_neighborhood = torch.min(neighborhood_min_depths, dim=1)
        hist_cand_pixels = torch.gather(neighborhood_pixels, 1, min_depth_indices_in_neighborhood.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        hist_cands_global = historical_di.unproject_pixels_batch(hist_cand_pixels)

        points_eval_batch = points_to_check_global[valid_mask]
        d_anchors_batch = d_anchors_of_points_to_check[valid_mask]
        
        occluding_mask_batch = mdetector_instance.check_occlusion_point_level_detailed_batch(
            points_eval_global=points_eval_batch,
            d_anchors_of_points_eval=d_anchors_batch,
            points_hist_cand_global=hist_cands_global,
            historical_di=historical_di,
            occlusion_type_to_check="OCCLUDING"
        )
        is_occluding_in_any_hist_di[valid_mask] |= occluding_mask_batch

    final_indices_to_update = candidate_indices[is_occluding_in_any_hist_di]
    new_labels = current_labels.clone()
    new_labels[final_indices_to_update] = OcclusionResult.OCCLUDING_IMAGE.value
    return new_labels