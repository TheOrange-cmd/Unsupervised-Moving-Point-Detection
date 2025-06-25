# src/core/m_detector/event_tests.py

import torch
from typing import TYPE_CHECKING
from ..constants import OcclusionResult
import logging

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
    
    candidate_indices = torch.where(current_labels == OcclusionResult.UNDETERMINED.value)[0]
    num_candidates = candidate_indices.numel()

    if num_candidates == 0:
        return current_labels
    
    TEST3_CANDIDATE_CAP = 5000 
    if num_candidates > TEST3_CANDIDATE_CAP:
        logging.warning(f"Test 3: Skipping due to excessive candidates ({num_candidates} > {TEST3_CANDIDATE_CAP}). "
                        f"Initial filter params are likely too lenient.")
        return current_labels

    points_to_check_global = current_di.get_points_global_by_idx(candidate_indices)
    d_anchors_of_points_to_check = current_di.get_depths_by_idx(candidate_indices)
    
    is_occluding_final = torch.zeros(candidate_indices.shape[0], dtype=torch.bool, device=device)

    for k in range(mdetector_instance.test3_M3_depth_images):
        hist_di_idx = di_to_process_idx - 1 - k
        if hist_di_idx < 0: continue
        
        historical_di = mdetector_instance.depth_image_library.get_image_by_index(hist_di_idx)
        if not historical_di or not historical_di.is_prepared_for_projection(): continue

        static_mask_hist = historical_di.get_static_points_mask(mdetector_instance.static_labels_for_map_check_values)
        if static_mask_hist is None or not torch.any(static_mask_hist):
            continue

        # Project the current candidate points into the historical frame.
        _, _, px_indices, valid_mask = historical_di.project_points_batch(points_to_check_global)
        if not torch.any(valid_mask):
            continue
        
        active_candidate_indices = torch.where(valid_mask)[0]
        is_occluding_batch = torch.zeros(active_candidate_indices.shape[0], dtype=torch.bool, device=device)

        # 1. Define the pixel neighborhood search grid.
        n_v, n_h = mdetector_instance.neighbor_search_pixels_v, mdetector_instance.neighbor_search_pixels_h
        dv = torch.arange(-n_v, n_v + 1, device=device)
        dh = torch.arange(-n_h, n_h + 1, device=device)
        grid_dv, grid_dh = torch.meshgrid(dv, dh, indexing='ij')
        offsets = torch.stack([grid_dv.flatten(), grid_dh.flatten()], dim=1) # Shape: [num_offsets, 2]

        # 2. Iterate through each active candidate point to check against its local neighborhood.
        # A loop here is clearer and often more efficient than complex tensor ops for sparse, variable-size problems.
        for i, cand_original_idx in enumerate(active_candidate_indices):
            # Get the pixel location of this specific candidate in the historical frame.
            cand_pixel = px_indices[cand_original_idx]
            
            # Find all pixels in its neighborhood.
            neighborhood_pixels = cand_pixel + offsets
            neighborhood_pixels[:, 0].clamp_(0, historical_di.num_pixels_v - 1)
            neighborhood_pixels[:, 1].clamp_(0, historical_di.num_pixels_h - 1)
            flat_neighborhood_indices = neighborhood_pixels[:, 0] * historical_di.num_pixels_h + neighborhood_pixels[:, 1]
            
            # 3. Use the pre-built map to get all original point indices within this pixel neighborhood.
            map_entries = historical_di.pixel_map_tensor[flat_neighborhood_indices]
            valid_map_entries = map_entries[map_entries[:, 1] > 0] # Filter out empty pixels

            if valid_map_entries.numel() == 0:
                continue

            # Gather all historical point indices from the neighborhood.
            hist_indices_in_neighborhood = torch.cat([
                historical_di.pixel_original_indices_tensor[start:start+count]
                for start, count in valid_map_entries
            ])

            # 4. Filter these neighbors to keep only the ones marked as STATIC.
            static_neighbor_mask = static_mask_hist[hist_indices_in_neighborhood]
            static_hist_indices = hist_indices_in_neighborhood[static_neighbor_mask]

            if static_hist_indices.numel() == 0:
                continue

            # 5. Perform the detailed occlusion check for this one candidate against its many static neighbors.
            point_eval_global = points_to_check_global[cand_original_idx].expand(static_hist_indices.numel(), -1)
            d_anchor_eval = d_anchors_of_points_to_check[cand_original_idx].expand(static_hist_indices.numel())
            points_hist_cand_global = historical_di.original_points_global_coords[static_hist_indices]

            occlusion_results = mdetector_instance.check_occlusion_point_level_detailed_batch(
                points_eval_global=point_eval_global,
                d_anchors_of_points_eval=d_anchor_eval,
                points_hist_cand_global=points_hist_cand_global,
                historical_di=historical_di,
                occlusion_type_to_check="OCCLUDING"
            )

            # If it occludes ANY of its local static neighbors, we're done with this point.
            if torch.any(occlusion_results):
                is_occluding_batch[i] = True

        # Update the final result mask for this historical DI.
        is_occluding_final[active_candidate_indices] |= is_occluding_batch

    final_indices_to_update = candidate_indices[is_occluding_final]
    new_labels = current_labels.clone()
    new_labels[final_indices_to_update] = OcclusionResult.OCCLUDING_IMAGE.value
    return new_labels