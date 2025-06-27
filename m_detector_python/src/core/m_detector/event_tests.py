# src/core/m_detector/event_tests.py

import torch
from typing import TYPE_CHECKING
from ..constants import OcclusionResult
import logging

if TYPE_CHECKING:
    from .base import MDetector

# A safety cap to prevent performance degradation in poorly tuned trials.
PERPENDICULAR_MOTION_CANDIDATE_CAP = 5000 

def execute_parallel_motion_away_test(mdetector_instance: 'MDetector',
                                  current_labels: torch.Tensor,
                                  di_to_process_idx: int) -> torch.Tensor:
    """
    Corrects mislabeled static points on objects moving away from the sensor.

    This test addresses a key failure mode: when an object moves directly away,
    its points are labeled as 'OCCLUDED_BY_IMAGE' by the initial pass, making it
    appear static. This test identifies these points by verifying the recursive
    occlusion property described in the M-Detector paper (Fig. 3b).

    Args:
        mdetector_instance (MDetector): The main detector instance.
        current_labels (torch.Tensor): The current labels for all points. Shape: (N,).
        di_to_process_idx (int): The index of the current DI in the library.

    Returns:
        torch.Tensor: The updated labels tensor. Shape: (N,).
    """
    device = mdetector_instance.device
    current_di = mdetector_instance.depth_image_library.get_image_by_index(di_to_process_idx)
    
    # 1. Correctly select candidates: points mislabeled as static (OCCLUDED_BY_IMAGE).
    candidate_indices = torch.where(current_labels == OcclusionResult.OCCLUDED_BY_IMAGE.value)[0]
    if candidate_indices.numel() == 0:
        return current_labels

    points_to_check_global = current_di.get_points_global_by_idx(candidate_indices)
    d_anchors_of_points_to_check = current_di.get_depths_by_idx(candidate_indices)
    
    # This mask will track if a candidate point satisfies the recursive occlusion property.
    # We initialize to True and set to False if any check fails.
    is_recursively_occluded = torch.ones(candidate_indices.shape[0], dtype=torch.bool, device=device)

    # 2. Iterate through recent historical DIs to check for consistent occlusion.
    # We need at least two historical frames to check the recursive property.
    history_length = mdetector_instance.parallel_motion_history_length
    if history_length < 1:
        return current_labels  # Cannot perform this test without history.

    for k in range(history_length):
        hist_di_idx = di_to_process_idx - 1 - k
        if hist_di_idx < 0: break
        
        historical_di = mdetector_instance.depth_image_library.get_image_by_index(hist_di_idx)
        if not historical_di or not historical_di.is_prepared_for_projection():
            is_recursively_occluded[:] = False # If a DI is missing, the chain is broken.
            break

        # 3. Find the corresponding historical point candidate in the neighborhood.
        _, _, px_indices, valid_mask = historical_di.project_points_batch(points_to_check_global)
        
        # If a point cannot be projected, it fails the consistency check.
        is_recursively_occluded[~valid_mask] = False
        if not torch.any(is_recursively_occluded): break # Early exit if no candidates remain

        # We only need to continue checking for points that are still valid candidates.
        active_mask = is_recursively_occluded & valid_mask
        if not torch.any(active_mask): break

        valid_px_indices = px_indices[active_mask]
        n_v, n_h = mdetector_instance.neighbor_search_pixels_v, mdetector_instance.neighbor_search_pixels_h
        dv = torch.arange(-n_v, n_v + 1, device=device); dh = torch.arange(-n_h, n_h + 1, device=device)
        grid_dv, grid_dh = torch.meshgrid(dv, dh, indexing='ij')
        offsets = torch.stack([grid_dv.flatten(), grid_dh.flatten()], dim=1)
        neighborhood_pixels = valid_px_indices.unsqueeze(1) + offsets
        neighborhood_pixels[:, :, 0].clamp_(0, historical_di.num_pixels_v - 1)
        neighborhood_pixels[:, :, 1].clamp_(0, historical_di.num_pixels_h - 1)
        
        flat_indices = neighborhood_pixels[..., 0] * historical_di.num_pixels_h + neighborhood_pixels[..., 1]
        # For "moving away", the historical point should be CLOSER, so we check against MIN depth.
        neighborhood_min_depths = historical_di.pixel_min_depth.view(-1)[flat_indices]
        neighborhood_min_depths[historical_di.pixel_count.view(-1)[flat_indices] == 0] = torch.inf
        
        _, min_depth_indices_in_neighborhood = torch.min(neighborhood_min_depths, dim=1)
        hist_cand_pixels = torch.gather(neighborhood_pixels, 1, min_depth_indices_in_neighborhood.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        hist_cands_global = historical_di.unproject_pixels_batch(hist_cand_pixels)

        # 4. Perform the detailed check: is our current point occluded by this historical point?
        occluded_mask_batch = mdetector_instance.check_occlusion_point_level_detailed_batch(
            points_eval_global=points_to_check_global[active_mask],
            d_anchors_of_points_eval=d_anchors_of_points_to_check[active_mask],
            points_hist_cand_global=hist_cands_global,
            historical_di=historical_di,
            occlusion_type_to_check="OCCLUDED_BY"
        )
        
        # Update the master mask: if a point was not occluded in this frame, it fails the test.
        temp_active_mask = torch.zeros_like(is_recursively_occluded)
        temp_active_mask[active_mask] = occluded_mask_batch
        is_recursively_occluded &= temp_active_mask

        if not torch.any(is_recursively_occluded): break # Early exit

    # 5. Final Decision: Any candidate that was consistently occluded through history is flipped to dynamic.
    final_indices_to_update = candidate_indices[is_recursively_occluded]
    
    new_labels = current_labels.clone()
    new_labels[final_indices_to_update] = OcclusionResult.OCCLUDING_IMAGE.value
    return new_labels

def execute_perpendicular_motion_test(mdetector_instance: 'MDetector',
                                       current_labels: torch.Tensor,
                                       di_to_process_idx: int) -> torch.Tensor:
    """
    Identifies points that are occluding previously known static points.

    This test handles objects moving perpendicularly to the sensor's line of sight.
    It checks if currently undetermined points are occluding parts of the map that
    were confidently labeled as static in the recent past.

    Args:
        mdetector_instance (MDetector): The main detector instance.
        current_labels (torch.Tensor): The current labels for all points. Shape: (N,).
        di_to_process_idx (int): The index of the current DI in the library.

    Returns:
        torch.Tensor: The updated labels tensor. Shape: (N,).
    """
    device = mdetector_instance.device
    current_di = mdetector_instance.depth_image_library.get_image_by_index(di_to_process_idx)
    
    # 1. Select candidate points: those that are still 'UNDETERMINED'.
    candidate_indices = torch.where(current_labels == OcclusionResult.UNDETERMINED.value)[0]
    if candidate_indices.numel() == 0:
        return current_labels
    
    if candidate_indices.numel() > PERPENDICULAR_MOTION_CANDIDATE_CAP:
        logging.warning(f"Test 3: Skipping due to excessive candidates ({candidate_indices.numel()} > {PERPENDICULAR_MOTION_CANDIDATE_CAP}).")
        return current_labels

    points_to_check_global = current_di.get_points_global_by_idx(candidate_indices)
    d_anchors_of_points_to_check = current_di.get_depths_by_idx(candidate_indices)
    
    # This mask will track if a candidate point occludes a static point in ANY historical frame.
    is_occluding_final = torch.zeros(candidate_indices.shape[0], dtype=torch.bool, device=device)

    # 2. Iterate through recent historical depth images.
    history_length = mdetector_instance.perpendicular_motion_history_length
    if history_length < 1:
        return current_labels 
    for k in range(history_length):
        hist_di_idx = di_to_process_idx - 1 - k
        if hist_di_idx < 0: continue
        
        historical_di = mdetector_instance.depth_image_library.get_image_by_index(hist_di_idx)
        if not historical_di or not historical_di.is_prepared_for_projection(): continue

        # Get the mask of all confidently static points from the historical frame.
        static_mask_hist = historical_di.get_static_points_mask(mdetector_instance.static_labels_for_map_check_values)
        if static_mask_hist is None or not torch.any(static_mask_hist):
            continue

        # 3. Project the current undetermined points into the historical frame.
        _, _, px_indices, valid_mask = historical_di.project_points_batch(points_to_check_global)
        if not torch.any(valid_mask):
            continue
        
        active_candidate_indices = torch.where(valid_mask)[0]
        is_occluding_batch = torch.zeros(active_candidate_indices.shape[0], dtype=torch.bool, device=device)

        # Define the pixel neighborhood search grid.
        n_v, n_h = mdetector_instance.neighbor_search_pixels_v, mdetector_instance.neighbor_search_pixels_h
        dv = torch.arange(-n_v, n_v + 1, device=device)
        dh = torch.arange(-n_h, n_h + 1, device=device)
        grid_dv, grid_dh = torch.meshgrid(dv, dh, indexing='ij')
        offsets = torch.stack([grid_dv.flatten(), grid_dh.flatten()], dim=1)

        # 4. For each candidate, check if it occludes any static points in its neighborhood.
        # A loop here is clearer and often more efficient than complex tensor ops for this sparse problem.
        for i, cand_original_idx in enumerate(active_candidate_indices):
            # Get the pixel location of this candidate in the historical frame.
            cand_pixel = px_indices[cand_original_idx]
            
            # Find all pixels in its neighborhood.
            neighborhood_pixels = cand_pixel + offsets
            neighborhood_pixels[:, 0].clamp_(0, historical_di.num_pixels_v - 1)
            neighborhood_pixels[:, 1].clamp_(0, historical_di.num_pixels_h - 1)
            flat_neighborhood_indices = neighborhood_pixels[:, 0] * historical_di.num_pixels_h + neighborhood_pixels[:, 1]
            
            # Use the pre-built map to get all original point indices within this pixel neighborhood.
            map_entries = historical_di.pixel_map_tensor[flat_neighborhood_indices]
            valid_map_entries = map_entries[map_entries[:, 1] > 0] # Filter out empty pixels
            if valid_map_entries.numel() == 0:
                continue

            # Gather all historical point indices from the neighborhood.
            hist_indices_in_neighborhood = torch.cat([
                historical_di.pixel_original_indices_tensor[start:start+count]
                for start, count in valid_map_entries
            ])

            # Filter these neighbors to keep only the ones marked as STATIC.
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

    # 6. Final decision: Update labels for points that occluded a static point.
    final_indices_to_update = candidate_indices[is_occluding_final]
    new_labels = current_labels.clone()
    new_labels[final_indices_to_update] = OcclusionResult.OCCLUDING_IMAGE.value
    return new_labels