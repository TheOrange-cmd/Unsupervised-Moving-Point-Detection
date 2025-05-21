# src/core/m_detector/occlusion_checks.py
# This file is imported into MDetector class

import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from ..depth_image import DepthImage
from ..constants import OcclusionResult

import logging
logger_oc = logging.getLogger(__name__) # Logger for this module

def check_occlusion_pixel_level(self,
                               current_point_global: np.ndarray,
                               historical_depth_image: DepthImage
                               ) -> Tuple[OcclusionResult, Optional[Tuple[int, int]], Optional[np.ndarray]]:
    """
    Performs pixel-level occlusion check for current_point_global against a historical_depth_image.
    (Corresponds to Test 1 - Fig. 10 in M-Detector paper, using min/max depths in pixel regions).

    Args:
        self: MDetector instance
        current_point_global (np.ndarray): The 3D point (in global frame) from the current scan.
        historical_depth_image (DepthImage): A historical depth image to compare against.

    Returns:
        Tuple[OcclusionResult, Optional[Tuple[int, int]], Optional[np.ndarray]]:
            - The occlusion result enum.
            - Pixel indices (v_idx, h_idx) in historical_depth_image if projected, else None.
            - Spherical coordinates (phi, theta, d_curr) of current_point_global in historical_depth_image's frame.
    """
    # 1. Project current_point_global into historical_depth_image's frame
    point_in_hist_di_frame, sph_coords_curr, pixel_indices_in_hist_di = \
        historical_depth_image.project_point_to_pixel_indices(current_point_global)

    if pixel_indices_in_hist_di is None or sph_coords_curr is None:
        return OcclusionResult.UNDETERMINED, None, None # Point projects outside historical DI's FoV

    v_idx_curr_proj, h_idx_curr_proj = pixel_indices_in_hist_di
    d_curr = sph_coords_curr[2] # Depth of current_point w.r.t. historical_depth_image's origin

    # 2. Gather min/max depths from the projected pixel and its neighbors in historical_depth_image
    # Define region bounds with clipping to image boundaries
    v_start = max(0, v_idx_curr_proj - self.neighbor_search_pixels_v)
    v_end = min(historical_depth_image.num_pixels_v, v_idx_curr_proj + self.neighbor_search_pixels_v + 1)
    h_start = max(0, h_idx_curr_proj - self.neighbor_search_pixels_h)
    h_end = min(historical_depth_image.num_pixels_h, h_idx_curr_proj + self.neighbor_search_pixels_h + 1)
    
    # Extract region data from arrays
    region_min_depths = historical_depth_image.pixel_min_depth[v_start:v_end, h_start:h_end]
    region_max_depths = historical_depth_image.pixel_max_depth[v_start:v_end, h_start:h_end]
    region_counts = historical_depth_image.pixel_count[v_start:v_end, h_start:h_end]
    
    # Check if any pixel in region has points
    has_data = region_counts > 0
    found_data_in_region = np.any(has_data)
    
    if not found_data_in_region:
        return OcclusionResult.EMPTY_IN_IMAGE, pixel_indices_in_hist_di, sph_coords_curr

    # Find min/max depths in the region (only where points exist)
    min_depth_in_region = np.min(region_min_depths[has_data])
    max_depth_in_region = np.max(region_max_depths[has_data])

    # 3. Check occlusion conditions (Eq. 7 and 8 in M-Detector paper, adapted)
    if d_curr > max_depth_in_region + self.epsilon_depth_occlusion:
        return OcclusionResult.OCCLUDED_BY_IMAGE, pixel_indices_in_hist_di, sph_coords_curr
    
    if d_curr < min_depth_in_region - self.epsilon_depth_occlusion:
        return OcclusionResult.OCCLUDING_IMAGE, pixel_indices_in_hist_di, sph_coords_curr

    return OcclusionResult.UNDETERMINED, pixel_indices_in_hist_di, sph_coords_curr

def check_occlusion_batch(self, 
                         points_global_batch: np.ndarray,
                         historical_depth_image: DepthImage) -> np.ndarray:
    """
    Batch occlusion check that reduces NumPy operations.
    """
    batch_size = points_global_batch.shape[0]
    
    # Project all points at once
    points_local, sph_coords, pixel_indices, valid_mask = historical_depth_image.project_points_batch(points_global_batch)
    
    # Pre-allocate results array with UNDETERMINED
    results = np.full(batch_size, OcclusionResult.UNDETERMINED.value, dtype=np.int32)
    
    if not np.any(valid_mask):
        # No valid points to process
        return np.array([OcclusionResult(int(r)) for r in results])
    
    # Extract valid indices for faster processing
    valid_indices = np.where(valid_mask)[0]
    valid_v = pixel_indices[valid_mask, 0].astype(np.int32)
    valid_h = pixel_indices[valid_mask, 1].astype(np.int32)
    valid_depths = sph_coords[valid_mask, 2]
    
    # Pre-fetch array dimensions for bounds checking
    num_v = historical_depth_image.num_pixels_v
    num_h = historical_depth_image.num_pixels_h
    
    # Get search windows (vectorized)
    v_start = np.maximum(0, valid_v - self.neighbor_search_pixels_v)
    v_end = np.minimum(num_v, valid_v + self.neighbor_search_pixels_v + 1)
    h_start = np.maximum(0, valid_h - self.neighbor_search_pixels_h)
    h_end = np.minimum(num_h, valid_h + self.neighbor_search_pixels_h + 1)
    
    # Loop over valid points (using direct indexing)
    min_depths = historical_depth_image.pixel_min_depth
    max_depths = historical_depth_image.pixel_max_depth
    counts = historical_depth_image.pixel_count
    
    for i, (idx, vs, ve, hs, he, d) in enumerate(zip(valid_indices, v_start, v_end, h_start, h_end, valid_depths)):
        # Get region data
        region_counts = counts[vs:ve, hs:he]
        
        # Quick check if any data exists
        if not np.any(region_counts > 0):
            results[idx] = OcclusionResult.EMPTY_IN_IMAGE.value
            continue
        
        # Get min/max depths only where counts > 0
        has_data = region_counts > 0
        region_min = min_depths[vs:ve, hs:he]
        region_max = max_depths[vs:ve, hs:he]
        
        # Avoid double-masking
        min_depth_values = region_min[has_data]
        max_depth_values = region_max[has_data]
        
        # Use direct min/max rather than np.min/np.max for faster access
        min_depth_in_region = min_depth_values.min() if min_depth_values.size > 0 else np.inf
        max_depth_in_region = max_depth_values.max() if max_depth_values.size > 0 else -np.inf
        
        # Apply occlusion logic
        if d > max_depth_in_region + self.epsilon_depth_occlusion:
            results[idx] = OcclusionResult.OCCLUDED_BY_IMAGE.value
        elif d < min_depth_in_region - self.epsilon_depth_occlusion:
            results[idx] = OcclusionResult.OCCLUDING_IMAGE.value
    
    # Convert to enum values
    return np.array([OcclusionResult(int(r)) for r in results])

def check_occlusion_point_level_detailed(
    self, # MDetector instance
    point_eval_global: np.ndarray,      # The point whose occlusion status relative to point_hist_cand_global is being checked
    point_hist_cand_global: np.ndarray, # A candidate point from a historical DI's neighborhood
    historical_di: DepthImage,          # The historical DI where point_hist_cand_global originated
    occlusion_type_to_check: str        # "OCCLUDING" (eval occludes hist_cand) or "OCCLUDED_BY" (eval occluded by hist_cand)
) -> bool:
    """
    Performs a detailed occlusion check between two specific global points,
    considering their projection into the historical_di.

    Args:
        self (MDetector): The MDetector instance (for config params).
        point_eval_global: Global coordinates of the point being evaluated.
        point_hist_cand_global: Global coordinates of the historical candidate point.
        historical_di: The DepthImage object for the historical frame.
        occlusion_type_to_check: 
            - "OCCLUDING": Checks if point_eval_global occludes point_hist_cand_global.
            - "OCCLUDED_BY": Checks if point_eval_global is occluded by point_hist_cand_global.

    Returns:
        bool: True if the specified occlusion condition is met, False otherwise.
    """
    if not historical_di.is_prepared_for_projection():
        logger_oc.warning(f"DetailedCheck: Historical DI (TS {historical_di.timestamp}) not prepared for projection. Cannot perform check.")
        return False

    # 1. Project point_eval_global into historical_di to get its spherical coordinates
    #    relative to the historical_di's sensor frame.
    _, sph_coords_eval_in_hist, px_indices_eval_in_hist = \
        historical_di.project_point_to_pixel_indices(point_eval_global)

    if sph_coords_eval_in_hist is None or px_indices_eval_in_hist is None:
        # point_eval_global is outside the FoV of historical_di
        logger_oc.debug(f"DetailedCheck: point_eval_global projects outside historical_di (TS {historical_di.timestamp}).")
        return False
    
    phi_eval_in_hist, theta_eval_in_hist, d_eval_in_hist = sph_coords_eval_in_hist

    # 2. Get spherical coordinates of point_hist_cand_global.
    #    These should be pre-calculated and stored in historical_di.local_sph_coords_for_points.
    #    We need to find the original index of point_hist_cand_global within historical_di.
    #    This is a bit tricky if only global coords are passed.
    #    A robust way is if point_hist_cand_global was identified by its original_index in historical_di.
    #    For now, let's assume we can find it or we pass its original_index.
    #    Let's modify the signature or assume point_hist_cand_global is an index.
    #    
    #    Alternative: Project point_hist_cand_global also into historical_di. This ensures both are in
    #    the same spherical coordinate system relative to historical_di.
    _, sph_coords_hist_cand_in_hist, px_indices_hist_cand_in_hist = \
        historical_di.project_point_to_pixel_indices(point_hist_cand_global)

    if sph_coords_hist_cand_in_hist is None or px_indices_hist_cand_in_hist is None:
        logger_oc.debug(f"DetailedCheck: point_hist_cand_global projects outside historical_di (TS {historical_di.timestamp}). Should not happen if it's from this DI.")
        return False
        
    phi_hist_cand_in_hist, theta_hist_cand_in_hist, d_hist_cand_in_hist = sph_coords_hist_cand_in_hist

    # 3. Check angular proximity
    angular_phi_match = abs(phi_eval_in_hist - phi_hist_cand_in_hist) <= self.detailed_check_angular_threshold_h_rad
    angular_theta_match = abs(theta_eval_in_hist - theta_hist_cand_in_hist) <= self.detailed_check_angular_threshold_v_rad

    if not (angular_phi_match and angular_theta_match):
        logger_oc.debug(f"DetailedCheck: Angular mismatch. Eval_sph({phi_eval_in_hist:.3f},{theta_eval_in_hist:.3f}) vs "
                        f"HistCand_sph({phi_hist_cand_in_hist:.3f},{theta_hist_cand_in_hist:.3f}).")
        return False

    # 4. Check depth condition based on occlusion_type_to_check
    depth_condition_met = False
    if occlusion_type_to_check == "OCCLUDING": # point_eval occludes point_hist_cand
        depth_condition_met = d_eval_in_hist < d_hist_cand_in_hist - self.detailed_check_epsilon_depth
        logger_oc.debug(f"DetailedCheck (OCCLUDING): d_eval={d_eval_in_hist:.3f}, d_hist_cand={d_hist_cand_in_hist:.3f}, "
                        f"eps={self.detailed_check_epsilon_depth:.3f}. Condition met: {depth_condition_met}")
    elif occlusion_type_to_check == "OCCLUDED_BY": # point_eval is occluded by point_hist_cand
        depth_condition_met = d_eval_in_hist > d_hist_cand_in_hist + self.detailed_check_epsilon_depth
        logger_oc.debug(f"DetailedCheck (OCCLUDED_BY): d_eval={d_eval_in_hist:.3f}, d_hist_cand={d_hist_cand_in_hist:.3f}, "
                        f"eps={self.detailed_check_epsilon_depth:.3f}. Condition met: {depth_condition_met}")
    else:
        raise ValueError(f"Invalid occlusion_type_to_check: {occlusion_type_to_check}")

    return depth_condition_met