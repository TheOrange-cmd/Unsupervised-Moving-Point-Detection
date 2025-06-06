# src/core/m_detector/interpolation_utils.py

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from itertools import combinations
from src.core.depth_image_legacy import DepthImage 
from src.core.constants import OcclusionResult 
import logging

logger_interp = logging.getLogger(__name__)

def interpolate_surface_depth_at_angle(
    target_phi_rad: float,
    target_theta_rad: float,
    historical_di: Any, # Use Any for now to avoid circular import with DepthImage if in same module
    static_labels_for_map_check_values: List[int],
    epsilon_phi_rad_search: float,
    epsilon_theta_rad_search: float,
    max_neighbors_to_consider: int,
    max_triplets_to_try: int
) -> Optional[float]:
    """
    Interpolates the depth of a static surface in a historical_di at a target angular location.

    Args:
        target_phi_rad: Target azimuthal angle (in historical_di's frame).
        target_theta_rad: Target polar angle (in historical_di's frame).
        historical_di: The historical DepthImage object to interpolate within.
        static_labels_for_map_check_values: List of integer enum values considered static.
        epsilon_phi_rad_search: Angular search window (horizontal) for neighbors.
        epsilon_theta_rad_search: Angular search window (vertical) for neighbors.
        max_neighbors_to_consider: Max static neighbors to use for forming triplets.
        max_triplets_to_try: Max triplet combinations to attempt solving.

    Returns:
        Optional[float]: Interpolated depth, or None if interpolation fails.
    """
    if logger_interp.isEnabledFor(logging.DEBUG):
        logger_interp.debug(f"MCC_Interp: Attempting interpolation at phi={target_phi_rad:.3f}, theta={target_theta_rad:.3f} in DI TS={historical_di.timestamp:.2f}")

    if not historical_di.is_prepared_for_projection() or \
       historical_di.local_sph_coords_for_points is None or \
       historical_di.mdet_labels_for_points is None:
        if logger_interp.isEnabledFor(logging.DEBUG):
            logger_interp.debug("MCC_Interp: Historical DI not fully prepared (missing sph_coords or labels). Cannot interpolate.")
        return None

    all_sph_coords_hist = historical_di.local_sph_coords_for_points
    all_labels_hist = historical_di.mdet_labels_for_points

    # 1. Find candidate static neighboring points
    # Ensure points are within the DI's valid range for spherical coordinates
    # This is implicitly handled if local_sph_coords_for_points only contains valid projections
    angular_mask_phi = np.abs(all_sph_coords_hist[:, 0] - target_phi_rad) <= epsilon_phi_rad_search
    angular_mask_theta = np.abs(all_sph_coords_hist[:, 1] - target_theta_rad) <= epsilon_theta_rad_search
    angular_mask = angular_mask_phi & angular_mask_theta
    
    static_mask = historical_di.get_static_points_mask(static_labels_for_map_check_values)

    if static_mask is None: # Check if the mask could be retrieved/generated
        if logger_interp.isEnabledFor(logging.DEBUG): # Check logging level
            logger_interp.debug(f"MCC_Interp: Could not get static_mask for DI TS={historical_di.timestamp:.2f}. Cannot interpolate.")
        return None
    
    valid_neighbor_indices = np.where(angular_mask & static_mask)[0]
    if logger_interp.isEnabledFor(logging.DEBUG):
        logger_interp.debug(f"MCC_Interp: Target phi={target_phi_rad:.3f}, theta={target_theta_rad:.3f}")
        logger_interp.debug(f"MCC_Interp: Epsilon_phi_search={epsilon_phi_rad_search:.3f}, Epsilon_theta_search={epsilon_theta_rad_search:.3f}")
    # Log a few of the all_sph_coords_hist and all_labels_hist to see what's there
    # For example, points around the target pixel in the historical DI

    if logger_interp.isEnabledFor(logging.DEBUG):
        # What are the labels of points that ARE in the angular mask?
        indices_in_angular_window = np.where(angular_mask)[0]
        if len(indices_in_angular_window) > 0:
            labels_in_angular_window = all_labels_hist[indices_in_angular_window]
            unique_labels_in_window, counts_in_window = np.unique(labels_in_angular_window, return_counts=True)
            logger_interp.debug(f"MCC_Interp: Points within angular window ({len(indices_in_angular_window)} total): Unique labels and counts: {dict(zip(unique_labels_in_window, counts_in_window))}")
            logger_interp.debug(f"MCC_Interp: Static labels we are looking for (values): {static_labels_for_map_check_values}")
        else:
            logger_interp.debug(f"MCC_Interp: NO points found within the angular search window at all.")

        logger_interp.debug(f"MCC_Interp: Number of points in angular neighborhood: {np.sum(angular_mask)}")
        logger_interp.debug(f"MCC_Interp: Number of static points in entire Depth Image: {np.sum(static_mask)}")
        logger_interp.debug(f"MCC_Interp: Number of valid_neighbor_indices (angular AND static): {len(valid_neighbor_indices)}")

    if len(valid_neighbor_indices) < 3:
        if logger_interp.isEnabledFor(logging.DEBUG):
            logger_interp.debug(f"MCC_Interp: Not enough static neighbors ({len(valid_neighbor_indices)}) within angular window. Need at least 3.")
        return None

    # Collect data for these neighbors: (phi, theta, depth, original_index)
    neighbor_data_tuples = [] 
    for idx in valid_neighbor_indices:
        neighbor_data_tuples.append(
            (all_sph_coords_hist[idx, 0], all_sph_coords_hist[idx, 1], all_sph_coords_hist[idx, 2], idx)
        )
    
    # Sort neighbors by angular distance to target (sum of absolute differences)
    neighbor_data_tuples.sort(key=lambda p_data: abs(target_phi_rad - p_data[0]) + abs(target_theta_rad - p_data[1]))

    if len(neighbor_data_tuples) > max_neighbors_to_consider:
        neighbor_data_tuples = neighbor_data_tuples[:max_neighbors_to_consider]
        if logger_interp.isEnabledFor(logging.DEBUG):
            logger_interp.debug(f"MCC_Interp: Limited neighbors to consider to {max_neighbors_to_consider}.")
    
    if len(neighbor_data_tuples) < 3:
        if logger_interp.isEnabledFor(logging.DEBUG):
            logger_interp.debug(f"MCC_Interp: Not enough static neighbors after limiting ({len(neighbor_data_tuples)}). Need at least 3.")
        return None

    # 2. Iterate through triplets to find one satisfying convex hull condition
    triplet_count = 0
    if logger_interp.isEnabledFor(logging.DEBUG):
        logger_interp.debug(f"MCC_Interp: Found {len(neighbor_data_tuples)} candidate static neighbors. Will try up to {max_triplets_to_try} triplets.")

    for p1_data, p2_data, p3_data in combinations(neighbor_data_tuples, 3):
        if triplet_count >= max_triplets_to_try:
            if logger_interp.isEnabledFor(logging.DEBUG):
                logger_interp.debug(f"MCC_Interp: Reached max triplets to try ({max_triplets_to_try}). Stopping triplet search.")
            break
        triplet_count += 1

        # p_data is (phi, theta, depth, original_idx)
        A_matrix = np.array([
            [p1_data[0], p2_data[0], p3_data[0]], # phis
            [p1_data[1], p2_data[1], p3_data[1]], # thetas
            [1,          1,          1         ]
        ], dtype=np.float64) # Use float64 for more stable solve
        b_vector = np.array([target_phi_rad, target_theta_rad, 1], dtype=np.float64)

        try:
            weights = np.linalg.solve(A_matrix, b_vector)
        except np.linalg.LinAlgError:
            # logger_interp.debug(f"MCC_Interp: Singular matrix for triplet (indices: {p1_data[3]},{p2_data[3]},{p3_data[3]}), skipping.")
            continue 

        # Check convex hull condition (w_i >= 0, allowing for small float inaccuracies)
        if np.all(weights >= -1e-9): # Allow very small negative due to float precision
            interpolated_depth = weights[0] * p1_data[2] + \
                                 weights[1] * p2_data[2] + \
                                 weights[2] * p3_data[2]
            if logger_interp.isEnabledFor(logging.DEBUG):
                logger_interp.debug(f"MCC_Interp: Success! Triplet (orig_indices: {p1_data[3]},{p2_data[3]},{p3_data[3]}), "
                                    f"Weights: {np.round(weights,3)}, Interp_Depth: {interpolated_depth:.3f}")
            return float(interpolated_depth)
        # else:
            # logger_interp.debug(f"MCC_Interp: Triplet (orig_indices: {p1_data[3]},{p2_data[3]},{p3_data[3]}) "
            #                     f"weights {np.round(weights,3)} not all >= 0 (convex hull fail).")
    if logger_interp.isEnabledFor(logging.DEBUG):
        logger_interp.debug("MCC_Interp: Failed to find a valid triplet for interpolation after trying allowed combinations.")
    return None
