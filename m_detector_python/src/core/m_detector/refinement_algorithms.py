# src/core/m_detector/refinement_algorithms.py

from enum import Enum
import logging
import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay
from hdbscan import HDBSCAN as CPU_HDBSCAN
import torch
import numpy as np

from ..constants import OcclusionResult
from ..depth_image import DepthImage

logger = logging.getLogger(__name__)

class ClusteringAlgorithm(Enum):
    HDBSCAN = "hdbscan"


def _refine_with_hdbscan_convex_hull(
    labels: torch.Tensor,
    current_di: DepthImage,
    cluster_params: dict
) -> torch.Tensor:
    """
    Refines dynamic point labels using HDBSCAN clustering followed by convex hull expansion.

    This two-step process first identifies dense clusters of dynamic points using
    HDBSCAN, filtering out sparse noise. Then, for each valid cluster, it computes
    its 3D convex hull and re-labels all points (including previously undetermined
    ones) that fall within this hull as dynamic.

    CPU based HDBSCAN was used as libraries with GPU versions were difficult to install in the current environment. 
    Future developers might try refactoring this to GPU, or using a different clustering method,
      as the current approach failed to improve the results ayways. 

    Args:
        labels (torch.Tensor): The current point labels. Shape: (N,).
        current_di (DepthImage): The DepthImage object for the current sweep.
        cluster_params (dict): Configuration for the clustering algorithm.

    Returns:
        torch.Tensor: The refined labels tensor. Shape: (N,).
    """
    dynamic_mask = (labels == OcclusionResult.OCCLUDING_IMAGE.value)
    if not torch.any(dynamic_mask):
        return labels

    dynamic_points_global_torch = current_di.original_points_global_coords[dynamic_mask].contiguous()
    min_points_param = cluster_params['min_points']
    min_cluster_size = cluster_params['min_cluster_size']

    # Extract cluster_selection_epsilon 
    cluster_selection_epsilon = cluster_params['cluster_selection_epsilon']

    if dynamic_points_global_torch.shape[0] < min_points_param:
        logger.debug(f"Skipping HDBSCAN: Not enough points ({dynamic_points_global_torch.shape[0]}) "
                     f"to meet min_points of {min_points_param}.")
        return labels
    if not torch.all(torch.isfinite(dynamic_points_global_torch)):
        logger.warning("Detected non-finite values (NaN or inf) in points for clustering. Skipping refinement.")
        return labels

    # ---- CPU HDBSCAN ----
    # 1. Move data from GPU tensor to CPU numpy array
    dynamic_points_np = dynamic_points_global_torch.cpu().numpy()
    
    # 2. Instantiate and run the CPU clusterer. Note the parameter `min_samples`.
    clusterer = CPU_HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_points_param, # The CPU library uses 'min_samples'
        cluster_selection_epsilon=cluster_selection_epsilon,
        core_dist_n_jobs=-1 # Use all available CPU cores
    )
    cluster_labels_np = clusterer.fit_predict(dynamic_points_np)
    
    # 3. Convert the resulting numpy labels back to a torch tensor on the correct device.
    cluster_labels_torch = torch.from_numpy(cluster_labels_np).to(labels.device)
    # ---- End CPU HDBSCAN ----

    refined_labels = labels.clone() # Start with a copy of the original labels

    # Handle HDBSCAN noise points (-1 label) 
    # Identify the original indices of points that were initially dynamic candidates
    # but were classified as noise by HDBSCAN.
    original_indices_of_dynamic_candidates = torch.where(dynamic_mask)[0]
    
    # Create a mask for noise points within the `dynamic_points_global_torch` subset
    noise_indices_in_dynamic_subset = (cluster_labels_torch == -1)
    
    # Get the actual indices in the *original* `labels` tensor that correspond to these noise points
    indices_to_revert_to_undetermined = original_indices_of_dynamic_candidates[noise_indices_in_dynamic_subset]
    
    if len(indices_to_revert_to_undetermined) > 0:
        refined_labels[indices_to_revert_to_undetermined] = OcclusionResult.UNDETERMINED.value
        logger.debug(f"Reverted {len(indices_to_revert_to_undetermined)} points from OCCLUDING_IMAGE to UNDETERMINED (HDBSCAN noise).")

    unique_cluster_ids = np.unique(cluster_labels_np) # Use np.unique on numpy array for consistency
    
    num_clusters = len(unique_cluster_ids) - (1 if -1 in unique_cluster_ids else 0)
    if num_clusters > 0:
        logger.debug(f"Found {num_clusters} dynamic clusters using CPU HDBSCAN.")

    all_points_global_np = current_di.original_points_global_coords.cpu().numpy()

    for cluster_id in unique_cluster_ids:
        if cluster_id == -1:
            continue # Noise points already handled above

        # Get mask relative to the dynamic_points_global_torch (subset of original points)
        points_in_cluster_mask_in_dynamic_subset = (cluster_labels_torch == cluster_id)
        
        # Get the actual original indices corresponding to this cluster
        original_indices_for_cluster = original_indices_of_dynamic_candidates[points_in_cluster_mask_in_dynamic_subset]
        
        # Get the global coordinates for points in this specific cluster
        cluster_points_np = current_di.original_points_global_coords[original_indices_for_cluster].cpu().numpy()

        # A 3D convex hull requires at least 4 points (d+1).
        if len(cluster_points_np) < 4: # Convex hull requires at least 4 points for 3D
            logger.debug(f"Skipping Convex Hull for cluster {cluster_id}: not enough points ({len(cluster_points_np)}).")
            # Points in small clusters (that pass min_cluster_size but fail convex hull)
            # will retain their original labels or UNDETERMINED if they were noise.
            # They are NOT re-labeled as OCCLUDING_IMAGE.value.
            continue

        try:
            hull = ConvexHull(cluster_points_np)
            # The Delaunay triangulation should be on the vertices of the convex hull, not all points.
            delaunay_hull = Delaunay(hull.points[hull.vertices])
        except Exception as e:
            logger.debug(f"Could not form convex hull for cluster {cluster_id}: {e}")
            # If hull fails, these points should not be confirmed dynamic
            continue

        min_coords = np.min(cluster_points_np, axis=0)
        max_coords = np.max(cluster_points_np, axis=0)
        
        # Find points from the *entire original point cloud* that are within the bounding box
        candidate_mask_np = np.all((all_points_global_np >= min_coords) & (all_points_global_np <= max_coords), axis=1)
        candidate_indices_np = np.where(candidate_mask_np)[0]
        points_to_check_np = all_points_global_np[candidate_indices_np]

        if len(points_to_check_np) == 0:
            continue

        points_in_hull_mask = delaunay_hull.find_simplex(points_to_check_np) >= 0
        final_indices_to_update_np = candidate_indices_np[points_in_hull_mask]

        if len(final_indices_to_update_np) > 0:
            final_indices_torch = torch.from_numpy(final_indices_to_update_np).long().to(labels.device)
            # Re-label all points within the convex hull of the cluster as dynamic
            refined_labels[final_indices_torch] = OcclusionResult.OCCLUDING_IMAGE.value
            
    return refined_labels


def apply_clustering_and_refinement(
    labels: torch.Tensor,
    current_di: DepthImage,
    refinement_params: dict
) -> torch.Tensor:
    """
    Main entry point for the frame refinement stage, dispatching to the
    configured clustering algorithm.

    Args:
        labels (torch.Tensor): The labels tensor after all geometric tests.
        current_di (DepthImage): The current DepthImage object.
        refinement_params (dict): Configuration for the entire refinement stage,
                                  including the 'enabled' flag and algorithm choice.

    Returns:
        torch.Tensor: The final, refined labels tensor.
    """
    if not refinement_params['enabled']:
        return labels

    clustering_config = refinement_params['clustering']
    algo_name = clustering_config['algorithm']
    
    if not algo_name or algo_name == 'none':
        return labels

    logger.debug(f"Applying frame refinement using algorithm: {algo_name}")

    if algo_name == ClusteringAlgorithm.HDBSCAN.value:
        algo_params = clustering_config[algo_name]
        return _refine_with_hdbscan_convex_hull(labels, current_di, algo_params)
    else:
        logger.error(f"Unknown or unsupported clustering algorithm specified: {algo_name}")
        return labels