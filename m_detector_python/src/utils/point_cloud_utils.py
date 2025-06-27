# src/utils/point_cloud_utils.py
import numpy as np
from typing import Dict

def filter_points(points_sensor_frame: np.ndarray, filter_params: Dict) -> np.ndarray:
    """
    Applies filtering to a point cloud based on the provided parameters.

    This function serves as the single source of truth for point filtering,
    ensuring that both the label generation and the M-Detector process
    the exact same subset of points from a raw sweep.

    Args:
        points_sensor_frame (np.ndarray): The (N, 3+) point cloud in the sensor's coordinate frame.
                                          Only the first 3 columns (x, y, z) are used.
        filter_params (Dict): A dictionary of filtering parameters, typically from the config.
                              Expected keys: 'min_range_meters', 'max_range_meters'.

    Returns:
        np.ndarray: A boolean mask of shape (N,) where True indicates a point
                    that should be KEPT after filtering.
    """
    if not filter_params.get('enabled', True):
        return np.ones(points_sensor_frame.shape[0], dtype=bool)

    # TODO (Phase 3): This is where the logic for a rectangular ego-vehicle filter will be added.
    # We can add a 'type' key to filter_params (e.g., 'radial' or 'rectangular')
    # and switch between filtering methods here.

    # Current implementation: Simple radial distance filtering.
    min_range = filter_params['min_range_meters']
    max_range = filter_params['max_range_meters']
    
    ranges = np.linalg.norm(points_sensor_frame[:, :3], axis=1)
    
    keep_mask = (ranges >= min_range) & (ranges <= max_range)
    
    return keep_mask

def transform_points_numpy(points_n3: np.ndarray, transf_matrix_4x4: np.ndarray) -> np.ndarray:
    """
    Transforms (N,3) points using a 4x4 transformation matrix.

    Args:
        points_n3 (np.ndarray): Array of N points, each with 3 coordinates.
        transf_matrix_4x4 (np.ndarray): 4x4 transformation matrix.

    Returns:
        np.ndarray: Transformed points as an (N,3) array.
    """
    if points_n3.ndim == 1: # Single point
        points_n3 = points_n3.reshape(1, -1)
    if points_n3.shape[1] != 3:
        raise ValueError("Input points must be of shape (N,3) or (3,)")
        
    points_n4_homogeneous = np.hstack((points_n3, np.ones((points_n3.shape[0], 1)))) 
    points_transformed_n4 = points_n4_homogeneous @ transf_matrix_4x4.T # (N,4)
    return points_transformed_n4[:, :3] # Back to (N,3)