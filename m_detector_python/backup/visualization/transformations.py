# src/utils/transformations.py
import numpy as np

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