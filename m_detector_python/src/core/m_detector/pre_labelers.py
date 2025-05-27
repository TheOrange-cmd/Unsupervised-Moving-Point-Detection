# src/core/m_detector/pre_labelers.py (New File)
import numpy as np
import torch
from typing import Dict, Optional, List, Callable, Tuple

from ..constants import OcclusionResult

# --- RANSAC Ground Detection (Adapted from your example) ---
@torch.no_grad()
def ransac_ground_prelabeler(
    points_global: np.ndarray, # Nx3 global coordinates
    points_lidar_frame: np.ndarray, # Nx3 lidar frame coordinates (for RANSAC)
    current_di_timestamp: float, # For context, if needed
    ransac_params: Optional[Dict] = None,
    device_str: str = 'auto' # 'cpu', 'cuda', or 'auto'
) -> np.ndarray:
    """
    Identifies ground points using RANSAC flat plane fitting.

    Args:
        points_global (np.ndarray): Global point cloud (Nx3).
        points_lidar_frame (np.ndarray): LiDAR point cloud in LiDAR frame (Nx3).
                                         RANSAC is typically done in sensor frame.
        current_di_timestamp (float): Timestamp of the current DI.
        ransac_params (dict): Hyperparameters for ground removal.
        device_str (str): 'cpu', 'cuda', or 'auto'.

    Returns:
        np.ndarray: Boolean mask (N,) where True indicates a ground point.
    """
    if points_lidar_frame.shape[0] == 0:
        return np.array([], dtype=bool)

    if device_str == 'auto':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(device_str)

    # Convert Nx3 to Nx4 for the RANSAC function
    pc_lidar_torch = torch.from_numpy(points_lidar_frame).float().to(device)
    ones_col = torch.ones((pc_lidar_torch.shape[0], 1), device=device, dtype=pc_lidar_torch.dtype)
    pc_lidar_torch_h = torch.cat([pc_lidar_torch, ones_col], dim=1)


    # Set default hyperparameters (move these to config eventually)
    default_params = {
        'xyradius_threshold': 20.0, # Example: Consider points within 20m radius for initial plane fit
        'z_min_threshold': -2.5,    # Example: Points below sensor (e.g. -2.0m for car-mounted lidar)
        'z_max_threshold': -0.5,    # Example: Points not too high above sensor
        'num_trials': 50,           # Increased for potentially more robustness
        'inlier_threshold': 0.20,   # Stricter for "conservative"
        'ground_threshold': 0.20,   # Stricter for "conservative"
    }
    current_ransac_params = ransac_params if ransac_params is not None else default_params

    # print(f"Using params for RANSAC: {current_ransac_params}")

    # --- Call your RANSAC function (adapted to take Nx4) ---
    # Assuming your ransac_flatplane is modified or wrapped to accept Nx4
    # For now, directly embedding the logic:
    assert pc_lidar_torch_h.ndim == 2 and pc_lidar_torch_h.shape[1] == 4, \
        f'pc_lidar must have shape (N,4), got {tuple(pc_lidar_torch_h.shape)}'
    
    R = current_ransac_params['xyradius_threshold']
    zmin = current_ransac_params['z_min_threshold']
    zmax = current_ransac_params['z_max_threshold']

    # Filter for ground disk in lidar frame
    bool_xyz_disk = ((pc_lidar_torch_h[:, 0]**2 + pc_lidar_torch_h[:, 1]**2 <= R**2) &
                     (pc_lidar_torch_h[:, 2] >= zmin) &
                     (pc_lidar_torch_h[:, 2] <= zmax))
    
    grounddisk_pc_lidar = pc_lidar_torch_h[bool_xyz_disk]

    if grounddisk_pc_lidar.shape[0] < 3: # Not enough points for RANSAC
        return np.zeros(points_lidar_frame.shape[0], dtype=bool)

    T = current_ransac_params['num_trials']
    # Ensure enough unique points for sampling if grounddisk is small
    num_points_in_disk = grounddisk_pc_lidar.shape[0]
    ids = torch.randint(0, num_points_in_disk, (T, 3), device=device)
    
    triplets = grounddisk_pc_lidar[ids,:3] # Use x,y,z for plane fitting
    v1, v2 = triplets[:,1]-triplets[:,0], triplets[:,2]-triplets[:,0]
    normals = torch.cross(v1, v2)
    norm_magnitudes = normals.norm(dim=1, keepdim=True)
    
    # Avoid division by zero for collinear points
    valid_normals_mask = norm_magnitudes.squeeze() > 1e-6
    if not torch.any(valid_normals_mask):
        return np.zeros(points_lidar_frame.shape[0], dtype=bool) # No valid planes found

    normals = normals[valid_normals_mask]
    norm_magnitudes = norm_magnitudes[valid_normals_mask]
    triplets = triplets[valid_normals_mask] # Filter triplets accordingly

    normals = normals / norm_magnitudes.clamp(min=1e-6) # Normalize valid normals
    ds = -(normals * triplets[:,0]).sum(dim=1, keepdim=True)
    plane_parameters = torch.cat([normals, ds], dim=1)
    
    # Compute residuals against all points in the disk (using x,y,z only for distance)
    res = plane_parameters.matmul(torch.cat([grounddisk_pc_lidar[:,:3], torch.ones((num_points_in_disk,1), device=device)], dim=1).T)
    inliers = (res.abs() <= current_ransac_params['inlier_threshold']).sum(dim=1)
    
    if inliers.shape[0] == 0: # No valid planes had inliers (e.g. if all normals were zero)
         return np.zeros(points_lidar_frame.shape[0], dtype=bool)

    best = inliers.argmax()
    best_plane = plane_parameters[best]
    
    # Enforce upward normal (normal vector's Z component should be positive in sensor frame)
    if best_plane[2] < 0:
        best_plane = -best_plane
        
    # Get boolean mask for ALL original points (pc_lidar_torch_h)
    distances = best_plane.matmul(pc_lidar_torch_h.T) # Distances for all points
    bool_ground_torch = distances.abs() <= current_ransac_params['ground_threshold'] # Use abs for distance to plane
    
    return bool_ground_torch.cpu().numpy()


# Type for a pre-labeler function
PreLabelerCallable = Callable[[np.ndarray, np.ndarray, float, Optional[Dict]], np.ndarray]

# List of active pre-labelers (can be extended)
ACTIVE_PRE_LABELERS: List[Tuple[str, PreLabelerCallable, Optional[Dict]]] = [
    ("ransac_ground", ransac_ground_prelabeler, None), # Params can be loaded from main config
    # Add other pre-labelers here, e.g., ("segmentation_based_static", some_other_func, params_for_it)
]