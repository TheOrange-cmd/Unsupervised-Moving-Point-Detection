# src/core/depth_image.py

import numpy as np
import torch
from typing import Tuple, Optional, Dict, List, Any
import logging
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter

from .constants import OcclusionResult
from ..utils.point_cloud_utils import filter_points

logger = logging.getLogger(__name__)

# A small constant to avoid division by zero or floating point issues.
FLOAT_PRECISION_GUARD = 1e-6

class DepthImage:
    """
    Represents a single LiDAR sweep as a spherical projection (depth image).

    This class is the core data structure for a single frame. It stores not only the
    raw point cloud data but also the 2D spherical projection, which includes
    pre-aggregated min/max depth and point counts per pixel. This pre-aggregation
    is key to the performance of the coarse occlusion checks.
    """
    def __init__(self,
                 image_pose_global: np.ndarray,
                 depth_image_params: Dict[str, Any],
                 timestamp: float,
                 device: torch.device):
        """
        Initializes the DepthImage for a single sweep.

        Args:
            image_pose_global (np.ndarray): The 4x4 pose matrix that transforms points
                                            from the sensor's frame to the global frame.
            depth_image_params (Dict[str, Any]): Configuration for the spherical projection.
            timestamp (float): The timestamp of the sweep in microseconds.
            device (torch.device): The PyTorch device (e.g., 'cuda:0' or 'cpu') for tensors.
        """
        self.device = device
        self.image_pose_global: torch.Tensor = torch.from_numpy(image_pose_global).float().to(self.device)
        self.timestamp: float = timestamp

        # --- Spherical Projection Parameters ---
        self.res_h_rad: float = np.deg2rad(depth_image_params['resolution_h_deg'])
        self.res_v_rad: float = np.deg2rad(depth_image_params['resolution_v_deg'])
        self.phi_min_rad: float = depth_image_params['phi_min_rad']
        self.phi_max_rad: float = depth_image_params['phi_max_rad']
        self.theta_min_rad: float = depth_image_params['theta_min_rad']
        self.theta_max_rad: float = depth_image_params['theta_max_rad']
        self.num_pixels_h: int = int(np.ceil((self.phi_max_rad - self.phi_min_rad) / self.res_h_rad))
        self.num_pixels_v: int = int(np.ceil((self.theta_max_rad - self.theta_min_rad) / self.res_v_rad))

        # --- Core Data Tensors ---
        # Stores the minimum depth of any point projecting to a pixel.
        self.pixel_min_depth: torch.Tensor = torch.full((self.num_pixels_v, self.num_pixels_h), float('inf'), dtype=torch.float32, device=self.device)
        # Stores the maximum depth of any point projecting to a pixel.
        self.pixel_max_depth: torch.Tensor = torch.full((self.num_pixels_v, self.num_pixels_h), float('-inf'), dtype=torch.float32, device=self.device)
        # Stores the number of points projecting to each pixel.
        self.pixel_count: torch.Tensor = torch.zeros((self.num_pixels_v, self.num_pixels_h), dtype=torch.int32, device=self.device)
        
        # --- Point Cloud Data (Populated by add_points_batch) ---
        self.num_points: int = 0
        self.original_points_global_coords: Optional[torch.Tensor] = None
        self.mdet_labels_for_points: Optional[torch.Tensor] = None
        self.original_indices_of_filtered_points: Optional[torch.Tensor] = None
        self.local_sph_coords_for_points: Optional[torch.Tensor] = None
        
        # --- Pre-computed Transformation ---
        self.matrix_local_from_global: torch.Tensor = torch.inverse(self.image_pose_global)

        # --- Pixel-to-Point-Index Mapping ---
        # A tensor storing original point indices, sorted by their flattened pixel location.
        self.pixel_original_indices_tensor: Optional[torch.Tensor] = None
        # A map where each entry [start, count] corresponds to a slice in the above tensor.
        self.pixel_map_tensor = torch.full((self.num_pixels_v * self.num_pixels_h, 2), -1, dtype=torch.long, device=self.device)

        # --- Caching for Static Point Mask ---
        self.static_points_mask: Optional[torch.Tensor] = None
        self._static_labels_used_for_cache: Optional[List[int]] = None

    def is_prepared_for_projection(self) -> bool:
        """Checks if the DI has been populated with points and is ready for use."""
        return self.num_points > 0 and self.local_sph_coords_for_points is not None

    def project_points_batch(self, points_global_batch: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects a batch of 3D points in global coordinates into this DI's local frame
        and calculates their spherical coordinates and pixel indices.

        Args:
            points_global_batch (torch.Tensor): Points to project, shape (N, 3).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - points_local (N, 3): Points in the DI's local sensor frame.
                - sph_coords (N, 3): Spherical coordinates (phi, theta, depth).
                - pixel_indices (N, 2): Pixel coordinates (v, h).
                - valid_mask (N,): Boolean mask indicating which points projected successfully.
        """
        batch_size = points_global_batch.shape[0]
        if not isinstance(points_global_batch, torch.Tensor):
            points_global_batch = torch.from_numpy(points_global_batch).float()
        points_global_batch = points_global_batch.to(self.device)

        # Homogeneous transformation to local frame
        points_global_h = torch.hstack([
            points_global_batch,
            torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
        ])
        points_local_h = points_global_h @ self.matrix_local_from_global.T
        points_local_all = points_local_h[:, :3]

        # Calculate spherical coordinates
        depths = torch.linalg.norm(points_local_all, dim=1)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        sph_coords_all = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
        pixel_indices_all = torch.zeros((batch_size, 2), dtype=torch.long, device=self.device)

        has_valid_depth_mask = depths > FLOAT_PRECISION_GUARD
        if not torch.any(has_valid_depth_mask):
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

        pl_valid_depth = points_local_all[has_valid_depth_mask]
        d_valid_depth = depths[has_valid_depth_mask]
        x, y, z = pl_valid_depth.T
        phi = torch.atan2(y, x)
        theta = torch.asin(torch.clip(z / d_valid_depth, -1.0, 1.0))

        # Filter points that are within the LiDAR's field of view
        in_fov_mask_subset = (
            (phi >= self.phi_min_rad - FLOAT_PRECISION_GUARD) & (phi <= self.phi_max_rad + FLOAT_PRECISION_GUARD) &
            (theta >= self.theta_min_rad - FLOAT_PRECISION_GUARD) & (theta <= self.theta_max_rad + FLOAT_PRECISION_GUARD)
        )
        original_indices_of_valid_points = torch.where(has_valid_depth_mask)[0][in_fov_mask_subset]
        if original_indices_of_valid_points.numel() == 0:
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

        valid_mask[original_indices_of_valid_points] = True
        sph_coords_all[original_indices_of_valid_points, 0] = phi[in_fov_mask_subset]
        sph_coords_all[original_indices_of_valid_points, 1] = theta[in_fov_mask_subset]
        sph_coords_all[original_indices_of_valid_points, 2] = d_valid_depth[in_fov_mask_subset]
        
        # Convert spherical coordinates to pixel indices
        phi_norm = phi[in_fov_mask_subset] - self.phi_min_rad
        theta_norm = theta[in_fov_mask_subset] - self.theta_min_rad
        h_idx = (phi_norm / self.res_h_rad).long()
        v_idx = (theta_norm / self.res_v_rad).long()
        h_idx.clamp_(0, self.num_pixels_h - 1)
        v_idx.clamp_(0, self.num_pixels_v - 1)
        pixel_indices_all[original_indices_of_valid_points, 0] = v_idx
        pixel_indices_all[original_indices_of_valid_points, 1] = h_idx
        
        return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

    def add_points_batch(self,
                        points_global_raw: np.ndarray,
                        points_sensor_raw: np.ndarray,
                        filter_params: Dict[str, Any]) -> int:
        """
        Populates the DepthImage with a point cloud from a single sweep.

        This method performs range filtering, projects the points to create the
        spherical depth image, and pre-aggregates min/max depths and counts per pixel.

        Args:
            points_global_raw (np.ndarray): Raw point cloud in global frame, shape (N, 3).
            points_sensor_raw (np.ndarray): Raw point cloud in sensor frame, shape (N, 3).
            filter_params (Dict[str, Any]): Parameters for point pre-filtering (e.g., min/max range).
            initial_labels_raw (Optional[np.ndarray]): Optional initial labels (e.g., from RANSAC), shape (N,).

        Returns:
            int: The number of points remaining after filtering.
        """
        if points_global_raw.shape[0] == 0:
            self.num_points = 0
            return 0

        # Filter out ego vehicle and extreme range points
        filter_mask = filter_points(points_sensor_raw, filter_params)
        
        self.original_indices_of_filtered_points = torch.from_numpy(np.where(filter_mask)[0]).int().to(self.device)
        
        points_global_filtered = points_global_raw[filter_mask]
        self.num_points = points_global_filtered.shape[0]

        if self.num_points == 0:
            return 0
            
        self.original_points_global_coords = torch.from_numpy(points_global_filtered).float().to(self.device)
        
        # Initialize labels to UNDETERMINED 
        self.mdet_labels_for_points = torch.full((self.num_points,), OcclusionResult.UNDETERMINED.value, dtype=torch.int8, device=self.device)
        
        # Project points and populate the spherical grid tensors
        _points_local, sph_coords, pixel_indices, valid_mask = \
            self.project_points_batch(self.original_points_global_coords)
        self.local_sph_coords_for_points = sph_coords
        
        valid_original_indices = torch.where(valid_mask)[0]
        if valid_original_indices.numel() > 0:
            valid_pixels = pixel_indices[valid_original_indices]
            valid_depths = sph_coords[valid_original_indices, 2]
            flat_pixel_indices = valid_pixels[:, 0] * self.num_pixels_h + valid_pixels[:, 1]

            # Use torch_scatter for efficient, parallel aggregation
            scatter_min(valid_depths, flat_pixel_indices, out=self.pixel_min_depth.view(-1))
            scatter_max(valid_depths, flat_pixel_indices, out=self.pixel_max_depth.view(-1))
            scatter_add(torch.ones_like(valid_depths, dtype=torch.int32), flat_pixel_indices, out=self.pixel_count.view(-1))
            
            # Build the pixel-to-point index map
            sorted_flat_pixels, sorted_indices = torch.sort(flat_pixel_indices)
            self.pixel_original_indices_tensor = valid_original_indices[sorted_indices]
            unique_pixels, counts = torch.unique_consecutive(sorted_flat_pixels, return_counts=True)
            start_indices = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(counts, dim=0)[:-1]])
            start_count_pairs = torch.stack([start_indices, counts], dim=1)
            scatter(start_count_pairs, unique_pixels.long(), dim=0, out=self.pixel_map_tensor)

        # Invalidate any cached static mask
        self.static_points_mask = None
        self._static_labels_used_for_cache = None
            
        return self.num_points
    
    def get_original_points_global(self) -> Optional[np.ndarray]:
        """Returns the filtered global points as a NumPy array."""
        if self.original_points_global_coords is None:
            return None
        return self.original_points_global_coords.cpu().numpy()

    def get_all_point_labels(self) -> Optional[np.ndarray]:
        """Returns the current point labels as a NumPy array."""
        if self.mdet_labels_for_points is None:
            return None
        return self.mdet_labels_for_points.cpu().numpy()
        
    def get_static_points_mask(self, static_config_values: List[int]) -> Optional[torch.Tensor]:
        """
        Returns a boolean mask indicating which points are considered static.
        Caches the result to avoid re-computation.

        Args:
            static_config_values (List[int]): A list of integer label values
                                              that are considered static.

        Returns:
            Optional[torch.Tensor]: A boolean tensor mask of shape (num_points,).
        """
        if self.mdet_labels_for_points is None:
            return None
        # Return cached mask if the defining labels haven't changed
        if self.static_points_mask is not None and self._static_labels_used_for_cache == static_config_values:
            return self.static_points_mask

        static_values_tensor = torch.tensor(static_config_values, device=self.device, dtype=self.mdet_labels_for_points.dtype)
        # Check if each point's label is in the list of static values
        self.static_points_mask = torch.any(self.mdet_labels_for_points.unsqueeze(1) == static_values_tensor, dim=1)
        self._static_labels_used_for_cache = static_config_values
        return self.static_points_mask

    def get_points_global_by_idx(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieves global coordinates for points at given indices."""
        return self.original_points_global_coords[indices]

    def get_depths_by_idx(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieves local depths for points at given indices."""
        return self.local_sph_coords_for_points[indices, 2]

    def unproject_pixels_batch(self, pixel_indices: torch.Tensor) -> torch.Tensor:
        """
        Unprojects a batch of 2D pixel indices back into 3D global coordinates.
        It uses the `pixel_max_depth` to reconstruct the 3D point.

        Args:
            pixel_indices (torch.Tensor): Pixel coordinates to unproject, shape (N, 2).

        Returns:
            torch.Tensor: The corresponding 3D points in global frame, shape (N, 3).
        """
        v_idx = pixel_indices[:, 0]
        h_idx = pixel_indices[:, 1]
        depths = self.pixel_max_depth[v_idx, h_idx]

        # Convert pixel indices back to spherical angles
        phi = (h_idx.float() + 0.5) * self.res_h_rad + self.phi_min_rad
        theta = (v_idx.float() + 0.5) * self.res_v_rad + self.theta_min_rad
        
        # Spherical to Cartesian conversion (local frame)
        cos_theta = torch.cos(theta)
        x_local = depths * cos_theta * torch.cos(phi)
        y_local = depths * cos_theta * torch.sin(phi)
        z_local = depths * torch.sin(theta)
        points_local = torch.stack([x_local, y_local, z_local], dim=1)
        
        # Transform from local to global frame
        points_local_h = torch.hstack([
            points_local,
            torch.ones((points_local.shape[0], 1), dtype=torch.float32, device=self.device)
        ])
        points_global_h = points_local_h @ self.image_pose_global.T
        return points_global_h[:, :3]