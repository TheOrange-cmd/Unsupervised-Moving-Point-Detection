# src/core/depth_image.py

import numpy as np
import torch
from typing import Tuple, Optional, Dict, List, Any
import collections
import logging
from .constants import OcclusionResult
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter

from ..utils.transformations import transform_points_numpy


logger = logging.getLogger(__name__)

class DepthImage:
    def __init__(self,
                 image_pose_global: np.ndarray,
                 depth_image_params: Dict[str, Any],
                 timestamp: float,
                 device: torch.device):
        self.device = device
        self.image_pose_global: torch.Tensor = torch.from_numpy(image_pose_global).float().to(self.device)
        self.timestamp: float = timestamp
        self.res_h_rad: float = np.deg2rad(depth_image_params['resolution_h_deg'])
        self.res_v_rad: float = np.deg2rad(depth_image_params['resolution_v_deg'])
        self.phi_min_rad: float = depth_image_params['phi_min_rad']
        self.phi_max_rad: float = depth_image_params['phi_max_rad']
        self.theta_min_rad: float = depth_image_params['theta_min_rad']
        self.theta_max_rad: float = depth_image_params['theta_max_rad']
        self.num_pixels_h: int = int(np.ceil((self.phi_max_rad - self.phi_min_rad) / self.res_h_rad))
        self.num_pixels_v: int = int(np.ceil((self.theta_max_rad - self.theta_min_rad) / self.res_v_rad))
        self.pixel_min_depth: torch.Tensor = torch.full((self.num_pixels_v, self.num_pixels_h), float('inf'), dtype=torch.float32, device=self.device)
        self.pixel_max_depth: torch.Tensor = torch.full((self.num_pixels_v, self.num_pixels_h), float('-inf'), dtype=torch.float32, device=self.device)
        self.pixel_count: torch.Tensor = torch.zeros((self.num_pixels_v, self.num_pixels_h), dtype=torch.int32, device=self.device)
        
        self.original_points_global_coords: Optional[torch.Tensor] = None
        self.mdet_labels_for_points: Optional[torch.Tensor] = None

        self.original_indices_of_filtered_points: Optional[torch.Tensor] = None
        self.local_sph_coords_for_points: Optional[torch.Tensor] = None
        self.raw_occlusion_results_vs_history: Optional[torch.Tensor] = None
        self.matrix_local_from_global: torch.Tensor = torch.inverse(self.image_pose_global)

        self.pixel_original_indices_tensor: Optional[torch.Tensor] = None
        self.pixel_map_tensor = torch.full((self.num_pixels_v * self.num_pixels_h, 2), -1, dtype=torch.long, device=self.device)

        self.static_points_mask: Optional[torch.Tensor] = None
        self._static_labels_for_mask_generation: Optional[List[int]] = None

    def is_prepared_for_projection(self) -> bool:
        return (self.original_points_global_coords is not None and
                self.local_sph_coords_for_points is not None)

    def project_points_batch(self, points_global_batch: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = points_global_batch.shape[0]
        if not isinstance(points_global_batch, torch.Tensor):
            points_global_batch = torch.from_numpy(points_global_batch).float()
        points_global_batch = points_global_batch.to(self.device)
        if points_global_batch.shape[1] == 3:
            points_global_h = torch.hstack([
                points_global_batch,
                torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
            ])
        else:
            points_global_h = points_global_batch
        points_local_h = points_global_h @ self.matrix_local_from_global.T
        points_local_all = points_local_h[:, :3]
        depths = torch.linalg.norm(points_local_all, dim=1)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        sph_coords_all = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
        pixel_indices_all = torch.zeros((batch_size, 2), dtype=torch.long, device=self.device)
        has_valid_depth_mask = depths > 1e-6
        if not torch.any(has_valid_depth_mask):
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask
        pl_valid_depth = points_local_all[has_valid_depth_mask]
        d_valid_depth = depths[has_valid_depth_mask]
        x, y, z = pl_valid_depth.T
        phi = torch.atan2(y, x)
        theta = torch.asin(torch.clip(z / d_valid_depth, -1.0, 1.0))
        epsilon_fov = 1e-6
        in_fov_mask_subset = (
            (phi >= self.phi_min_rad - epsilon_fov) & (phi <= self.phi_max_rad + epsilon_fov) &
            (theta >= self.theta_min_rad - epsilon_fov) & (theta <= self.theta_max_rad + epsilon_fov)
        )
        original_indices_of_valid_points = torch.where(has_valid_depth_mask)[0][in_fov_mask_subset]
        if original_indices_of_valid_points.numel() == 0:
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask
        valid_mask[original_indices_of_valid_points] = True
        sph_coords_all[original_indices_of_valid_points, 0] = phi[in_fov_mask_subset]
        sph_coords_all[original_indices_of_valid_points, 1] = theta[in_fov_mask_subset]
        sph_coords_all[original_indices_of_valid_points, 2] = d_valid_depth[in_fov_mask_subset]
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
                         filter_params: Dict[str, Any],
                         initial_labels_raw: Optional[np.ndarray] = None) -> int: # Renamed for clarity
        
        raw_batch_size = points_global_raw.shape[0]
        if raw_batch_size == 0:
            self.total_points_added_to_di_arrays = 0
            return 0

        ranges = np.linalg.norm(points_sensor_raw, axis=1)
        filter_mask = (ranges >= filter_params['min_range_meters']) & (ranges <= filter_params['max_range_meters'])
        
        self.original_indices_of_filtered_points = torch.from_numpy(np.where(filter_mask)[0]).int().to(self.device)
        
        points_global_filtered = points_global_raw[filter_mask]
        filtered_batch_size = points_global_filtered.shape[0]
        self.total_points_added_to_di_arrays = filtered_batch_size

        if filtered_batch_size == 0:
            return 0
            
        self.original_points_global_coords = torch.from_numpy(points_global_filtered).float().to(self.device)
        
        # This logic is now correct because initial_labels_raw is a proper integer array
        if initial_labels_raw is not None:
            initial_labels_filtered = initial_labels_raw[filter_mask]
            self.mdet_labels_for_points = torch.from_numpy(initial_labels_filtered).to(self.device)
        else:
            self.mdet_labels_for_points = torch.full((filtered_batch_size,), OcclusionResult.UNDETERMINED.value, dtype=torch.int8, device=self.device)

        # --- The rest of the function is unchanged ---
        _points_local, sph_coords, pixel_indices, valid_mask = \
            self.project_points_batch(self.original_points_global_coords)
        self.local_sph_coords_for_points = sph_coords
        
        valid_original_indices = torch.where(valid_mask)[0]
        if valid_original_indices.numel() > 0:
            valid_pixels = pixel_indices[valid_original_indices]
            valid_depths = sph_coords[valid_original_indices, 2]
            flat_pixel_indices = valid_pixels[:, 0] * self.num_pixels_h + valid_pixels[:, 1]
            pixel_min_depth_flat = self.pixel_min_depth.view(-1)
            pixel_max_depth_flat = self.pixel_max_depth.view(-1)
            pixel_count_flat = self.pixel_count.view(-1)
            scatter_min(valid_depths, flat_pixel_indices, out=pixel_min_depth_flat)
            scatter_max(valid_depths, flat_pixel_indices, out=pixel_max_depth_flat)
            scatter_add(torch.ones_like(valid_depths, dtype=torch.int32), flat_pixel_indices, out=pixel_count_flat)
            sorted_flat_pixels, sorted_indices = torch.sort(flat_pixel_indices)
            self.pixel_original_indices_tensor = valid_original_indices[sorted_indices]
            unique_pixels, counts = torch.unique_consecutive(sorted_flat_pixels, return_counts=True)
            start_indices = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(counts, dim=0)[:-1]])
            start_count_pairs = torch.stack([start_indices, counts], dim=1)
            scatter(start_count_pairs, unique_pixels.long(), dim=0, out=self.pixel_map_tensor)

        self.static_points_mask = None
        self._static_labels_used_for_cache = None
            
        return filtered_batch_size
    
    def get_original_points_global(self) -> Optional[np.ndarray]:
        if self.original_points_global_coords is None:
            return None
        return self.original_points_global_coords.cpu().numpy()

    def get_all_point_labels(self) -> Optional[np.ndarray]:
        if self.mdet_labels_for_points is None:
            return None
        return self.mdet_labels_for_points.cpu().numpy()
        
    def get_static_points_mask(self, static_config_values: List[int]) -> Optional[torch.Tensor]:
        if self.mdet_labels_for_points is None:
            return None
        if self.static_points_mask is not None and self._static_labels_used_for_cache == static_config_values:
            return self.static_points_mask
        static_values_tensor = torch.tensor(static_config_values, device=self.device, dtype=self.mdet_labels_for_points.dtype)
        self.static_points_mask = torch.any(self.mdet_labels_for_points.unsqueeze(1) == static_values_tensor, dim=1)
        self._static_labels_used_for_cache = static_config_values
        return self.static_points_mask

    def get_points_global_by_idx(self, indices: torch.Tensor) -> torch.Tensor:
        return self.original_points_global_coords[indices]

    def get_depths_by_idx(self, indices: torch.Tensor) -> torch.Tensor:
        return self.local_sph_coords_for_points[indices, 2]

    def unproject_pixels_batch(self, pixel_indices: torch.Tensor) -> torch.Tensor:
        v_idx = pixel_indices[:, 0]
        h_idx = pixel_indices[:, 1]
        depths = self.pixel_max_depth[v_idx, h_idx]
        phi = (h_idx.float() + 0.5) * self.res_h_rad + self.phi_min_rad
        theta = (v_idx.float() + 0.5) * self.res_v_rad + self.theta_min_rad
        cos_theta = torch.cos(theta)
        x_local = depths * cos_theta * torch.cos(phi)
        y_local = depths * cos_theta * torch.sin(phi)
        z_local = depths * torch.sin(theta)
        points_local = torch.stack([x_local, y_local, z_local], dim=1)
        points_local_h = torch.hstack([
            points_local,
            torch.ones((points_local.shape[0], 1), dtype=torch.float32, device=self.device)
        ])
        points_global_h = points_local_h @ self.image_pose_global.T
        return points_global_h[:, :3]