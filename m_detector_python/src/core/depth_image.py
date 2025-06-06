# src/core/depth_image.py (Refactored)

import numpy as np
import torch
from typing import Tuple, Optional, Dict, List, Any
import collections
import logging
from .constants import OcclusionResult
from torch_scatter import scatter_min, scatter_max, scatter_add


logger = logging.getLogger(__name__)

class DepthImage:
    def __init__(self,
                 image_pose_global: np.ndarray,
                 depth_image_params: Dict[str, Any],
                 timestamp: float,
                 device: torch.device):  # NEW: Pass the target device
        """
        Initializes a tensor-based DepthImage.
        """
        self.device = device
        self.image_pose_global: torch.Tensor = torch.from_numpy(image_pose_global).float().to(self.device) # CHANGED
        self.timestamp: float = timestamp

        # --- Configuration Parameters  ---
        self.res_h_rad: float = np.deg2rad(depth_image_params['resolution_h_deg'])
        self.res_v_rad: float = np.deg2rad(depth_image_params['resolution_v_deg'])
        self.phi_min_rad: float = depth_image_params['phi_min_rad']
        self.phi_min_rad: float = depth_image_params['phi_min_rad']
        self.phi_max_rad: float = depth_image_params['phi_max_rad']
        self.theta_min_rad: float = depth_image_params['theta_min_rad']
        self.theta_max_rad: float = depth_image_params['theta_max_rad']
        self.num_pixels_h: int = int(np.ceil((self.phi_max_rad - self.phi_min_rad) / self.res_h_rad))
        self.num_pixels_v: int = int(np.ceil((self.theta_max_rad - self.theta_min_rad) / self.res_v_rad))

        # --- Core Data Attributes (now PyTorch Tensors) ---
        # Pixel-level statistics
        self.pixel_min_depth: torch.Tensor = torch.full((self.num_pixels_v, self.num_pixels_h), float('inf'), dtype=torch.float32, device=self.device)
        self.pixel_max_depth: torch.Tensor = torch.full((self.num_pixels_v, self.num_pixels_h), float('-inf'), dtype=torch.float32, device=self.device)
        self.pixel_count: torch.Tensor = torch.zeros((self.num_pixels_v, self.num_pixels_h), dtype=torch.int32, device=self.device)

        # Per-point data arrays
        self.original_points_global_coords: Optional[torch.Tensor] = None
        self.mdet_labels_for_points: Optional[torch.Tensor] = None
        self.mdet_scores_for_points: Optional[torch.Tensor] = None
        self.local_sph_coords_for_points: Optional[torch.Tensor] = None
        self.raw_occlusion_results_vs_history: Optional[torch.Tensor] = None

        # --- Helper Attributes ---
        self.matrix_local_from_global: torch.Tensor = torch.inverse(self.image_pose_global)
        self.total_points_added_to_di_arrays: int = 0
        
        # This remains a CPU-based structure as it's complex to tensorize efficiently
        self.pixel_original_indices: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)

        # Cache for static points mask (will also become a tensor)
        self.static_points_mask: Optional[torch.Tensor] = None
        self._static_labels_for_mask_generation: Optional[List[int]] = None

# In class DepthImage:

    def project_points_batch(self, points_global_batch: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects a batch of global points to pixel coordinates using PyTorch.
        
        Args:
            points_global_batch (torch.Tensor): (N, 3) or (N, 4) tensor of points on self.device.

        Returns:
            Tuple of tensors (all on self.device):
            - points_local_all (N, 3)
            - sph_coords_all (N, 3) -> (phi, theta, depth)
            - pixel_indices_all (N, 2) -> (v_idx, h_idx)
            - valid_mask (N,) -> boolean mask
        """
        batch_size = points_global_batch.shape[0]

        # Ensure input is a tensor on the correct device
        if not isinstance(points_global_batch, torch.Tensor):
            points_global_batch = torch.from_numpy(points_global_batch).float()
        points_global_batch = points_global_batch.to(self.device)

        if points_global_batch.shape[1] == 3:
            points_global_h = torch.hstack([
                points_global_batch,
                torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)
            ])
        elif points_global_batch.shape[1] == 4:
            points_global_h = points_global_batch
        else:
            raise ValueError("points_global_batch must be Nx3 or Nx4.")

        # --- Core PyTorch Operations ---
        points_local_h = points_global_h @ self.matrix_local_from_global.T
        points_local_all = points_local_h[:, :3]

        depths = torch.linalg.norm(points_local_all, dim=1)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Initialize output tensors
        sph_coords_all = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)
        pixel_indices_all = torch.zeros((batch_size, 2), dtype=torch.long, device=self.device)

        has_valid_depth_mask = depths > 1e-6
        if not torch.any(has_valid_depth_mask):
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

        # Process only points with valid depth
        pl_valid_depth = points_local_all[has_valid_depth_mask]
        d_valid_depth = depths[has_valid_depth_mask]
        x, y, z = pl_valid_depth.T

        phi = torch.atan2(y, x)
        theta = torch.asin(torch.clip(z / d_valid_depth, -1.0, 1.0))

        # FOV check
        epsilon_fov = 1e-6
        in_fov_mask_subset = (
            (phi >= self.phi_min_rad - epsilon_fov) & (phi <= self.phi_max_rad + epsilon_fov) &
            (theta >= self.theta_min_rad - epsilon_fov) & (theta <= self.theta_max_rad + epsilon_fov)
        )

        # Get original indices of points that are valid in all steps
        original_indices_of_valid_points = torch.where(has_valid_depth_mask)[0][in_fov_mask_subset]
        
        if original_indices_of_valid_points.numel() == 0:
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask
        
        valid_mask[original_indices_of_valid_points] = True
        
        # Populate spherical coordinates for valid points
        sph_coords_all[original_indices_of_valid_points, 0] = phi[in_fov_mask_subset]
        sph_coords_all[original_indices_of_valid_points, 1] = theta[in_fov_mask_subset]
        sph_coords_all[original_indices_of_valid_points, 2] = d_valid_depth[in_fov_mask_subset]

        # Calculate pixel indices
        phi_norm = phi[in_fov_mask_subset] - self.phi_min_rad
        theta_norm = theta[in_fov_mask_subset] - self.theta_min_rad
        
        h_idx = (phi_norm / self.res_h_rad).long()
        v_idx = (theta_norm / self.res_v_rad).long()

        # Clip indices to be within bounds
        h_idx.clamp_(0, self.num_pixels_h - 1)
        v_idx.clamp_(0, self.num_pixels_v - 1)
        
        pixel_indices_all[original_indices_of_valid_points, 0] = v_idx
        pixel_indices_all[original_indices_of_valid_points, 1] = h_idx

        return points_local_all, sph_coords_all, pixel_indices_all, valid_mask
    


    def add_points_batch(self,
                         points_global_batch: np.ndarray,
                         initial_labels_for_points: Optional[np.ndarray] = None) -> int:
        
        batch_size = points_global_batch.shape[0]
        if batch_size == 0:
            # Initialize empty tensors
            self.original_points_global_coords = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            # ... initialize other tensors as empty ...
            return 0

        # 1. Convert inputs to Tensors and initialize data arrays
        self.original_points_global_coords = torch.from_numpy(points_global_batch).float().to(self.device)
        
        if initial_labels_for_points is not None:
            self.mdet_labels_for_points = torch.from_numpy(initial_labels_for_points).to(self.device)
        else:
            self.mdet_labels_for_points = torch.full((batch_size,), OcclusionResult.UNDETERMINED.value, dtype=torch.int8, device=self.device)
            
        self.mdet_scores_for_points = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.total_points_added_to_di_arrays = batch_size

        # 2. Project points using the new tensor-based method
        _points_local, sph_coords, pixel_indices, valid_mask = \
            self.project_points_batch(self.original_points_global_coords)
        self.local_sph_coords_for_points = sph_coords

        # 3. Update pixel statistics using scatter operations (NO PYTHON LOOP)
        valid_original_indices = torch.where(valid_mask)[0]
        if valid_original_indices.numel() > 0:
            # Get data only for points that were validly projected
            valid_pixels = pixel_indices[valid_original_indices]
            valid_depths = sph_coords[valid_original_indices, 2]
            
            # --- The Core Scatter Logic ---
            # To use scatter, we need a single 1D index for each pixel (v * width + h)
            flat_pixel_indices = valid_pixels[:, 0] * self.num_pixels_h + valid_pixels[:, 1]
            
            # Reshape pixel stat tensors to 1D for scatter operations
            num_pixels_total = self.num_pixels_v * self.num_pixels_h
            pixel_min_depth_flat = self.pixel_min_depth.view(-1)
            pixel_max_depth_flat = self.pixel_max_depth.view(-1)
            pixel_count_flat = self.pixel_count.view(-1)

            # Perform scatter operations
            scatter_min(valid_depths, flat_pixel_indices, out=pixel_min_depth_flat)
            scatter_max(valid_depths, flat_pixel_indices, out=pixel_max_depth_flat)
            scatter_add(torch.ones_like(valid_depths, dtype=torch.int32), flat_pixel_indices, out=pixel_count_flat)

            # --- Handle pixel_original_indices (still on CPU) ---
            # This part remains a loop as it's harder to vectorize.
            # We move the necessary data to CPU for this.
            valid_pixels_cpu = valid_pixels.cpu().numpy()
            valid_original_indices_cpu = valid_original_indices.cpu().numpy()
            for i in range(len(valid_original_indices_cpu)):
                v_idx, h_idx = valid_pixels_cpu[i]
                self.pixel_original_indices[(v_idx, h_idx)].append(valid_original_indices_cpu[i])

        # Reset cache
        self.static_points_mask = None
        self._static_labels_used_for_cache = None
            
        return batch_size
    
    def get_original_points_global(self) -> Optional[np.ndarray]:
        """Returns the (N,3) array of original global point coordinates as a NumPy array."""
        if self.original_points_global_coords is None:
            return None
        return self.original_points_global_coords.cpu().numpy()

    def get_all_point_labels(self) -> Optional[np.ndarray]:
        """Returns the (N,) array of M-Detector labels as a NumPy array."""
        if self.mdet_labels_for_points is None:
            return None
        return self.mdet_labels_for_points.cpu().numpy()
        
    def get_static_points_mask(self, static_config_values: List[int]) -> Optional[torch.Tensor]:
        """
        Returns a boolean TENSOR indicating static points.
        The caller will be responsible for moving to CPU if needed.
        """
        if self.mdet_labels_for_points is None:
            return None

        if self.static_points_mask is not None and self._static_labels_used_for_cache == static_config_values:
            return self.static_points_mask

        # Create a tensor for the values to check against
        static_values_tensor = torch.tensor(static_config_values, device=self.device, dtype=self.mdet_labels_for_points.dtype)
        # Use broadcasting to create the boolean mask
        self.static_points_mask = torch.any(self.mdet_labels_for_points.unsqueeze(1) == static_values_tensor, dim=1)
        self._static_labels_used_for_cache = static_config_values
        return self.static_points_mask