"""
Represents a depth image generated from a single LiDAR sweep, storing points
in a 2D pixel grid based on their spherical coordinates relative to the sensor.
"""

# src/core/depth_image.py

import numpy as np
from typing import Tuple, Optional, Dict, List, Any 
from .constants import OcclusionResult 
import logging 
import collections 

logger = logging.getLogger(__name__)

class DepthImage:
    def __init__(self, 
                 image_pose_global: np.ndarray, 
                 # MODIFIED: Accept specific depth image parameters
                 depth_image_params: Dict[str, Any], 
                 timestamp: float):
        """
        Initializes a DepthImage.

        Args:
            image_pose_global (np.ndarray): 4x4 homogeneous transformation matrix.
            depth_image_params (Dict[str, Any]): Configuration dictionary specifically for depth_image parameters.
                                                 Expected keys: 'resolution_h_deg', 'resolution_v_deg',
                                                                'phi_min_rad', 'phi_max_rad',
                                                                'theta_min_rad', 'theta_max_rad'.
            timestamp (float): Timestamp associated with this depth image.
        """
        self.image_pose_global: np.ndarray = image_pose_global
        self.timestamp: float = timestamp
        
        # Use the passed-in specific config for depth image parameters
        self.res_h_rad: float = np.deg2rad(depth_image_params['resolution_h_deg'])
        self.res_v_rad: float = np.deg2rad(depth_image_params['resolution_v_deg'])

        self.phi_min_rad: float = depth_image_params['phi_min_rad']
        self.phi_max_rad: float = depth_image_params['phi_max_rad']
        self.theta_min_rad: float = depth_image_params['theta_min_rad']
        self.theta_max_rad: float = depth_image_params['theta_max_rad']

        self.num_pixels_h: int = int(np.ceil((self.phi_max_rad - self.phi_min_rad) / self.res_h_rad))
        self.num_pixels_v: int = int(np.ceil((self.theta_max_rad - self.theta_min_rad) / self.res_v_rad))

        # Pixel-level statistics
        self.pixel_min_depth = np.full((self.num_pixels_v, self.num_pixels_h), np.inf, dtype=np.float32)
        self.pixel_max_depth = np.full((self.num_pixels_v, self.num_pixels_h), -np.inf, dtype=np.float32)
        self.pixel_count = np.zeros((self.num_pixels_v, self.num_pixels_h), dtype=np.int32)
        
        self.pixel_original_indices: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)

        self.original_points_global_coords: Optional[np.ndarray] = None
        self.mdet_labels_for_points: Optional[np.ndarray] = None
        self.mdet_scores_for_points: Optional[np.ndarray] = None
        self.local_sph_coords_for_points: Optional[np.ndarray] = None
        
        self.raw_occlusion_results_vs_history: Optional[np.ndarray] = None
        
        self.matrix_local_from_global: np.ndarray = np.linalg.inv(self.image_pose_global)
        self.total_points_added_to_di_arrays: int = 0

    def is_prepared_for_projection(self) -> bool:
        """
        Checks if the DepthImage has the necessary data populated for projection operations
        and accessing pixel-level information.
        Primarily, this means points have been added.
        """
        # If original_points_global_coords is not None, it implies add_points_batch was called,
        # and other essential arrays like local_sph_coords_for_points and pixel stats
        # would have been initialized or populated.
        return self.original_points_global_coords is not None

    def _apply_transformation_to_point(self, point_global: np.ndarray) -> np.ndarray:
        if point_global.shape == (3,):
            point_global_h = np.array([point_global[0], point_global[1], point_global[2], 1.0], dtype=point_global.dtype)
        elif point_global.shape == (4,):
            point_global_h = point_global
            if abs(point_global_h[3] - 1.0) > 1e-6:
                if abs(point_global_h[3]) < 1e-9:
                    raise ValueError("Homogeneous global point has w component near zero.")
                point_global_h = point_global_h / point_global_h[3]
        else:
            raise ValueError("Input point_global must be a 3D or 4D homogeneous vector.")
        
        point_local_frame_h = self.matrix_local_from_global @ point_global_h
        return point_local_frame_h[:3]

    def project_point_to_pixel_indices(self, point_global: np.ndarray) -> \
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        point_in_di_frame = self._apply_transformation_to_point(point_global)
        x, y, z = point_in_di_frame[0], point_in_di_frame[1], point_in_di_frame[2]
        d = np.linalg.norm(point_in_di_frame)

        if d < 1e-6: 
            return point_in_di_frame, None, None 

        phi = np.arctan2(y, x)
        theta = np.arcsin(np.clip(z / d, -1.0, 1.0)) # Added clip for robustness
        sph_coords = np.array([phi, theta, d], dtype=np.float32)

        epsilon_fov = 1e-6 
        if not (self.phi_min_rad - epsilon_fov <= phi <= self.phi_max_rad + epsilon_fov and
                self.theta_min_rad - epsilon_fov <= theta <= self.theta_max_rad + epsilon_fov):
            return point_in_di_frame, sph_coords, None

        phi_normalized = phi - self.phi_min_rad
        theta_normalized = theta - self.theta_min_rad
        
        h_idx = int(phi_normalized / self.res_h_rad)
        v_idx = int(theta_normalized / self.res_v_rad)
        
        h_idx = max(0, min(h_idx, self.num_pixels_h - 1))
        v_idx = max(0, min(v_idx, self.num_pixels_v - 1))
        
        pixel_indices = (v_idx, h_idx)
        return point_in_di_frame, sph_coords, pixel_indices

    def project_points_batch(self, points_global_batch: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size = points_global_batch.shape[0]
        
        points_local_all = np.zeros((batch_size, 3), dtype=np.float32) 
        sph_coords_all = np.zeros((batch_size, 3), dtype=np.float32) 
        pixel_indices_all = np.zeros((batch_size, 2), dtype=np.int32) 
        valid_mask = np.zeros(batch_size, dtype=bool)
        
        if points_global_batch.shape[1] == 3:
            points_global_h = np.hstack([points_global_batch, np.ones((batch_size, 1), dtype=np.float32)])
        elif points_global_batch.shape[1] == 4:
            points_global_h = points_global_batch.astype(np.float32)
        else:
            raise ValueError("points_global_batch must be Nx3 or Nx4.")

        points_local_h = points_global_h @ self.matrix_local_from_global.T
        points_local_all = points_local_h[:, :3]
        
        depths = np.linalg.norm(points_local_all, axis=1)
        has_valid_depth_mask = depths > 1e-6 
        
        if not np.any(has_valid_depth_mask):
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

        pl_valid_depth = points_local_all[has_valid_depth_mask]
        d_valid_depth = depths[has_valid_depth_mask]
        x, y, z = pl_valid_depth[:, 0], pl_valid_depth[:, 1], pl_valid_depth[:, 2]
        
        phi = np.arctan2(y, x)
        theta = np.arcsin(np.clip(z / d_valid_depth, -1.0, 1.0)) 
        
        epsilon_fov = 1e-6
        in_fov_mask_subset = (
            (self.phi_min_rad - epsilon_fov <= phi) & (phi <= self.phi_max_rad + epsilon_fov) &
            (self.theta_min_rad - epsilon_fov <= theta) & (theta <= self.theta_max_rad + epsilon_fov)
        )
        
        final_valid_indices_in_subset = np.where(in_fov_mask_subset)[0]
        original_indices_of_valid_points = np.where(has_valid_depth_mask)[0][final_valid_indices_in_subset]

        if original_indices_of_valid_points.size == 0:
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

        valid_mask[original_indices_of_valid_points] = True
        sph_coords_all[original_indices_of_valid_points, 0] = phi[final_valid_indices_in_subset]
        sph_coords_all[original_indices_of_valid_points, 1] = theta[final_valid_indices_in_subset]
        sph_coords_all[original_indices_of_valid_points, 2] = d_valid_depth[final_valid_indices_in_subset]
        
        phi_norm = phi[final_valid_indices_in_subset] - self.phi_min_rad
        theta_norm = theta[final_valid_indices_in_subset] - self.theta_min_rad
        
        h_idx_float = phi_norm / self.res_h_rad
        v_idx_float = theta_norm / self.res_v_rad

        h_idx = np.clip(h_idx_float.astype(np.int32), 0, self.num_pixels_h - 1)
        v_idx = np.clip(v_idx_float.astype(np.int32), 0, self.num_pixels_v - 1)
        
        pixel_indices_all[original_indices_of_valid_points, 0] = v_idx
        pixel_indices_all[original_indices_of_valid_points, 1] = h_idx
        
        return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

    def add_points_batch(self, 
                         points_global_batch: np.ndarray,
                         initial_labels_for_points: Optional[np.ndarray] = None
                        ) -> int:
        """
        Adds multiple global 3D points to the depth image using batch projection.
        Initializes main point data arrays and populates pixel-level statistics
        and original_index lists.

        Args:
            points_global_batch (np.ndarray): Nx3 array of points in global frame.
                                              (Labels and timestamps are no longer direct args here,
                                               as labels are initialized internally and DI has one timestamp)
        Returns:
            int: Number of points in the input batch (all are stored in main arrays).
        """
        batch_size = points_global_batch.shape[0]
        if batch_size == 0:
            self.original_points_global_coords = np.empty((0,3), dtype=np.float32)
            self.mdet_labels_for_points = np.empty(0, dtype=np.int8)
            self.mdet_scores_for_points = np.empty(0, dtype=np.float32)
            self.local_sph_coords_for_points = np.empty((0,3), dtype=np.float32)
            self.total_points_added_to_di_arrays = 0
            return 0

        # 1. Store Original Data & Initialize Label/Score Arrays
        self.original_points_global_coords = points_global_batch.astype(np.float32).copy()
        
        if initial_labels_for_points is not None and \
           initial_labels_for_points.shape[0] == batch_size:
            self.mdet_labels_for_points = initial_labels_for_points.copy()
        else:
            self.mdet_labels_for_points = np.full(batch_size, OcclusionResult.UNDETERMINED.value, dtype=np.int8)
            if initial_labels_for_points is not None: # Log warning if shape mismatch
                logger.warning("Shape mismatch for initial_labels_for_points or not provided. Defaulting all to UNDETERMINED.")

        self.mdet_scores_for_points = np.zeros(batch_size, dtype=np.float32)
        self.total_points_added_to_di_arrays = batch_size

        # 2. Project all points and get their local spherical coordinates in this DI's frame
        #    project_points_batch returns: points_local, sph_coords, pixel_indices, valid_mask
        #    sph_coords are (phi, theta, depth) in this DI's local frame.
        _points_local_frame, sph_coords_in_di_frame, pixel_indices_for_all, valid_projection_mask = \
            self.project_points_batch(self.original_points_global_coords)
        
        # Store all local spherical coordinates (even for points outside FoV, they'll have some value)
        self.local_sph_coords_for_points = sph_coords_in_di_frame.copy()

        # 3. Populate pixel_original_indices and update pixel statistics
        #    Only for points that were validly projected within FoV.
        valid_original_indices = np.where(valid_projection_mask)[0]
        
        # For optimizing updates later with np.ufunc.at
        # These are 1D arrays of v_coords, h_coords, and depths for *validly projected* points
        valid_v_coords = pixel_indices_for_all[valid_original_indices, 0]
        valid_h_coords = pixel_indices_for_all[valid_original_indices, 1]
        depths_of_valid_points = self.local_sph_coords_for_points[valid_original_indices, 2]

        # Update pixel_count (can be optimized with np.add.at later)
        for i in range(len(valid_original_indices)):
            original_idx = valid_original_indices[i]
            v_idx = valid_v_coords[i]
            h_idx = valid_h_coords[i]
            depth = depths_of_valid_points[i]

            pixel_key = (v_idx, h_idx)
            
            # Add original_index to the list for that pixel
            self.pixel_original_indices[pixel_key].append(original_idx)
            
            # Update pixel stats (will be optimized in P3)
            self.pixel_min_depth[v_idx, h_idx] = min(self.pixel_min_depth[v_idx, h_idx], depth)
            self.pixel_max_depth[v_idx, h_idx] = max(self.pixel_max_depth[v_idx, h_idx], depth)
            self.pixel_count[v_idx, h_idx] += 1
            
        return batch_size # Returns total points added to the main arrays

    def get_pixel_info(self, v_idx: int, h_idx: int) -> Dict[str, Any]:
        """ 
        Returns aggregated data and original indices for points in a specific pixel.
        """
        if not (0 <= v_idx < self.num_pixels_v and 0 <= h_idx < self.num_pixels_h):
            # logger.warning(f"Pixel indices ({v_idx}, {h_idx}) out of bounds.") # Can be noisy
            return {
                'min_depth': np.inf,
                'max_depth': -np.inf,
                'count': 0,
                'original_indices_in_pixel': []
            }
        
        pixel_key = (v_idx, h_idx)
        return {
            'min_depth': self.pixel_min_depth[v_idx, h_idx],
            'max_depth': self.pixel_max_depth[v_idx, h_idx],
            'count': self.pixel_count[v_idx, h_idx],
            'original_indices_in_pixel': self.pixel_original_indices.get(pixel_key, []) 
        }

    def __str__(self) -> str:
        pose_translation = self.image_pose_global[:3,3]
        return (f"DepthImage @ {self.timestamp:.2f}s, "
                f"Pose_xyz: [{pose_translation[0]:.2f}, {pose_translation[1]:.2f}, {pose_translation[2]:.2f}], "
                f"Dims: {self.num_pixels_v}x{self.num_pixels_h} pixels, "
                f"Total Points Stored: {self.total_points_added_to_di_arrays}") 
    
    def get_original_points_global(self) -> Optional[np.ndarray]:
        """Returns the (N,3) array of original global point coordinates."""
        return self.original_points_global_coords

    def get_all_point_labels(self) -> Optional[np.ndarray]:
        """Returns the (N,) array of M-Detector labels for each original point."""
        return self.mdet_labels_for_points

    def get_all_point_scores(self) -> Optional[np.ndarray]:
        """Returns the (N,) array of M-Detector scores for each original point."""
        return self.mdet_scores_for_points

    def get_local_sph_coords_for_all_points(self) -> Optional[np.ndarray]:
        """Returns the (N,3) array of (phi,theta,depth) in this DI's local frame for each original point."""
        return self.local_sph_coords_for_points

    # Old get_all_points method - to be deprecated or removed as assembly will use new getters
    def get_all_points_DEPRECATED(self, with_labels: bool = False) -> Dict[str, np.ndarray]:
        logger.warning("get_all_points_DEPRECATED is called. This should be replaced by new getter methods for assembly.")
        # This method would need significant rework to be compatible or should be removed.
        # For now, returning empty to signify it's not the way forward.
        return {
            'all': np.zeros((0, 3)), 'dynamic': np.zeros((0, 3)),
            'occluded': np.zeros((0, 3)), 'undetermined': np.zeros((0, 3))
        }
    
    def get_pixel_3d_corners_global(
        self,
        v_idx: int,
        h_idx: int,
        depth_for_visualization: float
    ) -> Optional[np.ndarray]:
        """
        Calculates the 3D global coordinates of the four corners of a given pixel.

        Args:
            v_idx (int): Vertical pixel index.
            h_idx (int): Horizontal pixel index.
            depth_for_visualization (float): The depth at which to project the pixel corners.

        Returns:
            Optional[np.ndarray]: A 4x3 NumPy array of [x,y,z] global coordinates
                                for the pixel corners, or None if indices are invalid.
                                Order of corners: bottom-left, bottom-right, top-right, top-left
                                (when viewed from sensor origin, looking outwards).
        """
        if not (0 <= v_idx < self.num_pixels_v and 0 <= h_idx < self.num_pixels_h):
            logger.warning(f"Pixel indices ({v_idx}, {h_idx}) out of bounds for DepthImage.")
            return None

        # Calculate angular boundaries for the pixel based on DI's properties
        # Horizontal angle (phi)
        phi_pixel_min = self.phi_min_rad + h_idx * self.res_h_rad
        phi_pixel_max = self.phi_min_rad + (h_idx + 1) * self.res_h_rad
        # Vertical angle (theta)
        theta_pixel_min = self.theta_min_rad + v_idx * self.res_v_rad
        theta_pixel_max = self.theta_min_rad + (v_idx + 1) * self.res_v_rad

        # Define the 4 corners in spherical coordinates (phi, theta, depth)
        # Order chosen for defining a quad for rendering (e.g., for k3d.lines later)
        # Corner 0: (phi_min, theta_min) -> "bottom-left" of the pixel in the spherical projection
        # Corner 1: (phi_max, theta_min) -> "bottom-right"
        # Corner 2: (phi_max, theta_max) -> "top-right"
        # Corner 3: (phi_min, theta_max) -> "top-left"
        sph_corners_local = np.array([
            [phi_pixel_min, theta_pixel_min, depth_for_visualization],
            [phi_pixel_max, theta_pixel_min, depth_for_visualization],
            [phi_pixel_max, theta_pixel_max, depth_for_visualization],
            [phi_pixel_min, theta_pixel_max, depth_for_visualization],
        ], dtype=np.float32)

        # Spherical to Cartesian conversion (sensor frame coordinates)
        
        d_vals = sph_corners_local[:, 2]
        phi_vals = sph_corners_local[:, 0]
        theta_vals = sph_corners_local[:, 1]

        x_sensor = d_vals * np.cos(theta_vals) * np.cos(phi_vals)
        y_sensor = d_vals * np.cos(theta_vals) * np.sin(phi_vals)
        z_sensor = d_vals * np.sin(theta_vals)

        cartesian_corners_sensor_frame = np.stack((x_sensor, y_sensor, z_sensor), axis=-1)

        # Transform corners to global frame
        if self.image_pose_global is not None:
            # Add a column of ones for homogenous transformation
            cartesian_corners_sensor_frame_hom = np.hstack(
                (cartesian_corners_sensor_frame, np.ones((4, 1), dtype=np.float32))
            )
            # Transform to global: P_global = T_global_sensor @ P_sensor_homogeneous.T
            global_corners_hom = (self.image_pose_global @ cartesian_corners_sensor_frame_hom.T).T
            return global_corners_hom[:, :3] # Return (4,3) array
        else:
            # This case should ideally not happen if we are visualizing in a global context
            logger.warning("DepthImage has no image_pose_global. Returning pixel corners in sensor frame.")
            return cartesian_corners_sensor_frame