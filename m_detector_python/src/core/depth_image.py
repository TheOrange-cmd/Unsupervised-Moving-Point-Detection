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
    def __init__(self, image_pose_global: np.ndarray, config: dict, timestamp: float):
        """
        Initializes a DepthImage.

        Args:
            image_pose_global (np.ndarray): 4x4 homogeneous transformation matrix representing the
                                            LiDAR sensor's pose in the global frame (global_T_lidar)
                                            at the moment this depth image is notionally created.
            config (dict): Configuration dictionary, expected to have a 'depth_image' sub-dictionary.
            timestamp (float): Timestamp associated with this depth image (e.g., start of sweep).
        """
        self.image_pose_global: np.ndarray = image_pose_global # Global pose of this DepthImage's sensor
        self.timestamp: float = timestamp
        
        # Store only the 'depth_image' sub-configuration for convenience
        di_config = config['depth_image']

        self.res_h_rad: float = np.deg2rad(di_config['resolution_h_deg'])
        self.res_v_rad: float = np.deg2rad(di_config['resolution_v_deg'])

        self.phi_min_rad: float = di_config['phi_min_rad']
        self.phi_max_rad: float = di_config['phi_max_rad']
        self.theta_min_rad: float = di_config['theta_min_rad']
        self.theta_max_rad: float = di_config['theta_max_rad']
        
        self.max_points_per_pixel: int = di_config['max_points_per_pixel']
        
        # Calculate number of pixels
        self.num_pixels_h: int = int(np.ceil((self.phi_max_rad - self.phi_min_rad) / self.res_h_rad))
        self.num_pixels_v: int = int(np.ceil((self.theta_max_rad - self.theta_min_rad) / self.res_v_rad))

        # Now we can initialize the optimized data structures since num_pixels_v and num_pixels_h are known
        self.pixel_min_depth = np.full((self.num_pixels_v, self.num_pixels_h), np.inf)
        self.pixel_max_depth = np.full((self.num_pixels_v, self.num_pixels_h), -np.inf)
        self.pixel_count = np.zeros((self.num_pixels_v, self.num_pixels_h), dtype=np.int32)
        
        # Keep storing individual points, but use a more efficient structure
        self.pixel_points: Dict[Tuple[int, int], List[Dict[str, Any]]] = collections.defaultdict(list)
        
        # This matrix transforms points from the GLOBAL frame TO this DepthImage's LOCAL sensor frame.
        self.matrix_local_from_global: np.ndarray = np.linalg.inv(self.image_pose_global)

        self.total_points_added: int = 0 # Initialize counter for total points in DI
        # self._projection_cache = {}  # Cache for projections

    def _apply_transformation_to_point(self, point_global: np.ndarray) -> np.ndarray:
        """
        Transforms a single 3D or 4D homogeneous point from the global coordinate frame
        to this DepthImage's local sensor coordinate frame using self.matrix_local_from_global.

        Args:
            point_global (np.ndarray): 3D point (x,y,z) or 4D homogeneous point (x,y,z,1) in global frame.

        Returns:
            np.ndarray: 3D point (x,y,z) in the local sensor frame of this DepthImage.
        """
        if point_global.shape == (3,):
            point_global_h = np.array([point_global[0], point_global[1], point_global[2], 1.0], dtype=point_global.dtype)
        elif point_global.shape == (4,):
            point_global_h = point_global
            if abs(point_global_h[3] - 1.0) > 1e-6:
                if abs(point_global_h[3]) < 1e-9:
                    raise ValueError("Homogeneous global point has w component near zero.")
                # logger.warning(f"Normalizing homogeneous global point with w={point_global_h[3]}.") # Optional logging
                point_global_h = point_global_h / point_global_h[3]
        else:
            raise ValueError("Input point_global must be a 3D or 4D homogeneous vector.")
        
        point_local_frame_h = self.matrix_local_from_global @ point_global_h
        return point_local_frame_h[:3]
    
    # def _compute_projection_batch(self, points_global_batch):
    #     """Project multiple points at once for better performance."""
    #     batch_size = points_global_batch.shape[0]
        
    #     # Pre-allocate result arrays
    #     points_local = np.zeros((batch_size, 3))
    #     sph_coords = np.zeros((batch_size, 3))
    #     pixel_indices = np.zeros((batch_size, 2), dtype=np.int32)
    #     valid_mask = np.zeros(batch_size, dtype=bool)
        
    #     # Make homogeneous coordinates for matrix multiplication
    #     points_global_h = np.hstack([points_global_batch, np.ones((batch_size, 1))])
        
    #     # Transform all points at once
    #     points_local_h = points_global_h @ self.matrix_local_from_global.T
    #     points_local = points_local_h[:, :3]
        
    #     # Calculate depths
    #     depths = np.linalg.norm(points_local, axis=1)
        
    #     # Avoid division by zero
    #     valid_depth = depths > 1e-6
    #     if not np.any(valid_depth):
    #         return points_local, sph_coords, pixel_indices, valid_mask
        
    #     # Calculate spherical coordinates only for valid depths
    #     x, y, z = points_local[valid_depth, 0], points_local[valid_depth, 1], points_local[valid_depth, 2]
    #     phi = np.arctan2(y, x)
    #     theta = np.arcsin(z / depths[valid_depth])
        
    #     # Check if points are within FOV
    #     in_fov = ((self.phi_min_rad <= phi) & (phi <= self.phi_max_rad) &
    #             (self.theta_min_rad <= theta) & (theta <= self.theta_max_rad))
        
    #     # Process only valid points
    #     valid_indices = np.where(valid_depth)[0][in_fov]
    #     if len(valid_indices) == 0:
    #         return points_local, sph_coords, pixel_indices, valid_mask
        
    #     # Store spherical coordinates for valid points
    #     sph_coords[valid_indices, 0] = phi[in_fov]
    #     sph_coords[valid_indices, 1] = theta[in_fov]
    #     sph_coords[valid_indices, 2] = depths[valid_depth][in_fov]
        
    #     # Calculate pixel indices
    #     phi_normalized = phi[in_fov] - self.phi_min_rad
    #     theta_normalized = theta[in_fov] - self.theta_min_rad
        
    #     h_idx = np.clip((phi_normalized / self.res_h_rad).astype(np.int32), 0, self.num_pixels_h - 1)
    #     v_idx = np.clip((theta_normalized / self.res_v_rad).astype(np.int32), 0, self.num_pixels_v - 1)
        
    #     # Store pixel indices
    #     pixel_indices[valid_indices, 0] = v_idx
    #     pixel_indices[valid_indices, 1] = h_idx
        
    #     # Mark valid points
    #     valid_mask[valid_indices] = True
        
    #     return points_local, sph_coords, pixel_indices, valid_mask
    
    # def project_points_batch(self, points_global):
    #     # Check cache
    #     cache_key = hash(points_global.tobytes())
    #     if cache_key in self._projection_cache:
    #         return self._projection_cache[cache_key]
        
    #     # Compute projection
    #     results = self._compute_projection_batch(points_global)
        
    #     # Store in cache
    #     self._projection_cache[cache_key] = results
        
    #     return results

    def project_point_to_pixel_indices(self, point_global: np.ndarray) -> \
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Projects a global 3D point into this DepthImage, determining its local coordinates,
        spherical coordinates, and pixel indices.

        Args:
            point_global (np.ndarray): The 3D point in global coordinates.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
                - point_in_di_frame (np.ndarray): The 3D point transformed into the DI's local frame.
                - sph_coords (np.ndarray): Spherical coordinates (phi, theta, depth) of the point
                                           in the DI's local frame.
                - pixel_indices (Tuple[int, int]): (v_idx, h_idx) of the pixel.
                Any of these can be None if the point cannot be projected or is out of FoV.
        """
        # 1. Transform Gp to Lp (point in DI's local frame) using the renamed method
        point_in_di_frame = self._apply_transformation_to_point(point_global)

        # 2. Convert point in DI's local frame to spherical coordinates (phi, theta, d)
        x, y, z = point_in_di_frame[0], point_in_di_frame[1], point_in_di_frame[2]
        d = np.linalg.norm(point_in_di_frame) # More robust way to get depth

        if d < 1e-6: # Avoid division by zero or issues with points at/near sensor origin
            return point_in_di_frame, None, None 

        phi = np.arctan2(y, x)    # Azimuth: from -pi to pi
        theta = np.arcsin(z / d)  # Elevation: from -pi/2 to pi/2

        sph_coords = np.array([phi, theta, d])

        # 3. Check if point is within the DI's configured Field of View (FoV)
        # A small epsilon is added to comparisons for robustness at boundaries
        epsilon_fov = 1e-6 
        if not (self.phi_min_rad - epsilon_fov <= phi <= self.phi_max_rad + epsilon_fov and
                self.theta_min_rad - epsilon_fov <= theta <= self.theta_max_rad + epsilon_fov):
            return point_in_di_frame, sph_coords, None

        # 4. Determine pixel indices (v_idx, h_idx)
        phi_normalized = phi - self.phi_min_rad
        theta_normalized = theta - self.theta_min_rad # theta_min_rad is the 'bottom' of the FoV

        h_idx_raw = phi_normalized / self.res_h_rad
        v_idx_raw = theta_normalized / self.res_v_rad
        
        h_idx = int(h_idx_raw)
        v_idx = int(v_idx_raw)

        # Clamp to be within valid pixel range.
        # This handles points exactly on max boundaries mapping to num_pixels-1
        h_idx = max(0, min(h_idx, self.num_pixels_h - 1))
        v_idx = max(0, min(v_idx, self.num_pixels_v - 1))
        
        pixel_indices = (v_idx, h_idx)
        
        return point_in_di_frame, sph_coords, pixel_indices

    def add_point(self, point_global: np.ndarray, label: str = "unknown", original_timestamp: Optional[float] = None):
        """
        Adds a global 3D point to the correct pixel in the depth image.
        Updates pixel statistics (min/max depth, point list, count).
        """
        point_in_di_frame, sph_coords, pixel_indices = self.project_point_to_pixel_indices(point_global)

        if pixel_indices is None or sph_coords is None:
            return 

        v_idx, h_idx = pixel_indices
        depth = sph_coords[2]

        self.pixel_min_depth[v_idx, h_idx] = min(self.pixel_min_depth[v_idx, h_idx], depth)
        self.pixel_max_depth[v_idx, h_idx] = max(self.pixel_max_depth[v_idx, h_idx], depth)
        
        pixel_key = (v_idx, h_idx)
        # if pixel_key not in self.pixel_points:
        #     self.pixel_points[pixel_key] = []
        
        # Only add point to list if not exceeding max_points_per_pixel
        if len(self.pixel_points[pixel_key]) < self.max_points_per_pixel:
            point_info = {
                'global_pt': point_global,
                'di_frame_pt': point_in_di_frame,
                'sph_coords': sph_coords,
                'label': label,
                'timestamp': original_timestamp if original_timestamp is not None else self.timestamp
            }
            self.pixel_points[pixel_key].append(point_info)
        
        # Update pixel count
        self.pixel_count[v_idx, h_idx] = len(self.pixel_points[pixel_key])
        point_info = {
            'global_pt': point_global,
            'di_frame_pt': point_in_di_frame,
            'sph_coords': sph_coords,
            'label': label,
            'timestamp': original_timestamp if original_timestamp is not None else self.timestamp
        }
        
        if len(self.pixel_points[pixel_key]) < self.max_points_per_pixel:
            self.pixel_points[pixel_key].append(point_info)
        
        self.pixel_count[v_idx, h_idx] += 1 # Count all points falling into this pixel
        self.total_points_added += 1  

    def get_pixel_info(self, v_idx: int, h_idx: int) -> Optional[Dict[str, Any]]:
        """ 
        Returns the data stored in a specific pixel.
        Uses (v_idx, h_idx) consistent with array indexing (row, column).
        """
        if not (0 <= v_idx < self.num_pixels_v and 0 <= h_idx < self.num_pixels_h):
            logger.warning(f"Pixel indices ({v_idx}, {h_idx}) out of bounds ({self.num_pixels_v}x{self.num_pixels_h}).")
            return None
        
        # Create a compatible dictionary structure
        pixel_key = (v_idx, h_idx)
        return {
            'min_depth': self.pixel_min_depth[v_idx, h_idx],
            'max_depth': self.pixel_max_depth[v_idx, h_idx],
            'points': self.pixel_points[pixel_key],
            'count': self.pixel_count[v_idx, h_idx]
        }

    def __str__(self) -> str:
        pose_translation = self.image_pose_global[:3,3] # Extract translation for brevity
        return (f"DepthImage @ {self.timestamp:.2f}s, "
                f"Pose_xyz: [{pose_translation[0]:.2f}, {pose_translation[1]:.2f}, {pose_translation[2]:.2f}], "
                f"Dims: {self.num_pixels_v}x{self.num_pixels_h} pixels, "
                f"Total Points Added: {self.total_points_added}")
    
    # Batch processing 
    def project_points_batch(self, points_global_batch: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects a batch of global 3D points into this DepthImage's local frame,
        calculates their spherical coordinates, and determines their pixel indices.

        Args:
            points_global_batch (np.ndarray): An Nx3 or Nx4 (homogeneous) array of
                                              points in the global coordinate frame.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - points_local (np.ndarray): Nx3 array of points in the DI's local sensor frame.
                - sph_coords (np.ndarray): Nx3 array of spherical coordinates (phi, theta, depth)
                                           for each point in the local frame.
                - pixel_indices (np.ndarray): Nx2 array of (v_idx, h_idx) for each point.
                - valid_mask (np.ndarray): A boolean array of shape (N,) where True indicates
                                           the point was successfully projected within FoV.
        """
        batch_size = points_global_batch.shape[0]
        
        points_local_all = np.zeros((batch_size, 3), dtype=np.float32) 
        sph_coords_all = np.zeros((batch_size, 3), dtype=np.float32) 
        pixel_indices_all = np.zeros((batch_size, 2), dtype=np.int32)
        valid_mask = np.zeros(batch_size, dtype=bool)
        
        if points_global_batch.shape[1] == 3:
            points_global_h = np.hstack([points_global_batch, np.ones((batch_size, 1), dtype=np.float32)])
        elif points_global_batch.shape[1] == 4:
            points_global_h = points_global_batch
            # Optional: Normalize if w is not 1, though matrix_local_from_global should handle it if it's a standard pose.
        else:
            raise ValueError("points_global_batch must be Nx3 or Nx4.")

        points_local_h = points_global_h @ self.matrix_local_from_global.T
        points_local_all = points_local_h[:, :3]
        
        depths = np.linalg.norm(points_local_all, axis=1)
        
        # Mask for points with valid depth (not at sensor origin)
        has_valid_depth_mask = depths > 1e-6 
        
        # Early exit if no points have valid depth
        if not np.any(has_valid_depth_mask):
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

        # Calculate spherical coordinates only for points with valid depth
        # Create views for calculations to avoid indexing multiple times
        pl_valid_depth = points_local_all[has_valid_depth_mask]
        d_valid_depth = depths[has_valid_depth_mask]

        x, y, z = pl_valid_depth[:, 0], pl_valid_depth[:, 1], pl_valid_depth[:, 2]
        
        phi = np.arctan2(y, x)
        # Clip argument to arcsin to prevent domain errors due to floating point inaccuracies
        theta = np.arcsin(np.clip(z / d_valid_depth, -1.0, 1.0)) 
        
        # FoV check
        epsilon_fov = 1e-6 # Small tolerance for floating point comparisons at boundaries
        in_fov_mask_subset = (
            (self.phi_min_rad - epsilon_fov <= phi) & (phi <= self.phi_max_rad + epsilon_fov) &
            (self.theta_min_rad - epsilon_fov <= theta) & (theta <= self.theta_max_rad + epsilon_fov)
        )
        
        # Combine valid_depth and in_fov masks
        # final_valid_indices_in_subset are indices relative to the subset of points that had valid depth
        final_valid_indices_in_subset = np.where(in_fov_mask_subset)[0]
        
        # Map these subset indices back to original batch indices
        original_indices_of_valid_points = np.where(has_valid_depth_mask)[0][final_valid_indices_in_subset]

        if original_indices_of_valid_points.size == 0:
            return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

        # Update the main valid_mask
        valid_mask[original_indices_of_valid_points] = True

        # Populate sph_coords for valid points
        sph_coords_all[original_indices_of_valid_points, 0] = phi[final_valid_indices_in_subset]
        sph_coords_all[original_indices_of_valid_points, 1] = theta[final_valid_indices_in_subset]
        sph_coords_all[original_indices_of_valid_points, 2] = d_valid_depth[final_valid_indices_in_subset]
        
        # Calculate pixel indices for these valid points
        phi_norm = phi[final_valid_indices_in_subset] - self.phi_min_rad
        theta_norm = theta[final_valid_indices_in_subset] - self.theta_min_rad
        
        h_idx_float = phi_norm / self.res_h_rad
        v_idx_float = theta_norm / self.res_v_rad

        h_idx = np.clip(h_idx_float.astype(np.int32), 0, self.num_pixels_h - 1)
        v_idx = np.clip(v_idx_float.astype(np.int32), 0, self.num_pixels_v - 1)
        
        pixel_indices_all[original_indices_of_valid_points, 0] = v_idx
        pixel_indices_all[original_indices_of_valid_points, 1] = h_idx
        
        return points_local_all, sph_coords_all, pixel_indices_all, valid_mask

    def add_points_batch(self, points_global_batch: np.ndarray, labels: Optional[List[str]] = None, 
                         timestamps: Optional[List[float]] = None) -> int:
        """
        Adds multiple global 3D points to the depth image using batch projection.
        Updates pixel statistics (min/max depth, point list, count).

        Args:
            points_global_batch (np.ndarray): Nx3 array of points in global frame.
            labels (Optional[List[str]]): List of labels for each point. If None,
                                          defaults to "unknown". Must match batch size.
            timestamps (Optional[List[float]]): List of timestamps for each point. If None,
                                                defaults to this DepthImage's timestamp.
                                                Must match batch size.
        Returns:
            int: Number of points successfully processed and added (or attempted to add) to pixels.
        """
        batch_size = points_global_batch.shape[0]
        
        if labels is None:
            labels_list = ["unknown"] * batch_size
        elif len(labels) != batch_size:
            raise ValueError("Length of labels list must match batch_size.")
        else:
            labels_list = labels # Use provided labels

        if timestamps is None:
            timestamps_list = [self.timestamp] * batch_size
        elif len(timestamps) != batch_size:
            raise ValueError("Length of timestamps list must match batch_size.")
        else:
            timestamps_list = timestamps # Use provided timestamps

        points_local, sph_coords, pixel_indices, valid_mask = self.project_points_batch(points_global_batch)
        
        points_processed_count = 0
        
        valid_original_indices = np.where(valid_mask)[0]

        for i in valid_original_indices: # Iterate using original indices of valid points
            v_idx, h_idx = pixel_indices[i, 0], pixel_indices[i, 1] # Already int from projection
            depth = sph_coords[i, 2]
            
            self.pixel_min_depth[v_idx, h_idx] = min(self.pixel_min_depth[v_idx, h_idx], depth)
            self.pixel_max_depth[v_idx, h_idx] = max(self.pixel_max_depth[v_idx, h_idx], depth)
            
            pixel_key = (v_idx, h_idx)
            if pixel_key not in self.pixel_points:
                self.pixel_points[pixel_key] = []
            
            if len(self.pixel_points[pixel_key]) < self.max_points_per_pixel:
                point_info = {
                    'global_pt': points_global_batch[i],
                    'di_frame_pt': points_local[i],
                    'sph_coords': sph_coords[i],
                    'label': labels_list[i], # Use from prepared list
                    'timestamp': timestamps_list[i] # Use from prepared list
                }
                self.pixel_points[pixel_key].append(point_info)
            
            self.pixel_count[v_idx, h_idx] += 1
            points_processed_count += 1
        
        self.total_points_added += points_processed_count # Increment total by number of valid points processed
        return points_processed_count
    
    def get_all_points(self, with_labels: bool = False) -> Dict[str, np.ndarray]:
        """
        Get all points in the depth image, optionally with their labels.
        
        Args:
            with_labels: If True, include point labels
            
        Returns:
            dict: Dictionary of categorized points
                {
                    'all': Nx3 array of all points,
                    'dynamic': Mx3 array of dynamic points,
                    'occluded': Px3 array of occluded points,
                    'undetermined': Qx3 array of points with no clear label
                }
        """
        all_points_list: List[np.ndarray] = [] # Explicitly list of arrays
        dynamic_points_list: List[np.ndarray] = []
        occluded_points_list: List[np.ndarray] = []
        undetermined_points_list: List[np.ndarray] = []
        
        # Iterate only where there are points
        v_indices, h_indices = np.where(self.pixel_count > 0)
        
        for v_idx, h_idx in zip(v_indices, h_indices):
            pixel_content = self.get_pixel_info(v_idx, h_idx)
            if pixel_content and 'points' in pixel_content and pixel_content['points']:
                for pt_info in pixel_content['points']:
                    pt = pt_info['global_pt']
                    all_points_list.append(pt)
                    
                    if with_labels and 'label' in pt_info:
                        if pt_info['label'] == OcclusionResult.OCCLUDING_IMAGE:
                            dynamic_points_list.append(pt)
                        elif pt_info['label'] == OcclusionResult.OCCLUDED_BY_IMAGE:
                            occluded_points_list.append(pt)
                        else:
                            undetermined_points_list.append(pt)
        
        # Convert to numpy arrays
        return {
            'all': np.array(all_points_list) if all_points_list else np.zeros((0, 3)),
            'dynamic': np.array(dynamic_points_list) if dynamic_points_list else np.zeros((0, 3)),
            'occluded': np.array(occluded_points_list) if occluded_points_list else np.zeros((0, 3)),
            'undetermined': np.array(undetermined_points_list) if undetermined_points_list else np.zeros((0, 3))
        }