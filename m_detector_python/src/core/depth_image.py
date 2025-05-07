# src/core/depth_image.py
import numpy as np
# import time # Not used in the provided snippet, can be removed if not needed elsewhere
from typing import Tuple, Optional, Dict, List, Any

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
        self.config_params: Dict[str, Any] = di_config 

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
        self.pixel_points = {}  # Will be a dictionary of lists, indexed by (v_idx, h_idx)
        
        # This matrix transforms points from the GLOBAL frame TO this DepthImage's LOCAL sensor frame.
        self.matrix_local_from_global: np.ndarray = np.linalg.inv(self.image_pose_global)

        self.total_points_added: int = 0 # Initialize counter for total points in DI

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
            point_global_h = np.append(point_global, 1.0) # Convert to homogeneous
        elif point_global.shape == (4,):
            point_global_h = point_global
            if abs(point_global_h[3] - 1.0) > 1e-6: # Check if w is approximately 1.0
                if abs(point_global_h[3]) < 1e-9: # w is zero or very close to zero
                    raise ValueError("Homogeneous global point has w component near zero, cannot transform as a point.")
                # Normalize if w is not 1 but also not zero (e.g. from a projection)
                # print(f"Warning: Normalizing homogeneous global point with w={point_global_h[3]}.")
                point_global_h = point_global_h / point_global_h[3]
        else:
            raise ValueError("Input point_global must be a 3D or 4D homogeneous vector.")
        
        # Apply the stored transformation matrix
        point_local_frame_h = self.matrix_local_from_global @ point_global_h
        return point_local_frame_h[:3] # Return as 3D Cartesian coordinates
    
    def project_points_batch(self, points_global_batch):
        """Project multiple points at once for better performance."""
        batch_size = points_global_batch.shape[0]
        
        # Pre-allocate result arrays
        points_local = np.zeros((batch_size, 3))
        sph_coords = np.zeros((batch_size, 3))
        pixel_indices = np.zeros((batch_size, 2), dtype=np.int32)
        valid_mask = np.zeros(batch_size, dtype=bool)
        
        # Make homogeneous coordinates for matrix multiplication
        points_global_h = np.hstack([points_global_batch, np.ones((batch_size, 1))])
        
        # Transform all points at once
        points_local_h = points_global_h @ self.matrix_local_from_global.T
        points_local = points_local_h[:, :3]
        
        # Calculate depths
        depths = np.linalg.norm(points_local, axis=1)
        
        # Avoid division by zero
        valid_depth = depths > 1e-6
        if not np.any(valid_depth):
            return points_local, sph_coords, pixel_indices, valid_mask
        
        # Calculate spherical coordinates only for valid depths
        x, y, z = points_local[valid_depth, 0], points_local[valid_depth, 1], points_local[valid_depth, 2]
        phi = np.arctan2(y, x)
        theta = np.arcsin(z / depths[valid_depth])
        
        # Check if points are within FOV
        in_fov = ((self.phi_min_rad <= phi) & (phi <= self.phi_max_rad) &
                (self.theta_min_rad <= theta) & (theta <= self.theta_max_rad))
        
        # Process only valid points
        valid_indices = np.where(valid_depth)[0][in_fov]
        if len(valid_indices) == 0:
            return points_local, sph_coords, pixel_indices, valid_mask
        
        # Store spherical coordinates for valid points
        sph_coords[valid_indices, 0] = phi[in_fov]
        sph_coords[valid_indices, 1] = theta[in_fov]
        sph_coords[valid_indices, 2] = depths[valid_depth][in_fov]
        
        # Calculate pixel indices
        phi_normalized = phi[in_fov] - self.phi_min_rad
        theta_normalized = theta[in_fov] - self.theta_min_rad
        
        h_idx = np.clip((phi_normalized / self.res_h_rad).astype(np.int32), 0, self.num_pixels_h - 1)
        v_idx = np.clip((theta_normalized / self.res_v_rad).astype(np.int32), 0, self.num_pixels_v - 1)
        
        # Store pixel indices
        pixel_indices[valid_indices, 0] = v_idx
        pixel_indices[valid_indices, 1] = h_idx
        
        # Mark valid points
        valid_mask[valid_indices] = True
        
        return points_local, sph_coords, pixel_indices, valid_mask

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
        if not (self.phi_min_rad <= phi <= self.phi_max_rad and
                self.theta_min_rad <= theta <= self.theta_max_rad):
            return point_in_di_frame, sph_coords, None # Point is outside FoV

        # 4. Determine pixel indices (v_idx, h_idx)
        phi_normalized = phi - self.phi_min_rad
        theta_normalized = theta - self.theta_min_rad # Assuming theta_min is "lower" visually

        # Calculate raw indices
        h_idx_raw = phi_normalized / self.res_h_rad
        v_idx_raw = theta_normalized / self.res_v_rad
        
        # Convert to integer and clamp to be within valid pixel range
        # Clamping is crucial due to floating point inaccuracies, especially at FoV boundaries.
        h_idx = max(0, min(int(h_idx_raw), self.num_pixels_h - 1))
        v_idx = max(0, min(int(v_idx_raw), self.num_pixels_v - 1))
        
        pixel_indices = (v_idx, h_idx)
        
        return point_in_di_frame, sph_coords, pixel_indices

    def add_point(self, point_global: np.ndarray, label: str = "unknown", original_timestamp: Optional[float] = None):
        """
        Adds a global 3D point to the correct pixel in the depth image.
        Updates pixel statistics (min/max depth, point list, count).
        """
        point_in_di_frame, sph_coords, pixel_indices = self.project_point_to_pixel_indices(point_global)

        if pixel_indices is None or sph_coords is None:
            # Point is outside FoV, too close to origin, or other projection issue
            return 

        v_idx, h_idx = pixel_indices
        depth = sph_coords[2] # Depth of the point in the DI's local frame

        # Update min/max depth with direct array access
        self.pixel_min_depth[v_idx, h_idx] = min(self.pixel_min_depth[v_idx, h_idx], depth)
        self.pixel_max_depth[v_idx, h_idx] = max(self.pixel_max_depth[v_idx, h_idx], depth)
        
        # Create or update the point list for this pixel
        pixel_key = (v_idx, h_idx)
        if pixel_key not in self.pixel_points:
            self.pixel_points[pixel_key] = []
        
        # Store detailed information about the point
        point_info = {
            'global_pt': point_global,          # Original global coordinates
            'di_frame_pt': point_in_di_frame,   # Coordinates in this DI's local sensor frame
            'sph_coords': sph_coords,           # Spherical coords (phi,theta,d) in local frame
            'label': label,
            'timestamp': original_timestamp if original_timestamp is not None else self.timestamp
        }
        
        # Add point to list, respecting max_points_per_pixel
        if len(self.pixel_points[pixel_key]) < self.max_points_per_pixel:
            self.pixel_points[pixel_key].append(point_info)
        
        # Update pixel count
        self.pixel_count[v_idx, h_idx] += 1
        self.total_points_added += 1 # Increment total points in the entire DepthImage
        
        # Only store if under the maximum points per pixel
        if len(self.pixel_points[pixel_key]) < self.max_points_per_pixel:
            point_info = {
                'global_pt': point_global,
                'di_frame_pt': point_in_di_frame,
                'sph_coords': sph_coords,
                'label': label,
                'timestamp': original_timestamp if original_timestamp is not None else self.timestamp
            }
            self.pixel_points[pixel_key].append(point_info)
        
        # Update count
        self.pixel_count[v_idx, h_idx] += 1
        self.total_points_added += 1

    def get_pixel_info(self, v_idx: int, h_idx: int) -> Optional[Dict[str, Any]]:
        """ 
        Returns the data stored in a specific pixel.
        Uses (v_idx, h_idx) consistent with array indexing (row, column).
        """
        if not (0 <= v_idx < self.num_pixels_v and 0 <= h_idx < self.num_pixels_h):
            print(f"Warning: Pixel indices ({v_idx}, {h_idx}) out of bounds ({self.num_pixels_v}x{self.num_pixels_h}).")
            return None
        
        # Create a compatible dictionary structure
        pixel_key = (v_idx, h_idx)
        return {
            'min_depth': self.pixel_min_depth[v_idx, h_idx],
            'max_depth': self.pixel_max_depth[v_idx, h_idx],
            'points': self.pixel_points.get(pixel_key, []),
            'count': self.pixel_count[v_idx, h_idx]
        }

    def __str__(self) -> str:
        pose_translation = self.image_pose_global[:3,3] # Extract translation for brevity
        return (f"DepthImage @ {self.timestamp:.2f}s, "
                f"Pose_xyz: [{pose_translation[0]:.2f}, {pose_translation[1]:.2f}, {pose_translation[2]:.2f}], "
                f"Dims: {self.num_pixels_v}x{self.num_pixels_h} pixels, "
                f"Total Points Added: {self.total_points_added}")
    
    # Batch processing 
    def project_points_batch(self, points_global_batch: np.ndarray):
        """
        Projects multiple points at once for better performance.
        
        Args:
            points_global_batch (np.ndarray): Nx3 array of points in global frame
            
        Returns:
            tuple: (points_local, sph_coords, pixel_indices, valid_mask)
                - points_local (np.ndarray): Nx3 array of points in local frame
                - sph_coords (np.ndarray): Nx3 array of (phi, theta, depth) for each point
                - pixel_indices (np.ndarray): Nx2 array of (v_idx, h_idx) for each point
                - valid_mask (np.ndarray): N boolean array indicating which points are valid (in FoV)
        """
        batch_size = points_global_batch.shape[0]
        
        # Pre-allocate result arrays
        points_local = np.zeros((batch_size, 3))
        sph_coords = np.zeros((batch_size, 3))
        pixel_indices = np.zeros((batch_size, 2), dtype=np.int32)
        valid_mask = np.zeros(batch_size, dtype=bool)
        
        # Convert all points to homogeneous coordinates
        if points_global_batch.shape[1] == 3:  # Input points are [x,y,z]
            points_global_h = np.hstack([points_global_batch, np.ones((batch_size, 1))])
        else:  # Input points are already homogeneous [x,y,z,1]
            points_global_h = points_global_batch
        
        # Transform all points at once
        points_local_h = points_global_h @ self.matrix_local_from_global.T
        points_local = points_local_h[:, :3]
        
        # Calculate depths
        depths = np.linalg.norm(points_local, axis=1)
        
        # Avoid division by zero
        valid_depth = depths > 1e-6
        if not np.any(valid_depth):
            return points_local, sph_coords, pixel_indices, valid_mask
        
        # Calculate spherical coordinates only for valid depths
        x, y, z = points_local[valid_depth, 0], points_local[valid_depth, 1], points_local[valid_depth, 2]
        phi = np.arctan2(y, x)  # Azimuth
        theta = np.arcsin(np.clip(z / depths[valid_depth], -1.0, 1.0))  # Elevation, clip to avoid numerical issues
        
        # Check if points are within FOV
        in_fov = ((self.phi_min_rad <= phi) & (phi <= self.phi_max_rad) &
                (self.theta_min_rad <= theta) & (theta <= self.theta_max_rad))
        
        # If no points in FOV, return early
        if not np.any(in_fov):
            return points_local, sph_coords, pixel_indices, valid_mask
        
        # Process only valid points
        valid_indices = np.where(valid_depth)[0][in_fov]
        if len(valid_indices) == 0:
            return points_local, sph_coords, pixel_indices, valid_mask
        
        # Store spherical coordinates for valid points
        sph_coords[valid_indices, 0] = phi[in_fov]
        sph_coords[valid_indices, 1] = theta[in_fov]
        sph_coords[valid_indices, 2] = depths[valid_depth][in_fov]
        
        # Calculate pixel indices
        phi_normalized = phi[in_fov] - self.phi_min_rad
        theta_normalized = theta[in_fov] - self.theta_min_rad
        
        h_idx = np.clip((phi_normalized / self.res_h_rad).astype(np.int32), 0, self.num_pixels_h - 1)
        v_idx = np.clip((theta_normalized / self.res_v_rad).astype(np.int32), 0, self.num_pixels_v - 1)
        
        # Store pixel indices
        pixel_indices[valid_indices, 0] = v_idx
        pixel_indices[valid_indices, 1] = h_idx
        
        # Mark valid points
        valid_mask[valid_indices] = True
        
        return points_local, sph_coords, pixel_indices, valid_mask

    def add_points_batch(self, points_global_batch: np.ndarray, labels=None, timestamps=None):
        """
        Add multiple points at once for better performance.
        
        Args:
            points_global_batch (np.ndarray): Nx3 array of points in global frame
            labels (list, optional): List of labels for each point. Default is "unknown".
            timestamps (list, optional): List of timestamps for each point. Default is self.timestamp.
            
        Returns:
            int: Number of points successfully added
        """
        batch_size = points_global_batch.shape[0]
        
        if labels is None:
            labels = ["unknown"] * batch_size
        if timestamps is None:
            timestamps = [self.timestamp] * batch_size
        
        # Project all points at once
        points_local, sph_coords, pixel_indices, valid_mask = self.project_points_batch(points_global_batch)
        
        # Count of points added
        points_added = 0
        
        # Process only valid points
        valid_indices = np.where(valid_mask)[0]
        for i in valid_indices:
            v_idx, h_idx = pixel_indices[i].astype(int)
            depth = sph_coords[i, 2]
            
            # Update min/max depth
            self.pixel_min_depth[v_idx, h_idx] = min(self.pixel_min_depth[v_idx, h_idx], depth)
            self.pixel_max_depth[v_idx, h_idx] = max(self.pixel_max_depth[v_idx, h_idx], depth)
            
            # Add to point list
            pixel_key = (v_idx, h_idx)
            if pixel_key not in self.pixel_points:
                self.pixel_points[pixel_key] = []
            
            if len(self.pixel_points[pixel_key]) < self.max_points_per_pixel:
                point_info = {
                    'global_pt': points_global_batch[i],
                    'di_frame_pt': points_local[i],
                    'sph_coords': sph_coords[i],
                    'label': labels[i],
                    'timestamp': timestamps[i]
                }
                self.pixel_points[pixel_key].append(point_info)
            
            # Update count
            self.pixel_count[v_idx, h_idx] += 1
            points_added += 1
        
        self.total_points_added += points_added
        return points_added