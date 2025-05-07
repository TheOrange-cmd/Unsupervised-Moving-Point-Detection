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

        # Initialize pixels array
        self.pixels: np.ndarray = np.empty((self.num_pixels_v, self.num_pixels_h), dtype=object)
        for r in range(self.num_pixels_v):
            for c in range(self.num_pixels_h):
                self.pixels[r, c] = {
                    'min_depth': float('inf'),
                    'max_depth': float('-inf'),
                    'points': [], # Stores point_info dictionaries
                    'count': 0 # Explicit count, can also use len(points)
                }
        
        # This matrix transforms points from the GLOBAL frame TO this DepthImage's LOCAL sensor frame.
        # Renamed from your 'transform_global_to_lidar' attribute to avoid conflict with the method.
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

        # The pixel initialization in __init__ ensures self.pixels[v_idx, h_idx] is always a dict.
        pixel_data = self.pixels[v_idx, h_idx]

        # Store detailed information about the point
        point_info = {
            'global_pt': point_global,          # Original global coordinates
            'di_frame_pt': point_in_di_frame,   # Coordinates in this DI's local sensor frame
            'sph_coords': sph_coords,           # Spherical coords (phi,theta,d) in local frame
            'label': label,
            'timestamp': original_timestamp if original_timestamp is not None else self.timestamp
        }
        
        # Add point to list, respecting max_points_per_pixel
        if len(pixel_data['points']) < self.max_points_per_pixel:
            pixel_data['points'].append(point_info)
        # else:
            # Optional: Implement a strategy if max_points_per_pixel is reached
            # (e.g., replace oldest, random replacement, or simply don't add).
            # Currently, it just stops adding more raw point details to this pixel.

        # Update pixel statistics
        pixel_data['min_depth'] = min(pixel_data['min_depth'], depth)
        pixel_data['max_depth'] = max(pixel_data['max_depth'], depth)
        pixel_data['count'] = len(pixel_data['points']) # Update count based on actual stored points

        self.total_points_added += 1 # Increment total points in the entire DepthImage

    def get_pixel_info(self, v_idx: int, h_idx: int) -> Optional[Dict[str, Any]]:
        """ 
        Returns the data stored in a specific pixel.
        Uses (v_idx, h_idx) consistent with array indexing (row, column).
        """
        if 0 <= v_idx < self.num_pixels_v and 0 <= h_idx < self.num_pixels_h:
            return self.pixels[v_idx, h_idx]
        print(f"Warning: Pixel indices ({v_idx}, {h_idx}) out of bounds ({self.num_pixels_v}x{self.num_pixels_h}).")
        return None

    def __str__(self) -> str:
        pose_translation = self.image_pose_global[:3,3] # Extract translation for brevity
        return (f"DepthImage @ {self.timestamp:.2f}s, "
                f"Pose_xyz: [{pose_translation[0]:.2f}, {pose_translation[1]:.2f}, {pose_translation[2]:.2f}], "
                f"Dims: {self.num_pixels_v}x{self.num_pixels_h} pixels, "
                f"Total Points Added: {self.total_points_added}")