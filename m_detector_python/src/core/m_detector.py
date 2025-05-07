# src/core/m_detector.py
from enum import Enum
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from .depth_image import DepthImage
from .depth_image_library import DepthImageLibrary

class OcclusionResult(Enum):
    """
    Enumerates the possible outcomes of a pixel-level occlusion check.
    """
    UNDETERMINED = 0
    OCCLUDED_BY_IMAGE = 1  # current_point is further than max_depth in historical DI pixel region
    OCCLUDING_IMAGE = 2    # current_point is nearer than min_depth in historical DI pixel region
    EMPTY_IN_IMAGE = 3     # The corresponding region in the historical DI was empty (no points)

class MDetector:
    """
    Implements the core logic of the M-Detector algorithm.
    Manages a library of DepthImages and performs occlusion checks,
    map consistency, and event detection.
    """
    def __init__(self, config: Dict):
        """
        Initializes the MDetector.

        Args:
            config (Dict): The configuration dictionary.
        """
        self.config = config
        self.depth_image_library = DepthImageLibrary(
            max_size=config['depth_image']['library_size']
        )
        self.min_sweeps_for_processing = config['initialization']['num_initial_sweeps_for_map']

        # Occlusion check parameters from config
        oc_cfg = config['occlusion_check']
        self.epsilon_depth_occlusion: float = oc_cfg['epsilon_depth_occlusion']
        self.neighbor_search_pixels_h: int = oc_cfg['neighbor_search_pixels_h']
        self.neighbor_search_pixels_v: int = oc_cfg['neighbor_search_pixels_v']
        self.angular_threshold_rad_h: float = np.deg2rad(oc_cfg['angular_threshold_deg_h'])
        self.angular_threshold_rad_v: float = np.deg2rad(oc_cfg['angular_threshold_deg_v'])

    def add_sweep_and_create_depth_image(self, 
                                        points_lidar_frame: np.ndarray, 
                                        T_global_lidar: np.ndarray, 
                                        lidar_timestamp: float) -> DepthImage:
        """
        Creates a DepthImage from a new LiDAR sweep and adds it to the library,
        using batch processing for better performance.
        """
        current_di = DepthImage(
            image_pose_global=T_global_lidar,
            config=self.config,
            timestamp=lidar_timestamp
        )

        # Transform points to global frame for DI storage
        points_lidar_frame_h = np.hstack((points_lidar_frame, np.ones((points_lidar_frame.shape[0], 1))))
        points_global_h = (T_global_lidar @ points_lidar_frame_h.T).T
        points_global = points_global_h[:, :3]

        # Filter points based on range
        max_range = self.config['filtering']['max_point_range_meters']
        min_range = self.config['filtering']['min_point_range_meters']
        
        # Calculate ranges of all points at once
        ranges = np.linalg.norm(points_lidar_frame, axis=1)
        range_mask = (min_range <= ranges) & (ranges <= max_range)
        
        # Only keep points within range
        filtered_points_global = points_global[range_mask]
        
        # Set labels in batch
        label = "non_event" if not self.is_ready_for_processing() else "pending_classification"
        batch_labels = [label] * len(filtered_points_global)
        
        # Add all points at once
        current_di.add_points_batch(
            points_global_batch=filtered_points_global,
            labels=batch_labels
        )
        
        self.depth_image_library.add_image(current_di)
        # print(f"Created DI: {str(current_di)}. Library size: {len(self.depth_image_library)}")
        return current_di
    
    def check_occlusion_batch(self, 
                            points_global_batch: np.ndarray,
                            historical_depth_image: DepthImage) -> np.ndarray:
        """
        Perform occlusion checks on a batch of points.
        
        Args:
            points_global_batch (np.ndarray): Nx3 array of points in global frame
            historical_depth_image (DepthImage): Historical depth image to compare against
            
        Returns:
            np.ndarray: Array of OcclusionResult enum values for each input point
        """
        batch_size = points_global_batch.shape[0]
        results = np.zeros(batch_size, dtype=np.int32)
        
        # Project all points at once
        points_local, sph_coords, pixel_indices, valid_mask = historical_depth_image.project_points_batch(points_global_batch)
        
        # Process each valid point
        for i in np.where(valid_mask)[0]:
            v_idx, h_idx = pixel_indices[i].astype(int)
            d_curr = sph_coords[i, 2]
            
            # Get region bounds
            v_start = max(0, v_idx - self.neighbor_search_pixels_v)
            v_end = min(historical_depth_image.num_pixels_v, v_idx + self.neighbor_search_pixels_v + 1)
            h_start = max(0, h_idx - self.neighbor_search_pixels_h)
            h_end = min(historical_depth_image.num_pixels_h, h_idx + self.neighbor_search_pixels_h + 1)
            
            # Get region data
            region_min_depths = historical_depth_image.pixel_min_depth[v_start:v_end, h_start:h_end]
            region_max_depths = historical_depth_image.pixel_max_depth[v_start:v_end, h_start:h_end]
            region_counts = historical_depth_image.pixel_count[v_start:v_end, h_start:h_end]
            
            # Check if region has data
            has_data = region_counts > 0
            found_data_in_region = np.any(has_data)
            
            if not found_data_in_region:
                results[i] = OcclusionResult.EMPTY_IN_IMAGE.value
                continue
                
            # Get min/max depths in region
            min_depth_in_region = np.min(region_min_depths[has_data])
            max_depth_in_region = np.max(region_max_depths[has_data])
            
            # Check occlusion conditions
            if d_curr > max_depth_in_region + self.epsilon_depth_occlusion:
                results[i] = OcclusionResult.OCCLUDED_BY_IMAGE.value
            elif d_curr < min_depth_in_region - self.epsilon_depth_occlusion:
                results[i] = OcclusionResult.OCCLUDING_IMAGE.value
            else:
                results[i] = OcclusionResult.UNDETERMINED.value
        
        # Set UNDETERMINED for points not in FoV
        results[~valid_mask] = OcclusionResult.UNDETERMINED.value
        
        # Convert to enum values
        enum_results = np.array([OcclusionResult(int(r)) for r in results])
        return enum_results

    # def add_sweep_and_create_depth_image(self, 
    #                                      points_lidar_frame: np.ndarray, 
    #                                      T_global_lidar: np.ndarray, 
    #                                      lidar_timestamp: float) -> DepthImage:
    #     """
    #     Creates a DepthImage from a new LiDAR sweep and adds it to the library.

    #     Args:
    #         points_lidar_frame (np.ndarray): Points in the LiDAR sensor's coordinate frame.
    #         T_global_lidar (np.ndarray): Global pose of the LiDAR sensor for this sweep.
    #         lidar_timestamp (float): Timestamp of the LiDAR sweep.

    #     Returns:
    #         DepthImage: The newly created and populated DepthImage.
    #     """
    #     current_di = DepthImage(
    #         image_pose_global=T_global_lidar,
    #         config=self.config,
    #         timestamp=lidar_timestamp
    #     )

    #     # Transform points to global frame for DI storage
    #     points_lidar_frame_h = np.hstack((points_lidar_frame, np.ones((points_lidar_frame.shape[0], 1))))
    #     points_global_h = (T_global_lidar @ points_lidar_frame_h.T).T
    #     points_global = points_global_h[:, :3]

    #     max_range = self.config['filtering']['max_point_range_meters']
    #     min_range = self.config['filtering']['min_point_range_meters']

    #     for pt_idx, pt_g in enumerate(points_global):
    #         range_val = np.linalg.norm(points_lidar_frame[pt_idx]) # Range check in sensor frame
    #         if min_range <= range_val <= max_range:
    #             # During initialization, all points are provisionally non-event.
    #             # Otherwise, label will be determined by full M-detector logic later.
    #             label = "non_event" if not self.is_ready_for_processing() else "pending_classification"
    #             current_di.add_point(pt_g, label=label)
        
    #     self.depth_image_library.add_image(current_di)
    #     # print(f"Created DI: {str(current_di)}. Library size: {len(self.depth_image_library)}")
    #     return current_di

    def is_ready_for_processing(self) -> bool:
        """Checks if enough DepthImages have been collected to start full processing."""
        return self.depth_image_library.is_ready_for_processing(self.min_sweeps_for_processing)

    def check_occlusion_pixel_level(self,
                                    current_point_global: np.ndarray,
                                    historical_depth_image: DepthImage
                                    ) -> Tuple[OcclusionResult, Optional[Tuple[int, int]], Optional[np.ndarray]]:
        """
        Performs pixel-level occlusion check for current_point_global against a historical_depth_image.
        (Corresponds to Test 1 - Fig. 10 in M-Detector paper, using min/max depths in pixel regions).

        Args:
            current_point_global (np.ndarray): The 3D point (in global frame) from the current scan.
            historical_depth_image (DepthImage): A historical depth image to compare against.

        Returns:
            Tuple[OcclusionResult, Optional[Tuple[int, int]], Optional[np.ndarray]]:
                - The occlusion result enum.
                - Pixel indices (v_idx, h_idx) in historical_depth_image if projected, else None.
                - Spherical coordinates (phi, theta, d_curr) of current_point_global in historical_depth_image's frame.
        """
        # 1. Project current_point_global into historical_depth_image's frame
        point_in_hist_di_frame, sph_coords_curr, pixel_indices_in_hist_di = \
            historical_depth_image.project_point_to_pixel_indices(current_point_global)

        if pixel_indices_in_hist_di is None or sph_coords_curr is None:
            return OcclusionResult.UNDETERMINED, None, None # Point projects outside historical DI's FoV

        v_idx_curr_proj, h_idx_curr_proj = pixel_indices_in_hist_di
        d_curr = sph_coords_curr[2] # Depth of current_point w.r.t. historical_depth_image's origin

        # 2. Gather min/max depths from the projected pixel and its neighbors in historical_depth_image
        # Define region bounds with clipping to image boundaries
        v_start = max(0, v_idx_curr_proj - self.neighbor_search_pixels_v)
        v_end = min(historical_depth_image.num_pixels_v, v_idx_curr_proj + self.neighbor_search_pixels_v + 1)
        h_start = max(0, h_idx_curr_proj - self.neighbor_search_pixels_h)
        h_end = min(historical_depth_image.num_pixels_h, h_idx_curr_proj + self.neighbor_search_pixels_h + 1)
        
        # Extract region data from arrays
        region_min_depths = historical_depth_image.pixel_min_depth[v_start:v_end, h_start:h_end]
        region_max_depths = historical_depth_image.pixel_max_depth[v_start:v_end, h_start:h_end]
        region_counts = historical_depth_image.pixel_count[v_start:v_end, h_start:h_end]
        
        # Check if any pixel in region has points
        has_data = region_counts > 0
        found_data_in_region = np.any(has_data)
        
        if not found_data_in_region:
            return OcclusionResult.EMPTY_IN_IMAGE, pixel_indices_in_hist_di, sph_coords_curr

        # Find min/max depths in the region (only where points exist)
        min_depth_in_region = np.min(region_min_depths[has_data])
        max_depth_in_region = np.max(region_max_depths[has_data])

        # 3. Check occlusion conditions (Eq. 7 and 8 in M-Detector paper, adapted)
        if d_curr > max_depth_in_region + self.epsilon_depth_occlusion:
            return OcclusionResult.OCCLUDED_BY_IMAGE, pixel_indices_in_hist_di, sph_coords_curr
        
        if d_curr < min_depth_in_region - self.epsilon_depth_occlusion:
            return OcclusionResult.OCCLUDING_IMAGE, pixel_indices_in_hist_di, sph_coords_curr

        return OcclusionResult.UNDETERMINED, pixel_indices_in_hist_di, sph_coords_curr
    
    def process_and_label_di(self,
                             current_di: DepthImage,
                             historical_di: Optional[DepthImage]) -> int:
        """
        Processes points in current_di against historical_di to determine and store
        occlusion labels within current_di's point_info structures.

        Args:
            current_di (DepthImage): The DepthImage whose points are to be labeled.
            historical_di (Optional[DepthImage]): The historical DepthImage to compare against.
                                                 If None, points will be labeled as UNDETERMINED.

        Returns:
            int: The number of points for which labels were updated/assigned.
        """
        if not isinstance(current_di, DepthImage):
            raise TypeError("current_di must be a DepthImage object.")

        points_labeled_count = 0
        if not historical_di or historical_di.timestamp >= current_di.timestamp:
            print(f"Warning: No valid (older) historical DI provided for DI @ {current_di.timestamp/1e6:.2f}s. "
                  f"Points will be labeled as UNDETERMINED.")
            # Label all points in current_di as UNDETERMINED
            for v_idx in range(current_di.num_pixels_v):
                for h_idx in range(current_di.num_pixels_h):
                    # Use get_pixel_info to get pixel data
                    pixel_content = current_di.get_pixel_info(v_idx, h_idx)
                    if pixel_content and pixel_content['points']:
                        for pt_info in pixel_content['points']:
                            pt_info['label'] = OcclusionResult.UNDETERMINED
                            points_labeled_count += 1
            return points_labeled_count

        # Process all points in current_di against historical_di
        for v_idx in range(current_di.num_pixels_v):
            for h_idx in range(current_di.num_pixels_h):
                # Use get_pixel_info to get pixel data
                pixel_content = current_di.get_pixel_info(v_idx, h_idx)
                if pixel_content and pixel_content['points']:
                    for pt_info in pixel_content['points']:
                        global_pt = pt_info['global_pt']
                        
                        # Perform the pixel-level occlusion check
                        result, _, _ = self.check_occlusion_pixel_level(global_pt, historical_di)
                        
                        # Update the label in the point_info dictionary
                        pt_info['label'] = result 
                        points_labeled_count += 1
        
        return points_labeled_count
