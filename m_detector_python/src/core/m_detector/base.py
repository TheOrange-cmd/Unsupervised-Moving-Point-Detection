"""
Defines the main MDetector class, which orchestrates the motion detection
process using depth images, occlusion checks, and map consistency.
"""

# src/core/m_detector/base.py

from enum import Enum
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging 

from ..depth_image import DepthImage
from ..depth_image_library import DepthImageLibrary
from ..constants import OcclusionResult

logger = logging.getLogger(__name__) # Module-level logger

class MDetector:
    """
    Implements the core logic of the M-Detector algorithm.
    Manages a library of DepthImages and performs occlusion checks,
    map consistency analysis, and temporal processing to label points
    as dynamic, static, or undetermined.

    The class methods are organized into separate files (occlusion_checks.py,
    map_consistency.py, temporal.py, processing.py) and imported into
    this base class definition.
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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}") # 

        # Load configuration sections
        self._load_occlusion_check_config()
        self._load_map_consistency_config()
        self._load_temporal_processing_config()
        self.logger.info("MDetector initialized successfully.")
    
    def _load_occlusion_check_config(self):
        """Load occlusion check parameters from config."""
        oc_cfg = self.config['occlusion_check']
        self.epsilon_depth_occlusion = oc_cfg['epsilon_depth_occlusion']
        self.neighbor_search_pixels_h = oc_cfg['neighbor_search_pixels_h']
        self.neighbor_search_pixels_v = oc_cfg['neighbor_search_pixels_v']
        self.angular_threshold_rad_h = np.deg2rad(oc_cfg['angular_threshold_deg_h'])
        self.angular_threshold_rad_v = np.deg2rad(oc_cfg['angular_threshold_deg_v'])
    
    def _load_map_consistency_config(self):
        """Load map consistency check parameters from config."""
        mc_cfg = self.config.get('map_consistency_check', {})
        self.map_consistency_enabled = mc_cfg.get('enabled', True)
        self.num_historical_di_for_map_check = mc_cfg.get('num_historical_di_for_map_check', 3)
        self.num_future_di_for_map_check = mc_cfg.get('num_future_di_for_map_check', 3)
        self.epsilon_phi_map_rad = np.deg2rad(mc_cfg.get('epsilon_phi_map_deg', 1.0))
        self.epsilon_theta_map_rad = np.deg2rad(mc_cfg.get('epsilon_theta_map_deg', 1.0))
        self.epsilon_depth_forward_map = mc_cfg.get('epsilon_depth_forward_map', 0.3)
        self.epsilon_depth_backward_map = mc_cfg.get('epsilon_depth_backward_map', 0.3)
        
        # Convert static labels from config to a set for faster lookups
        config_label_strings = mc_cfg.get('static_labels_for_map_check', [])
        self.static_labels_for_map_check = []

        if not isinstance(config_label_strings, list):
            print(f"Warning: 'static_labels_for_map_check' in config is not a list. Found: {config_label_strings}. Map consistency might not work as expected.")
            config_label_strings = [] # Ensure it's a list to iterate

        for label_str_from_config in config_label_strings:
            normalized_config_str = label_str_from_config.strip().upper() # Normalize for comparison

            if normalized_config_str == "NON_EVENT":
                # add_sweep_and_create_depth_image uses "non_event" (lowercase)
                self.static_labels_for_map_check.append("non_event")
            elif normalized_config_str == "PENDING_CLASSIFICATION":
                # add_sweep_and_create_depth_image uses "pending_classification" (lowercase)
                self.static_labels_for_map_check.append("pending_classification")
            elif normalized_config_str == "UNDETERMINED":
                self.static_labels_for_map_check.append(OcclusionResult.UNDETERMINED) # Store as Enum
            elif normalized_config_str == "STATIC_CONFIRMED_BY_MAP":
                # This label would be a string. Decide on its canonical form (e.g., uppercase)
                self.static_labels_for_map_check.append("STATIC_CONFIRMED_BY_MAP")
            else:
                # Fallback: if it's a different string that might be an enum name
                try:
                    enum_member = OcclusionResult[normalized_config_str]
                    self.static_labels_for_map_check.append(enum_member)
                except KeyError:
                    print(f"Warning: Label '{label_str_from_config}' in static_labels_for_map_check config"
                        f" is not a recognized special string or OcclusionResult member. It will be ignored.")

        if not self.static_labels_for_map_check and config_label_strings:
            print("Warning: 'static_labels_for_map_check' was configured in YAML, but the parsing logic resulted in an empty list. "
                "No points will be considered static for map consistency.")
        elif not config_label_strings:
            print("Info: 'static_labels_for_map_check' is not configured or is empty in YAML. "
                "Map consistency will rely on default behavior (if any) or fail to find static points.")

        self.logger.debug(f"Parsed static_labels_for_map_check: {self.static_labels_for_map_check}")
    
    def _load_temporal_processing_config(self):
        """Load temporal processing parameters from config."""
        temp_cfg = self.config.get('temporal_processing', {})
        self.use_bidirectional = temp_cfg.get('bidirectional', False)
        self.past_window_size = temp_cfg.get('past_window_size', 5)
        self.future_window_size = temp_cfg.get('future_window_size', 5)
        self.past_weight = temp_cfg.get('past_weight', 1.0)
        self.future_weight = temp_cfg.get('future_weight', 1.0)
    
    def is_ready_for_processing(self) -> bool:
        """Checks if enough DepthImages have been collected to start full processing."""
        return self.depth_image_library.is_ready_for_processing(self.min_sweeps_for_processing)
    
    def add_sweep_and_create_depth_image(self, 
                                        points_lidar_frame: np.ndarray, 
                                        T_global_lidar: np.ndarray, 
                                        lidar_timestamp: float) -> DepthImage:
        """
        Creates a DepthImage from a new LiDAR sweep and adds it to the library.
        """
        # Create new depth image
        current_di = DepthImage(
            image_pose_global=T_global_lidar,
            config=self.config,
            timestamp=lidar_timestamp
        )

        # Transform points to global frame
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
        
        # Add to library and return
        self.depth_image_library.add_image(current_di)
        return current_di
        
    # Import methods from other modules
    from .occlusion_checks import (
        check_occlusion_pixel_level,
        check_occlusion_batch
    )
    
    from .map_consistency import (
        is_map_consistent
    )
    
    from .temporal import (
        process_and_label_di_bidirectional,
        _determine_final_label,
        _determine_final_label_bidirectional_simplified
    )
    
    from .processing import (
        process_and_label_di,
        process_frame
    )