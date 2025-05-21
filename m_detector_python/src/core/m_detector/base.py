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
from ..debug_collector import PointDebugCollector
from ...config_loader import MDetectorConfigAccessor
from .pre_labelers import ACTIVE_PRE_LABELERS 

logger = logging.getLogger(__name__) # Module-level logger

class MDetector:
    """
    Implements the core logic of the M-Detector algorithm.
    Manages a library of DepthImages and performs occlusion checks,
    map consistency analysis, and temporal processing to label points
    as dynamic, static, or undetermined.
    """
    def __init__(self, config_accessor: MDetectorConfigAccessor):
        self.config_accessor = config_accessor 
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Get DI library size from accessor
        di_lib_params = self.config_accessor.get_depth_image_params()
        library_max_size = di_lib_params.get('library_size', 20) # Default if not in config
        
        self.depth_image_library = DepthImageLibrary(max_size=library_max_size)
        
        init_params = self.config_accessor.get_initialization_phase_params()
        self.min_sweeps_for_processing = init_params.get('num_sweeps_for_initial_map', 1)
        
        self._load_occlusion_check_config()
        self._load_map_consistency_config()
        self._load_event_tests_config()
        self._load_pre_labeler_configs()

        # State for processing flow
        self.timestamp_of_last_processed_di: Optional[float] = None 
        self.current_lidar_sd_token: Optional[str] = None # Still useful for MDetector's internal reference if needed
        self.debug_collector: Optional[PointDebugCollector] = None # Add this line

        self.logger.info("MDetector initialized successfully.")

    def set_debug_collector(self, collector: Optional[PointDebugCollector]) -> None:
        """
        Sets or clears the debug collector for tracing point processing.
        """
        self.debug_collector = collector
    
    def reset_scene_state(self):
        """Resets internal state for processing a new scene."""
        self.logger.critical(f"!!!!!! MDetector.reset_scene_state CALLED. Library will be cleared. Current lib size: {len(self.depth_image_library._images)} !!!!!!") # Use CRITICAL to make it stand out
        self.depth_image_library.clear()
        self.timestamp_of_last_processed_di = None
        self.current_lidar_sd_token = None
        # self.logger.info("MDetector state reset for new scene.") # Already have a similar log

    def _load_occlusion_check_config(self):
        oc_params = self.config_accessor.get_occlusion_determination_params()
        self.epsilon_depth_occlusion = oc_params.get('epsilon_depth', 0.3)
        self.neighbor_search_pixels_h = oc_params.get('pixel_neighborhood_h', 1)
        self.neighbor_search_pixels_v = oc_params.get('pixel_neighborhood_v', 1)
        # Angular thresholds are not used in the provided batch check, but keep if used elsewhere
        # self.angular_threshold_rad_h = np.deg2rad(oc_cfg.get('angular_threshold_deg_h', 2.0))
        # self.angular_threshold_rad_v = np.deg2rad(oc_cfg.get('angular_threshold_deg_v', 2.0))
    
    def _load_map_consistency_config(self):
        mc_params = self.config_accessor.get_map_consistency_params()
        self.map_consistency_enabled = mc_params.get('enabled', True)
        self.map_consistency_time_window_past_s = mc_params.get('time_window_past_s', 0.25)
        self.map_consistency_time_window_future_s = mc_params.get('map_consistency_time_window_future_s', 0.25) 
        self.epsilon_phi_map_rad = np.deg2rad(mc_params.get('epsilon_phi_map_deg', 1.0))
        self.epsilon_theta_map_rad = np.deg2rad(mc_params.get('epsilon_theta_map_deg', 1.0))
        self.epsilon_depth_forward_map = mc_params.get('epsilon_depth_forward_map', 0.3)
        self.epsilon_depth_backward_map = mc_params.get('epsilon_depth_backward_map', 0.3)
        self.mc_threshold_mode = mc_params.get('threshold_mode', 'count').lower()
        self.mc_threshold_value_count = mc_params.get('threshold_value_count', 1)
        self.mc_threshold_value_ratio = mc_params.get('threshold_value_ratio', 0.5)

        config_label_strings = mc_params.get('static_labels_for_map_check', [])
        self.static_labels_for_map_check = []
        if not isinstance(config_label_strings, list):
            self.logger.warning(f"'static_labels_for_map_check' in config is not a list. Found: {config_label_strings}.")
            config_label_strings = []
        for label_str in config_label_strings:
            norm_str = label_str.strip().upper()
            if norm_str == "NON_EVENT": self.static_labels_for_map_check.append("non_event") # Keep as strings if preferred
            elif norm_str == "PENDING_CLASSIFICATION": self.static_labels_for_map_check.append("pending_classification")
            elif norm_str == "PRELABELED_STATIC_GROUND": self.static_labels_for_map_check.append(OcclusionResult.PRELABELED_STATIC_GROUND) # Use enum
            elif hasattr(OcclusionResult, norm_str): self.static_labels_for_map_check.append(OcclusionResult[norm_str])
            else: self.static_labels_for_map_check.append(label_str)
        self.logger.debug(f"Parsed static_labels_for_map_check: {self.static_labels_for_map_check}")

    def _load_event_tests_config(self):
        test1_params = self.config_accessor.get_test1_perpendicular_params()
        self.test1_N_historical_DIs = test1_params.get('num_historical_DIs_N', 5)
        self.test1_M1_threshold = test1_params.get('min_occluding_DIs_M1', 2)

        test2_params = self.config_accessor.get_test2_parallel_away_params()
        self.test2_M2_depth_images = test2_params.get('num_historical_DIs_M2', 3)

        test3_params = self.config_accessor.get_test3_parallel_towards_params()
        self.test3_M3_depth_images = test3_params.get('num_historical_DIs_M3', 3)
        
        # Detailed check parameters are under 'occlusion_determination' in your config
        oc_params = self.config_accessor.get_occlusion_determination_params()
        self.detailed_check_angular_threshold_h_rad = np.deg2rad(oc_params.get('angular_neighborhood_h_deg', 0.5))
        self.detailed_check_angular_threshold_v_rad = np.deg2rad(oc_params.get('angular_neighborhood_v_deg', 0.5))
        # For detailed_check_epsilon_depth, let's use the general epsilon_depth for now,
        # or you can add a specific 'detailed_check_epsilon_depth_m' to your config.
        self.detailed_check_epsilon_depth = oc_params.get('epsilon_depth', 0.5) # Reusing general epsilon_depth

        self.logger.info(f"Event Test Config: Test1_N={self.test1_N_historical_DIs}, Test1_M1={self.test1_M1_threshold}, "
                         f"Test2_M2={self.test2_M2_depth_images}, Test3_M3={self.test3_M3_depth_images}")
        self.logger.info(f"Detailed Occlusion Check Config: AngH_rad={self.detailed_check_angular_threshold_h_rad:.3f}, "
                         f"AngV_rad={self.detailed_check_angular_threshold_v_rad:.3f}, EpsDepth={self.detailed_check_epsilon_depth:.2f}")
        
    def _load_pre_labeler_configs(self):
        self.pre_labeler_params = {}
        ransac_cfg = self.config_accessor.get_ransac_ground_params()
        if ransac_cfg:
            self.pre_labeler_params["ransac_ground"] = ransac_cfg
        else:
            self.pre_labeler_params["ransac_ground"] = None # Will use defaults in ransac_ground_prelabeler
    
    def is_ready_for_processing(self) -> bool:
        """
        Checks if enough DepthImages have been collected for basic (causal) processing.
        Bidirectional processing has its own readiness check within decide_and_process_frame.
        """
        return len(self.depth_image_library._images) >= self.min_sweeps_for_processing

    def add_sweep_and_create_depth_image(self,
                                        points_lidar_frame: np.ndarray, # Nx3
                                        T_global_lidar: np.ndarray,
                                        lidar_timestamp: float,
                                        lidar_sd_token: Optional[str] = None) -> DepthImage:

        self.current_lidar_sd_token = lidar_sd_token
        depth_image_constructor_params = self.config_accessor.get_depth_image_params()
        current_di = DepthImage(
            image_pose_global=T_global_lidar,
            depth_image_params=depth_image_constructor_params,
            timestamp=lidar_timestamp,
        )

        # Original points in lidar frame (Nx3) are passed directly
        # Transform to global for storage and some checks
        points_lidar_frame_h = np.hstack((points_lidar_frame, np.ones((points_lidar_frame.shape[0], 1))))
        points_global_h = (T_global_lidar @ points_lidar_frame_h.T).T
        points_global = points_global_h[:, :3] # Nx3 global

        # --- Point Pre-Filtering (Range) ---
        point_filter_params = self.config_accessor.get_point_pre_filtering_params()
        max_range = point_filter_params.get('max_range_meters', 80.0)
        min_range = point_filter_params.get('min_range_meters', 1.0)
        ranges = np.linalg.norm(points_lidar_frame, axis=1)
        range_mask = (min_range <= ranges) & (ranges <= max_range)

        # Apply range mask to both global and lidar frame points
        filtered_points_global = points_global[range_mask]
        filtered_points_lidar_frame = points_lidar_frame[range_mask] # Needed for RANSAC
        
        if filtered_points_global.shape[0] == 0:
            current_di.add_points_batch(np.empty((0,3), dtype=np.float32))
            self.depth_image_library.add_image(current_di)
            return current_di

        # --- Pre-Labeling Phase ---
        # Initialize labels for filtered points
        num_filtered_points = filtered_points_global.shape[0]
        preliminary_labels = np.full(num_filtered_points, OcclusionResult.UNDETERMINED.value, dtype=np.int8)

        for pre_labeler_name, pre_labeler_func, _ in ACTIVE_PRE_LABELERS: # Use _ for default params for now
            self.logger.debug(f"Applying pre-labeler: {pre_labeler_name}")
            # Get specific params for this pre-labeler if configured
            specific_params = self.pre_labeler_params.get(pre_labeler_name, None)
            
            # RANSAC needs points in lidar frame
            if pre_labeler_name == "ransac_ground":
                # Pass filtered_points_lidar_frame to RANSAC
                ground_mask = pre_labeler_func(
                    filtered_points_global, # Pass global for context if func needs it
                    filtered_points_lidar_frame, # Main input for RANSAC
                    lidar_timestamp, 
                    specific_params # Pass configured RANSAC params
                )
                preliminary_labels[ground_mask] = OcclusionResult.PRELABELED_STATIC_GROUND.value
                self.logger.info(f"RANSAC pre-labeled {np.sum(ground_mask)} points as ground.")
            # Add other pre-labelers here with elif
            # else:
            #    # For generic pre-labelers that might operate on global points
            #    # some_mask = pre_labeler_func(filtered_points_global, lidar_timestamp, specific_params)
            #    # preliminary_labels[some_mask] = Some_Appropriate_Label.value
        
        # --- Add points to DepthImage ---
        # add_points_batch will store filtered_points_global
        # We need to pass the preliminary_labels to be stored in current_di.mdet_labels_for_points
        current_di.add_points_batch(
            points_global_batch=filtered_points_global,
            initial_labels_for_points=preliminary_labels 
        )
        
        self.depth_image_library.add_image(current_di)
        self.logger.info(f"ADD_SWEEP: Added DI for TS {lidar_timestamp}. Lib size now: {len(self.depth_image_library._images)}. Max size: {self.depth_image_library.max_size}") # Changed log level for visibility
        return current_di
        
    # --- Method Imports from other modules ---

    from .occlusion_checks import (
        check_occlusion_pixel_level, 
        check_occlusion_batch,
        check_occlusion_point_level_detailed
    )
    
    from .map_consistency import (
        is_map_consistent
    )

    from .event_tests import (
        execute_test2_parallel_motion,
        execute_test3_perpendicular_motion
    )
    
    # --- Core Processing Logic (from processing.py) ---
    from .processing import process_and_label_di

    def _process_causal_di_wrapper(self, di_to_process_idx: int) -> Dict:
        """Wrapper for causal processing of a specific DI."""
        current_di = self.depth_image_library.get_image_by_index(di_to_process_idx)
        if not current_di:
            self.logger.error(f"Causal Wrapper: Could not get DI at index {di_to_process_idx}.")
            # Return a structure indicating failure but including a processed_di=None
            return {'success': False, 'reason': 'DI not found at index', 
                    'processed_frame_timestamp': None, 'processed_di': None}

        # Pass current_di and its index in the library
        result = self.process_and_label_di(current_di, di_to_process_idx)
        
        result.setdefault('frame_index', di_to_process_idx)
        return result

    # --- The main decision-making method (from processing.py or defined here) ---
    
    def decide_and_process_frame(self, is_end_of_sequence: bool = False) -> Optional[Dict]:
        library_len = len(self.depth_image_library._images)
        # Use INFO or DEBUG, but make it distinct
        self.logger.info(f"DECIDE_FRAME_START: Lib len: {library_len}, Min sweeps needed: {self.min_sweeps_for_processing}, Last processed TS: {self.timestamp_of_last_processed_di}")

        if not self.is_ready_for_processing():
            self.logger.warning(f"DECIDE_FRAME_NOT_READY: Lib len {library_len} < min_sweeps {self.min_sweeps_for_processing}. Returning 'not ready'.") # More specific log
            return {'success': False, 'reason': 'Causal: MDetector not ready (min sweeps)', 'processed_frame_timestamp': None}
        if library_len == 0:
            return {'success': False, 'reason': 'Causal: No images in library', 'processed_frame_timestamp': None}
        
        target_idx_in_deque = library_len - 1 # Index of the latest frame in the deque
        target_di_candidate = self.depth_image_library._images[target_idx_in_deque]

        # Check if this latest DI is newer than the last processed one
        if self.timestamp_of_last_processed_di is None or \
        target_di_candidate.timestamp > self.timestamp_of_last_processed_di:
            
            self.logger.debug(f"Processing CAUSAL DI lib_idx: {target_idx_in_deque} (TS: {target_di_candidate.timestamp}), lib_len: {library_len}, last_proc_TS: {self.timestamp_of_last_processed_di}")
            self.timestamp_of_last_processed_di = target_di_candidate.timestamp # Update with current DI's timestamp
            return self._process_causal_di_wrapper(target_idx_in_deque)
        else:
            # This implies the "latest" frame added was already processed or has same timestamp.
            # Should be rare in normal operation if timestamps are strictly increasing.
            self.logger.info(f"MDetector info: Causal target DI (idx {target_idx_in_deque}, TS: {target_di_candidate.timestamp}) "
                            f"is not newer than last processed TS ({self.timestamp_of_last_processed_di}).")
            return {'success': False, 'reason': 'Causal target DI not newer than last processed', 'processed_frame_timestamp': None}
        

