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
    """
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.depth_image_library = DepthImageLibrary(
            max_size=config.get('depth_image', {}).get('library_size', 20) # Default if not specified
        )
        self.min_sweeps_for_processing = config.get('initialization', {}).get('num_initial_sweeps_for_map', 1)
        
        # Load configuration sections
        self._load_occlusion_check_config()
        self._load_map_consistency_config()
        self._load_temporal_processing_config() # This will set self.use_bidirectional

        # State for processing flow
        self.timestamp_of_last_processed_di: Optional[float] = None 
        self.current_lidar_sd_token: Optional[str] = None # Still useful for MDetector's internal reference if needed

        self.logger.info("MDetector initialized successfully.")
    
    def reset_scene_state(self):
        """Resets internal state for processing a new scene."""
        self.depth_image_library.clear()
        self.timestamp_of_last_processed_di = None
        self.current_lidar_sd_token = None
        self.logger.info("MDetector state reset for new scene.")

    def _load_occlusion_check_config(self):
        oc_cfg = self.config.get('occlusion_check', {})
        self.epsilon_depth_occlusion = oc_cfg.get('epsilon_depth_occlusion', 0.3)
        self.neighbor_search_pixels_h = oc_cfg.get('neighbor_search_pixels_h', 1)
        self.neighbor_search_pixels_v = oc_cfg.get('neighbor_search_pixels_v', 1)
        # Angular thresholds are not used in the provided batch check, but keep if used elsewhere
        # self.angular_threshold_rad_h = np.deg2rad(oc_cfg.get('angular_threshold_deg_h', 2.0))
        # self.angular_threshold_rad_v = np.deg2rad(oc_cfg.get('angular_threshold_deg_v', 2.0))
    
    def _load_map_consistency_config(self):
        mc_cfg = self.config.get('map_consistency_check', {})
        self.map_consistency_enabled = mc_cfg.get('enabled', True)
        self.num_historical_di_for_map_check = mc_cfg.get('num_historical_di_for_map_check', 3)
        self.num_future_di_for_map_check = mc_cfg.get('num_future_di_for_map_check', 3) # Used by is_map_consistent
        self.epsilon_phi_map_rad = np.deg2rad(mc_cfg.get('epsilon_phi_map_deg', 1.0))
        self.epsilon_theta_map_rad = np.deg2rad(mc_cfg.get('epsilon_theta_map_deg', 1.0))
        self.epsilon_depth_forward_map = mc_cfg.get('epsilon_depth_forward_map', 0.3)
        self.epsilon_depth_backward_map = mc_cfg.get('epsilon_depth_backward_map', 0.3)
        self.map_consistency_threshold = mc_cfg.get('consistency_threshold', 0.5) # Used by is_map_consistent

        config_label_strings = mc_cfg.get('static_labels_for_map_check', [])
        self.static_labels_for_map_check = []
        if not isinstance(config_label_strings, list):
            self.logger.warning(f"'static_labels_for_map_check' in config is not a list. Found: {config_label_strings}.")
            config_label_strings = []
        for label_str in config_label_strings:
            norm_str = label_str.strip().upper()
            if norm_str == "NON_EVENT": self.static_labels_for_map_check.append("non_event")
            elif norm_str == "PENDING_CLASSIFICATION": self.static_labels_for_map_check.append("pending_classification")
            elif hasattr(OcclusionResult, norm_str): self.static_labels_for_map_check.append(OcclusionResult[norm_str])
            else: self.static_labels_for_map_check.append(label_str) # Keep as string if not an enum
        self.logger.debug(f"Parsed static_labels_for_map_check: {self.static_labels_for_map_check}")

    def _load_temporal_processing_config(self):
        temp_cfg = self.config.get('temporal_processing', {})
        self.use_bidirectional = temp_cfg.get('bidirectional', False)
        
        # Bidirectional window parameters (used by decide_and_process_frame)
        self.bidirectional_window_size = temp_cfg.get('bidirectional_window_size', 3)
        if self.bidirectional_window_size < 3 or self.bidirectional_window_size % 2 == 0:
            self.logger.warning(f"Bidirectional window size must be an odd number >= 3. Got {self.bidirectional_window_size}. Defaulting to 3.")
            self.bidirectional_window_size = 3
        # Offset from the latest frame in a full window to the center frame.
        # e.g., window [F0, F1, F2], latest is F2 (idx 2). Center F1 (idx 1). Offset = 2-1 = 1.
        self.bidirectional_center_offset_from_latest = (self.bidirectional_window_size - 1) // 2

        # These seem like parameters for a different temporal aggregation logic (_determine_final_label)
        # Keep them if that logic is still used, or remove if superseded by _determine_final_label_bidirectional_simplified
        self.past_window_size_temporal_agg = temp_cfg.get('past_window_size', 5) # Renamed to avoid clash
        self.future_window_size_temporal_agg = temp_cfg.get('future_window_size', 5) # Renamed
        self.past_weight_temporal_agg = temp_cfg.get('past_weight', 1.0) # Renamed
        self.future_weight_temporal_agg = temp_cfg.get('future_weight', 1.0) # Renamed
    
    def is_ready_for_processing(self) -> bool:
        """
        Checks if enough DepthImages have been collected for basic (causal) processing.
        Bidirectional processing has its own readiness check within decide_and_process_frame.
        """
        return len(self.depth_image_library._images) >= self.min_sweeps_for_processing

    def add_sweep_and_create_depth_image(self, 
                                        points_lidar_frame: np.ndarray, 
                                        T_global_lidar: np.ndarray, 
                                        lidar_timestamp: float, # Changed to _us for clarity
                                        lidar_sd_token: Optional[str] = None) -> DepthImage: # Added token
        """
        Creates a DepthImage from a new LiDAR sweep and adds it to the library.
        """
        self.current_lidar_sd_token = lidar_sd_token # Store if needed for context

        current_di = DepthImage(
            image_pose_global=T_global_lidar,
            config=self.config, # Pass DI specific config
            timestamp=lidar_timestamp,
            # Optionally pass lidar_sd_token to DepthImage if it needs to store it
            # lidar_sd_token=lidar_sd_token 
        )

        points_lidar_frame_h = np.hstack((points_lidar_frame, np.ones((points_lidar_frame.shape[0], 1))))
        points_global_h = (T_global_lidar @ points_lidar_frame_h.T).T
        points_global = points_global_h[:, :3]

        max_range = self.config.get('filtering',{}).get('max_point_range_meters', 80.0)
        min_range = self.config.get('filtering',{}).get('min_point_range_meters', 1.0)
        
        ranges = np.linalg.norm(points_lidar_frame, axis=1) # Ranges calculated in sensor frame
        range_mask = (min_range <= ranges) & (ranges <= max_range)
        
        filtered_points_global = points_global[range_mask]
        
        # Initial label for points in a new DI
        # This might be "pending_classification" or "non_event" based on readiness
        # The actual processing functions (causal/bidirectional) will update these.
        initial_label_for_new_di_points = "pending_classification" 
        # Or: initial_label_for_new_di_points = OcclusionResult.UNDETERMINED
        # This initial label is important for map_consistency checks if they run before full processing.
        # The `process_and_label_di` and `process_and_label_di_bidirectional` functions
        # are responsible for setting the final pt_info['label'].
        
        batch_labels = [initial_label_for_new_di_points] * len(filtered_points_global)
        
        current_di.add_points_batch(
            points_global_batch=filtered_points_global,
            labels=batch_labels # Pass the initial labels
        )
        
        self.depth_image_library.add_image(current_di)
        self.logger.debug(f"Added DI for timestamp {lidar_timestamp}, library size: {len(self.depth_image_library._images)}")
        return current_di
        
    # --- Method Imports from other modules ---
    # These will bind the functions from those files as methods of this class.
    # Make sure the first argument of these functions is `self`.

    from .occlusion_checks import (
        check_occlusion_pixel_level, # If still used directly
        check_occlusion_batch
    )
    
    from .map_consistency import (
        is_map_consistent
    )
    
    # --- Core Processing Logic (from processing.py) ---
    # `process_and_label_di` is the main causal processing function.
    # `process_frame` is replaced by `decide_and_process_frame`.
    # `_process_causal_di` and `_process_bidirectional_di` are new helper wrappers.
    
    # This is your original causal processing function.
    # It will be called by _process_causal_di_wrapper.
    from .processing import process_and_label_di as actual_causal_processing_logic

    def _process_causal_di_wrapper(self, di_to_process_idx: int) -> Dict:
        """Wrapper for causal processing of a specific DI."""
        current_di = self.depth_image_library._images[di_to_process_idx]
        historical_di = None
        if di_to_process_idx > 0:
            historical_di = self.depth_image_library._images[di_to_process_idx - 1]
        
        # Call the imported actual_causal_processing_logic
        # This function (process_and_label_di) needs to be defined in processing.py
        # and take (self, current_di, historical_di)
        result = self.actual_causal_processing_logic(current_di, historical_di) 
        
        # Ensure result dict has standard fields
        result['processed_frame_timestamp'] = current_di.timestamp
        result['frame_index'] = di_to_process_idx
        result.setdefault('success', True) # Assume success if no explicit failure
        result.setdefault('label_counts', {}) # Ensure label_counts exists
        return result

    # --- Temporal Processing Logic (from temporal.py) ---
    # `process_and_label_di_bidirectional` is the main bidirectional function.
    # `_determine_final_label` and `_determine_final_label_bidirectional_simplified` are helpers.
    from .temporal import (
        process_and_label_di_bidirectional as actual_bidirectional_processing_logic,
        _determine_final_label, # If used by a different temporal strategy
        _determine_final_label_bidirectional_simplified
    )

    def _process_bidirectional_di_wrapper(self, di_to_process_idx: int) -> Dict:
        """Wrapper for bidirectional processing of a specific DI."""
        # Call the imported actual_bidirectional_processing_logic
        # This function (process_and_label_di_bidirectional) needs to be defined in temporal.py
        # and take (self, center_index) which is di_to_process_idx here.
        result = self.actual_bidirectional_processing_logic(di_to_process_idx)
        
        # Ensure result dict has standard fields
        if result.get('success'): # Only add these if successful
            result['processed_frame_timestamp'] = self.depth_image_library._images[di_to_process_idx].timestamp
            # 'frame_index' should already be set by actual_bidirectional_processing_logic
            result.setdefault('label_counts', {}) # Ensure label_counts exists
        return result

    # --- The main decision-making method (from processing.py or defined here) ---
    # This was previously named `decide_and_process_frame`
    # Let's assume it's defined in processing.py and imported, or defined directly here.
    # For clarity, defining it here based on the previous logic.
    
    def decide_and_process_frame(self, is_end_of_sequence: bool = False) -> Optional[Dict]:
        library_len = len(self.depth_image_library._images)

        if self.use_bidirectional:
            # --- Bidirectional Logic ---
            if is_end_of_sequence:
                # --- Flushing Logic (Timestamp-based) ---
                next_di_to_flush: Optional[DepthImage] = None
                idx_of_next_di_to_flush: Optional[int] = None
                min_eligible_timestamp = float('inf')

                current_library_snapshot = list(self.depth_image_library._images) # Iterate a snapshot

                for i, di_candidate in enumerate(current_library_snapshot):
                    # Check if this DI is newer than the last one processed
                    is_newer = (self.timestamp_of_last_processed_di is None or 
                                di_candidate.timestamp > self.timestamp_of_last_processed_di)
                    
                    if is_newer and di_candidate.timestamp < min_eligible_timestamp:
                        min_eligible_timestamp = di_candidate.timestamp
                        # Find the actual current index in the deque, as snapshot indices might not match if deque changed
                        try:
                            # This index is relative to the current state of the deque
                            current_idx_in_deque = self.depth_image_library._images.index(di_candidate)
                            next_di_to_flush = di_candidate
                            idx_of_next_di_to_flush = current_idx_in_deque
                        except ValueError:
                            self.logger.warning(f"Flush: DI (TS {di_candidate.timestamp}) from snapshot not found in current deque. Skipping.")
                            continue
                
                if next_di_to_flush is not None and idx_of_next_di_to_flush is not None:
                    self.logger.debug(f"Flushing BI   DI lib_idx: {idx_of_next_di_to_flush} (TS: {next_di_to_flush.timestamp}), lib_len: {library_len}, last_proc_TS: {self.timestamp_of_last_processed_di}")
                    self.timestamp_of_last_processed_di = next_di_to_flush.timestamp # Update with current DI's timestamp
                    return self._process_bidirectional_di_wrapper(idx_of_next_di_to_flush)
                else:
                    self.logger.info("Bidirectional flush: No more DIs to process or library empty.")
                    return None # Fully flushed or nothing to flush initially

            else: # --- Normal Bidirectional Operation (Not end_of_sequence) ---
                if library_len < self.bidirectional_window_size:
                    return {'success': False, 'reason': 'Bidirectional buffer not full yet', 'processed_frame_timestamp': None}
                
                # This index is into the current state of the deque
                potential_target_idx_in_deque = library_len - 1 - self.bidirectional_center_offset_from_latest
                
                # Safety check for the calculated index (should always be valid if lib_len >= window_size)
                if not (0 <= potential_target_idx_in_deque < library_len):
                    self.logger.error(f"Logic error: potential_target_idx {potential_target_idx_in_deque} out of bounds for lib_len {library_len}. Window: {self.bidirectional_window_size}, Offset: {self.bidirectional_center_offset_from_latest}")
                    return {'success': False, 'reason': 'Internal logic error calculating target index', 'processed_frame_timestamp': None}

                target_di_candidate = self.depth_image_library._images[potential_target_idx_in_deque]
                
                # Check if the DI at the target slot is newer than the last one processed
                if self.timestamp_of_last_processed_di is None or \
                target_di_candidate.timestamp > self.timestamp_of_last_processed_di:
                    
                    self.logger.debug(f"Processing BI   DI lib_idx: {potential_target_idx_in_deque} (TS: {target_di_candidate.timestamp}), lib_len: {library_len}, last_proc_TS: {self.timestamp_of_last_processed_di}")
                    self.timestamp_of_last_processed_di = target_di_candidate.timestamp # Update with current DI's timestamp
                    return self._process_bidirectional_di_wrapper(potential_target_idx_in_deque)
                else:
                    # The DI at the target slot is not newer. This means the window hasn't "slid" enough
                    # to present a new frame for processing at that slot, or timestamps are equal.
                    self.logger.info(f"MDetector info: Bidirectional target DI (idx {potential_target_idx_in_deque}, TS: {target_di_candidate.timestamp}) "
                                    f"is not newer than last processed TS ({self.timestamp_of_last_processed_di}).")
                    return {'success': False, 'reason': 'Bidirectional target DI not newer than last processed', 'processed_frame_timestamp': None}

        else: # --- Causal Logic (Process latest frame) ---
            if not self.is_ready_for_processing(): # Basic readiness (e.g. min sweeps)
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
