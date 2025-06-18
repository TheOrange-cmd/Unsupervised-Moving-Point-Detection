# src/core/m_detector/base.py

import logging
from typing import Deque, Dict, Any, Optional, List
import collections
import time

import numpy as np
import torch

from ..depth_image import DepthImage
from ..depth_image_library import DepthImageLibrary
from ..constants import OcclusionResult
from src.config_loader import MDetectorConfigAccessor
# from ..debug_collector import PointDebugCollector, NoOpDebugCollector

logger = logging.getLogger(__name__)

class MDetector:
    """
    Main class for the Moving Point Detector (M-Detector).
    """

    # --- Method Bindings ---
    from .occlusion_checks import (
        check_occlusion_batch,
        check_occlusion_point_level_detailed_batch
    )
    from .map_consistency import is_map_consistent
    from .processing import process_and_label_di
    from .event_tests import execute_test2_parallel_motion, execute_test3_perpendicular_motion

    def __init__(self,
                 config_accessor: MDetectorConfigAccessor,
                 is_legacy: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initializes the M-Detector instance.
        """
        self.config = config_accessor
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.is_legacy = is_legacy
        self.logger = logging.getLogger(f"{__name__}.{'Legacy' if is_legacy else 'Torch'}")

        # --- Core Components ---
        self.depth_image_library = DepthImageLibrary(
            max_size=self.config.get_depth_image_params()['library_size']
        )
        self.depth_image_params = self.config.get_depth_image_params()

        # --- Debugging ---
        # self.debug_collector: PointDebugCollector = NoOpDebugCollector()

        # --- Configuration Parameters ---
        self._load_configuration()

    def _load_configuration(self):
        """Loads parameters from the config accessor into instance attributes."""
        # CORRECTED: All keys now match the provided config.yml file exactly.
        
        # General
        self.min_sweeps_for_processing = self.config.get_initialization_phase_params()['num_sweeps_for_initial_map']

        # Occlusion Checks
        occ_params = self.config.get_occlusion_determination_params()
        self.epsilon_depth_occlusion = occ_params['epsilon_depth']
        self.neighbor_search_pixels_v = occ_params['pixel_neighborhood_v']
        self.neighbor_search_pixels_h = occ_params['pixel_neighborhood_h']

        # Detailed Occlusion Check (for Event Tests)
        self.detailed_check_epsilon_depth = occ_params['epsilon_depth'] # Uses the same base epsilon
        self.detailed_check_angular_threshold_h_rad = np.deg2rad(occ_params['angular_neighborhood_h_deg'])
        self.detailed_check_angular_threshold_v_rad = np.deg2rad(occ_params['angular_neighborhood_v_deg'])
        
        # Adaptive Epsilon
        self.adaptive_eps_config_occ_depth = self.config.get_adaptive_epsilon_config_for_occlusion_depth()

        # Test 1 & 4
        test4_params = self.config.get_test4_perpendicular_params()
        self.test1_N_historical_DIs = test4_params['num_historical_DIs_N']
        self.test1_M4_threshold = test4_params['min_occluding_DIs_M4']

        # Event Tests (Test 2 & 3)
        self.test2_M2_depth_images = self.config.get_test2_parallel_away_params()['num_historical_DIs_M2']
        self.test3_M3_depth_images = self.config.get_test3_parallel_towards_params()['num_historical_DIs_M3']

        # Map Consistency Check (MCC)
        self.mcc_config = self.config.get_map_consistency_params()
        self.map_consistency_enabled = self.mcc_config['enabled']
        self.num_past_sweeps_for_mcc = self.mcc_config['num_past_sweeps_for_mcc'] 
        self.static_labels_for_map_check_enums = [
            OcclusionResult[label] for label in self.mcc_config['static_labels_for_map_check']
        ]
        self.static_labels_for_map_check_values = [
            label.value for label in self.static_labels_for_map_check_enums
        ]

    # def set_debug_collector(self, collector: PointDebugCollector):
    #     """Sets a debug collector for tracing specific points."""
    #     self.logger.info(f"Setting debug collector: {collector}")
    #     self.debug_collector = collector

    def add_sweep(self,
                  points_global: np.ndarray,
                  pose_global: np.ndarray,
                  timestamp: float,
                  prelabeled_mask: Optional[np.ndarray] = None) -> DepthImage:
        """
        Adds a new sweep to the detector.
        """
        start_time = time.time()

        di = DepthImage(
            image_pose_global=pose_global,
            depth_image_params=self.depth_image_params,
            timestamp=timestamp,
            device=self.device
        )

        initial_labels = None
        if prelabeled_mask is not None:
            initial_labels = np.full(points_global.shape[0], OcclusionResult.UNDETERMINED.value, dtype=np.int8)
            initial_labels[prelabeled_mask] = OcclusionResult.PRELABELED_STATIC_GROUND.value

        num_added = di.add_points_batch(points_global, initial_labels)
        self.depth_image_library.add_image(di)

        end_time = time.time()
        self.logger.debug(
            f"ADD_SWEEP: Added DI with {num_added} points. "
            f"TS: {timestamp:.2f}, Lib size: {len(self.depth_image_library)}. "
            f"Took {(end_time - start_time) * 1000:.2f} ms."
        )
        return di

    def decide_and_process_frame(self) -> Dict[str, Any]:
        """
        Decides which frame in the library to process and initiates processing.
        """
        lib_len = len(self.depth_image_library)
        if lib_len < self.min_sweeps_for_processing:
            reason = f"Lib len {lib_len} < min_sweeps {self.min_sweeps_for_processing}"
            self.logger.info(f"DECIDE_FRAME_NOT_READY: {reason}. Returning 'not ready'.")
            return {'success': False, 'reason': reason}

        target_idx_in_deque = lib_len - 1
        di_to_process = self.depth_image_library.get_image_by_index(target_idx_in_deque)
        if di_to_process is None:
            reason = f"Could not retrieve DI at index {target_idx_in_deque} from library."
            self.logger.error(f"DECIDE_FRAME_ERROR: {reason}")
            return {'success': False, 'reason': reason}

        self.logger.info(
            f"DECIDE_FRAME_READY: Processing DI at index {target_idx_in_deque} "
            f"(TS: {di_to_process.timestamp:.2f})."
        )
        return self._process_causal_di_wrapper(target_idx_in_deque)

    def _process_causal_di_wrapper(self, di_to_process_idx: int) -> Dict[str, Any]:
        """
        A wrapper around the main processing logic that handles timing and error catching.
        """
        current_di = self.depth_image_library.get_image_by_index(di_to_process_idx)
        if current_di is None:
            return {'success': False, 'reason': f"DI at index {di_to_process_idx} is None."}

        start_time = time.time()
        try:
            result = self.process_and_label_di(current_di, di_to_process_idx)
        except Exception as e:
            self.logger.exception(f"Exception during process_and_label_di for DI @ TS {current_di.timestamp:.2f}")
            return {'success': False, 'reason': str(e), 'timestamp': current_di.timestamp}
        finally:
            end_time = time.time()
            self.logger.info(
                f"PROCESS_DI_TIMING: Processing DI @ TS {current_di.timestamp:.2f} "
                f"took {(end_time - start_time) * 1000:.2f} ms."
            )
        return result