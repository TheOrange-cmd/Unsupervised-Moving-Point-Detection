# src/core/m_detector/base.py

import logging
from typing import Dict, Any, Optional, List
import time

import numpy as np
import torch

from ..depth_image import DepthImage
from ..depth_image_library import DepthImageLibrary
from ..constants import OcclusionResult
from src.config_loader import MDetectorConfigAccessor
from .pre_labelers import ransac_ground_prelabeler

logger = logging.getLogger(__name__)

class MDetector:
    """
    Main class for the Moving Point Detector (M-Detector).

    This class orchestrates the entire detection pipeline, from managing the
    history of LiDAR sweeps (DepthImages) to applying the sequence of geometric
    tests and refinements to label points as moving or static.
    """

    # --- Method Bindings from other files ---
    # This pattern keeps the base class file clean while logically grouping methods.
    from .occlusion_checks import (
        check_occlusion_batch,
        check_occlusion_point_level_detailed_batch
    )
    from .map_consistency import (
        is_map_consistent, 
        _perform_direct_comparison_gpu
    )
    from .processing import (
        forward,
        _perform_initial_occlusion_pass,
        _apply_map_consistency_check,
        _run_event_test_sequence,
        _apply_frame_refinement,
        _forward_geometric_only,
        _run_single_event_test_with_mcc
    )
    from .event_tests import (
        execute_parallel_motion_away_test, 
        execute_perpendicular_motion_test
    )

    def __init__(self,
                 config_accessor: MDetectorConfigAccessor,
                 device: Optional[torch.device] = None,
                 logger_name: Optional[str] = None):
        """
        Initializes the M-Detector instance.

        Args:
            config_accessor (MDetectorConfigAccessor): An accessor object to retrieve all
                                                       configuration parameters.
            device (Optional[torch.device]): The PyTorch device to run computations on.
                                             Defaults to CPU if not specified.
            logger_name (Optional[str]): A specific name for the logger instance.
        """
        self.config = config_accessor
        self.device = device if device else torch.device('cpu')
        
        log_name_to_use = logger_name if logger_name else f"{__name__}"
        self.logger = logging.getLogger(log_name_to_use)

        # --- Core Components ---
        init_params = self.config.get_initialization_phase_params()
        self.depth_image_library = DepthImageLibrary(
            max_size=init_params['num_sweeps_for_initial_map'] + 1
        )
        self.depth_image_params = self.config.get_depth_image_params()
        self.ransac_params = self.config.get_ransac_ground_params()

        # --- Load all algorithm parameters from config ---
        self._load_configuration()



    def _load_configuration(self):
        """Loads parameters from the config accessor into instance attributes for performance."""
        
        # General
        init_params = self.config.get_initialization_phase_params()
        self.min_sweeps_for_processing = init_params['num_sweeps_for_initial_map']

        # Occlusion Checks
        occ_params = self.config.get_occlusion_determination_params()
        self.epsilon_depth_occlusion = occ_params['epsilon_depth']
        self.neighbor_search_pixels_v = occ_params['pixel_neighborhood_v']
        self.neighbor_search_pixels_h = occ_params['pixel_neighborhood_h']

        # Detailed Occlusion Check (for Event Tests)
        detailed_check_params = self.config.get_detailed_occlusion_check_params()
        self.detailed_check_epsilon_depth = detailed_check_params['epsilon_depth']
        self.detailed_check_angular_threshold_h_rad = np.deg2rad(detailed_check_params['angular_neighborhood_h_deg'])
        self.detailed_check_angular_threshold_v_rad = np.deg2rad(detailed_check_params['angular_neighborhood_v_deg'])
        
        # Adaptive Epsilon
        self.adaptive_eps_config_occ_depth = self.config.get_adaptive_epsilon_config_for_occlusion_depth()

        # Geometric Test Parameters ---
        geo_tests_config = self.config.get_geometric_tests_params()

        # Initial Pass (formerly Test 1 & 4)
        initial_pass_params = geo_tests_config['initial_occlusion_pass']
        self.initial_pass_history_length = initial_pass_params['history_length']
        self.initial_pass_min_occlusion_count = initial_pass_params['min_occlusion_count']

        # Event Tests (formerly Test 2 & 3)
        event_tests_params = geo_tests_config['event_tests']
        
        parallel_away_params = event_tests_params['parallel_motion_away']
        self.parallel_motion_history_length = parallel_away_params['history_length']
        self.parallel_motion_apply_mcc = parallel_away_params['apply_mcc_filter']

        perpendicular_params = event_tests_params['perpendicular_motion']
        self.perpendicular_motion_history_length = perpendicular_params['history_length']
        self.perpendicular_motion_apply_mcc = perpendicular_params['apply_mcc_filter']

        # Map Consistency Check (MCC)
        self.mcc_config = self.config.get_map_consistency_params()
        self.map_consistency_enabled = self.mcc_config['enabled']
        self.num_past_sweeps_for_mcc = self.mcc_config['num_past_sweeps_for_mcc'] 
        self.static_labels_for_map_check_enums = [
            OcclusionResult["OCCLUDED_BY_IMAGE"], OcclusionResult["PRELABELED_STATIC_GROUND"], 
        ]
        self.static_labels_for_map_check_values = [
            label.value for label in self.static_labels_for_map_check_enums
        ]
        self.mc_static_confidence_threshold = self.mcc_config['static_confidence_threshold']

        # Epsilons
        self.epsilon_phi_map_rad = np.deg2rad(self.mcc_config['epsilon_phi_deg'])
        self.epsilon_theta_map_rad = np.deg2rad(self.mcc_config['epsilon_theta_deg'])
        self.epsilon_depth_forward_map = self.mcc_config['epsilon_depth_forward_m']
        self.epsilon_depth_backward_map = self.mcc_config['epsilon_depth_backward_m']
        self.adaptive_eps_config_mc_fwd = self.config.get_adaptive_epsilon_config_for_map_consistency_forward()
        self.adaptive_eps_config_mc_bwd = self.config.get_adaptive_epsilon_config_for_map_consistency_backward()

        # Interpolation in empty regions
        self.mc_interp_enabled = self.mcc_config['interpolation_enabled']
        self.mc_interp_max_neighbors = self.mcc_config['interpolation_max_neighbors_to_consider']
        self.mc_interp_max_triplets = self.mcc_config['interpolation_max_triplets_to_try']

    def add_sweep(self,
                  points_global_raw: np.ndarray,
                  points_sensor_raw: np.ndarray,
                  pose_global: np.ndarray,
                  timestamp: float) -> DepthImage:
        """
        Adds a new LiDAR sweep to the detector's history.

        This method creates a new DepthImage, populates it with the provided
        point cloud, and adds it to the internal library.

        Args:
            points_global_raw (np.ndarray): Raw point cloud in global frame, shape (N, 3).
            points_sensor_raw (np.ndarray): Raw point cloud in sensor frame, shape (N, 3).
            pose_global (np.ndarray): 4x4 pose matrix for this sweep.
            timestamp (float): Timestamp of the sweep in microseconds.
        
        Returns:
            DepthImage: The newly created and populated DepthImage object.
        """
        start_time = time.time()

        di = DepthImage(
            image_pose_global=pose_global,
            depth_image_params=self.depth_image_params,
            timestamp=timestamp,
            device=self.device
        )

        # The DI will initialize all points to UNDETERMINED by default.
        num_added = di.add_points_batch(
            points_global_raw,
            points_sensor_raw,
            self.config.get_point_pre_filtering_params()
        )
        self.depth_image_library.add_image(di)

        end_time = time.time()
        self.logger.debug(f"ADD_SWEEP: Added DI with {num_added} points. Took {(end_time - start_time) * 1000:.2f} ms.")
        return di
    
    def _run_pre_labeling_pipeline(self, current_di: DepthImage):
        """
        Internal method to run all configured pre-labelers on a DepthImage.
        This modifies the labels within the DepthImage object in-place.
        """
        # --- RANSAC Ground Pre-labeling ---
        if self.ransac_params.get('enabled', False) and current_di.num_points > 0:
            # RANSAC needs the sensor-frame points, which the DI doesn't store.
            # We can re-transform the global points back to the sensor frame.
            # This is a small price for better encapsulation.
            points_global_filtered = current_di.original_points_global_coords
            points_local_filtered_h = torch.cat([points_global_filtered, torch.ones_like(points_global_filtered[:, :1])], dim=1) @ current_di.matrix_local_from_global.T
            points_sensor_filtered_np = points_local_filtered_h[:, :3].cpu().numpy()

            ground_mask_filtered = ransac_ground_prelabeler(
                points_lidar_frame=points_sensor_filtered_np,
                ransac_params=self.ransac_params,
                device_str=self.device.type
            )
            
            # Update the labels in the DepthImage directly
            current_di.mdet_labels_for_points[ground_mask_filtered] = OcclusionResult.PRELABELED_STATIC_GROUND.value
            self.logger.debug(f"PRE-LABELING: Applied RANSAC, found {np.sum(ground_mask_filtered)} ground points.")

        # --- Future Pre-labelers would go here ---
        # if self.wall_detector_params.get('enabled', False):
        #     ...

    def process_latest_sweep(self) -> Dict[str, Any]:
        """
        Processes the most recently added sweep in the depth image library.

        This method checks if there is sufficient historical data, retrieves the
        latest sweep, and then executes the full forward pass of the M-Detector
        algorithm on it.

        Returns:
            Dict[str, Any]: A dictionary containing the processing results.
                            If successful, includes 'success': True, 'processed_di',
                            and label counts. If not ready, includes 'success': False
                            and a 'reason' string.
        """
        # 1. Check if the library has enough history to provide context.
        lib_len = len(self.depth_image_library)
        if lib_len < self.min_sweeps_for_processing:
            return {
                'success': False,
                'reason': f"Cannot process: Library size ({lib_len}) is less than min required ({self.min_sweeps_for_processing})."
            }

        # 2. Retrieve the most recent sweep to be processed.
        # The DI to process is always the last one in the causal sequence.
        di_to_process_idx = lib_len - 1
        current_di = self.depth_image_library.get_image_by_index(di_to_process_idx)

        if current_di is None:
            reason = f"Could not retrieve DI at index {di_to_process_idx} from library."
            self.logger.error(f"PROCESS_SWEEP_ERROR: {reason}")
            # This indicates a critical logic error, so we should raise an exception.
            raise RuntimeError(reason)
        
        # Run pre-labeling as the first step of processing ---
        self._run_pre_labeling_pipeline(current_di)

        self.logger.debug(f"PROCESS_SWEEP_READY: Processing DI at index {di_to_process_idx} (TS: {current_di.timestamp:.2f}).")

        # 3. Execute the forward pass 
        result_dict = self.forward(current_di, di_to_process_idx)
        
        self.logger.debug(f"Forward pass for DI @ TS {current_di.timestamp:.2f} DONE.")
        
        return result_dict
    
    def get_labels_before_refinement(self, current_di: 'DepthImage', current_di_idx_in_lib: int) -> torch.Tensor:
        """
        Runs the geometric pipeline but stops before the final refinement step.
        This is used by the 'bake' process to cache intermediate results.

        Args:
            current_di (DepthImage): The DI to process.
            current_di_idx_in_lib (int): The index of the DI in the library.

        Returns:
            torch.Tensor: The labels after all geometric tests, shape (N,).
        """
        return self._forward_geometric_only(current_di, current_di_idx_in_lib)