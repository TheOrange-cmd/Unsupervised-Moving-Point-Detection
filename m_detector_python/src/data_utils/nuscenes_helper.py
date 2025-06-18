# src/data_utils/nuscenes_helper.py

import numpy as np
from typing import Dict, Optional
import logging
import ray
import math

# Project-specific imports
from ..core.m_detector.base import MDetector
from ..config_loader import MDetectorConfigAccessor
from ..core.m_detector.pre_labelers import ransac_ground_prelabeler
from ..core.constants import OcclusionResult

class NuScenesProcessor:
    """
    Handles the core logic for processing a single NuScenes scene for one tuning configuration.
    This class is instantiated within a Ray actor.
    """
    def __init__(self, data_actor: ray.actor.ActorHandle, config_accessor: MDetectorConfigAccessor,
                 progress_actor: Optional[ray.actor.ActorHandle], 
                 worker_id: int, logger_name: str):
        """
        Initializes the processor.
        """
        self.data_actor = data_actor
        self.config_accessor = config_accessor 
        self.ransac_params = self.config_accessor.get_ransac_ground_params()
        self.ransac_enabled = self.ransac_params['enabled']
        self.filter_params = self.config_accessor.get_point_pre_filtering_params()
        self.filtering_enabled = self.filter_params['enabled']
        
        self.progress_actor = progress_actor
        self.worker_id = worker_id
        
        self.logger = logging.getLogger(logger_name)

    def process_scene(self, scene_index: int, detector: MDetector, task_description: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Processes all sweeps in a given scene, fetches corresponding ground truth from the
        data actor, and returns a dictionary containing both predictions and GT for metric calculation.
        """
        # --- 1. Setup and Sweep Selection ---
        scene_rec = ray.get(self.data_actor.get_scene_record.remote(scene_index))
        scene_token = scene_rec['token']
        scene_name = scene_rec['name']
        
        processing_settings = self.config_accessor.get_processing_settings()
        skip_frames = processing_settings['skip_frames']
        max_frames_config = processing_settings['max_frames']
        
        nuscenes_params = self.config_accessor.get_nuscenes_params()
        lidar_name = nuscenes_params['lidar_sensor_name']

        all_sweep_tokens = ray.get(self.data_actor.get_scene_sweep_tokens.remote(scene_token, lidar_name=lidar_name))
        
        if not all_sweep_tokens:
            self.logger.warning(f"No sweeps found for scene {scene_name}. Skipping.")
            return None

        start_idx = min(skip_frames, len(all_sweep_tokens))
        end_idx = len(all_sweep_tokens)
        if max_frames_config is not None and max_frames_config >= 0:
            end_idx = min(start_idx + max_frames_config, len(all_sweep_tokens))
        sweeps_to_process_tokens = all_sweep_tokens[start_idx:end_idx]

        if not sweeps_to_process_tokens:
            self.logger.warning(f"No sweeps selected for processing in scene {scene_name} based on config.")
            return None
            
        # --- 2. Main Processing Loop ---
        all_processed_predictions = []
        all_processed_gt_labels = []

        self.logger.debug(f"Prefetching data for {len(sweeps_to_process_tokens)} sweeps...")
        sweep_data_futures = [
            self.data_actor.get_sweep_data_by_token.remote(token) 
            for token in sweeps_to_process_tokens
        ]
        all_sweep_data = ray.get(sweep_data_futures)
        self.logger.debug("Sweep data prefetching complete.")

        for i, sweep_data in enumerate(all_sweep_data):
            # --- Point Filtering (same as before) ---
            points_sensor_raw = sweep_data['points_sensor_frame']
            points_sensor_filtered = points_sensor_raw
            if self.filtering_enabled and points_sensor_raw.shape[0] > 0:
                ranges = np.linalg.norm(points_sensor_raw[:, :3], axis=1)
                min_range = self.filter_params['min_range_meters']
                max_range = self.filter_params['max_range_meters']
                range_mask = (ranges >= min_range) & (ranges <= max_range)
                points_sensor_filtered = points_sensor_raw[range_mask]

            if points_sensor_filtered.shape[0] == 0:
                continue

            # --- RANSAC Pre-labeling (same as before) ---
            T_global_sensor = sweep_data['T_global_lidar']
            points_global = (T_global_sensor[:3, :3] @ points_sensor_filtered.T).T + T_global_sensor[:3, 3]
            prelabeled_mask = None
            if self.ransac_enabled:
                prelabeled_mask = ransac_ground_prelabeler(
                    points_global=points_global, 
                    points_lidar_frame=points_sensor_filtered, 
                    current_di_timestamp=float(sweep_data['timestamp']),
                    ransac_params=self.ransac_params, 
                    device_str=detector.device.type
                )

            # --- Run M-Detector (same as before) ---
            detector.add_sweep(
                points_global=points_global.astype(np.float32),
                pose_global=T_global_sensor.astype(np.float32),
                timestamp=float(sweep_data['timestamp']),
                prelabeled_mask=prelabeled_mask 
            )
            mdet_result = detector.decide_and_process_frame()
            
            # --- 3. Package Predictions with Ground Truth ---
            if mdet_result and mdet_result.get('success'):
                processed_di = mdet_result.get('processed_di')
                if processed_di and processed_di.total_points_added_to_di_arrays > 0:
                    
                    # A. Get predictions from the detector
                    prediction_labels = processed_di.mdet_labels_for_points.cpu().numpy()
                    
                    # B. Get corresponding GT data from the actor's cache
                    current_sweep_abs_idx = start_idx + i
                    gt_labels_for_sweep = ray.get(self.data_actor.get_ground_truth_slice.remote(scene_token, current_sweep_abs_idx))

                    # C. Validate and store the paired data
                    if gt_labels_for_sweep is None:
                        self.logger.warning(f"No GT data found in cache for scene {scene_name}, sweep index {current_sweep_abs_idx}. Skipping sweep.")
                        continue
                    
                    if gt_labels_for_sweep.shape[0] != len(prediction_labels):
                        self.logger.warning(f"FATAL ALIGNMENT ERROR in scene {scene_name}, sweep {i}. "
                                            f"GT has {gt_labels_for_sweep.shape[0]} points, but "
                                            f"MDet has {len(prediction_labels)} points. This indicates a "
                                            f"discrepancy between devkit and GT generation. Skipping sweep.")
                        continue

                    all_processed_gt_labels.append(gt_labels_for_sweep)
                    all_processed_predictions.append(prediction_labels)

        # --- 4. Final Aggregation ---
        if not all_processed_predictions:
            self.logger.warning(f"No successful M-Detector outputs with matching GT were collected for scene {scene_name}.")
            return None

        # Concatenate all the collected data into two large arrays
        final_predictions = np.concatenate(all_processed_predictions)
        final_gt_labels = np.concatenate(all_processed_gt_labels)

        # Return a simple dictionary containing the final, aligned arrays
        data_to_save = {
            'predictions': final_predictions,
            'ground_truth': final_gt_labels,
            'scene_name': scene_name
        }
        return data_to_save