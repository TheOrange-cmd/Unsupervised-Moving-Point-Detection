# src/data_utils/nuscenes_helper.py

import numpy as np
from typing import Dict, Optional
import logging
import ray
import torch

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
        
        self.progress_actor = progress_actor
        self.worker_id = worker_id
        
        self.logger = logging.getLogger(logger_name)

    def _get_prelabeled_ground_mask(self, points_sensor_raw: np.ndarray, points_global_raw: np.ndarray, 
                                    timestamp: float, device: torch.device) -> Optional[np.ndarray]:
        """
        Orchestrates the pre-labeling of ground points. This involves:
        1. Filtering the raw point cloud to the desired range.
        2. Running a ground segmentation algorithm (RANSAC) on the filtered points.
        3. Reconstructing a boolean mask aligned with the original raw point cloud.

        This keeps the core `process_scene` loop clean and maintains separation of concerns.

        Returns:
            An optional boolean numpy array with the same size as `points_sensor_raw`,
            where `True` indicates a point pre-labeled as ground.
        """
        if not self.ransac_enabled:
            return None

        # 1. Filter the points. This is the place to modify for future ego-vehicle models.
        filter_params = self.config_accessor.get_point_pre_filtering_params()
        ranges = np.linalg.norm(points_sensor_raw, axis=1)
        filter_mask = (ranges >= filter_params['min_range_meters']) & (ranges <= filter_params['max_range_meters'])
        
        points_sensor_filtered = points_sensor_raw[filter_mask]
        points_global_filtered = points_global_raw[filter_mask]
        
        if points_sensor_filtered.shape[0] == 0:
            return np.zeros(points_sensor_raw.shape[0], dtype=bool)

        # 2. Run RANSAC on the filtered data.
        ground_mask_filtered = ransac_ground_prelabeler(
            points_global=points_global_filtered, 
            points_lidar_frame=points_sensor_filtered,
            current_di_timestamp=timestamp,
            ransac_params=self.ransac_params, 
            device_str=device.type
        )

        # 3. Reconstruct the mask to align with the raw point cloud.
        ground_mask_raw = np.zeros(points_sensor_raw.shape[0], dtype=bool)
        ground_mask_raw[filter_mask] = ground_mask_filtered
        
        return ground_mask_raw

    def process_scene(self, scene_index: int, detector: MDetector, task_description: str) -> Optional[Dict[str, np.ndarray]]:
        scene_rec = ray.get(self.data_actor.get_scene_record.remote(scene_index))
        scene_token = scene_rec['token']
        scene_name = scene_rec['name']
        processing_settings = self.config_accessor.get_processing_settings()
        skip_frames = processing_settings['skip_frames']
        max_frames_config = processing_settings['max_frames']
        lidar_name = self.config_accessor.get_nuscenes_params()['lidar_sensor_name']
        all_sweep_tokens = ray.get(self.data_actor.get_scene_sweep_tokens.remote(scene_token, lidar_name=lidar_name))
        if not all_sweep_tokens: return None
        start_idx = min(skip_frames, len(all_sweep_tokens))
        end_idx = len(all_sweep_tokens)
        if max_frames_config is not None and max_frames_config >= 0:
            end_idx = min(start_idx + max_frames_config, len(all_sweep_tokens))
        sweeps_to_process_tokens = all_sweep_tokens[start_idx:end_idx]
        if not sweeps_to_process_tokens: return None
        results_for_validation = []
        sweep_data_futures = [self.data_actor.get_sweep_data_by_token.remote(token) for token in sweeps_to_process_tokens]
        all_sweep_data = ray.get(sweep_data_futures)

        for i, sweep_data in enumerate(all_sweep_data):
            points_sensor_raw = sweep_data['points_sensor_frame']
            if points_sensor_raw.shape[0] == 0:
                continue

            T_global_sensor = sweep_data['T_global_lidar']
            points_global_raw = (T_global_sensor[:3, :3] @ points_sensor_raw.T).T + T_global_sensor[:3, 3]
            
            # --- CLEANED UP PRE-LABELING STEP ---
            prelabeled_mask_raw = self._get_prelabeled_ground_mask(
                points_sensor_raw=points_sensor_raw,
                points_global_raw=points_global_raw,
                timestamp=float(sweep_data['timestamp']),
                device=detector.device
            )

            # Pass the raw points and the correctly aligned raw mask to the detector.
            detector.add_sweep(
                points_global_raw=points_global_raw.astype(np.float32),
                points_sensor_raw=points_sensor_raw.astype(np.float32),
                pose_global=T_global_sensor.astype(np.float32),
                timestamp=float(sweep_data['timestamp']),
                prelabeled_mask_raw=prelabeled_mask_raw
            )
            mdet_result = detector.decide_and_process_frame()
            
            # --- The rest of the packaging logic is unchanged ---
            if mdet_result and mdet_result.get('success'):
                processed_di = mdet_result.get('processed_di')
                if processed_di and processed_di.total_points_added_to_di_arrays > 0:
                    prediction_labels = processed_di.mdet_labels_for_points.cpu().numpy()
                    original_indices_map = processed_di.original_indices_of_filtered_points.cpu().numpy()
                    
                    current_sweep_abs_idx = start_idx + i
                    gt_sparse_indices = ray.get(self.data_actor.get_ground_truth_slice.remote(scene_token, current_sweep_abs_idx))

                    if gt_sparse_indices is None:
                        continue

                    results_for_validation.append({
                        'predictions': prediction_labels,
                        'original_indices_map': original_indices_map,
                        'gt_sparse_indices': gt_sparse_indices
                    })

        if not results_for_validation:
            self.logger.warning(f"No successful M-Detector outputs with matching GT for scene {scene_name}.")
            return None

        return {
            'validation_data': results_for_validation,
            'scene_name': scene_name
        }