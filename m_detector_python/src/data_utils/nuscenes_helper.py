# src/data_utils/nuscenes_helper.py

import numpy as np
from typing import Dict, Optional
import logging
import ray

# Project-specific imports
from ..core.m_detector.base import MDetector
from ..config_loader import MDetectorConfigAccessor

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
        
        self.progress_actor = progress_actor
        self.worker_id = worker_id
        
        self.logger = logging.getLogger(logger_name)

    def process_scene(self, scene_index: int, detector: MDetector, task_description: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Processes an entire scene sweep-by-sweep using the provided M-Detector instance.

        For each sweep, it loads data, runs pre-labeling, adds the sweep to the
        detector, and processes it. It collects all valid predictions and their
        corresponding ground truth data for later validation.

        Returns:
            A dictionary containing the validation data for the entire scene, structured as:
            {
                'validation_data': [
                    {
                        'predictions': np.ndarray,
                        'original_indices_map': np.ndarray,
                        'gt_sparse_indices': np.ndarray
                    }, ...
                ],
                'scene_name': str
            }
            Returns None if no valid data could be processed.
        """
        scene_rec = ray.get(self.data_actor.get_scene_record.remote(scene_index))
        scene_token = scene_rec['token']
        scene_name = scene_rec['name']
        processing_settings = self.config_accessor.get_processing_settings()
        skip_frames = processing_settings['skip_frames']
        max_frames_config = processing_settings['max_frames']
        all_sweep_tokens = ray.get(self.data_actor.get_scene_sweep_tokens.remote(scene_token))
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

            # Pass the raw points to the detector.
            detector.add_sweep(
                points_global_raw=points_global_raw.astype(np.float32),
                points_sensor_raw=points_sensor_raw.astype(np.float32),
                pose_global=T_global_sensor.astype(np.float32),
                timestamp=float(sweep_data['timestamp'])
            )
            mdet_result = detector.process_latest_sweep()
            
            # Package results
            if mdet_result and mdet_result['success']:
                processed_di = mdet_result['processed_di']
                if processed_di and processed_di.num_points > 0:
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
    
    def process_scene_for_baking(self, scene_index: int, detector: MDetector):
        """
        A generator that processes a scene frame-by-frame and yields the
        intermediate data needed for the 'tune-refinement' stage.
        """
        scene_rec = ray.get(self.data_actor.get_scene_record.remote(scene_index))
        scene_token = scene_rec['token']
        all_sweep_tokens = ray.get(self.data_actor.get_scene_sweep_tokens.remote(scene_token))
        
        if not all_sweep_tokens:
            return

        sweep_data_futures = [self.data_actor.get_sweep_data_by_token.remote(token) for token in all_sweep_tokens]
        all_sweep_data = ray.get(sweep_data_futures)

        for i, sweep_data in enumerate(all_sweep_data):
            points_sensor_raw = sweep_data['points_sensor_frame']
            if points_sensor_raw.shape[0] == 0:
                continue

            T_global_sensor = sweep_data['T_global_lidar']
            points_global_raw = (T_global_sensor[:3, :3] @ points_sensor_raw.T).T + T_global_sensor[:3, 3]

            di = detector.add_sweep(
                points_global_raw.astype(np.float32),
                points_sensor_raw.astype(np.float32),
                T_global_sensor.astype(np.float32),
                float(sweep_data['timestamp'])
            )
            
            # Check if the detector is ready to process
            if len(detector.depth_image_library) < detector.min_sweeps_for_processing:
                continue

            # --- This is the core of the baking process ---
            lib_len = len(detector.depth_image_library)
            di_to_process_idx = lib_len - 1
            di_to_process = detector.depth_image_library.get_image_by_index(di_to_process_idx)

            if di_to_process and di_to_process.num_points > 0:
                # 1. Run geometry pipeline
                labels_before_refinement = detector.get_labels_before_refinement(di_to_process, di_to_process_idx)
                
                # 2. Get corresponding ground truth
                gt_sparse_indices = ray.get(self.data_actor.get_ground_truth_slice.remote(scene_token, i))
                if gt_sparse_indices is None:
                    continue

                # 3. Yield all necessary data (on CPU to be pickle-friendly)
                yield {
                    'labels_before_refinement': labels_before_refinement.cpu(),
                    'points_global': di_to_process.original_points_global_coords.cpu(),
                    'original_indices_map': di_to_process.original_indices_of_filtered_points.cpu().numpy(),
                    'gt_sparse_indices': gt_sparse_indices
                }