# src/data_utils/nuscenes_helper.py
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator
import os
import json

from nuscenes.utils.geometry_utils import transform_matrix

from ..core.m_detector.base import MDetector
from ..core.m_detector.processing import extract_mdetector_points
from ..core.constants import OcclusionResult
from ..config_loader import MDetectorConfigAccessor

import logging 

logger = logging.getLogger(__name__) # Module-level logger

def get_lidar_sweep_data(nusc: NuScenes, lidar_sd_token: str) -> Tuple[np.ndarray, np.ndarray, int, str, str, bool, str]: # Added str for sample_token
    """
    Fetches LiDAR point cloud data, ego pose, timestamp, and other info for a given sweep token.
    Returns points in sensor frame, global pose of sensor, timestamp in microseconds,
    calibrated_sensor_token, lidar_sd_token itself, is_key_frame flag, and sample_token.
    """
    sweep_rec = nusc.get('sample_data', lidar_sd_token)
    cs_rec = nusc.get('calibrated_sensor', sweep_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sweep_rec['ego_pose_token'])

    pc_filepath = os.path.join(nusc.dataroot, sweep_rec['filename'])
    if not os.path.exists(pc_filepath): # Handle missing files gracefully
        # print(f"Warning: LiDAR file not found: {pc_filepath} for token {lidar_sd_token}")
        points_sensor_frame = np.empty((0, 3))
    else:
        pc = LidarPointCloud.from_file(pc_filepath)
        points_sensor_frame = pc.points[:3, :].T  # (N, 3) in XYZ

    # Calculate T_global_sensor (sensor pose in global frame)
    sens_to_ego_rot = Quaternion(cs_rec['rotation']).rotation_matrix
    sens_to_ego_trans = np.array(cs_rec['translation'])
    T_sensor_ego = np.eye(4)
    T_sensor_ego[:3, :3] = sens_to_ego_rot
    T_sensor_ego[:3, 3] = sens_to_ego_trans

    ego_to_glob_rot = Quaternion(pose_rec['rotation']).rotation_matrix
    ego_to_glob_trans = np.array(pose_rec['translation'])
    T_ego_global = np.eye(4)
    T_ego_global[:3, :3] = ego_to_glob_rot
    T_ego_global[:3, 3] = ego_to_glob_trans
    
    T_global_sensor = T_ego_global @ T_sensor_ego

    return (
        points_sensor_frame,
        T_global_sensor, # This is T_global_lidar
        sweep_rec['timestamp'], # Microseconds
        sweep_rec['calibrated_sensor_token'],
        lidar_sd_token,
        sweep_rec['is_key_frame'],
        sweep_rec['sample_token'] 
    )

def get_scene_sweep_data_sequence(nusc: NuScenes, scene_token: str, lidar_name: str = 'LIDAR_TOP') -> Iterator[Dict]:
    """
    Yields a sequence of data dictionaries for ALL LiDAR sweeps (keyframes and non-keyframes)
    for the specified sensor in a scene, ordered by timestamp.
    Now includes 'sample_token'.
    """
    scene_rec = nusc.get('scene', scene_token)
    
    first_sample_in_scene_token = scene_rec['first_sample_token']
    first_sample_rec = nusc.get('sample', first_sample_in_scene_token)
    initial_sd_token_for_sensor = first_sample_rec['data'].get(lidar_name)

    if not initial_sd_token_for_sensor:
        _s_token = scene_rec['first_sample_token']
        while _s_token:
            _s_rec = nusc.get('sample', _s_token)
            if lidar_name in _s_rec['data']:
                initial_sd_token_for_sensor = _s_rec['data'][lidar_name]
                break
            _s_token = _s_rec['next']
        if not initial_sd_token_for_sensor:
            return 

    current_sd_token_for_sensor = initial_sd_token_for_sensor
    while True:
        sd_rec_temp = nusc.get('sample_data', current_sd_token_for_sensor)
        if sd_rec_temp['prev']:
            current_sd_token_for_sensor = sd_rec_temp['prev']
        else:
            break 

    all_sweeps_for_sensor: List[Dict[str, Any]] = []
    
    temp_sd_token: Optional[str] = current_sd_token_for_sensor
    while temp_sd_token:
        sweep_rec_header = nusc.get('sample_data', temp_sd_token) # Just to get cs_token for sensor check
        cs_rec_of_current_sweep = nusc.get('calibrated_sensor', sweep_rec_header['calibrated_sensor_token'])
        sensor_rec_of_current_sweep = nusc.get('sensor', cs_rec_of_current_sweep['sensor_token'])

        if sensor_rec_of_current_sweep['channel'] == lidar_name:
            # Fetch all data for this sweep, now including sample_token
            points_sf, T_global_lidar_np, ts_us, cs_token, sd_token, is_kf, sample_tok = \
                get_lidar_sweep_data(nusc, sweep_rec_header['token']) # Use sweep_rec_header['token'] which is temp_sd_token
            
            all_sweeps_for_sensor.append({
                'points_sensor_frame': points_sf,
                'T_global_lidar': T_global_lidar_np,
                'timestamp': ts_us, # Renamed from timestamp for consistency
                'calibrated_sensor_token': cs_token,
                'lidar_sd_token': sd_token,
                'is_key_frame': is_kf,
                'sample_token': sample_tok # <-- ADDED THIS
            })
        
        temp_sd_token = sweep_rec_header['next']

    for sweep_data_dict in all_sweeps_for_sensor:
        yield sweep_data_dict


class NuScenesProcessor:
    def __init__(self, nusc: NuScenes, config_accessor: MDetectorConfigAccessor):
        self.nusc = nusc
        self.config_accessor = config_accessor 
        self.logger = logging.getLogger(__name__)

    def process_scene(self, scene_index: int, detector: MDetector, with_progress: bool = True) -> Optional[Dict[str, np.ndarray]]:
        scene_rec = self.nusc.scene[scene_index]
        self.logger.critical(f"@@@@@@ NuScenesProcessor.process_scene ENTERED for scene_idx: {scene_index}, scene_token: {scene_rec['token']} @@@@@@")
        
        scene_rec = self.nusc.scene[scene_index]

        # Get config params
        processing_settings = self.config_accessor.get_processing_settings()
        skip_frames_config = processing_settings.get('skip_frames', 0)
        max_frames_config = processing_settings.get('max_frames', -1) 
        
        # Ensure it's an integer; if not, default to -1 (process all) and log warning
        if not isinstance(max_frames_config, int):
            self.logger.warning(
                f"Invalid type for 'max_frames' in config: {type(max_frames_config)} "
                f"(value: '{max_frames_config}'). Defaulting to process all frames (-1)."
            )
            max_frames_to_process = -1
        else:
            max_frames_to_process = max_frames_config

        self.logger.critical(f"DEBUG: scene_idx={scene_index}, max_frames_config RAW FROM GETTER: {max_frames_config}, TYPE: {type(max_frames_config)}")
        nuscenes_params = self.config_accessor.get_nuscenes_params()
        lidar_name_to_use = nuscenes_params.get('lidar_sensor_name', 'LIDAR_TOP')
        log_max_frames_msg = "all (due to -1 or default)" if max_frames_to_process < 0 else str(max_frames_to_process)
        self.logger.info(f"Skipping first {skip_frames_config} frames. Processing max: {log_max_frames_msg} frames for LiDAR: {lidar_name_to_use}.")

        #  Get all sweeps
        all_scene_sweep_data_dicts = list(get_scene_sweep_data_sequence(self.nusc, scene_rec['token'], lidar_name=lidar_name_to_use))
        
        if not all_scene_sweep_data_dicts:
            tqdm.write(f"No sweeps found for scene {scene_rec['name']}. Skipping M-Detector processing.")
            return None

        # Limit sequences as configured
        start_idx = min(skip_frames_config, len(all_scene_sweep_data_dicts))
        end_idx = len(all_scene_sweep_data_dicts) # Default to all

        # If max_frames_to_process is a non-negative integer, apply the limit
        if max_frames_to_process >= 0: # Check for non-negative (0 or more)
            end_idx = min(start_idx + max_frames_to_process, len(all_scene_sweep_data_dicts))
        # If max_frames_to_process is negative (e.g., -1), end_idx remains len(all_scene_sweep_data_dicts)
        
        sweeps_to_feed_list = all_scene_sweep_data_dicts[start_idx:end_idx]
        num_sweeps_to_feed = len(sweeps_to_feed_list)

        if num_sweeps_to_feed == 0:
            self.logger.warning(f"No sweeps selected to feed to M-Detector for scene {scene_rec['name']} based on skip/max frames.")
            return None

        collected_mdetector_outputs = [] 
        fed_sweep_data_by_timestamp: Dict[int, Dict] = {} 

        if hasattr(detector, 'reset_scene_state') and callable(detector.reset_scene_state):
            detector.reset_scene_state()
        else:
            self.logger.warning("MDetector does not have a 'reset_scene_state' method. State might carry over.")

        desc = f"Feeding sweeps to M-Detector for Scene {scene_rec['name']}"
        iterator_for_feeding = tqdm(sweeps_to_feed_list, total=num_sweeps_to_feed, desc=desc, disable=not with_progress)
        
        for sweep_data in iterator_for_feeding:
            fed_sweep_data_by_timestamp[sweep_data['timestamp']] = sweep_data
            
            detector.add_sweep_and_create_depth_image(
                points_lidar_frame=sweep_data['points_sensor_frame'], 
                T_global_lidar=sweep_data['T_global_lidar'], 
                lidar_timestamp=sweep_data['timestamp'],
                lidar_sd_token=sweep_data['lidar_sd_token']
            )
            
            mdet_result = detector.decide_and_process_frame(is_end_of_sequence=False)
            
            if mdet_result and mdet_result.get('success'):
                processed_di_object = mdet_result.get('processed_di')
                processed_timestamp = mdet_result.get('timestamp')

                if processed_di_object and processed_timestamp is not None:
                    original_sweep_for_this_output = fed_sweep_data_by_timestamp.get(processed_timestamp)
                    
                    if original_sweep_for_this_output and processed_di_object.timestamp == processed_timestamp:
                        collected_mdetector_outputs.append({
                            'original_sweep_data': original_sweep_for_this_output,
                            'mdet_success_flag': True,
                            'processed_di_object_ref': processed_di_object
                        })
                    else:
                        self.logger.warning(f"Timestamp/data mismatch. Processed TS: {processed_timestamp}. Original sweep found: {'Yes' if original_sweep_for_this_output else 'No'}. DI TS: {processed_di_object.timestamp if processed_di_object else 'N/A'}. Output skipped.")
                else:
                    self.logger.warning(f"MDetector success but missing processed_di or timestamp. Output skipped. Result: {mdet_result}")
            elif mdet_result and not mdet_result.get('success'):
                pass # self.logger.info(f"MDetector info for scene {scene_rec['name']}: {mdet_result.get('reason', 'No specific reason given by MDetector')}")


        if not collected_mdetector_outputs:
            self.logger.warning(f"No successful M-Detector outputs collected for scene {scene_rec['name']} to save to hdf5.")
            return None

        collected_mdetector_outputs.sort(key=lambda x: x['original_sweep_data']['timestamp'])

        sweep_lidar_sd_tokens_list: List[str] = []
        sweep_timestamps_us_list: List[int] = []
        all_points_predictions_list: List[np.ndarray] = [] 
        points_predictions_indices: List[int] = [0] 
        structured_array_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('mdet_label', 'i2'), 
            ('mdet_score', 'f4')
        ]

        for output_item in collected_mdetector_outputs:
            original_sweep_ref = output_item['original_sweep_data']
            processed_di = output_item.get('processed_di_object_ref') 
            sweep_lidar_sd_tokens_list.append(original_sweep_ref['lidar_sd_token'])
            sweep_timestamps_us_list.append(original_sweep_ref['timestamp'])
            valid_di = False
            if processed_di and processed_di.original_points_global_coords is not None and \
               processed_di.mdet_labels_for_points is not None:
                if processed_di.original_points_global_coords.shape[0] == 0 and \
                   processed_di.mdet_labels_for_points.shape[0] == 0:
                    valid_di = True
                elif processed_di.original_points_global_coords.shape[0] > 0 and \
                     processed_di.mdet_labels_for_points.shape[0] == processed_di.original_points_global_coords.shape[0]:
                    valid_di = True
            
            if valid_di:
                points_xyz_sweep = processed_di.original_points_global_coords
                labels_sweep_numeric = processed_di.mdet_labels_for_points
                num_points_in_sweep = points_xyz_sweep.shape[0]
                
                sweep_structured_array = np.empty(num_points_in_sweep, dtype=structured_array_dtype)
                if num_points_in_sweep > 0:
                    sweep_structured_array['x'] = points_xyz_sweep[:, 0]
                    sweep_structured_array['y'] = points_xyz_sweep[:, 1]
                    sweep_structured_array['z'] = points_xyz_sweep[:, 2]
                    sweep_structured_array['mdet_label'] = labels_sweep_numeric
                    
                    if hasattr(processed_di, 'mdet_scores_for_points') and \
                       processed_di.mdet_scores_for_points is not None and \
                       processed_di.mdet_scores_for_points.shape[0] == num_points_in_sweep:
                        sweep_structured_array['mdet_score'] = processed_di.mdet_scores_for_points
                    else:
                        sweep_structured_array['mdet_score'] = 0.0 
                
                all_points_predictions_list.append(sweep_structured_array)
                points_predictions_indices.append(points_predictions_indices[-1] + num_points_in_sweep)
            else: 
                self.logger.warning(f"Scene {scene_rec['name']}, sweep {original_sweep_ref['lidar_sd_token']}: DI data inconsistent, skipping sweep output. Points: {processed_di.original_points_global_coords.shape[0] if processed_di and processed_di.original_points_global_coords is not None else 'N/A'}, Labels: {processed_di.mdet_labels_for_points.shape[0] if processed_di and processed_di.mdet_labels_for_points is not None else 'N/A'}")
                points_predictions_indices.append(points_predictions_indices[-1])


        final_all_points_predictions = np.concatenate(all_points_predictions_list) if all_points_predictions_list else \
                                       np.empty(0, dtype=structured_array_dtype)
        output_data = {
            'scene_token': np.array([scene_rec['token']], dtype='S36'),
            'sweep_lidar_sd_tokens': np.array([s.encode('utf-8') for s in sweep_lidar_sd_tokens_list], dtype='S36'),
            'sweep_timestamps_us': np.array(sweep_timestamps_us_list, dtype=np.int64),
            'all_points_predictions': final_all_points_predictions,
            'points_predictions_indices': np.array(points_predictions_indices, dtype=np.int64)
        }
        return output_data