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
    def __init__(self, nusc: NuScenes, config: Dict):
        self.nusc = nusc
        self.config = config
        self.logger = logging.getLogger(__name__) 

    def process_scene(self,
                      scene_index: int,
                      detector: MDetector, 
                      with_progress: bool = True) -> Optional[Dict[str, np.ndarray]]:
        
        scene_rec = self.nusc.scene[scene_index]
        processing_cfg = self.config.get('processing', {})
        skip_frames_config = processing_cfg.get('skip_frames', 0)
        max_frames_config = processing_cfg.get('max_frames', None) 
        logger.info(f"Skipping first {skip_frames_config} frames. Processing max: {max_frames_config} frames.")

        all_scene_sweep_data_dicts = list(get_scene_sweep_data_sequence(self.nusc, scene_rec['token']))
        
        if not all_scene_sweep_data_dicts:
            tqdm.write(f"No sweeps found for scene {scene_rec['name']}. Skipping M-Detector processing.")
            return None

        start_idx = min(skip_frames_config, len(all_scene_sweep_data_dicts))
        end_idx = len(all_scene_sweep_data_dicts)
        if max_frames_config is not None:
            end_idx = min(start_idx + max_frames_config, len(all_scene_sweep_data_dicts))
        
        sweeps_to_feed_list = all_scene_sweep_data_dicts[start_idx:end_idx]
        num_sweeps_to_feed = len(sweeps_to_feed_list)

        if num_sweeps_to_feed == 0:
            tqdm.write(f"No sweeps selected to feed to M-Detector for scene {scene_rec['name']} based on skip/max frames.")
            return None

        collected_mdetector_outputs = [] 
        fed_sweep_data_by_timestamp: Dict[int, Dict] = {} 

        if hasattr(detector, 'reset_scene_state') and callable(detector.reset_scene_state):
            detector.reset_scene_state()
        else:
            tqdm.write("Warning: MDetector does not have a 'reset_scene_state' method. State might carry over.")

        desc = f"Feeding sweeps to M-Detector for Scene {scene_rec['name']}"
        iterator_for_feeding = tqdm(sweeps_to_feed_list, total=num_sweeps_to_feed, desc=desc) if with_progress else sweeps_to_feed_list
        
        # --- Phase 1: Feed sweeps and collect M-Detector outputs ---
        for sweep_data in iterator_for_feeding:
            # 1. Store a reference to the full sweep_data, keyed by its unique timestamp
            fed_sweep_data_by_timestamp[sweep_data['timestamp']] = sweep_data
            
            # 2. Add sweep to MDetector's internal library
            detector.add_sweep_and_create_depth_image(
                points_lidar_frame=sweep_data['points_sensor_frame'], 
                T_global_lidar=sweep_data['T_global_lidar'], 
                lidar_timestamp=sweep_data['timestamp'],
                lidar_sd_token=sweep_data['lidar_sd_token'] # Pass token if your MDetector uses it
            )
            
            # 3. Ask MDetector to process whatever frame it deems ready now
            mdet_result = detector.decide_and_process_frame(is_end_of_sequence=False)
            
            if mdet_result and mdet_result.get('success'):
                processed_di_object = mdet_result.get('processed_di')
                processed_timestamp = mdet_result.get('processed_frame_timestamp')

                if processed_di_object and processed_timestamp is not None:
                    original_sweep_for_this_output = fed_sweep_data_by_timestamp.get(processed_timestamp)
                    
                    if original_sweep_for_this_output and processed_di_object.timestamp == processed_timestamp:
                        collected_mdetector_outputs.append({
                            'original_sweep_data': original_sweep_for_this_output,
                            'mdet_success_flag': True,
                            'processed_di_object_ref': processed_di_object
                        })
                    else:
                        tqdm.write(f"Warning (Main Loop): Timestamp/data mismatch. Processed TS: {processed_timestamp}. Original sweep found: {'Yes' if original_sweep_for_this_output else 'No'}. DI TS: {processed_di_object.timestamp if processed_di_object else 'N/A'}. Output skipped.")
                else:
                    tqdm.write(f"Warning (Main Loop): MDetector success but missing processed_di or timestamp. Output skipped. Result: {mdet_result}")
            elif mdet_result:
                if with_progress: tqdm.write(f"MDetector info: {mdet_result.get('reason', 'No specific reason given by MDetector')}")
        
        # --- Phase 2: Flush MDetector's buffer ---
        if hasattr(detector, 'use_bidirectional') and detector.use_bidirectional:
            if with_progress: tqdm.write("Flushing MDetector bidirectional buffer...")
            flush_counter = 0
            max_flush_attempts = len(detector.depth_image_library._images) + detector.bidirectional_window_size + 5 # Adjusted safety break
            
            while flush_counter < max_flush_attempts:
                flush_counter += 1
                mdet_result = detector.decide_and_process_frame(is_end_of_sequence=True)
                
                if mdet_result and mdet_result.get('success'):
                    processed_di_object = mdet_result.get('processed_di')
                    processed_timestamp = mdet_result.get('processed_frame_timestamp')

                    if processed_di_object and processed_timestamp is not None:
                        original_sweep_for_this_output = fed_sweep_data_by_timestamp.get(processed_timestamp)
                        if original_sweep_for_this_output and processed_di_object.timestamp == processed_timestamp:
                            collected_mdetector_outputs.append({
                                'original_sweep_data': original_sweep_for_this_output,
                                'mdet_success_flag': True,
                                'processed_di_object_ref': processed_di_object
                            })
                        else:
                            tqdm.write(f"Warning (Flush): Timestamp/data mismatch. Processed TS: {processed_timestamp}. Original sweep found: {'Yes' if original_sweep_for_this_output else 'No'}. DI TS: {processed_di_object.timestamp if processed_di_object else 'N/A'}. Output skipped.")
                    else:
                        tqdm.write(f"Warning (Flush): MDetector success but missing processed_di or timestamp. Output skipped. Result: {mdet_result}")
                elif mdet_result and not mdet_result.get('success'):
                    if with_progress: tqdm.write(f"MDetector info (flush): {mdet_result.get('reason', 'Failed or nothing more to process during flush')}")
                    if mdet_result.get('reason') != 'Bidirectional buffer not full yet': # Break if it's not just waiting for more frames (which it won't get)
                        break 
                elif not mdet_result: 
                    if with_progress: tqdm.write("MDetector flush complete (returned None).")
                    break
                if flush_counter >= max_flush_attempts:
                    tqdm.write("Warning: Max flush attempts reached. Breaking flush loop.")
                    break
        self.logger.debug(f"--- INSPECTING collected_mdetector_outputs_for_npz_assembly (count: {len(collected_mdetector_outputs)}) ---")
        for idx, output_item_debug in enumerate(collected_mdetector_outputs):
            original_sweep_ref_debug = output_item_debug['original_sweep_data']
            processed_di_debug = output_item_debug.get('processed_di_object_ref')
            self.logger.debug(f"Item {idx}, Sweep Token: {original_sweep_ref_debug['lidar_sd_token']}, Timestamp: {original_sweep_ref_debug['timestamp']}")
            if processed_di_debug:
                points_arr = processed_di_debug.original_points_global_coords
                labels_arr = processed_di_debug.mdet_labels_for_points
                scores_arr = processed_di_debug.mdet_scores_for_points if hasattr(processed_di_debug, 'mdet_scores_for_points') else None

                points_info = "None" if points_arr is None else f"Shape {points_arr.shape}"
                labels_info = "None" if labels_arr is None else f"Shape {labels_arr.shape}"
                scores_info = "None" if scores_arr is None else f"Shape {scores_arr.shape}"
                self.logger.debug(f"  DI Valid: True. Points: {points_info}, Labels: {labels_info}, Scores: {scores_info}")
                if points_arr is not None and labels_arr is not None and points_arr.shape[0] != labels_arr.shape[0] and points_arr.shape[0] > 0 : # Only warn if points exist but shapes mismatch
                     self.logger.error(f"  CRITICAL SHAPE MISMATCH for sweep {original_sweep_ref_debug['lidar_sd_token']}")
                if points_arr is not None and points_arr.shape[0] > 0 and labels_arr is None:
                    self.logger.error(f"  CRITICAL LABELS ARE NONE for sweep {original_sweep_ref_debug['lidar_sd_token']} but points exist.")

            else:
                self.logger.debug(f"  DI Valid: False (processed_di_object_ref is None)")
        self.logger.debug("--- END INSPECTION ---")

         # --- Phase 3: Assemble NPZ data from collected outputs ---
        if not collected_mdetector_outputs:
            tqdm.write(f"No successful M-Detector outputs collected for scene {scene_rec['name']} to save to NPZ.")
            return None

        collected_mdetector_outputs.sort(key=lambda x: x['original_sweep_data']['timestamp'])

        npz_sweep_lidar_sd_tokens_list: List[str] = []
        npz_sweep_timestamps_us_list: List[int] = []
        # Add other per-sweep metadata if needed, e.g., calibrated sensor tokens, T_global_lidar

        # For the new structured array:
        all_points_predictions_list: List[np.ndarray] = [] # List of structured arrays, one per sweep
        points_predictions_indices: List[int] = [0] # Start/end indices for each sweep's data in the concatenated array

        # Define the dtype for the structured array
        # Ensure mdet_label can hold OcclusionResult.value (e.g., int8 or int16)
        # Ensure mdet_score is float32
        structured_array_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('mdet_label', 'i2'), # Or 'i1' if OcclusionResult values fit
            ('mdet_score', 'f4')  # Add if your MDetector generates scores
        ]

        for output_item in collected_mdetector_outputs:
            original_sweep_ref = output_item['original_sweep_data']
            processed_di = output_item.get('processed_di_object_ref') 

            npz_sweep_lidar_sd_tokens_list.append(original_sweep_ref['lidar_sd_token'])
            npz_sweep_timestamps_us_list.append(original_sweep_ref['timestamp'])

            # --- More detailed check and logging ---
            valid_di_for_npz = False
            if processed_di:
                if processed_di.original_points_global_coords is not None and \
                   processed_di.mdet_labels_for_points is not None:
                    # Check if shapes match if both exist and are not empty
                    if processed_di.original_points_global_coords.shape[0] == 0 and \
                       processed_di.mdet_labels_for_points.shape[0] == 0:
                        valid_di_for_npz = True # Empty but consistent DI
                    elif processed_di.original_points_global_coords.shape[0] > 0 and \
                         processed_di.mdet_labels_for_points.shape[0] == processed_di.original_points_global_coords.shape[0]:
                        valid_di_for_npz = True # Non-empty and consistent
                    else:
                        self.logger.warning(
                            f"NPZ Assembly: Shape mismatch for sweep {original_sweep_ref['lidar_sd_token']}. "
                            f"Points shape: {processed_di.original_points_global_coords.shape}, "
                            f"Labels shape: {processed_di.mdet_labels_for_points.shape}. Treating as invalid."
                        )
                else:
                    points_status = "None" if processed_di.original_points_global_coords is None else f"Shape {processed_di.original_points_global_coords.shape}"
                    labels_status = "None" if processed_di.mdet_labels_for_points is None else f"Shape {processed_di.mdet_labels_for_points.shape}"
                    self.logger.warning(
                        f"NPZ Assembly: Point or Label array is None for sweep {original_sweep_ref['lidar_sd_token']}. "
                        f"Points: {points_status}, Labels: {labels_status}. Treating as invalid."
                    )
            else:
                self.logger.warning(f"NPZ Assembly: processed_di object itself is None for sweep {original_sweep_ref['lidar_sd_token']}. This should not happen if it's in collected_outputs.")
            # --- End detailed check ---

            if valid_di_for_npz:
                points_xyz_sweep = processed_di.original_points_global_coords
                labels_sweep_numeric = processed_di.mdet_labels_for_points
                num_points_in_sweep = points_xyz_sweep.shape[0]

                # Create structured array even if num_points_in_sweep is 0
                sweep_structured_array = np.empty(num_points_in_sweep, dtype=structured_array_dtype)
                
                if num_points_in_sweep > 0:
                    sweep_structured_array['x'] = points_xyz_sweep[:, 0]
                    sweep_structured_array['y'] = points_xyz_sweep[:, 1]
                    sweep_structured_array['z'] = points_xyz_sweep[:, 2]
                    sweep_structured_array['mdet_label'] = labels_sweep_numeric
                    
                    # Score handling (ensure mdet_scores_for_points exists and matches shape)
                    if hasattr(processed_di, 'mdet_scores_for_points') and \
                       processed_di.mdet_scores_for_points is not None and \
                       processed_di.mdet_scores_for_points.shape[0] == num_points_in_sweep:
                        sweep_structured_array['mdet_score'] = processed_di.mdet_scores_for_points
                    else:
                        sweep_structured_array['mdet_score'] = 0.0 # Default
                
                all_points_predictions_list.append(sweep_structured_array)
                points_predictions_indices.append(points_predictions_indices[-1] + num_points_in_sweep)
            
            else: # Case where DI is not valid for NPZ point data
                points_predictions_indices.append(points_predictions_indices[-1])

        # Concatenate all structured arrays for the scene
        final_all_points_predictions = np.concatenate(all_points_predictions_list) if all_points_predictions_list else \
                                       np.empty(0, dtype=structured_array_dtype)

        # Save config to NPZ
        config_str = json.dumps(self.config, sort_keys=True, indent=4)

        output_data_for_npz = {
            'scene_token': np.array([scene_rec['token']], dtype='S36'), # For reference
            'sweep_lidar_sd_tokens': np.array(npz_sweep_lidar_sd_tokens_list, dtype='S36'),
            'sweep_timestamps_us': np.array(npz_sweep_timestamps_us_list, dtype=np.int64),
            # Add other per-sweep metadata arrays if needed (e.g., T_global_lidar)
            
            'all_points_predictions': final_all_points_predictions, # The main structured array
            'points_predictions_indices': np.array(points_predictions_indices, dtype=np.int64),
            '_config_json_str': np.array(config_str) # Save config as a string
        }
        return output_data_for_npz