# src/data_utils/nuscenes_helper.py
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator
import os

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

    def process_scene(self,
                      scene_index: int,
                      detector: MDetector, 
                      with_progress: bool = True) -> Optional[Dict[str, np.ndarray]]:
        
        scene_rec = self.nusc.scene[scene_index]
        # ... (initial setup, skip_frames, max_frames etc. remains the same) ...
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
            # === CORRECTED ORDER AND ADDITIONS START ===
            # 1. Store a reference to the full sweep_data, keyed by its unique timestamp
            fed_sweep_data_by_timestamp[sweep_data['timestamp']] = sweep_data
            
            # 2. Add sweep to MDetector's internal library
            detector.add_sweep_and_create_depth_image(
                points_lidar_frame=sweep_data['points_sensor_frame'], 
                T_global_lidar=sweep_data['T_global_lidar'], 
                lidar_timestamp=sweep_data['timestamp'],
                lidar_sd_token=sweep_data['lidar_sd_token'] # Pass token if your MDetector uses it
            )
            # === CORRECTED ORDER AND ADDITIONS END ===
            
            # 3. Ask MDetector to process whatever frame it deems ready now
            mdet_result = detector.decide_and_process_frame(is_end_of_sequence=False)
            
            if mdet_result and mdet_result.get('success'):
                processed_di_object = mdet_result.get('processed_di')
                processed_timestamp = mdet_result.get('processed_frame_timestamp')

                if processed_di_object and processed_timestamp is not None:
                    original_sweep_for_this_output = fed_sweep_data_by_timestamp.get(processed_timestamp)
                    
                    if original_sweep_for_this_output and processed_di_object.timestamp == processed_timestamp:
                        all_points_global = processed_di_object.get_original_points_global()
                        all_labels_numeric = processed_di_object.get_all_point_labels()

                        mdet_points_for_npz = {'dynamic': [], 'occluded_by_mdet': [], 'undetermined_by_mdet': []}
                        actual_label_counts = {label: 0 for label in OcclusionResult}

                        if all_points_global is not None and all_labels_numeric is not None and \
                           all_points_global.shape[0] == all_labels_numeric.shape[0]:
                            for i in range(all_points_global.shape[0]):
                                point_global = all_points_global[i]
                                label_enum = OcclusionResult(all_labels_numeric[i])
                                actual_label_counts[label_enum] += 1
                                if label_enum == OcclusionResult.OCCLUDING_IMAGE:
                                    mdet_points_for_npz['dynamic'].append(point_global)
                                elif label_enum == OcclusionResult.OCCLUDED_BY_IMAGE:
                                    mdet_points_for_npz['occluded_by_mdet'].append(point_global)
                                elif label_enum == OcclusionResult.UNDETERMINED:
                                    mdet_points_for_npz['undetermined_by_mdet'].append(point_global)
                            for key in mdet_points_for_npz:
                                mdet_points_for_npz[key] = np.array(mdet_points_for_npz[key], dtype=np.float32) if mdet_points_for_npz[key] else np.empty((0,3), dtype=np.float32)
                        else:
                            tqdm.write(f"Warning (Main Loop): Missing points/labels in DI for TS {processed_timestamp} or mismatched shapes.")
                            for key in mdet_points_for_npz: mdet_points_for_npz[key] = np.empty((0,3), dtype=np.float32)
                        
                        collected_mdetector_outputs.append({
                            'original_sweep_data': original_sweep_for_this_output,
                            'mdet_output_points': mdet_points_for_npz,
                            'mdet_label_counts': actual_label_counts,
                            'mdet_success_flag': True 
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
                            all_points_global = processed_di_object.get_original_points_global()
                            all_labels_numeric = processed_di_object.get_all_point_labels()

                            mdet_points_for_npz = {'dynamic': [], 'occluded_by_mdet': [], 'undetermined_by_mdet': []}
                            actual_label_counts = {label: 0 for label in OcclusionResult}

                            if all_points_global is not None and all_labels_numeric is not None and \
                               all_points_global.shape[0] == all_labels_numeric.shape[0]:
                                for i in range(all_points_global.shape[0]):
                                    point_global = all_points_global[i]
                                    label_enum = OcclusionResult(all_labels_numeric[i])
                                    actual_label_counts[label_enum] += 1
                                    if label_enum == OcclusionResult.OCCLUDING_IMAGE:
                                        mdet_points_for_npz['dynamic'].append(point_global)
                                    elif label_enum == OcclusionResult.OCCLUDED_BY_IMAGE:
                                        mdet_points_for_npz['occluded_by_mdet'].append(point_global)
                                    elif label_enum == OcclusionResult.UNDETERMINED:
                                        mdet_points_for_npz['undetermined_by_mdet'].append(point_global)
                                for key in mdet_points_for_npz:
                                    mdet_points_for_npz[key] = np.array(mdet_points_for_npz[key], dtype=np.float32) if mdet_points_for_npz[key] else np.empty((0,3), dtype=np.float32)
                            else:
                                tqdm.write(f"Warning (Flush): Missing points/labels in DI for TS {processed_timestamp} or mismatched shapes.")
                                for key in mdet_points_for_npz: mdet_points_for_npz[key] = np.empty((0,3), dtype=np.float32)

                            collected_mdetector_outputs.append({
                                'original_sweep_data': original_sweep_for_this_output,
                                'mdet_output_points': mdet_points_for_npz,
                                'mdet_label_counts': actual_label_counts,
                                'mdet_success_flag': True
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
        
        # --- Phase 3: Assemble NPZ data from collected outputs ---
        if not collected_mdetector_outputs:
            tqdm.write(f"No successful M-Detector outputs collected for scene {scene_rec['name']} to save to NPZ.")
            return None # This was causing the TypeError

        # ... (rest of NPZ assembly logic remains the same) ...
        collected_mdetector_outputs.sort(key=lambda x: x['original_sweep_data']['timestamp'])

        npz_tokens, npz_timestamps, npz_cs_tokens, npz_T_mats, npz_success_flags = [], [], [], [], []
        npz_counts_dyn, npz_counts_occ, npz_counts_und = [], [], []
        
        npz_all_dyn_pts_list, npz_all_occ_pts_list, npz_all_und_pts_list = [], [], []
        npz_dyn_indices, npz_occ_indices, npz_und_indices = [0], [0], [0]

        for output_item in collected_mdetector_outputs:
            sweep_ref = output_item['original_sweep_data']
            points = output_item['mdet_output_points']
            counts = output_item['mdet_label_counts']

            npz_tokens.append(sweep_ref['lidar_sd_token'])
            npz_timestamps.append(sweep_ref['timestamp'])
            npz_cs_tokens.append(sweep_ref['calibrated_sensor_token'])
            npz_T_mats.append(sweep_ref['T_global_lidar'])
            npz_success_flags.append(output_item['mdet_success_flag'])
            
            npz_counts_dyn.append(counts.get(OcclusionResult.OCCLUDING_IMAGE, 0))
            npz_counts_occ.append(counts.get(OcclusionResult.OCCLUDED_BY_IMAGE, 0))
            npz_counts_und.append(counts.get(OcclusionResult.UNDETERMINED, 0))

            dyn_pts_arr = points.get('dynamic', np.empty((0,3), dtype=np.float32))
            if dyn_pts_arr.ndim == 2 and dyn_pts_arr.shape[0] > 0: npz_all_dyn_pts_list.append(dyn_pts_arr)
            npz_dyn_indices.append(npz_dyn_indices[-1] + dyn_pts_arr.shape[0])
            
            occ_pts_arr = points.get('occluded_by_mdet', np.empty((0,3), dtype=np.float32))
            if occ_pts_arr.ndim == 2 and occ_pts_arr.shape[0] > 0: npz_all_occ_pts_list.append(occ_pts_arr)
            npz_occ_indices.append(npz_occ_indices[-1] + occ_pts_arr.shape[0])

            und_pts_arr = points.get('undetermined_by_mdet', np.empty((0,3), dtype=np.float32))
            if und_pts_arr.ndim == 2 and und_pts_arr.shape[0] > 0: npz_all_und_pts_list.append(und_pts_arr)
            npz_und_indices.append(npz_und_indices[-1] + und_pts_arr.shape[0])

        output_data_for_npz = {
            'sweep_lidar_sd_tokens': np.array(npz_tokens, dtype='S36'),
            'sweep_timestamps_us': np.array(npz_timestamps, dtype=np.int64),
            'sweep_calibrated_sensor_tokens': np.array(npz_cs_tokens, dtype='S36'),
            'T_global_lidar_matrices': np.array(npz_T_mats, dtype=np.float32),
            'mdet_success_flags': np.array(npz_success_flags, dtype=bool), # bool_ is alias for bool
            'mdet_label_counts_dynamic': np.array(npz_counts_dyn, dtype=np.int32),
            'mdet_label_counts_occluded': np.array(npz_counts_occ, dtype=np.int32),
            'mdet_label_counts_undetermined': np.array(npz_counts_und, dtype=np.int32),
            'all_dynamic_points': np.concatenate(npz_all_dyn_pts_list, axis=0) if npz_all_dyn_pts_list else np.empty((0,3), dtype=np.float32),
            'dynamic_points_indices': np.array(npz_dyn_indices, dtype=np.int64),
            'all_occluded_points': np.concatenate(npz_all_occ_pts_list, axis=0) if npz_all_occ_pts_list else np.empty((0,3), dtype=np.float32),
            'occluded_points_indices': np.array(npz_occ_indices, dtype=np.int64),
            'all_undetermined_points': np.concatenate(npz_all_und_pts_list, axis=0) if npz_all_und_pts_list else np.empty((0,3), dtype=np.float32),
            'undetermined_points_indices': np.array(npz_und_indices, dtype=np.int64),
        }
        return output_data_for_npz