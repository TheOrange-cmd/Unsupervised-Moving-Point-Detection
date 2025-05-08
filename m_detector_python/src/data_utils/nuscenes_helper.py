# src/data_utils/nuscenes_helper.py
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any, Callable

from ..core.m_detector.base import MDetector

def get_lidar_sweep_data(nusc: NuScenes, lidar_token: str):
    """
    Loads LiDAR point cloud data and its pose information for a given token.

    Args:
        nusc (NuScenes): NuScenes SDK object.
        lidar_token (str): LiDAR sample_data token.

    Returns:
        tuple: (points, T_global_lidar, lidar_timestamp)
               points (np.ndarray): Nx3 array of LiDAR points (x,y,z) in LiDAR frame.
               T_global_lidar (np.ndarray): 4x4 transformation matrix from LiDAR to global frame.
               lidar_timestamp (float): Timestamp of the LiDAR sweep.
    """
    lidar_sample_data = nusc.get('sample_data', lidar_token)
    lidar_timestamp = lidar_sample_data['timestamp'] # Keep in microseconds for precise timing

    # Load point cloud
    pcl_path = nusc.get_sample_data_path(lidar_token)
    if lidar_sample_data['sensor_modality'] == 'lidar':
        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points[:3, :].T  # Get X, Y, Z coordinates, transpose to Nx3
    else:
        raise ValueError(f"Token {lidar_token} is not a LiDAR sample.")

    # Get LiDAR sensor pose in global frame
    calibrated_sensor = nusc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', lidar_sample_data['ego_pose_token'])

    # Transformation from sensor (LiDAR) to ego vehicle
    T_vehicle_lidar = np.eye(4)
    T_vehicle_lidar[:3, :3] = Quaternion(calibrated_sensor['rotation']).rotation_matrix
    T_vehicle_lidar[:3, 3] = np.array(calibrated_sensor['translation'])

    # Transformation from ego vehicle to global
    T_global_vehicle = np.eye(4)
    T_global_vehicle[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
    T_global_vehicle[:3, 3] = np.array(ego_pose['translation'])

    # Transformation from LiDAR to global
    T_global_lidar = T_global_vehicle @ T_vehicle_lidar
    
    return points, T_global_lidar, lidar_timestamp


class NuScenesProcessor:
    """
    Processes NuScenes data with MDetector, one sweep at a time.
    """
    
    def __init__(self, nusc: NuScenes):
        """
        Initialize the processor.
        
        Args:
            nusc: NuScenes instance
        """
        self.nusc = nusc
    
    def get_scene_tokens(self, scene_index: int) -> List[str]:
        """
        Get all LiDAR tokens for a scene.
        
        Args:
            scene_index: Index of the scene
            
        Returns:
            List of LiDAR tokens
        """
        scene = self.nusc.scene[scene_index]
        sample_token = scene['first_sample_token']
        sample = self.nusc.get('sample', sample_token)
        
        # Get first LiDAR token
        lidar_token = sample['data']['LIDAR_TOP']
        
        # Build token chain
        token_chain = []
        current_token = lidar_token
        
        while current_token != '':
            token_chain.append(current_token)
            lidar_data = self.nusc.get('sample_data', current_token)
            current_token = lidar_data['next']
        
        return token_chain
    
    def process_scene(self, 
                      scene_index: int, 
                      detector, 
                      frame_callback: Optional[Callable] = None,
                      skip_frames: int = 0,
                      max_frames: Optional[int] = None,
                      with_progress: bool = True) -> List[Dict]:
        """
        Process a NuScenes scene with MDetector.
        
        Args:
            scene_index: Scene to process
            detector: MDetector instance
            frame_callback: Optional callback for each processed frame
            skip_frames: Number of frames to skip at the beginning
            max_frames: Maximum number of frames to process
            with_progress: Show progress bar
            
        Returns:
            List of processing results
        """
        # Get tokens for the scene
        tokens = self.get_scene_tokens(scene_index)
        
        # Apply frame limits
        tokens = tokens[skip_frames:]
        if max_frames is not None:
            tokens = tokens[:max_frames]
        
        # Process each frame
        results = []
        
        # Create progress iterator if needed
        token_iter = tqdm(tokens, desc="Processing frames") if with_progress else tokens
        
        for token in token_iter:
            # Get data for this sweep
            points, pose, timestamp = get_lidar_sweep_data(self.nusc, token)
            
            # Add to detector
            di = detector.add_sweep_and_create_depth_image(points, pose, timestamp)
            
            # Process the current sweep
            if detector.is_ready_for_processing():
                # Get the index of this sweep in the detector's library
                sweep_index = len(detector.depth_image_library._images) - 1
                
                # Process this frame with detector
                process_result = detector.process_frame(sweep_index)
                results.append(process_result)
                
                # Call frame callback if provided
                if frame_callback is not None:
                    frame_callback(detector, sweep_index, process_result)
            else:
                # Not enough data for processing yet
                results.append({
                    'success': False,
                    'reason': 'Not enough initial data',
                    'timestamp': timestamp
                })
        
        return results