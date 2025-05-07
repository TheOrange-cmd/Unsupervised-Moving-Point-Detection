# src/data_utils/nuscenes_helper.py
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion

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
    lidar_timestamp = lidar_sample_data['timestamp'] / 1e6 # Convert microseconds to seconds

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

# You can add the get_scene_information function from your utils here if needed
# For now, we'll focus on single sweep loading.