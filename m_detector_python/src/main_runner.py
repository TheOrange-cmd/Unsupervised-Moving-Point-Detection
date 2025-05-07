# src/main_runner.py
import yaml
import numpy as np
from nuscenes.nuscenes import NuScenes

from core.depth_image import DepthImage
from data_utils.nuscenes_helper import get_lidar_sweep_data
from utils.visualization import plot_lidar_sweep_with_depth_image_pixels # Updated import

def run_single_sweep_test(config_path: str):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize NuScenes
    nusc = NuScenes(version=config['nuscenes']['version'], dataroot=config['nuscenes']['dataroot'], verbose=True)

    # --- Select a sample LiDAR sweep ---
    # Option 1: Use a predefined token from config (if set)
    lidar_token = config['nuscenes'].get('lidar_token_to_load')
    
    # Option 2: Or get the first LiDAR sweep from a specific scene or the first scene
    if not lidar_token:
        scene_name_to_load = config['nuscenes'].get('scene_name_to_load')
        if scene_name_to_load:
            scenes_by_name = {s['name']: s for s in nusc.scene}
            if scene_name_to_load not in scenes_by_name:
                print(f"Error: Scene '{scene_name_to_load}' not found. Available scenes: {list(scenes_by_name.keys())[:5]}")
                return
            scene_token = scenes_by_name[scene_name_to_load]['token']
        else: # Default to the first scene in the dataset
            scene_token = nusc.scene[0]['token']
        
        scene_info = nusc.get('scene', scene_token)
        first_sample_token = scene_info['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        print(f"Using LiDAR token: {lidar_token} from scene: {nusc.get('scene', scene_token)['name']}")

    # Load LiDAR data and pose for the selected sweep
    points_lidar_frame, T_global_lidar, lidar_timestamp = get_lidar_sweep_data(nusc, lidar_token)
    print(f"Loaded {points_lidar_frame.shape[0]} points for sweep at {lidar_timestamp:.2f}s.")
    print(f"LiDAR global pose (T_global_lidar):\n{T_global_lidar}")

    # Create a DepthImage instance
    # The pose for the DepthImage is the LiDAR's global pose at the time of the sweep
    di = DepthImage(image_pose_global=T_global_lidar, config=config, timestamp=lidar_timestamp)
    print(str(di))

    # Add points to the DepthImage
    # Points are in LiDAR frame, but DepthImage.add_point expects global points.
    # So, transform points_lidar_frame to global frame first.
    points_lidar_frame_h = np.hstack((points_lidar_frame, np.ones((points_lidar_frame.shape[0], 1)))) # Nx4
    points_global_h = (T_global_lidar @ points_lidar_frame_h.T).T # Nx4
    points_global = points_global_h[:, :3] # Nx3

    # Filter points by range before adding to depth image
    max_range = config['filtering']['max_point_range_meters']
    min_range = config['filtering']['min_point_range_meters']
    
    for pt_idx, pt_g in enumerate(points_global):
        # Calculate range from LiDAR origin (which is T_global_lidar[:3,3])
        # Or, more simply, use range from original points_lidar_frame
        range_val = np.linalg.norm(points_lidar_frame[pt_idx])
        if min_range <= range_val <= max_range:
            di.add_point(pt_g, label="non_event") # Initially all points are non_event

    print(f"After adding points to DepthImage: {str(di)}")

    # Visualize (optional)
    # For visualization, we can pass points in the LiDAR frame as that's often
    # the natural frame for the k3d camera setup.
    plot_lidar_sweep_with_depth_image_pixels(
        points_lidar_frame, # Pass original points in LiDAR frame for plotting
        di,
        point_size=0.05,
        max_range=max_range
    )

if __name__ == "__main__":
    # Make sure to create the config file and update 'dataroot'
    config_file_path = 'config/m_detector_config.yaml' 
    run_single_sweep_test(config_file_path)