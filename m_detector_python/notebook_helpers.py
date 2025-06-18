# notebook_helpers.py

import open3d as o3d
import numpy as np
import ray
import logging

# Copy these functions from visualize_detector.py into this file
from visualize_detector import map_labels_to_colors, get_sweep_data_and_colors
from src.config_loader import MDetectorConfigAccessor
from src.core.m_detector.base import MDetector
from src.ray_scripts.ray_actors import NuScenesDataActor

def setup_for_notebook(config_path: str):
    """Initializes all necessary components for debugging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if ray.is_initialized():
        ray.shutdown()
    ray.init(logging_level=logging.ERROR)

    config_accessor = MDetectorConfigAccessor(config_path=config_path)
    detector = MDetector(config_accessor=config_accessor)
    
    nuscenes_params = config_accessor.get_nuscenes_params()
    data_actor = NuScenesDataActor.remote(
        version=nuscenes_params['version'],
        dataroot=nuscenes_params['dataroot'],
        config_path=config_path
    )
    
    scene_index = config_accessor.get_mdetector_output_paths().get('scene_indices_to_run', [0])[0]
    scene_rec = ray.get(data_actor.get_scene_record.remote(scene_index))
    scene_token = scene_rec['token']
    lidar_name = nuscenes_params.get('lidar_sensor_name', 'LIDAR_TOP')
    all_sweep_tokens = ray.get(data_actor.get_scene_sweep_tokens.remote(scene_token, lidar_name))
    
    print(f"Setup complete. Found {len(all_sweep_tokens)} sweeps in scene '{scene_rec['name']}'.")
    
    return config_accessor, detector, data_actor, all_sweep_tokens

def process_and_get_pcd(frame_index: int, all_sweep_tokens, data_actor, detector, config_accessor):
    """
    Processes all frames up to a target index and returns a combined point cloud 
    of all processed sweeps with their final labels.
    """
    if frame_index >= len(all_sweep_tokens):
        print(f"Error: Frame index {frame_index} is out of bounds.")
        return None
        
    print(f"Building detector state up to frame {frame_index}...")
    final_result = None
    for i in range(frame_index + 1):
        current_sweep_token = all_sweep_tokens[i]
        sweep_data = ray.get(data_actor.get_sweep_data_by_token.remote(current_sweep_token))
        
        # We still need to call this to run the detector logic for each frame
        get_sweep_data_and_colors(sweep_data, detector, config_accessor, i)

    # --- NEW LOGIC: After the loop, get the final state from the detector ---
    # The detector's library now holds all the processed sweeps.
    # We need to get all points and all corresponding labels from it.
    
    all_points_combined = []
    all_labels_combined = []
    
    # The detector holds a library of DepthImage objects
    for di in detector.depth_image_library.get_all_images():
        if di.total_points_added_to_di_arrays > 0:
            all_points_combined.append(di.original_points_global_coords.cpu().numpy())
            all_labels_combined.append(di.mdet_labels_for_points.cpu().numpy())

    if not all_points_combined:
        print("No points found in the detector's library.")
        return None

    # Concatenate all points and labels into single large arrays
    final_points = np.concatenate(all_points_combined, axis=0)
    final_labels = np.concatenate(all_labels_combined, axis=0)

    print(f"State built. Visualizing {final_points.shape[0]} total points from {len(detector.depth_image_library)} sweeps.")
    
    # Now, points and labels will have the same size.
    final_colors = map_labels_to_colors(final_labels)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    return pcd