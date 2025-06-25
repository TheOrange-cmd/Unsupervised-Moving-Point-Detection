# visualize_detector.py (Optimized for In-Memory Rendering, preserving original structure)

import open3d as o3d
import numpy as np
import logging
import argparse
import time
import ray
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm
import yaml
import torch

# Project-specific imports
from ..core.m_detector.base import MDetector
from ..config_loader import MDetectorConfigAccessor
from ..ray_scripts.ray_actors import NuScenesDataActor
from ..core.constants import OcclusionResult
from ..core.m_detector.pre_labelers import ransac_ground_prelabeler
from ..ray_scripts.shared_utils import deep_update_dict 

# --- Color Constants (Unchanged) ---
RANSAC_GROUND_COLOR = [0.1, 0.9, 0.1]
OCCLUDING_COLOR     = [1.0, 0.0, 0.0]
DEFAULT_COLOR       = [0.7, 0.7, 0.7]
INITIALIZING_COLOR  = [0.0, 0.7, 1.0]

def map_labels_to_colors(labels: np.ndarray) -> np.ndarray:
    # This function had a small bug where it used mdet_labels_for_points,
    # which we removed. It should use final_labels.
    num_points = len(labels)
    colors = np.full((num_points, 3), DEFAULT_COLOR, dtype=np.float64)
    colors[labels == OcclusionResult.PRELABELED_STATIC_GROUND.value] = RANSAC_GROUND_COLOR
    colors[labels == OcclusionResult.OCCLUDING_IMAGE.value] = OCCLUDING_COLOR
    # Explicitly color other labels for clarity
    colors[labels == OcclusionResult.OCCLUDED_BY_IMAGE.value] = DEFAULT_COLOR
    colors[labels == OcclusionResult.UNDETERMINED.value] = DEFAULT_COLOR
    return colors

def get_colored_pcd_for_sweep(sweep_data, detector, config_accessor, sweep_index):
    # (This function is largely the same, but with the critical fix)
    points_sensor_raw = sweep_data['points_sensor_frame']
    T_global_sensor = sweep_data['T_global_lidar']
    filter_params = config_accessor.get_point_pre_filtering_params()
    min_range, max_range = filter_params['min_range_meters'], filter_params['max_range_meters']
    ranges = np.linalg.norm(points_sensor_raw[:, :3], axis=1)
    range_mask = (ranges >= min_range) & (ranges <= max_range)
    points_to_process_sensor = points_sensor_raw[range_mask]
    points_to_process_global = (T_global_sensor[:3, :3] @ points_to_process_sensor.T).T + T_global_sensor[:3, 3]

    prelabeled_mask = None
    ransac_params = config_accessor.get_ransac_ground_params()
    if ransac_params.get('enabled') and points_to_process_sensor.shape[0] > 0:
        prelabeled_mask = ransac_ground_prelabeler(points_global=points_to_process_global, points_lidar_frame=points_to_process_sensor, current_di_timestamp=float(sweep_data['timestamp']), ransac_params=ransac_params, device_str=detector.device.type)

    detector.add_sweep(points_global=points_to_process_global.astype(np.float32), pose_global=T_global_sensor.astype(np.float32), timestamp=float(sweep_data['timestamp']), prelabeled_mask=prelabeled_mask)
    mdet_result = detector.decide_and_process_frame()

    pcd = o3d.geometry.PointCloud()
    
    init_sweeps = config_accessor.get_initialization_phase_params()['num_sweeps_for_initial_map']
    if sweep_index < init_sweeps:
        pcd.points = o3d.utility.Vector3dVector(points_to_process_global)
        pcd.paint_uniform_color(INITIALIZING_COLOR)
    elif mdet_result and mdet_result.get('success'):
        processed_di = mdet_result.get('processed_di')
        if processed_di and processed_di.total_points_added_to_di_arrays > 0:
            final_points = processed_di.original_points_global_coords.cpu().numpy()
            
            final_labels = processed_di.mdet_labels_for_points.cpu().numpy()

            pcd.points = o3d.utility.Vector3dVector(final_points)
            pcd.colors = o3d.utility.Vector3dVector(map_labels_to_colors(final_labels))
        else:
            pcd.points = o3d.utility.Vector3dVector(points_to_process_global)
            pcd.paint_uniform_color(DEFAULT_COLOR)
            
    return pcd

def render_scene_to_video_in_memory(config_accessor: MDetectorConfigAccessor, detector: MDetector, config_path: str, output_path: str, fps: int, max_frames: int = None):
    logging.info(f"Starting in-memory render. Video will be saved to: {output_path}")
    
    nuscenes_params = config_accessor.get_nuscenes_params()
    data_actor = NuScenesDataActor.remote(version=nuscenes_params['version'], dataroot=nuscenes_params['dataroot'], config_path=config_path)
    scene_index = config_accessor.get_processing_settings().get('scene_indices_to_run', [0])[0]
    scene_rec = ray.get(data_actor.get_scene_record.remote(scene_index))
    scene_token = scene_rec['token']
    lidar_name = 'LIDAR_TOP'
    all_sweep_tokens = ray.get(data_actor.get_scene_sweep_tokens.remote(scene_token, lidar_name))
    
    if max_frames is not None and max_frames > 0:
        all_sweep_tokens = all_sweep_tokens[:max_frames]

    width, height = 1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.1, 0.1, 0.1, 1.0])
    renderer.scene.set_lighting(renderer.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for i, sweep_token in enumerate(tqdm(all_sweep_tokens, desc="Processing and Rendering")):
            sweep_data = ray.get(data_actor.get_sweep_data_by_token.remote(sweep_token))
            pcd_to_render = get_colored_pcd_for_sweep(sweep_data, detector, config_accessor, i)
            
            vehicle_position = sweep_data['T_global_lidar'][:3, 3]
            eye = vehicle_position + np.array([-40, -40, 20])
            center = vehicle_position
            up = [0, 0, 1]
            renderer.setup_camera(60.0, center, eye, up)
            
            renderer.scene.clear_geometry()
            renderer.scene.add_geometry(f"pcd_{i}", pcd_to_render, mat)
            
            img_o3d = renderer.render_to_image()
            img_np = np.asarray(img_o3d)
            video_writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
    finally:
        video_writer.release()
        logging.info(f"In-memory rendering complete. Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize M-Detector outputs using Open3D.")
    parser.add_argument('--config', type=str, default='config/m_detector_config.yaml', help='Path to the base YAML configuration file.')
    parser.add_argument('--params', type=str, default=None, help='Path to a YAML file with override parameters (e.g., from Optuna).')
    parser.add_argument('--video-file', type=str, default='mdetector_output.mp4', help='Path to save or play the video file.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the output video.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if ray.is_initialized(): ray.shutdown()
    ray.init(logging_level=logging.ERROR)
    
    try:
        config_path = args.config
        logging.info(f"Loading configuration from: {config_path}")
        if args.params:
            logging.info(f"Applying override parameters from: {args.params}")
            with open(args.config, 'r') as f: base_config = yaml.safe_load(f)
            with open(args.params, 'r') as f: override_params = yaml.safe_load(f)
            final_config = deep_update_dict(base_config, override_params)
            config_accessor = MDetectorConfigAccessor(config_dict=final_config)
        else:
            config_accessor = MDetectorConfigAccessor(config_path=config_path)
        
        # Use cuda:0 if available, otherwise fallback to cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        detector = MDetector(config_accessor=config_accessor, device=device)

        render_scene_to_video_in_memory(config_accessor, detector, config_path, args.video_file, args.fps)

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)
    finally:
        if ray.is_initialized():
            ray.shutdown()
            logging.info("Ray has been shut down.")

if __name__ == "__main__":
    main()