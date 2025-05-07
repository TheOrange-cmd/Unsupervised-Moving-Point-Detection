# scripts/generate_mdetector_bev_video.py
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box # For type hinting
from pyquaternion import Quaternion
from tqdm import tqdm

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    # Use tqdm.write instead of print to avoid interfering with progress bar
    tqdm.write(f"Added project root to sys.path: {PROJECT_ROOT}")

from src.core.m_detector import MDetector, OcclusionResult
from src.data_utils.nuscenes_helper import get_lidar_sweep_data
from src.utils.visualization import draw_bev_box_on_ax, mpl_fig_to_opencv_bgr
from src.utils.transformations import transform_points_numpy

def generate_video(config_path: str, scene_index: int, output_filename: str, start_sweep_num: int = 0):
    """
    Generates a BEV video showing M-Detector's dynamic point detections for a NuScenes scene.
    """
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)

    tqdm.write(f"Loading NuScenes {config_yaml['nuscenes']['version']} from {config_yaml['nuscenes']['dataroot']}...")
    nusc = NuScenes(version=config_yaml['nuscenes']['version'],
                    dataroot=config_yaml['nuscenes']['dataroot'],
                    verbose=False)

    tqdm.write("Initializing MDetector...")
    m_detector = MDetector(config=config_yaml)

    my_scene = nusc.scene[scene_index]
    tqdm.write(f"Targeting scene: {my_scene['name']} (Token: {my_scene['token']})")

    first_sample_token = my_scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    current_lidar_data_token = first_sample['data'][config_yaml['nuscenes']['lidar_sensor_name']]
    
    # Count total number of frames for the progress bar
    total_frames = 0
    frame_count_token = current_lidar_data_token
    while frame_count_token:
        lidar_data_rec = nusc.get('sample_data', frame_count_token)
        frame_count_token = lidar_data_rec['next']
        total_frames += 1
    
    # Skip to start_sweep_num if specified
    sweep_counter_for_start = 0
    if start_sweep_num > 0:
        tqdm.write(f"Skipping to LiDAR sweep approx #{start_sweep_num}...")
        skip_pbar = tqdm(total=start_sweep_num, desc="Skipping frames")
        while current_lidar_data_token and sweep_counter_for_start < start_sweep_num:
            lidar_data_rec = nusc.get('sample_data', current_lidar_data_token)
            current_lidar_data_token = lidar_data_rec['next']
            sweep_counter_for_start += 1
            skip_pbar.update(1)
            if not current_lidar_data_token:
                tqdm.write(f"Error: Reached end of scene before reaching start sweep {start_sweep_num}")
                skip_pbar.close()
                return # Exit if start sweep is beyond scene length
        skip_pbar.close()
        tqdm.write(f"Starting video generation from LiDAR sweep approx #{start_sweep_num}")

    # Video settings from config or defaults
    video_cfg = config_yaml.get('video_generation', {})
    video_fps = video_cfg.get('fps', 20)
    bev_range_meters = video_cfg.get('bev_range_meters', 50)
    point_size_all_lidar = video_cfg.get('point_size_all_lidar', 0.2)
    point_size_dynamic = video_cfg.get('point_size_dynamic', 2.0)
    historical_di_lag_seconds = video_cfg.get('historical_di_lag_seconds', 0.5)
    plot_gt_boxes = video_cfg.get('plot_ground_truth_boxes', True) # Option to turn off GT boxes

    # Initialize VideoWriter
    fig_init, ax_init = plt.subplots(figsize=(10, 10))
    frame_bgr = mpl_fig_to_opencv_bgr(fig_init)
    frame_height, frame_width, _ = frame_bgr.shape
    plt.close(fig_init)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, video_fps, (frame_width, frame_height))
    tqdm.write(f"Output video: {output_filename} ({frame_width}x{frame_height} @ {video_fps} FPS)")

    fig_main, ax_main = plt.subplots(figsize=(10, 10))
    processed_video_frames = 0
    current_dynamic_points_global_to_plot = []

    # Initialize progress bar for video generation
    remaining_frames = total_frames - sweep_counter_for_start
    video_pbar = tqdm(total=remaining_frames, desc="Generating video frames")

    while current_lidar_data_token:
        lidar_data = nusc.get('sample_data', current_lidar_data_token)
        ego_pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        sample_for_annotations = nusc.get('sample', lidar_data['sample_token'])

        # Update progress bar description with frame info
        video_pbar.set_postfix({
            "Frame": processed_video_frames + 1,
            "TS": f"{lidar_data['timestamp']/1e6:.2f}s"
        })

        points_lidar_sensor_frame, T_global_lidar, lidar_timestamp_sec = \
            get_lidar_sweep_data(nusc, current_lidar_data_token)

        current_di = m_detector.add_sweep_and_create_depth_image(
            points_lidar_sensor_frame, T_global_lidar, lidar_data['timestamp'] # MDetector uses microsec ts
        )
        
        newly_detected_dynamic_points_global = []
        if m_detector.is_ready_for_processing() and current_di:
            historical_di = None
            if len(m_detector.depth_image_library) >= 2:
                # Attempt to find a DI around historical_di_lag_seconds ago
                # This is a simple heuristic, could be improved by searching timestamps
                target_hist_ts = current_di.timestamp - (historical_di_lag_seconds * 1e6)
                best_match_di = None
                min_ts_diff = float('inf')
                for i in range(len(m_detector.depth_image_library) -1): # Exclude current_di
                    img = m_detector.depth_image_library.get_image_by_index(i)
                    if img and img.timestamp < current_di.timestamp :
                        ts_diff = abs(img.timestamp - target_hist_ts)
                        if ts_diff < min_ts_diff:
                            min_ts_diff = ts_diff
                            best_match_di = img
                historical_di = best_match_di if best_match_di else m_detector.depth_image_library.get_image_by_index(-2)


            if historical_di and historical_di.timestamp < current_di.timestamp:
                for v_idx in range(current_di.num_pixels_v):
                    for h_idx in range(current_di.num_pixels_h):
                        pixel_content = current_di.pixels[v_idx, h_idx]
                        if pixel_content and pixel_content['points']:
                            for pt_info in pixel_content['points']:
                                global_pt_curr = pt_info['global_pt']
                                pixel_res, _, _ = m_detector.check_occlusion_pixel_level(global_pt_curr, historical_di)
                                if pixel_res == OcclusionResult.OCCLUDING_IMAGE:
                                    newly_detected_dynamic_points_global.append(global_pt_curr)
                current_dynamic_points_global_to_plot = newly_detected_dynamic_points_global
        
        ax_main.clear()
        points_global_plot = transform_points_numpy(points_lidar_sensor_frame, T_global_lidar)
        ax_main.scatter(points_global_plot[:, 0], points_global_plot[:, 1], s=point_size_all_lidar, color='lightgrey', alpha=0.6)

        ego_translation_global = np.array(ego_pose_data['translation'])
        ego_rotation_global = Quaternion(ego_pose_data['rotation'])
        ax_main.plot(ego_translation_global[0], ego_translation_global[1], 'o', markersize=8, color='blue', label='Ego Vehicle')
        ego_front_direction = ego_rotation_global.rotate(np.array([2.0, 0, 0]))
        ax_main.arrow(ego_translation_global[0], ego_translation_global[1],
                       ego_front_direction[0], ego_front_direction[1],
                       head_width=0.8, head_length=1.0, fc='blue', ec='blue')

        if plot_gt_boxes:
            for ann_token in sample_for_annotations['anns']:
                ann_record = nusc.get('sample_annotation', ann_token)
                if 'vehicle' in ann_record['category_name'] or 'human' in ann_record['category_name']:
                    box = Box(center=ann_record['translation'], size=ann_record['size'], orientation=Quaternion(ann_record['rotation']))
                    draw_bev_box_on_ax(ax_main, box, color='red', linewidth=1)
        
        if current_dynamic_points_global_to_plot:
            dynamic_pts_np = np.array(current_dynamic_points_global_to_plot)
            ax_main.scatter(dynamic_pts_np[:, 0], dynamic_pts_np[:, 1], s=point_size_dynamic, color='green', label='MDet: OCCLUDING', zorder=5)

        ax_main.set_xlim(ego_translation_global[0] - bev_range_meters, ego_translation_global[0] + bev_range_meters)
        ax_main.set_ylim(ego_translation_global[1] - bev_range_meters, ego_translation_global[1] + bev_range_meters)
        ax_main.set_aspect('equal', adjustable='box')
        ax_main.set_xlabel("Global X (meters)")
        ax_main.set_ylabel("Global Y (meters)")
        title_str = (f"Scene: {my_scene['name']} - LiDAR Frame: {sweep_counter_for_start + processed_video_frames}\n"
                     f"TS: {lidar_data['timestamp']/1e6:.2f}s - MDet Ready: {m_detector.is_ready_for_processing()} "
                     f"DI Lib: {len(m_detector.depth_image_library)}")
        if current_dynamic_points_global_to_plot: title_str += f" DynPts: {len(current_dynamic_points_global_to_plot)}"
        ax_main.set_title(title_str, fontsize=9)
        ax_main.grid(True, linestyle='--', alpha=0.5)
        if processed_video_frames == 0: ax_main.legend(loc='upper right', fontsize='small')

        video_frame_bgr = mpl_fig_to_opencv_bgr(fig_main)
        video_writer.write(video_frame_bgr)

        processed_video_frames += 1
        current_lidar_data_token = lidar_data['next']
        video_pbar.update(1)

    video_pbar.close()
    plt.close(fig_main)
    video_writer.release()
    tqdm.write(f"\nVideo generation complete. Saved {processed_video_frames} frames to {output_filename}")

if __name__ == '__main__':
    # --- Configuration for running the script ---
    # It's better to use argparse for command-line arguments for flexibility
    # For now, hardcoding for simplicity:
    CONFIG_FILE = '../config/m_detector_config.yaml' # Relative to this script's location
    SCENE_TO_PROCESS = 1 # Index of the scene in nusc.scene
    OUTPUT_VIDEO_FILE = f'mdetector_scene_{SCENE_TO_PROCESS}_dynamic_pts_scripted.mp4'
    START_LIDAR_SWEEP = 0 # Start from the beginning of the scene

    # Ensure config path is correct relative to the script
    abs_config_path = os.path.join(SCRIPT_DIR, CONFIG_FILE)
    
    generate_video(config_path=abs_config_path,
                   scene_index=SCENE_TO_PROCESS,
                   output_filename=OUTPUT_VIDEO_FILE,
                   start_sweep_num=START_LIDAR_SWEEP)