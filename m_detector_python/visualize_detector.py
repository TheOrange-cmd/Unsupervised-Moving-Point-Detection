# visualize_detector.py (Version 2)

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

# Project-specific imports
from src.core.m_detector.base import MDetector
from src.config_loader import MDetectorConfigAccessor
from src.ray_scripts.ray_actors import NuScenesDataActor
from src.core.constants import OcclusionResult
from src.core.m_detector.pre_labelers import ransac_ground_prelabeler

# --- Color Constants ---
RANSAC_GROUND_COLOR = [0.1, 0.9, 0.1]
OCCLUDING_COLOR     = [1.0, 0.0, 0.0]
DEFAULT_COLOR       = [0.7, 0.7, 0.7]
INITIALIZING_COLOR  = [0.0, 0.7, 1.0]

def map_labels_to_colors(labels: np.ndarray) -> np.ndarray:
    """Maps an array of M-Detector labels to a corresponding RGB color array."""
    num_points = len(labels)
    colors = np.full((num_points, 3), DEFAULT_COLOR, dtype=np.float64)
    colors[labels == OcclusionResult.PRELABELED_STATIC_GROUND.value] = RANSAC_GROUND_COLOR
    colors[labels == OcclusionResult.OCCLUDING_IMAGE.value] = OCCLUDING_COLOR
    colors[labels == OcclusionResult.OCCLUDED_BY_IMAGE.value] = DEFAULT_COLOR
    colors[labels == OcclusionResult.UNDETERMINED.value] = DEFAULT_COLOR
    return colors


def get_colored_pcd_for_sweep(sweep_data, detector, config_accessor, sweep_index):
    """
    Processes a single sweep and returns an Open3D PointCloud object for THAT sweep,
    colored with the latest labels from the detector.
    """
    # 1. Filter points and run RANSAC pre-labeler
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
        prelabeled_mask = ransac_ground_prelabeler(
            points_global=points_to_process_global,
            points_lidar_frame=points_to_process_sensor,
            current_di_timestamp=float(sweep_data['timestamp']),
            ransac_params=ransac_params,
            device_str=detector.device.type
        )

    # 2. Add sweep to detector and let it process the frame
    detector.add_sweep(
        points_global=points_to_process_global.astype(np.float32),
        pose_global=T_global_sensor.astype(np.float32),
        timestamp=float(sweep_data['timestamp']),
        prelabeled_mask=prelabeled_mask
    )
    mdet_result = detector.decide_and_process_frame()

    pcd = o3d.geometry.PointCloud()
    
    # 3. Create the PointCloud for visualization
    init_sweeps = config_accessor.get_initialization_phase_params()['num_sweeps_for_initial_map']
    if sweep_index < init_sweeps:
        # During initialization, just show the points being added
        pcd.points = o3d.utility.Vector3dVector(points_to_process_global)
        pcd.paint_uniform_color(INITIALIZING_COLOR)
    elif mdet_result and mdet_result.get('success'):
        processed_di = mdet_result.get('processed_di')
        if processed_di and processed_di.total_points_added_to_di_arrays > 0:
            # --- GUARANTEED ALIGNMENT ---
            # Get points and labels from the *same source* (the processed DepthImage)
            final_points = processed_di.original_points_global_coords.cpu().numpy()
            final_labels = processed_di.mdet_labels_for_points.cpu().numpy()
            
            pcd.points = o3d.utility.Vector3dVector(final_points)
            pcd.colors = o3d.utility.Vector3dVector(map_labels_to_colors(final_labels))
        else:
            # If processing fails or DI is empty, show points as default
            pcd.points = o3d.utility.Vector3dVector(points_to_process_global)
            pcd.paint_uniform_color(DEFAULT_COLOR)
            
    return pcd

def render_scene_to_images(config_accessor: MDetectorConfigAccessor, detector: MDetector, config_path: str, temp_dir: Path, max_frames: int = None):
    """Renders each sweep of a scene to an individual image file."""
    # (Setup code is the same...)
    logging.info(f"Starting offline render. Frames will be saved to: {temp_dir}")
    temp_dir.mkdir(exist_ok=True)
    nuscenes_params = config_accessor.get_nuscenes_params()
    data_actor = NuScenesDataActor.remote(version=nuscenes_params['version'], dataroot=nuscenes_params['dataroot'], config_path=config_path)
    scene_index = config_accessor.get_mdetector_output_paths().get('scene_indices_to_run', [0])[0]
    scene_rec = ray.get(data_actor.get_scene_record.remote(scene_index))
    scene_token = scene_rec['token']
    lidar_name = nuscenes_params.get('lidar_sensor_name', 'LIDAR_TOP')
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
    
    for i, sweep_token in enumerate(tqdm(all_sweep_tokens, desc="Rendering Frames")):
        sweep_data = ray.get(data_actor.get_sweep_data_by_token.remote(sweep_token))
        
        # Get the colored point cloud for just this sweep
        pcd_to_render = get_colored_pcd_for_sweep(sweep_data, detector, config_accessor, i)
        
        # Use a "chase cam" that follows the vehicle
        vehicle_position = sweep_data['T_global_lidar'][:3, 3]
        eye = vehicle_position + np.array([-40, -40, 20])
        center = vehicle_position
        up = [0, 0, 1]
        renderer.setup_camera(60.0, center, eye, up)
        
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry(f"pcd_{i}", pcd_to_render, mat)
        img = renderer.render_to_image()
        img_path = str(temp_dir / f"frame_{i:05d}.png")
        o3d.io.write_image(img_path, img, 9)
    logging.info("Offline rendering complete.")

# (create_video_from_images, play_video, and run_realtime_visualization are unchanged)
def create_video_from_images(image_folder: Path, output_path: str, fps: int):
    logging.info(f"Creating video from frames in {image_folder}...")
    images = sorted(list(image_folder.glob("*.png")))
    if not images:
        logging.error("No images found to create a video.")
        return
    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image_path in tqdm(images, desc="Encoding Video"):
        frame = cv2.imread(str(image_path))
        video_writer.write(frame)
    video_writer.release()
    logging.info(f"Video saved successfully to: {output_path}")

def play_video(video_path: str, fps: int):
    if not Path(video_path).exists():
        logging.error(f"Video file not found at: {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return
    logging.info(f"Playing video: {video_path}. Press 'q' to quit.")
    delay = int(1000 / fps)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('M-Detector Visualization', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

def run_realtime_visualization(config_accessor: MDetectorConfigAccessor, detector: MDetector, config_path: str, max_frames: int = None):
    """Runs an interactive, real-time visualization of one sweep at a time."""
    # (Setup code is the same...)
    nuscenes_params = config_accessor.get_nuscenes_params()
    data_actor = NuScenesDataActor.remote(version=nuscenes_params['version'], dataroot=nuscenes_params['dataroot'], config_path=config_path)
    scene_index = config_accessor.get_mdetector_output_paths().get('scene_indices_to_run', [0])[0]
    scene_rec = ray.get(data_actor.get_scene_record.remote(scene_index))
    scene_token = scene_rec['token']
    scene_name = scene_rec['name']
    lidar_name = nuscenes_params.get('lidar_sensor_name', 'LIDAR_TOP')
    all_sweep_tokens = ray.get(data_actor.get_scene_sweep_tokens.remote(scene_token, lidar_name))

    if max_frames is not None and max_frames > 0:
        all_sweep_tokens = all_sweep_tokens[:max_frames]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"M-Detector Visualization: {scene_name}")
    render_options = vis.get_render_option()
    render_options.background_color = np.asarray([0.1, 0.1, 0.1])
    render_options.point_size = 2.0
    pcd_to_show = o3d.geometry.PointCloud()
    is_first_frame = True

    try:
        for i, sweep_token in enumerate(tqdm(all_sweep_tokens, desc="Processing Real-time Frames")):
            sweep_data = ray.get(data_actor.get_sweep_data_by_token.remote(sweep_token))
            
            pcd_for_sweep = get_colored_pcd_for_sweep(sweep_data, detector, config_accessor, i)
            
            pcd_to_show.points = pcd_for_sweep.points
            pcd_to_show.colors = pcd_for_sweep.colors

            if not pcd_to_show.has_points():
                continue

            if is_first_frame:
                vis.add_geometry(pcd_to_show)
                is_first_frame = False
            else:
                vis.update_geometry(pcd_to_show)
            
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05)
    except KeyboardInterrupt:
        logging.info("Visualization stopped by user.")
    finally:
        vis.destroy_window()

# --- ROBUST MAIN FUNCTION ---
def main():
    parser = argparse.ArgumentParser(description="Visualize M-Detector outputs using Open3D.")
    parser.add_argument('--config', type=str, default='best_tuning.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--render', action='store_true', help='Enable offline rendering to create a video.')
    parser.add_argument('--play', action='store_true', help='Play the resulting video after rendering (or play an existing one).')
    parser.add_argument('--video-file', type=str, default='mdetector_output.mp4', help='Path to save or play the video file.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the output video.')
    parser.add_argument('--keep-frames', action='store_true', help='Do not delete the temporary folder of rendered image frames.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.play and not args.render:
        play_video(args.video_file, args.fps)
        return

    # --- Robust Ray Initialization and Shutdown ---
    if ray.is_initialized():
        ray.shutdown()
    ray.init(logging_level=logging.ERROR)
    
    try:
        config_path = args.config
        logging.info(f"Loading configuration from: {config_path}")
        config_accessor = MDetectorConfigAccessor(config_path=config_path)
        detector = MDetector(config_accessor=config_accessor)

        if args.render:
            temp_dir = Path("./temp_render_frames")
            try:
                render_scene_to_images(config_accessor, detector, config_path, temp_dir)
                create_video_from_images(temp_dir, args.video_file, args.fps)
                if args.play:
                    play_video(args.video_file, args.fps)
            finally:
                if not args.keep_frames and temp_dir.exists():
                    logging.info(f"Cleaning up temporary frame directory: {temp_dir}")
                    shutil.rmtree(temp_dir)
        else:
            run_realtime_visualization(config_accessor, detector, config_path)

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)
    finally:
        # This ensures Ray is always shut down, even if errors occur.
        if ray.is_initialized():
            ray.shutdown()
            logging.info("Ray has been shut down.")

if __name__ == "__main__":
    main()