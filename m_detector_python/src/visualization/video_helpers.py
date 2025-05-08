# src/visualization/video_helpers.py
import os
import cv2
import numpy as np
from typing import Dict, Optional

from ..core.m_detector import MDetector
from ..core.constants import OcclusionResult
from ..data_utils.nuscenes_helper import NuScenesProcessor

def generate_video(nusc, scene_index, detector, output_path, config):
    """
    Generate a video visualizing M-detector results.
    
    Args:
        nusc: NuScenes instance
        scene_index: Index of the scene to process
        detector: Configured MDetector instance
        output_path: Path to save the video
        config: Configuration dictionary
    """
    # Video settings
    fps = config.get('video_generation', {}).get('fps', 10)
    width = 1280  # Default width if not specified
    height = 720  # Default height if not specified

    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create a test frame to get dimensions
    import matplotlib.pyplot as plt
    from src.utils.visualization import mpl_fig_to_opencv_bgr
    fig_init, ax_init = plt.subplots(figsize=(10, 10))
    frame_bgr = mpl_fig_to_opencv_bgr(fig_init)
    height, width, _ = frame_bgr.shape
    plt.close(fig_init)
    
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get scene tokens
    processor = NuScenesProcessor(nusc)
    scene_tokens = processor.get_scene_tokens(scene_index)
    
    # Track frame index
    current_frame = 0
    
    # Frame processing callback
    def process_frame(detector, frame_index, process_result):
        nonlocal current_frame
        
        # Get the current token
        token_idx = current_frame + config.get('processing', {}).get('skip_frames', 0)
        if token_idx < len(scene_tokens):
            current_token = scene_tokens[token_idx]
        else:
            # Safety check
            return
        
        # Get the depth image that was just processed
        depth_image = detector.depth_image_library._images[frame_index]
        
        # Extract dynamic points
        dynamic_points = extract_dynamic_points(depth_image)
        
        # Create visualization frame
        frame = create_visualization_frame(
            depth_image=depth_image, 
            points=dynamic_points, 
            frame_index=current_frame,
            width=width, 
            height=height,
            config=config,
            nusc=nusc,
            lidar_token=current_token
        )
        
        # Add to video
        video_writer.write(frame)
        
        # Increment frame counter
        current_frame += 1
    
    # Process the scene
    results = processor.process_scene(
        scene_index=scene_index,
        detector=detector,
        frame_callback=process_frame,
        skip_frames=config.get('processing', {}).get('skip_frames', 0),
        max_frames=config.get('processing', {}).get('max_frames', None),
        with_progress=True
    )
    
    # Finalize video
    video_writer.release()
    
    return {
        'video_path': output_path,
        'frames_processed': current_frame,
        'results': results
    }

def extract_dynamic_points(depth_image):
    """
    Extract points by label from a depth image.
    
    Args:
        depth_image: Processed depth image
        
    Returns:
        Dict with points by category
    """
    dynamic_points = []
    occluded_points = []
    undetermined_points = []
    
    # Extract points by category
    for pixel_key, points_list in depth_image.pixel_points.items():
        for pt_info in points_list:
            label = pt_info.get('label')
            point = pt_info['global_pt']
            
            if label == OcclusionResult.OCCLUDING_IMAGE:
                dynamic_points.append(point)
            elif label == OcclusionResult.OCCLUDED_BY_IMAGE:
                occluded_points.append(point)
            elif label == OcclusionResult.UNDETERMINED:
                undetermined_points.append(point)
    
    return {
        'dynamic': np.array(dynamic_points) if dynamic_points else np.empty((0, 3)),
        'occluded': np.array(occluded_points) if occluded_points else np.empty((0, 3)),
        'undetermined': np.array(undetermined_points) if undetermined_points else np.empty((0, 3))
    }

def create_visualization_frame(depth_image, points, frame_index, width, height, config=None, nusc=None, lidar_token=None):
    """
    Create a visualization frame for the video.
    
    Args:
        depth_image: Processed depth image
        points: Dictionary with categorized points ('dynamic', 'occluded', 'undetermined')
        frame_index: Current frame index
        width: Frame width
        height: Frame height
        config: Configuration dictionary
        nusc: NuScenes instance (for ground truth boxes)
        lidar_token: Current lidar token
        
    Returns:
        np.ndarray: BGR visualization frame
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pyquaternion import Quaternion
    
    # Extract visualization settings from config
    video_cfg = config.get('video_generation', {}) if config else {}
    bev_range_meters = video_cfg.get('bev_range_meters', 50)
    point_size_all_lidar = video_cfg.get('point_size_all_lidar', 0.2)
    point_size_dynamic = video_cfg.get('point_size_dynamic', 2.0)
    plot_gt_boxes = video_cfg.get('plot_ground_truth_boxes', True) and nusc is not None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get original lidar points and transform to global frame
    original_points_global = None
    ego_translation_global = None
    ego_rotation_global = None
    
    if nusc and lidar_token:
        # Get lidar data and ego pose
        lidar_data = nusc.get('sample_data', lidar_token)
        ego_pose_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_translation_global = np.array(ego_pose_data['translation'])
        ego_rotation_global = Quaternion(ego_pose_data['rotation'])
        
        # Get all points in global frame
        from src.data_utils.nuscenes_helper import get_lidar_sweep_data
        from src.utils.transformations import transform_points_numpy
        
        points_lidar_sensor_frame, T_global_lidar, _ = get_lidar_sweep_data(nusc, lidar_token)
        original_points_global = transform_points_numpy(points_lidar_sensor_frame, T_global_lidar)
        
        # Plot all lidar points in grey
        ax.scatter(
            original_points_global[:, 0], 
            original_points_global[:, 1], 
            s=point_size_all_lidar, 
            color='lightgrey', 
            alpha=0.6
        )
        
        # Plot ego vehicle
        ax.plot(
            ego_translation_global[0], 
            ego_translation_global[1], 
            'o', 
            markersize=8, 
            color='blue', 
            label='Ego Vehicle'
        )
        
        # Add direction arrow for ego vehicle
        ego_front_direction = ego_rotation_global.rotate(np.array([2.0, 0, 0]))
        ax.arrow(
            ego_translation_global[0], 
            ego_translation_global[1],
            ego_front_direction[0], 
            ego_front_direction[1],
            head_width=0.8, 
            head_length=1.0, 
            fc='blue', 
            ec='blue'
        )
        
    else:
        # If nusc is not available, use depth_image for the ego pose
        ego_translation_global = depth_image.image_pose_global[:3, 3]
    
    # If points is None, extract points from depth_image
    if points is None and depth_image:
        points = depth_image.get_all_points(with_labels=True)
    
    # Plot dynamic points
    if points['dynamic'].shape[0] > 0:
        ax.scatter(
            points['dynamic'][:, 0], 
            points['dynamic'][:, 1], 
            s=point_size_dynamic, 
            color='green', 
            label='MDet: OCCLUDING', 
            zorder=5
        )
    
    # Plot occluded points if desired
    if points['occluded'].shape[0] > 0 and video_cfg.get('show_occluded_points', False):
        ax.scatter(
            points['occluded'][:, 0], 
            points['occluded'][:, 1], 
            s=point_size_dynamic * 0.8, 
            color='red', 
            label='MDet: OCCLUDED', 
            zorder=4
        )
    
    # Set plot limits centered on ego vehicle
    ax.set_xlim(ego_translation_global[0] - bev_range_meters, ego_translation_global[0] + bev_range_meters)
    ax.set_ylim(ego_translation_global[1] - bev_range_meters, ego_translation_global[1] + bev_range_meters)
    ax.set_aspect('equal', adjustable='box')
    
    # Add labels and title
    ax.set_xlabel("Global X (meters)")
    ax.set_ylabel("Global Y (meters)")
    
    # Create informative title
    title_str = f"LiDAR Frame: {frame_index}"
    if depth_image:
        title_str += f" - TS: {depth_image.timestamp/1e6:.2f}s"
    if points['dynamic'].shape[0] > 0:
        title_str += f" - Dynamic Points: {points['dynamic'].shape[0]}"
    
    ax.set_title(title_str, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend on first frame
    # if frame_index == 0:
    ax.legend(loc='upper right', fontsize='small')
    
    # Convert matplotlib figure to OpenCV image
    from src.utils.visualization import mpl_fig_to_opencv_bgr
    frame_bgr = mpl_fig_to_opencv_bgr(fig)
    
    # Close the figure to prevent memory leaks
    plt.close(fig)
    
    return frame_bgr