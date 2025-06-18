# src/visualization/video_helpers.py

import os
import cv2
import numpy as np
from typing import Dict, Optional, List, Any, Callable
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes # For type hinting

# M-Detector and constants
from ..core.m_detector.base import MDetector # For type hinting
from ..core.constants import OcclusionResult, POINT_LABEL_DTYPE 
from ..core.m_detector.processing import extract_mdetector_points

# Data utilities
from ..data_utils.nuscenes_helper import NuScenesProcessor, get_scene_sweep_data_sequence, get_lidar_sweep_data
from .transformations import transform_points_numpy
from ..utils.validation_utils import get_gt_dynamic_points_for_sweep # For GT dynamic points

# Visualization utilities
from ..visualization.visualization_utils import mpl_fig_to_opencv_bgr

# --- Core BEV Plotting Function ---
def plot_bev_frame(
    ax: plt.Axes,
    all_points_global: np.ndarray,
    ego_translation_global: np.ndarray,
    ego_rotation_global: Quaternion,
    points_to_highlight: Dict[str, np.ndarray],
    highlight_configs: Dict[str, Dict[str, Any]], # e.g., {'label': {'color':'red', 's':5, ...}}
    general_plot_config: Dict[str, Any],
    is_right_subplot: bool = False # To adjust labels for side-by-side
    ):
    """
    Populates a single Matplotlib Axes with a BEV plot.
    """
    bev_range = general_plot_config['bev_range_meters']
    bg_point_size = general_plot_config['point_size_all_lidar']
    bg_point_color = general_plot_config['point_color_all_lidar']
    bg_point_alpha = general_plot_config['point_alpha_all_lidar']

    # 1. Plot all LiDAR points as background
    if all_points_global.shape[0] > 0:
        ax.scatter(
            all_points_global[:, 0], all_points_global[:, 1],
            s=bg_point_size, color=bg_point_color, alpha=bg_point_alpha, zorder=1,
            label='_nolegend_' # Avoid legend entry for background points unless specified
        )

    # 2. Plot highlighted points
    legend_handles = []
    for label, points in points_to_highlight.items():
        if points.shape[0] > 0:
            config = highlight_configs.get(label, {}) # Get specific config for this label
            handle = ax.scatter(
                points[:, 0], points[:, 1],
                s=config['s'],
                color=config['color'],
                alpha=config['alpha'],
                label=config['legend_label'] + f" ({points.shape[0]})", # Add count to legend
                zorder=config['zorder']
            )
            legend_handles.append(handle)

    # 3. Plot Ego Vehicle
    ego_config = general_plot_config['ego_vehicle']
    ax.plot(
        ego_translation_global[0], ego_translation_global[1],
        marker=ego_config['marker'],
        markersize=ego_config['markersize'],
        color=ego_config['color'],
        label='_nolegend_' # Ego vehicle often doesn't need separate legend if consistent
    )
    ego_front_direction = ego_rotation_global.rotate(np.array([ego_config['arrow_length'], 0, 0]))
    ax.arrow(
        ego_translation_global[0], ego_translation_global[1],
        ego_front_direction[0], ego_front_direction[1],
        head_width=ego_config['arrow_head_width'],
        head_length=ego_config['arrow_head_length'],
        fc=ego_config['arrow_fc'],
        ec=ego_config['arrow_ec'],
        zorder=ego_config['arrow_zorder']
    )

    # 4. Aesthetics
    ax.set_xlim(ego_translation_global[0] - bev_range, ego_translation_global[0] + bev_range)
    ax.set_ylim(ego_translation_global[1] - bev_range, ego_translation_global[1] + bev_range)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Global X (m)")
    if not is_right_subplot:
        ax.set_ylabel("Global Y (m)")
    else:
        ax.set_yticklabels([]) # Clean up y-axis for right plot
    ax.set_title(general_plot_config['subplot_title'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    if legend_handles: # Only add legend if there are items to show
        ax.legend(handles=legend_handles, loc='upper right', fontsize='x-small')

# --- Frame Creation for Comparison Video ---
def create_comparison_visualization_frame(
    nusc: NuScenes,
    sweep_data_dict: Dict, # Dict from get_scene_sweep_data_sequence
    mdetector_output_points: Dict[str, np.ndarray], # Output from extract_mdetector_points
    config: Dict,
    frame_idx_in_video: int # For unique titles or conditional plotting
) -> np.ndarray:
    """
    Creates a side-by-side BEV visualization frame (GT vs. M-Detector).
    """
    video_cfg = config['video_generation']
    validation_cfg = config['validation']
    nuscenes_cfg = config['nuscenes']

    figure_size = video_cfg.get('figure_size_side_by_side', (20, 10)) # Inches
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=figure_size)
    
    gt_vel_thresh = validation_cfg['vel_threshold']
    fig.suptitle(f'Ground Truth (vel>={gt_vel_thresh}m/s) vs. M-Detector Predictions - Frame {frame_idx_in_video}', fontsize=14)

    # --- Common Data for Both Subplots ---
    points_sensor_frame = sweep_data_dict['points_sensor_frame']
    T_global_lidar = sweep_data_dict['T_global_lidar']
    all_points_global = transform_points_numpy(points_sensor_frame, T_global_lidar)
    
    # Ego pose from T_global_lidar (sensor to global)
    # To get ego pose, we need T_global_vehicle.
    # T_global_lidar = T_global_vehicle @ T_vehicle_lidar
    # We need T_vehicle_lidar (calibrated sensor).
    cs_rec = nusc.get('calibrated_sensor', sweep_data_dict['calibrated_sensor_token'])
    T_vehicle_lidar_np = np.eye(4)
    T_vehicle_lidar_np[:3,:3] = Quaternion(cs_rec['rotation']).rotation_matrix
    T_vehicle_lidar_np[:3,3] = np.array(cs_rec['translation'])
    T_lidar_vehicle_np = np.linalg.inv(T_vehicle_lidar_np) # Transform from vehicle to lidar
    
    T_global_vehicle = T_global_lidar @ T_lidar_vehicle_np # Global from Vehicle
    
    ego_translation_global = T_global_vehicle[:3, 3]
    ego_rotation_global = Quaternion(matrix=T_global_vehicle)


    # --- 1. Ground Truth Plot (Left Subplot: ax_gt) ---
    gt_points = get_gt_dynamic_points_for_sweep(
        nusc, sweep_data_dict, all_points_global,
        nuscenes_cfg['label_path'], gt_vel_thresh
    ) # Returns {'dynamic': ..., 'static': ...}

    gt_highlight_points = {'GT Dynamic': gt_points['dynamic']} # Only highlight dynamic for GT
    # If you also want to show GT static points differently from background, add here:
    # gt_highlight_points['GT Static'] = gt_points['static']
    
    gt_highlight_configs = {
        'GT Dynamic': {'color': 'blue', 's': video_cfg['point_size_dynamic_gt'], 'legend_label': 'GT Dynamic', 'zorder': 6},
        # 'GT Static': {'color': 'cyan', 's': 0.5, 'legend_label': 'GT Static', 'zorder': 2}
    }
    gt_general_config = {
        'bev_range_meters': video_cfg['bev_range_meters'],
        'point_size_all_lidar': video_cfg['point_size_all_lidar'],
        'subplot_title': f'Ground Truth (vel >= {gt_vel_thresh:.1f} m/s)',
        'ego_vehicle': {'color': 'darkred', 'arrow_fc': 'darkred', 'arrow_ec': 'darkred'} # Different ego color
    }
    plot_bev_frame(ax_gt, all_points_global, ego_translation_global, ego_rotation_global,
                   gt_highlight_points, gt_highlight_configs, gt_general_config, is_right_subplot=False)

    # --- 2. M-Detector Predictions Plot (Right Subplot: ax_pred) ---
    # mdetector_output_points is already {'dynamic': ..., 'occluded_by_mdet': ...}
    pred_highlight_points = {'MDet Dynamic': mdetector_output_points.get('dynamic', np.empty((0,3)))}
    if video_cfg['show_mdet_occluded_points']: # Configurable
        pred_highlight_points['MDet Occluded'] = mdetector_output_points.get('occluded_by_mdet', np.empty((0,3)))
    
    pred_highlight_configs = {
        'MDet Dynamic': {'color': 'green', 's': video_cfg['point_size_dynamic_pred'], 'legend_label': 'MDet: Dynamic', 'zorder': 5},
        'MDet Occluded': {'color': 'orange', 's': video_cfg['point_size_occluded_pred'], 'legend_label': 'MDet: Occluded', 'zorder': 4}
    }
    pred_general_config = {
        'bev_range_meters': video_cfg['bev_range_meters'],
        'point_size_all_lidar': video_cfg['point_size_all_lidar'],
        'subplot_title': 'M-Detector Predictions',
        'ego_vehicle': {'color': 'darkred', 'arrow_fc': 'darkred', 'arrow_ec': 'darkred'}
    }
    plot_bev_frame(ax_pred, all_points_global, ego_translation_global, ego_rotation_global,
                   pred_highlight_points, pred_highlight_configs, pred_general_config, is_right_subplot=True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    frame_bgr = mpl_fig_to_opencv_bgr(fig)
    plt.close(fig)
    return frame_bgr


# --- Main Video Generation Function (Orchestrator) ---
def generate_comparison_video(
    nusc: NuScenes,
    scene_index: int,
    mdetector_results_npz_filepath: str, 
    output_path: str,
    config: Dict
    ):
    """
    Generates a side-by-side comparison video: GT vs. M-Detector.
    """
    video_cfg = config['video_generation']
    processing_cfg = config['processing'] # For skip_frames, max_frames
    
    fps = video_cfg['fps']
    figure_size_inches = video_cfg.get('figure_size_side_by_side', (20, 10)) # [width, height]

    # Initialize NuScenesProcessor (which now uses get_scene_sweep_data_sequence)
    processor = NuScenesProcessor(nusc, config) # Pass config if NuScenesProcessor needs it for skip/max frames

    # To get video dimensions, create a dummy frame
    # Need a sample sweep_data_dict for the dummy frame. Get first one from the scene.
    scene_rec_for_init = nusc.scene[scene_index]
    temp_sweep_iter = get_scene_sweep_data_sequence(nusc, scene_rec_for_init['token'])
    try:
        first_sweep_data_for_init = next(temp_sweep_iter)
    except StopIteration:
        tqdm.write(f"Error: No sweeps found in scene {scene_rec_for_init['name']} for video dimension initialization.")
        return {'video_path': output_path, 'frames_processed': 0, 'detector_results': None}
    
    dummy_mdet_output = extract_mdetector_points(None) # Pass None or a dummy DepthImage if needed
    temp_frame_bgr = create_comparison_visualization_frame(nusc, first_sweep_data_for_init, dummy_mdet_output, config, 0)
    height, width, _ = temp_frame_bgr.shape
    
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    video_frame_idx_counter = 0 # Tracks frames written to video

    # Frame processing callback for NuScenesProcessor
    # It receives: detector_instance, index_of_frame_processed_by_detector, detector_result
    def frame_processing_callback(detector_instance: MDetector, processed_detector_frame_idx: int, _):
        nonlocal video_frame_idx_counter
        
        current_lidar_sd_token = detector_instance.current_lidar_sd_token
        if not current_lidar_sd_token:
            tqdm.write(f"Warning: current_lidar_sd_token not found in detector instance for frame {processed_detector_frame_idx}. Skipping video frame.")
            return

        current_sweep_data_dict = getattr(detector_instance, 'current_sweep_data_dict_processed', None)
        #     }
        
        # Get MDetector's output for the current frame
        try:
            # processed_detector_frame_idx is the index into detector.depth_image_library._images
            mdetector_depth_image_output = detector_instance.depth_image_library._images[processed_detector_frame_idx]
        except IndexError:
            tqdm.write(f"Warning: MDetector output not found for its frame index {processed_detector_frame_idx}. Skipping video frame.")
            return
        
        mdetector_points = extract_mdetector_points(mdetector_depth_image_output)
        
        # Create the comparison frame
        bgr_comparison_frame = create_comparison_visualization_frame(
            nusc,
            current_sweep_data_dict,
            mdetector_points,
            config,
            video_frame_idx_counter 
        )
        
        video_writer.write(bgr_comparison_frame)
        video_frame_idx_counter += 1

    # Process the scene using NuScenesProcessor
    detector_run_results = processor.process_scene(
        scene_index=scene_index, 
        detector=detector,
        frame_callback=frame_processing_callback,
        with_progress=True 
        # skip_frames and max_frames are handled by NuScenesProcessor via config
    )
    
    video_writer.release()
    
    return {
        'video_path': output_path,
        'frames_processed_for_video': video_frame_idx_counter,
        'detector_results': detector_run_results 
    }