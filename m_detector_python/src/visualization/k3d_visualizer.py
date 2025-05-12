# src/visualization/k3d_visualizer.py
import os
import numpy as np
import k3d # Make sure k3d is installed
import matplotlib.pyplot as plt # For colormaps
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesDataClassesBox # For drawing boxes
from pyquaternion import Quaternion
from typing import Optional, Dict
from tqdm import tqdm 

# Utilities
from ..data_utils.nuscenes_helper import get_scene_sweep_data_sequence # To get original sweep data
from ..utils.transformations import transform_points_numpy
from ..data_utils.label_generation import get_interpolated_extrapolated_boxes_for_instance, find_instances_in_scene 

# Define default colors (can be moved to a constants file)
K3D_COLOR_BACKGROUND = 0xAAAAAA  # Light grey
K3D_COLOR_GT_DYNAMIC = 0x0000FF  # Blue
K3D_COLOR_GT_STATIC = 0x00FFFF   # Cyan (example)
K3D_COLOR_MDET_DYNAMIC = 0x00FF00 # Green
K3D_COLOR_MDET_OCCLUDED = 0xFFA500 # Orange
K3D_COLOR_MDET_UNDETERMINED = 0xFFFF00 # Yellow
K3D_COLOR_TP = 0x32CD32 # LimeGreen
K3D_COLOR_FP = 0xFF0000 # Red
K3D_COLOR_FN_POINTS = 0xFFC0CB # Pink (for GT points missed by detector)


def _get_data_from_npz_for_sweep(
    npz_data: np.lib.npyio.NpzFile, 
    target_sweep_token: str,
    token_key: str, # e.g., 'sweep_lidar_sd_tokens'
    data_array_key: str, # e.g., 'all_gt_point_labels' or 'all_dynamic_points'
    indices_array_key: str # e.g., 'gt_point_labels_indices' or 'dynamic_points_indices'
) -> Optional[np.ndarray]:
    """Helper to extract data for a specific sweep from an NPZ file."""
    try:
        all_tokens = npz_data[token_key].astype(str)
        matches = np.where(all_tokens == target_sweep_token)[0]
        if not matches.size > 0:
            return None
        idx_in_token_list = matches[0]
        
        start_idx = npz_data[indices_array_key][idx_in_token_list]
        end_idx = npz_data[indices_array_key][idx_in_token_list + 1]
        return npz_data[data_array_key][start_idx:end_idx]
    except KeyError:
        return None # Or raise error
    except IndexError: # If idx_in_token_list + 1 is out of bounds for indices_array_key
        tqdm.write(f"Warning: Index issue for token {target_sweep_token} in NPZ with key {indices_array_key}")
        return None


def visualize_sweep_k3d(
    nusc: NuScenes,
    scene_token: str,
    target_sweep_lidar_sd_token: str,
    gt_labels_npz_path: Optional[str] = None,      # Path to the GT point labels NPZ
    mdet_results_npz_path: Optional[str] = None, # Path to M-Detector results NPZ
    config: Optional[Dict] = None,               # For visualization settings
    show_gt_boxes: bool = True,
    show_gt_points: bool = True,
    show_mdet_points: bool = True,
    # Future: show_tp_fp_fn: bool = False,
    point_size: float = 0.05,
    downsample_factor: int = 1 # Downsample raw LiDAR for background display
):
    """
    Visualizes a single LiDAR sweep in 3D using K3D, optionally showing
    GT labels (points and boxes) and M-Detector classified points.
    """
    viz_cfg = config.get('visualization', {}).get('k3d_plot', {}) if config else {}
    
    # --- 1. Get original LiDAR sweep data ---
    original_sweep_data = None
    for sweep_d in get_scene_sweep_data_sequence(nusc, scene_token):
        if sweep_d['lidar_sd_token'] == target_sweep_lidar_sd_token:
            original_sweep_data = sweep_d
            break
    if not original_sweep_data:
        print(f"Error: LiDAR sweep {target_sweep_lidar_sd_token} not found in scene {scene_token}.")
        return None
    
    points_sensor_frame = original_sweep_data['points_sensor_frame']
    T_global_lidar = original_sweep_data['T_global_lidar']
    raw_points_global = transform_points_numpy(points_sensor_frame, T_global_lidar)

    if raw_points_global.shape[0] == 0:
        print(f"No points in sweep {target_sweep_lidar_sd_token}.")
        return None

    plot = k3d.plot(name=f"Scene: {nusc.get('scene', scene_token)['name']} - Sweep: {target_sweep_lidar_sd_token[:8]}", 
                    grid_visible=viz_cfg.get('grid_visible', True),
                    camera_auto_fit=viz_cfg.get('camera_auto_fit', True))

    # --- Plot Raw LiDAR as Background ---
    bg_points_to_plot = raw_points_global[::downsample_factor]
    plot += k3d.points(positions=bg_points_to_plot.astype(np.float32),
                       color=viz_cfg.get('background_color', K3D_COLOR_BACKGROUND),
                       point_size=point_size * viz_cfg.get('background_point_scale', 0.7), 
                       shader='3d', name='Background_LiDAR')

    # --- 2. Load and Process GT Labels (if path provided) ---
    gt_point_labels_for_sweep: Optional[np.ndarray] = None
    if gt_labels_npz_path and show_gt_points:
        try:
            gt_npz = np.load(gt_labels_npz_path, allow_pickle=True)
            gt_point_labels_for_sweep = _get_data_from_npz_for_sweep(
                gt_npz, target_sweep_lidar_sd_token,
                'sweep_lidar_sd_tokens', 'all_gt_point_labels', 'gt_point_labels_indices'
            )
            if gt_point_labels_for_sweep is not None and gt_point_labels_for_sweep.shape[0] == raw_points_global.shape[0]:
                print(f"Loaded {gt_point_labels_for_sweep.shape[0]} GT point labels for the sweep.")
                
                # Plot GT dynamic points
                gt_dynamic_mask = (gt_point_labels_for_sweep['velocity_x']**2 + 
                                   gt_point_labels_for_sweep['velocity_y']**2 > 
                                   viz_cfg.get('gt_dynamic_vel_threshold_sq', 0.1**2)) # Example threshold
                
                gt_dynamic_pts = raw_points_global[gt_dynamic_mask]
                if gt_dynamic_pts.shape[0] > 0:
                    plot += k3d.points(positions=gt_dynamic_pts.astype(np.float32),
                                       color=viz_cfg.get('gt_dynamic_color', K3D_COLOR_GT_DYNAMIC),
                                       point_size=point_size, shader='3d', name='GT_Dynamic_Points')
                
                # Plot GT static points (points with instance label but not dynamic)
                gt_instance_labeled_mask = (gt_point_labels_for_sweep['instance_token'] != b'')
                gt_static_labeled_mask = gt_instance_labeled_mask & ~gt_dynamic_mask
                gt_static_pts = raw_points_global[gt_static_labeled_mask]
                if gt_static_pts.shape[0] > 0:
                     plot += k3d.points(positions=gt_static_pts.astype(np.float32),
                                       color=viz_cfg.get('gt_static_color', K3D_COLOR_GT_STATIC),
                                       point_size=point_size * viz_cfg.get('gt_static_point_scale', 0.8), 
                                       shader='3d', name='GT_Static_Points')
            else:
                print(f"Warning: GT point labels for sweep {target_sweep_lidar_sd_token} not found or mismatch count.")
        except Exception as e:
            print(f"Error loading GT labels NPZ {gt_labels_npz_path}: {e}")

    # --- 3. Load and Process M-Detector Results (if path provided) ---
    if mdet_results_npz_path and show_mdet_points:
        try:
            mdet_npz = np.load(mdet_results_npz_path, allow_pickle=True)
            # Get MDet dynamic points
            mdet_dyn_pts = _get_data_from_npz_for_sweep(
                mdet_npz, target_sweep_lidar_sd_token,
                'sweep_lidar_sd_tokens', 'all_dynamic_points', 'dynamic_points_indices'
            )
            if mdet_dyn_pts is not None and mdet_dyn_pts.shape[0] > 0:
                plot += k3d.points(positions=mdet_dyn_pts.astype(np.float32),
                                   color=viz_cfg.get('mdet_dynamic_color', K3D_COLOR_MDET_DYNAMIC),
                                   point_size=point_size, shader='3d', name='MDet_Dynamic_Points')

            # Get MDet occluded points
            mdet_occ_pts = _get_data_from_npz_for_sweep(
                mdet_npz, target_sweep_lidar_sd_token,
                'sweep_lidar_sd_tokens', 'all_occluded_points', 'occluded_points_indices'
            )
            if mdet_occ_pts is not None and mdet_occ_pts.shape[0] > 0:
                plot += k3d.points(positions=mdet_occ_pts.astype(np.float32),
                                   color=viz_cfg.get('mdet_occluded_color', K3D_COLOR_MDET_OCCLUDED),
                                   point_size=point_size, shader='3d', name='MDet_Occluded_Points')
            # Add undetermined if needed
        except Exception as e:
            print(f"Error loading M-Detector results NPZ {mdet_results_npz_path}: {e}")

    # --- 4. Show GT Boxes (re-calculating them for the specific sweep) ---
    if show_gt_boxes:
        instance_tokens_in_scene = find_instances_in_scene(nusc, scene_token, min_annotations=1)
        # We need all_sweep_data_dicts for get_interpolated_extrapolated_boxes_for_instance
        # and the index of the current sweep within that list.
        all_scene_sweeps_list = list(get_scene_sweep_data_sequence(nusc, scene_token))
        current_sweep_idx_in_list = -1
        for idx, sdd in enumerate(all_scene_sweeps_list):
            if sdd['lidar_sd_token'] == target_sweep_lidar_sd_token:
                current_sweep_idx_in_list = idx
                break
        
        if current_sweep_idx_in_list != -1:
            color_map_instances = plt.cm.get_cmap(viz_cfg.get('instance_colormap', 'tab20'), 20)
            inst_color_idx = 0
            for inst_token in instance_tokens_in_scene:
                boxes_for_inst, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
                    nusc, inst_token, all_scene_sweeps_list
                )
                gt_box_at_sweep: Optional[NuScenesDataClassesBox] = boxes_for_inst[current_sweep_idx_in_list]

                if gt_box_at_sweep:
                    instance_color_rgb = color_map_instances(inst_color_idx % 20)[:3] # Cycle through colormap
                    instance_color_hex = int(instance_color_rgb[0]*255)<<16 | int(instance_color_rgb[1]*255)<<8 | int(instance_color_rgb[2]*255)
                    inst_color_idx += 1
                    
                    corners = gt_box_at_sweep.corners() # (3,8)
                    box_edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7)
                    ]
                    for start_idx, end_idx in box_edges:
                        segment_vertices = corners[:, [start_idx, end_idx]].T.astype(np.float32)
                        plot += k3d.line(segment_vertices, shader='simple', color=instance_color_hex, 
                                         width=viz_cfg.get('gt_box_line_width', 0.03), 
                                         name=f'GTBox_{inst_token[:6]}')
        else:
            print(f"Could not determine sweep index for {target_sweep_lidar_sd_token} to draw GT boxes.")
            
    # --- 5. Future: TP/FP/FN Visualization ---
    # if show_tp_fp_fn:
    #    # This would involve:
    #    # 1. Matching GT points to MDet points (e.g., using a bipartite matching on dynamic points).
    #    # 2. Classifying MDet points as TP or FP.
    #    # 3. Classifying GT dynamic points not matched as FN.
    #    # Then plot these sets with distinct colors.
    #    pass

    return plot