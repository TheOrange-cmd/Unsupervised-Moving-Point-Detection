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
import h5py

# Utilities
from ..data_utils.nuscenes_helper import get_scene_sweep_data_sequence # To get original sweep data
from ..utils.transformations import transform_points_numpy
from ..data_utils.label_generation import get_interpolated_extrapolated_boxes_for_instance, find_instances_in_scene 
from ..core.constants import OcclusionResult
from ..config_loader import MDetectorConfigAccessor

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


def _get_data_from_hdf5_for_sweep(
    h5_file_handle: h5py.File,
    target_sweep_token: str,
    token_key: str,
    data_array_key: str,
    indices_array_key: str
) -> Optional[np.ndarray]:
    """Helper to extract data for a specific sweep from an open HDF5 file handle."""
    try:
        all_tokens_bytes = h5_file_handle[token_key][:] # Load tokens
        # Tokens in HDF5 might be bytes, convert target_sweep_token to bytes if not already
        target_token_bytes = target_sweep_token.encode('utf-8') if isinstance(target_sweep_token, str) else target_sweep_token

        # Decode tokens from HDF5 if they are bytes, for comparison
        all_tokens_str = [t.decode('utf-8') for t in all_tokens_bytes]

        matches = np.where(np.array(all_tokens_str) == target_sweep_token)[0]

        if not matches.size > 0:
            return None
        idx_in_token_list = matches[0]

        indices_array = h5_file_handle[indices_array_key][:] # Load indices
        start_idx = indices_array[idx_in_token_list]
        # Ensure idx_in_token_list + 1 is a valid index for indices_array
        if idx_in_token_list + 1 >= len(indices_array):
            tqdm.write(f"Warning: Index issue for token {target_sweep_token}. "
                       f"idx_in_token_list+1 ({idx_in_token_list + 1}) is out of bounds for "
                       f"indices_array_key '{indices_array_key}' (len: {len(indices_array)}).")
            return None # Or handle as end of list if appropriate for your data structure
        end_idx = indices_array[idx_in_token_list + 1]

        return h5_file_handle[data_array_key][start_idx:end_idx]
    except KeyError as e:
        # tqdm.write(f"KeyError in _get_data_from_hdf5_for_sweep: {e} for token {target_sweep_token}")
        return None
    except IndexError:
        tqdm.write(f"Warning: Index issue for token {target_sweep_token} in HDF5 with key {indices_array_key}")
        return None
    except Exception as e_gen:
        tqdm.write(f"Generic error in _get_data_from_hdf5_for_sweep for {target_sweep_token}: {e_gen}")
        return None


def visualize_sweep_k3d(
    nusc: NuScenes,
    scene_token: str,
    target_sweep_lidar_sd_token: str,
    gt_labels_hdf5_path: Optional[str] = None,
    mdet_results_hdf5_path: Optional[str] = None,
    # --- MODIFIED: Accept config_accessor ---
    config_accessor: Optional[MDetectorConfigAccessor] = None,
    show_gt_boxes: bool = True,
    show_gt_points: bool = True,
    show_mdet_points: bool = True,
    point_size: float = 0.05, # This can also be moved to config
    downsample_factor: int = 1
):
    # --- Use config_accessor to get k3d_plot parameters ---
    k3d_plot_cfg = {}
    if config_accessor:
        k3d_plot_cfg = config_accessor.get_k3d_plot_params()
    # --- End config access changes ---

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
                    grid_visible=k3d_plot_cfg.get('grid_visible', True),
                    camera_auto_fit=k3d_plot_cfg.get('camera_auto_fit', True),
                    background_color=k3d_plot_cfg.get('plot_background_color_hex', K3D_COLOR_BACKGROUND))


    bg_points_to_plot = raw_points_global[::downsample_factor]
    plot += k3d.points(positions=bg_points_to_plot.astype(np.float32),
                       color=k3d_plot_cfg.get('background_points_color_hex', K3D_COLOR_BACKGROUND),
                       point_size=point_size * k3d_plot_cfg.get('background_points_scale_factor', 0.7),
                       shader='3d', name='Background_LiDAR')

    if gt_labels_hdf5_path and show_gt_points:
        try:
            with h5py.File(gt_labels_hdf5_path, 'r') as gt_h5:
                gt_point_labels_for_sweep = _get_data_from_hdf5_for_sweep(
                    gt_h5, target_sweep_lidar_sd_token,
                    'sweep_lidar_sd_tokens', 'all_gt_point_labels', 'gt_point_labels_indices'
                )
                if gt_point_labels_for_sweep is not None and \
                   gt_point_labels_for_sweep.shape[0] == raw_points_global.shape[0]:
                    
                    # Use velocity threshold from validation_params via config_accessor if available
                    # Fallback to a default or a k3d_plot_cfg specific one
                    vel_thresh_for_gt_dynamic = 0.1 # Default
                    if config_accessor:
                        validation_p = config_accessor.get_validation_params()
                        vel_thresh_for_gt_dynamic = validation_p.get('gt_velocity_threshold', 0.1)
                    
                    gt_dynamic_mask = (gt_point_labels_for_sweep['velocity_x']**2 +
                                       gt_point_labels_for_sweep['velocity_y']**2 >
                                       vel_thresh_for_gt_dynamic**2) # Use fetched threshold
                    
                    gt_dynamic_pts = raw_points_global[gt_dynamic_mask]
                    if gt_dynamic_pts.shape[0] > 0:
                        plot += k3d.points(positions=gt_dynamic_pts.astype(np.float32),
                                           color=k3d_plot_cfg.get('gt_dynamic_points_color_hex', K3D_COLOR_GT_DYNAMIC),
                                           point_size=point_size * k3d_plot_cfg.get('gt_dynamic_points_scale_factor', 1.0), 
                                           shader='3d', name='GT_Dynamic_Points')

                    gt_instance_labeled_mask = (gt_point_labels_for_sweep['instance_token'] != b'')
                    gt_static_labeled_mask = gt_instance_labeled_mask & ~gt_dynamic_mask
                    gt_static_pts = raw_points_global[gt_static_labeled_mask]
                    if gt_static_pts.shape[0] > 0:
                         plot += k3d.points(positions=gt_static_pts.astype(np.float32),
                                           color=k3d_plot_cfg.get('gt_static_points_color_hex', K3D_COLOR_GT_STATIC),
                                           point_size=point_size * k3d_plot_cfg.get('gt_static_points_scale_factor', 0.8),
                                           shader='3d', name='GT_Static_Points')
        except Exception as e:
            print(f"Error loading GT labels HDF5 {gt_labels_hdf5_path}: {e}")

    if mdet_results_hdf5_path and show_mdet_points:
        try:
            with h5py.File(mdet_results_hdf5_path, 'r') as mdet_h5:
                mdet_all_preds_for_sweep = _get_data_from_hdf5_for_sweep(
                    mdet_h5, target_sweep_lidar_sd_token,
                    'sweep_lidar_sd_tokens', 'all_points_predictions', 'points_predictions_indices'
                )
                if mdet_all_preds_for_sweep is not None:
                    mdet_dynamic_val = k3d_plot_cfg.get('mdet_dynamic_label_value', OcclusionResult.OCCLUDING_IMAGE.value)
                    mdet_occluded_val = k3d_plot_cfg.get('mdet_occluded_label_value', OcclusionResult.OCCLUDED_BY_IMAGE.value)
                    mdet_undetermined_val = k3d_plot_cfg.get('mdet_undetermined_label_value', OcclusionResult.UNDETERMINED.value)


                    mdet_dynamic_mask = (mdet_all_preds_for_sweep['mdet_label'] == mdet_dynamic_val)
                    mdet_dyn_pts_global = np.stack((mdet_all_preds_for_sweep['x'][mdet_dynamic_mask],
                                                    mdet_all_preds_for_sweep['y'][mdet_dynamic_mask],
                                                    mdet_all_preds_for_sweep['z'][mdet_dynamic_mask]), axis=-1)
                    if mdet_dyn_pts_global.shape[0] > 0:
                        plot += k3d.points(positions=mdet_dyn_pts_global.astype(np.float32),
                                           color=k3d_plot_cfg.get('mdet_dynamic_points_color_hex', K3D_COLOR_MDET_DYNAMIC),
                                           point_size=point_size * k3d_plot_cfg.get('mdet_dynamic_points_scale_factor', 1.0), 
                                           shader='3d', name='MDet_Dynamic_Points')

                    mdet_occluded_mask = (mdet_all_preds_for_sweep['mdet_label'] == mdet_occluded_val)
                    mdet_occ_pts_global = np.stack((mdet_all_preds_for_sweep['x'][mdet_occluded_mask],
                                                    mdet_all_preds_for_sweep['y'][mdet_occluded_mask],
                                                    mdet_all_preds_for_sweep['z'][mdet_occluded_mask]), axis=-1)
                    if mdet_occ_pts_global.shape[0] > 0:
                        plot += k3d.points(positions=mdet_occ_pts_global.astype(np.float32),
                                           color=k3d_plot_cfg.get('mdet_occluded_points_color_hex', K3D_COLOR_MDET_OCCLUDED),
                                           point_size=point_size * k3d_plot_cfg.get('mdet_occluded_points_scale_factor', 0.9), 
                                           shader='3d', name='MDet_Occluded_Points')
                    
                    mdet_undetermined_mask = (mdet_all_preds_for_sweep['mdet_label'] == mdet_undetermined_val)
                    mdet_und_pts_global = np.stack((mdet_all_preds_for_sweep['x'][mdet_undetermined_mask],
                                                    mdet_all_preds_for_sweep['y'][mdet_undetermined_mask],
                                                    mdet_all_preds_for_sweep['z'][mdet_undetermined_mask]), axis=-1)
                    if mdet_und_pts_global.shape[0] > 0 and k3d_plot_cfg.get('show_mdet_undetermined_points', False):
                        plot += k3d.points(positions=mdet_und_pts_global.astype(np.float32),
                                           color=k3d_plot_cfg.get('mdet_undetermined_points_color_hex', K3D_COLOR_MDET_UNDETERMINED),
                                           point_size=point_size * k3d_plot_cfg.get('mdet_undetermined_points_scale_factor', 0.7),
                                           shader='3d', name='MDet_Undetermined_Points')
        except Exception as e:
            print(f"Error loading M-Detector results HDF5 {mdet_results_hdf5_path}: {e}")

    if show_gt_boxes:
        instance_tokens_in_scene = find_instances_in_scene(nusc, scene_token, min_annotations=1)
        all_scene_sweeps_list = list(get_scene_sweep_data_sequence(nusc, scene_token)) # Expensive if called repeatedly
        current_sweep_idx_in_list = -1
        for idx, sdd_box in enumerate(all_scene_sweeps_list):
            if sdd_box['lidar_sd_token'] == target_sweep_lidar_sd_token:
                current_sweep_idx_in_list = idx
                break
        if current_sweep_idx_in_list != -1:
            num_instance_colors = k3d_plot_cfg.get('num_instance_box_colors', 20)
            instance_colormap_name = k3d_plot_cfg.get('instance_box_colormap', 'tab20')
            color_map_instances = plt.cm.get_cmap(instance_colormap_name, num_instance_colors)
            inst_color_idx = 0
            for inst_token in instance_tokens_in_scene:
                boxes_for_inst, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
                    nusc, inst_token, all_scene_sweeps_list
                )
                gt_box_at_sweep: Optional[NuScenesDataClassesBox] = boxes_for_inst[current_sweep_idx_in_list]
                if gt_box_at_sweep:
                    instance_color_rgb = color_map_instances(inst_color_idx % num_instance_colors)[:3]
                    instance_color_hex = int(instance_color_rgb[0]*255)<<16 | int(instance_color_rgb[1]*255)<<8 | int(instance_color_rgb[2]*255)
                    inst_color_idx += 1
                    corners = gt_box_at_sweep.corners()
                    box_edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7)
                    ]
                    for start_idx, end_idx in box_edges:
                        segment_vertices = corners[:, [start_idx, end_idx]].T.astype(np.float32)
                        plot += k3d.line(segment_vertices, shader='simple', 
                                         color=instance_color_hex,
                                         width=k3d_plot_cfg.get('gt_box_line_width', 0.03),
                                         name=f'GTBox_{inst_token[:6]}')
    return plot