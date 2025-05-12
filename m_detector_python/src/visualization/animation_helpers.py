# src/visualization/animation_helpers.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon # Ensure Polygon is imported
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from pyquaternion import Quaternion
# from scipy.spatial.transform import Slerp, Rotation as R # Not used in these animation funcs
# import json # Not used
# import inspect # Not used
# from PIL import Image # Not used
# import k3d # Not used in these animation funcs

from nuscenes.nuscenes import NuScenes # For type hinting
# from nuscenes.utils.data_classes import LidarPointCloud # Used by load_lidar_points_global
from nuscenes.utils.data_classes import Box as NuScenesDataClassesBox
# from nuscenes.utils.geometry_utils import transform_matrix, view_points # Not directly used here

# Import necessary functions that were previously in utils/refactor.py or need to be accessible
# Specifically, get_interpolated_extrapolated_boxes_for_instance and a point loader.

# Option 1: If label_generation is now the canonical source for box interpolation
from ..data_utils.label_generation import get_interpolated_extrapolated_boxes_for_instance
# Option 2: If point loading is now via nuscenes_helper
from ..data_utils.nuscenes_helper import get_lidar_sweep_data # For points in sensor frame + transform
from ..utils.transformations import transform_points_numpy   # For transforming points to global

# --- Animation Component Functions ---

def _prepare_animation_data(nusc: NuScenes, instance_tokens: list, all_sweep_data_dicts: list):
    instance_animation_data = {}
    # Initialize with valid numbers, but ones that will be easily overridden
    min_x_all, max_x_all = float('inf'), float('-inf')
    min_y_all, max_y_all = float('inf'), float('-inf')
    
    any_instance_has_boxes = False # Flag to track if any valid box was processed
    
    instance_categories = {}
    for token in instance_tokens:
        instance_rec = nusc.get('instance', token)
        first_ann_token = instance_rec['first_annotation_token']
        if first_ann_token:
            first_ann_rec = nusc.get('sample_annotation', first_ann_token)
            instance_categories[token] = first_ann_rec['category_name']
        else:
            instance_categories[token] = "N/A_NoAnns"

    if not instance_tokens: # If there are no instances to process
        print("Warning: No instance tokens provided to _prepare_animation_data.")
        # Return default plot limits and empty data
        default_plot_lims = {'xlim': (-50, 50), 'ylim': (-50, 50)}
        return {}, all_sweep_data_dicts, default_plot_lims


    for token in instance_tokens:
        boxes_at_sweeps, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
            nusc, token, all_sweep_data_dicts
        )
        
        instance_animation_data[token] = {
            'boxes_at_sweeps': boxes_at_sweeps, 
            'category': instance_categories.get(token, "N/A_CatError"),
        }

        # This loop calculates the overall min/max extents
        for box in boxes_at_sweeps: # boxes_at_sweeps is a list of Box objects or None
            if box is not None: # CRITICAL: Only process if a box object exists for this sweep
                any_instance_has_boxes = True # Mark that we found at least one box
                corners_xy = box.bottom_corners()[:2, :] # (2,4) array for BEV
                
                current_min_x = np.min(corners_xy[0, :])
                current_max_x = np.max(corners_xy[0, :])
                current_min_y = np.min(corners_xy[1, :])
                current_max_y = np.max(corners_xy[1, :])

                # Ensure these are finite before updating global min/max
                if np.isfinite(current_min_x):
                    min_x_all = min(min_x_all, current_min_x)
                if np.isfinite(current_max_x):
                    max_x_all = max(max_x_all, current_max_x)
                if np.isfinite(current_min_y):
                    min_y_all = min(min_y_all, current_min_y)
                if np.isfinite(current_max_y):
                    max_y_all = max(max_y_all, current_max_y)

    # After iterating through all instances and their boxes:
    if not any_instance_has_boxes:
        # This case means that although there might be instance_tokens,
        # none of them had any valid (non-None) boxes across all sweeps.
        # This could happen if instances are only present in parts of the scene
        # not covered by all_sweep_data_dicts, or if interpolation/extrapolation fails.
        print("Warning: No valid bounding boxes found for any instance to determine plot limits. Using defaults.")
        plot_lims = {'xlim': (-50, 50), 'ylim': (-50, 50)}
    elif not (np.isfinite(min_x_all) and np.isfinite(max_x_all) and \
              np.isfinite(min_y_all) and np.isfinite(max_y_all)):
        # This is a fallback: if any_instance_has_boxes was true, but min/max are still inf/nan
        # This shouldn't happen if the np.isfinite checks inside the loop are correct, but as a safeguard:
        print(f"Warning: Plot limits are still non-finite even after processing boxes. "
              f"min_x:{min_x_all}, max_x:{max_x_all}, min_y:{min_y_all}, max_y:{max_y_all}. Using defaults.")
        plot_lims = {'xlim': (-50, 50), 'ylim': (-50, 50)}
    else:
        # All good, calculate limits with padding
        padding = 15.0
        # Ensure min < max after padding, can happen if only one point or very small extent
        final_min_x = min_x_all - padding
        final_max_x = max_x_all + padding
        final_min_y = min_y_all - padding
        final_max_y = max_y_all + padding

        if final_min_x >= final_max_x: # If min is not strictly less than max
            final_min_x = final_min_x - 1 # Add some default spread
            final_max_x = final_max_x + 1
        if final_min_y >= final_max_y:
            final_min_y = final_min_y - 1
            final_max_y = final_max_y + 1
            
        plot_lims = {
            'xlim': (final_min_x, final_max_x),
            'ylim': (final_min_y, final_max_y)
        }
    
    return instance_animation_data, all_sweep_data_dicts, plot_lims



def _initialize_plot_artists(fig, ax, instance_animation_data, instance_colors, animation_lidar_sweeps_data_dicts):
    # animation_lidar_sweeps_data_dicts is the list of dicts from get_scene_sweep_data_sequence
    artists_dict = {
        'box_polys': {}, 'orientation_lines': {}, 'lidar_scatter': None,
        'time_text': None, 'legend_elements': [], 'lidar_cache': {}
    }

    for token, data in instance_animation_data.items():
        color = instance_colors.get(token, plt.cm.tab10.colors[0]) # Default color
        edgecolor = tuple(0.7 * np.array(color[:3])) # Darker edge
        poly = Polygon(np.zeros((4, 2)), closed=True, facecolor=color, edgecolor=edgecolor,
                       alpha=0.6, linewidth=1.5, zorder=10, visible=False)
        ax.add_patch(poly)
        artists_dict['box_polys'][token] = poly
        
        line, = ax.plot([], [], color=edgecolor, linewidth=2, zorder=11, visible=False)
        artists_dict['orientation_lines'][token] = line
        
        artists_dict['legend_elements'].append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, 
                       label=data.get('category', 'Unknown')) # Use .get for safety
        )

    artists_dict['lidar_scatter'] = ax.scatter([], [], s=1.5, color='dimgray', alpha=0.6, zorder=1)
    artists_dict['time_text'] = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                                        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=15)
    
    # Initialize with the first frame if data exists
    if animation_lidar_sweeps_data_dicts and instance_animation_data: # Check both
        # Get first box for each instance if available
        for token, data in instance_animation_data.items():
            if data['boxes_at_sweeps'] and data['boxes_at_sweeps'][0] is not None:
                initial_box = data['boxes_at_sweeps'][0]
                poly = artists_dict['box_polys'][token]
                poly.set_xy(initial_box.bottom_corners()[:2, :].T)
                poly.set_visible(True)
                
                line = artists_dict['orientation_lines'][token]
                # Calculate orientation vector (e.g., along length)
                # NuScenes Box: wlh is width, length, height. Local X is along length.
                front_vec_local = np.array([initial_box.wlh[1] / 2.0, 0, 0]) 
                front_vec_global = initial_box.orientation.rotation_matrix @ front_vec_local
                line.set_data(
                    [initial_box.center[0], initial_box.center[0] + front_vec_global[0]],
                    [initial_box.center[1], initial_box.center[1] + front_vec_global[1]]
                )
                line.set_visible(True)
    return artists_dict

def _update_animation_frame(frame_idx, nusc, artists_dict, instance_animation_data, 
                            animation_lidar_sweeps_data_dicts, 
                            plot_lims, scene_name, point_downsample):
    
    current_sweep_data = animation_lidar_sweeps_data_dicts[frame_idx] 
    returned_artists = []
    visible_instances_count = 0

    for inst_token, data in instance_animation_data.items():
        box = data['boxes_at_sweeps'][frame_idx]
        poly = artists_dict['box_polys'][inst_token]
        line = artists_dict['orientation_lines'][inst_token]

        if box is not None:
            visible_instances_count += 1
            poly.set_xy(box.bottom_corners()[:2, :].T); poly.set_visible(True)
            
            front_vec_local = np.array([box.wlh[1] / 2.0, 0, 0])
            front_vec_global = box.orientation.rotation_matrix @ front_vec_local
            line.set_data([box.center[0], box.center[0] + front_vec_global[0]],
                          [box.center[1], box.center[1] + front_vec_global[1]]); line.set_visible(True)
        else:
            poly.set_visible(False); line.set_visible(False)
        returned_artists.extend([poly, line])

    lidar_sd_token = current_sweep_data['lidar_sd_token'] 
    
    if lidar_sd_token not in artists_dict['lidar_cache']:
        # Use the new point loading mechanism
        points_sensor_frame, T_global_lidar, *_ = get_lidar_sweep_data(nusc, lidar_sd_token)
        if points_sensor_frame.shape[0] > 0:
            points_global = transform_points_numpy(points_sensor_frame, T_global_lidar)
            points_xy = points_global[:, :2] # Get XY for BEV plot
            if point_downsample > 1:
                points_xy = points_xy[::point_downsample, :]
            artists_dict['lidar_cache'][lidar_sd_token] = points_xy
        else:
            artists_dict['lidar_cache'][lidar_sd_token] = np.zeros((0,2)) # Cache empty if no points

    artists_dict['lidar_scatter'].set_offsets(artists_dict['lidar_cache'].get(lidar_sd_token, np.zeros((0,2))))
    returned_artists.append(artists_dict['lidar_scatter'])
    
    # Use the new timestamp key
    start_ts_us = animation_lidar_sweeps_data_dicts[0]['timestamp']
    current_ts_us = current_sweep_data['timestamp']
    relative_time_s = (current_ts_us - start_ts_us) / 1e6
    
    artists_dict['time_text'].set_text( # Updated text content for clarity
        f'Scene: {scene_name}\nFrame: {frame_idx}/{len(animation_lidar_sweeps_data_dicts)-1}\n'
        f'Time: {relative_time_s:.2f}s (Sweep: {lidar_sd_token[:6]}...)\n'
        f'Visible Inst: {visible_instances_count}/{len(instance_animation_data)}'
    )
    returned_artists.append(artists_dict['time_text'])

    ax = artists_dict['lidar_scatter'].axes 
    ax.set_xlim(plot_lims['xlim']); ax.set_ylim(plot_lims['ylim'])
    ax.set_aspect('equal', adjustable='box'); ax.set_xlabel("Global X (m)"); ax.set_ylabel("Global Y (m)")
    ax.set_title(f"NuScenes: {scene_name} - Interpolated/Extrapolated Annotations"); ax.grid(True, alpha=0.6)
       
    return returned_artists


def create_synchronized_animation(nusc: NuScenes, instance_tokens: list, 
                                  all_sweep_data_dicts: list, 
                                  scene_name: str ="Scene",
                                  interval_ms: int =50, figsize: tuple =(10, 10), point_downsample: int =1,
                                  save_path: str =None, save_writer: str =None, save_fps: int =20, save_dpi: int =200):
    
    if not all_sweep_data_dicts: # Check if the input list is empty
        print("Error: No sweep data provided for animation.")
        return None, None

    prepared_data = _prepare_animation_data(nusc, instance_tokens, all_sweep_data_dicts)
    
    if prepared_data is None or not prepared_data[0]: 
        print("Animation preparation failed or no instance data. No data to animate.")
        return None, None
        
    instance_animation_data, animation_sweeps_to_iterate, plot_lims = prepared_data
    # animation_sweeps_to_iterate is the same as all_sweep_data_dicts passed in

    if not animation_sweeps_to_iterate: # Should be redundant due to earlier check
        print("No LiDAR sweeps to animate after preparation.")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    # Generate colors for instances
    num_instances = len(instance_animation_data)
    # Ensure tab10 provides enough distinct colors, or use a different colormap for many instances
    colors_palette = plt.cm.get_cmap('tab10', max(10, num_instances)) 
    instance_colors = { token: colors_palette(i % num_instances) for i, token in enumerate(instance_animation_data.keys()) }

    artists_dict = _initialize_plot_artists(fig, ax, instance_animation_data, instance_colors, animation_sweeps_to_iterate)

    update_wrapper = lambda frame_idx: _update_animation_frame(
        frame_idx, nusc, artists_dict, instance_animation_data,
        animation_sweeps_to_iterate, plot_lims, scene_name, point_downsample
    )

    anim = FuncAnimation(fig, update_wrapper, frames=len(animation_sweeps_to_iterate), 
                         interval=interval_ms, blit=False) # blit=False is often more robust

    if save_path:
        actual_save_fps = save_fps if save_fps is not None else (1000 / interval_ms)
        try:
            anim.save(save_path, writer=save_writer, fps=actual_save_fps, dpi=save_dpi)
            print(f"Animation saved successfully to {save_path}")
            plt.close(fig); return None, anim # Return None for HTML if saved
        except Exception as e:
            print(f"Error saving animation: {e}. Generating HTML output instead.")
            html_output = HTML(anim.to_jshtml()); plt.close(fig); return html_output, anim
    else:
        html_output = HTML(anim.to_jshtml()); plt.close(fig); return html_output, anim