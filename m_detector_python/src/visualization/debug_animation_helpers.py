# src/visualization/debug_animation_helpers.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML # For notebook display
from pyquaternion import Quaternion
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm # For progress if any long loops are involved

from nuscenes.nuscenes import NuScenes

from ..data_utils.nuscenes_helper import get_lidar_sweep_data
from ..utils.transformations import transform_points_numpy
from ..core.depth_image import DepthImage # To create historical DI for context points
from ..core.m_detector.base import MDetector # For type hinting and accessing config

# Define some default colors for debug animation (can be moved to constants or config)
DEBUG_ANIM_COLOR_BACKGROUND = 'lightgrey'
DEBUG_ANIM_COLOR_DEBUG_POINT = 'lime' # Bright green for the point being debugged
DEBUG_ANIM_COLOR_CONTEXT_POINTS = 'deepskyblue' # Blue for context points from historical frame
DEBUG_ANIM_COLOR_EGO = 'red'


def _prepare_debug_animation_data(
    nusc: NuScenes,
    target_sweep_sd_token: str,     # SD token of the 'current_di_from_detector_logic'
    mdetector: MDetector,           # MDetector instance
    current_di_from_detector_logic: DepthImage, # The actual 'current_di' object
    pt_idx_in_current_di: int,      # The 'pt_idx' being debugged in current_di
    idx_of_current_di_in_lib: int,  # Library index of current_di_from_detector_logic
    num_past_sweeps_to_show: int = 1 # Typically 1 for this comparison
) -> Optional[Dict[str, Any]]:
    """
    Prepares data for the debug animation.
    Crucially, it now uses the provided 'current_di_from_detector_logic' to get the
    debug point's global coordinates and retrieves the historical DI directly from
    the mdetector's library.
    """
    prepared_data = {
        'animation_sweeps_data_dicts': [], # For plotting background sweeps
        'debug_point_global_coord': None,
        'debug_point_depth_in_hist_frame': None,
        'projected_pixel_in_hist_di': None, # Store (v,h) for comparison
        'context_points_global': np.empty((0,3)),
        'min_depth_context_point_global_xy': None,
        'min_depth_context_value_in_hist_frame': None,
        'max_depth_context_point_global_xy': None,
        'max_depth_context_value_in_hist_frame': None,
        'target_sweep_sd_token': target_sweep_sd_token,
        'historical_sweep_sd_token_for_context': None,
        'plot_lims': {'xlim': (-30, 30), 'ylim': (-30, 30)},
        'ego_poses': {}
    }
    print(f"    ANIM_HELPER: Preparing animation data for target SD token: {target_sweep_sd_token}, pt_idx: {pt_idx_in_current_di}")

    # 1. Get target sweep data & debug point global coordinate
    try:
        if current_di_from_detector_logic.original_points_global_coords is None or \
           pt_idx_in_current_di >= current_di_from_detector_logic.original_points_global_coords.shape[0]:
            print(f"    ANIM_HELPER Error: pt_idx_in_current_di ({pt_idx_in_current_di}) is out of bounds for current_di.")
            return None
        
        # Use the exact global coordinate from the DI M-Detector is using for the debug point
        prepared_data['debug_point_global_coord'] = current_di_from_detector_logic.original_points_global_coords[pt_idx_in_current_di, :3].copy()
        print(f"    ANIM_HELPER: Definitive Debug Point Global Coords (from current_di): {np.round(prepared_data['debug_point_global_coord'], 6).tolist()}")

        # For plotting the target sweep's background/ego, get its metadata
        _, target_T_global_lidar, target_ts, _, _, _, _ = \
            get_lidar_sweep_data(nusc, target_sweep_sd_token) # target_sweep_sd_token is from current_di

        target_sweep_plot_dict = {
            'T_global_lidar': target_T_global_lidar,
            'timestamp': target_ts,
            'lidar_sd_token': target_sweep_sd_token,
            'points_sensor_frame': transform_points_numpy( # Points for background
                current_di_from_detector_logic.original_points_global_coords,
                np.linalg.inv(target_T_global_lidar)
            ) if current_di_from_detector_logic.original_points_global_coords is not None else np.empty((0,3))
        }
        target_ego_pose_rec = nusc.get('ego_pose', nusc.get('sample_data', target_sweep_sd_token)['ego_pose_token'])
        prepared_data['ego_poses'][target_sweep_sd_token] = (
            np.array(target_ego_pose_rec['translation']), Quaternion(target_ego_pose_rec['rotation'])
        )
    except Exception as e:
        print(f"    ANIM_HELPER Error preparing target sweep data for {target_sweep_sd_token}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 2. Get the *actual* historical DepthImage from M-Detector's library for context calculation
    #    and its metadata for plotting.
    immediate_hist_di_from_lib: Optional[DepthImage] = None
    hist_sd_token_for_plot: Optional[str] = None # SD token of the historical sweep

    if num_past_sweeps_to_show > 0: # Only if we need to show/use past sweeps
        idx_of_immediate_hist_di_in_lib = idx_of_current_di_in_lib - 1
        if 0 <= idx_of_immediate_hist_di_in_lib < len(mdetector.depth_image_library.get_all_images()):
            immediate_hist_di_from_lib = mdetector.depth_image_library.get_image_by_index(idx_of_immediate_hist_di_in_lib)
            if immediate_hist_di_from_lib:
                print(f"    ANIM_HELPER: Retrieved historical DI from library (TS: {immediate_hist_di_from_lib.timestamp}) for context.")
                # Try to find the sd_token for this historical DI for plotting its background
                # This assumes unique timestamps or that the DI stores its original sd_token (which it doesn't currently)
                # A more robust way: if DepthImage stored its sd_token.
                # For now, trace back from target_sweep_sd_token.
                temp_sd_token_trace = target_sweep_sd_token
                for _ in range(idx_of_current_di_in_lib - idx_of_immediate_hist_di_in_lib): # Should be 1 step for immediate past
                    prev_sdd_rec = nusc.get('sample_data', temp_sd_token_trace)
                    if not prev_sdd_rec['prev']:
                        hist_sd_token_for_plot = None; break
                    temp_sd_token_trace = prev_sdd_rec['prev']
                hist_sd_token_for_plot = temp_sd_token_trace
                
                if hist_sd_token_for_plot:
                    prepared_data['historical_sweep_sd_token_for_context'] = hist_sd_token_for_plot
                    # Get metadata for plotting the historical sweep's background/ego
                    _, hist_T_global_lidar_plot, hist_ts_plot, _, _, _, _ = \
                        get_lidar_sweep_data(nusc, hist_sd_token_for_plot)
                    
                    hist_sweep_plot_dict = {
                        'T_global_lidar': hist_T_global_lidar_plot,
                        'timestamp': hist_ts_plot,
                        'lidar_sd_token': hist_sd_token_for_plot,
                        'points_sensor_frame': transform_points_numpy( # Points for background
                            immediate_hist_di_from_lib.original_points_global_coords,
                            np.linalg.inv(hist_T_global_lidar_plot)
                        ) if immediate_hist_di_from_lib.original_points_global_coords is not None else np.empty((0,3))
                    }
                    prepared_data['animation_sweeps_data_dicts'].append(hist_sweep_plot_dict)
                    hist_ego_pose_rec = nusc.get('ego_pose', nusc.get('sample_data', hist_sd_token_for_plot)['ego_pose_token'])
                    prepared_data['ego_poses'][hist_sd_token_for_plot] = (
                        np.array(hist_ego_pose_rec['translation']), Quaternion(hist_ego_pose_rec['rotation'])
                    )
                else:
                    print("    ANIM_HELPER Warning: Could not determine sd_token for historical DI for plotting background.")
                    immediate_hist_di_from_lib = None # Can't use if we can't plot its background
            else:
                print(f"    ANIM_HELPER: Historical DI at index {idx_of_immediate_hist_di_in_lib} is None.")
        else:
            print(f"    ANIM_HELPER: No valid historical DI index ({idx_of_immediate_hist_di_in_lib}) in library.")
    
    # Add target sweep's plot dict last for animation order
    prepared_data['animation_sweeps_data_dicts'].append(target_sweep_plot_dict)

    # 3. Identify context points using the historical DI from the library
    if immediate_hist_di_from_lib:
        # Ensure the historical DI is prepared
        if immediate_hist_di_from_lib.local_sph_coords_for_points is None or \
           not immediate_hist_di_from_lib.pixel_original_indices: # Check if defaultdict is empty
            print("    ANIM_HELPER Warning: Historical DI from library seems unprepared (missing local_sph_coords or pixel_original_indices). Context search might be incomplete.")
            # Potentially, one could try to prepare it here if it's missing, but it should have been done by MDetector's processing.
            # For now, proceed, but be aware results might be skewed if it's truly unprepared.

        hist_local_sph_coords_all = immediate_hist_di_from_lib.get_local_sph_coords_for_all_points()

        _point_in_hist_di_frame, sph_coords_debug_in_hist_di, pixel_indices_proj = \
            immediate_hist_di_from_lib.project_point_to_pixel_indices(
                prepared_data['debug_point_global_coord'] # Uses the coord from current_di_from_detector_logic
            )
        
        prepared_data['projected_pixel_in_hist_di'] = pixel_indices_proj # Store for comparison
        print(f"    ANIM_HELPER: Projected (v,h) in past DI: {pixel_indices_proj}") # <--- KEY PRINT
            
        if sph_coords_debug_in_hist_di is not None:
            prepared_data['debug_point_depth_in_hist_frame'] = sph_coords_debug_in_hist_di[2]
            print(f"    ANIM_HELPER: Debug point depth in hist frame (d_curr for animation): {prepared_data['debug_point_depth_in_hist_frame']:.3f}")


        if pixel_indices_proj and hist_local_sph_coords_all is not None:
            v_proj, h_proj = pixel_indices_proj
            context_indices_in_hist_di_list = []
            neigh_v = mdetector.neighbor_search_pixels_v
            neigh_h = mdetector.neighbor_search_pixels_h
            print(f"    ANIM_HELPER: Neighbor search window for context: v +/- {neigh_v}, h +/- {neigh_h}")


            for v_offset in range(-neigh_v, neigh_v + 1):
                for h_offset in range(-neigh_h, neigh_h + 1):
                    check_v, check_h = v_proj + v_offset, h_proj + h_offset
                    if 0 <= check_v < immediate_hist_di_from_lib.num_pixels_v and \
                       0 <= check_h < immediate_hist_di_from_lib.num_pixels_h:
                        pixel_info = immediate_hist_di_from_lib.get_pixel_info(check_v, check_h)
                        context_indices_in_hist_di_list.extend(pixel_info.get('original_indices_in_pixel', []))
            
            if context_indices_in_hist_di_list:
                unique_context_indices = np.unique(context_indices_in_hist_di_list).astype(int)
                print(f"    ANIM_HELPER: Found {len(unique_context_indices)} unique context point indices in historical DI region.")
                
                if immediate_hist_di_from_lib.original_points_global_coords is not None:
                    # Filter unique_context_indices to be within bounds of original_points_global_coords
                    valid_unique_context_indices = unique_context_indices[unique_context_indices < immediate_hist_di_from_lib.original_points_global_coords.shape[0]]
                    if len(valid_unique_context_indices) != len(unique_context_indices):
                        print(f"    ANIM_HELPER Warning: Some unique_context_indices were out of bounds for historical DI's original_points_global_coords.")
                    
                    if valid_unique_context_indices.size > 0:
                        prepared_data['context_points_global'] = immediate_hist_di_from_lib.original_points_global_coords[valid_unique_context_indices, :3]
                        
                        # Also filter hist_local_sph_coords_all by valid_unique_context_indices
                        # before accessing depths, ensuring indices align.
                        if hist_local_sph_coords_all.shape[0] > np.max(valid_unique_context_indices): # Basic check
                            context_depths_in_hist_frame = hist_local_sph_coords_all[valid_unique_context_indices, 2]

                            if context_depths_in_hist_frame.size > 0:
                                min_depth_val = np.min(context_depths_in_hist_frame)
                                prepared_data['min_depth_context_value_in_hist_frame'] = min_depth_val
                                print(f"    ANIM_HELPER: Min context depth for animation: {min_depth_val:.3f}")

                                max_depth_val = np.max(context_depths_in_hist_frame)
                                prepared_data['max_depth_context_value_in_hist_frame'] = max_depth_val
                                print(f"    ANIM_HELPER: Max context depth for animation: {max_depth_val:.3f}")

                                # Store XY for min/max depth points if needed for plotting text
                                idx_of_min_depth_in_subset = np.argmin(context_depths_in_hist_frame)
                                original_idx_of_min_depth_pt = valid_unique_context_indices[idx_of_min_depth_in_subset]
                                prepared_data['min_depth_context_point_global_xy'] = immediate_hist_di_from_lib.original_points_global_coords[original_idx_of_min_depth_pt, :2]

                                idx_of_max_depth_in_subset = np.argmax(context_depths_in_hist_frame)
                                original_idx_of_max_depth_pt = valid_unique_context_indices[idx_of_max_depth_in_subset]
                                prepared_data['max_depth_context_point_global_xy'] = immediate_hist_di_from_lib.original_points_global_coords[original_idx_of_max_depth_pt, :2]
                            else:
                                print("    ANIM_HELPER: No valid context depths found after filtering indices.")
                        else:
                            print("    ANIM_HELPER Warning: Max of valid_unique_context_indices is out of bounds for hist_local_sph_coords_all.")
                    else:
                        print("    ANIM_HELPER: No valid unique context indices after bounds check.")
                else:
                    print("    ANIM_HELPER Warning: Historical DI from library has no original_points_global_coords for context point visualization.")
                    prepared_data['context_points_global'] = np.empty((0,3))
            else:
                print("    ANIM_HELPER: No context point indices found in the historical DI region.")
        else:
            print("    ANIM_HELPER: Debug point did not project into historical DI, or historical DI has no local_sph_coords. Cannot find context points.")
    else:
        print("    ANIM_HELPER: No valid immediate historical DI from library to use for context.")


    # 4. Calculate plot limits
    if prepared_data['debug_point_global_coord'] is not None:
        debug_point_xy = prepared_data['debug_point_global_coord'][:2].reshape(1, 2)
        all_relevant_points_for_lims_xy = [debug_point_xy]
        if prepared_data['context_points_global'].shape[0] > 0:
            context_points_xy = prepared_data['context_points_global'][:, :2]
            all_relevant_points_for_lims_xy.append(context_points_xy)
        
        ego_trans_target_for_lim, _ = prepared_data['ego_poses'].get(target_sweep_sd_token, (np.array([0,0,0]), Quaternion()))
        ego_xy_for_lim = ego_trans_target_for_lim[:2].reshape(1,2)
        all_relevant_points_for_lims_xy.append(ego_xy_for_lim)
        
        if all_relevant_points_for_lims_xy: # Check if list is not empty before vstack
            combined_points_xy = np.vstack(all_relevant_points_for_lims_xy)
            if combined_points_xy.shape[0] > 0:
                center_x, center_y = np.mean(combined_points_xy, axis=0)
                max_dev_x = np.max(np.abs(combined_points_xy[:, 0] - center_x)) if combined_points_xy.shape[0] > 0 else 0
                max_dev_y = np.max(np.abs(combined_points_xy[:, 1] - center_y)) if combined_points_xy.shape[0] > 0 else 0
                max_extent = max(max_dev_x, max_dev_y, 10.0) 
                
                vis_params_from_accessor = mdetector.config_accessor.get_visualization_params()
                debug_anim_style_cfg = vis_params_from_accessor.get('debug_animation_bev_style', {})
                padding = debug_anim_style_cfg.get('padding', 15.0)
                half_range = max(max_extent + padding, padding) 

                prepared_data['plot_lims'] = {
                    'xlim': (center_x - half_range, center_x + half_range),
                    'ylim': (center_y - half_range, center_y + half_range)
                }
            else: # Fallback if combined_points_xy is empty after vstack (should not happen if debug_point_xy is always there)
                ego_x, ego_y = ego_trans_target_for_lim[0], ego_trans_target_for_lim[1]
                vis_params_from_accessor = mdetector.config_accessor.get_visualization_params()
                debug_anim_style_cfg = vis_params_from_accessor.get('debug_animation_bev_style', {})
                default_range = debug_anim_style_cfg.get('padding', 15.0)
                prepared_data['plot_lims'] = {
                    'xlim': (ego_x - default_range, ego_x + default_range),
                    'ylim': (ego_y - default_range, ego_y + default_range)
                }
        else: # Fallback if all_relevant_points_for_lims_xy is empty
            ego_x, ego_y = ego_trans_target_for_lim[0], ego_trans_target_for_lim[1]
            vis_params_from_accessor = mdetector.config_accessor.get_visualization_params()
            debug_anim_style_cfg = vis_params_from_accessor.get('debug_animation_bev_style', {})
            default_range = debug_anim_style_cfg.get('padding', 15.0)
            prepared_data['plot_lims'] = {
                'xlim': (ego_x - default_range, ego_x + default_range),
                'ylim': (ego_y - default_range, ego_y + default_range)
            }

    return prepared_data

def _initialize_debug_plot_artists(
    fig: plt.Figure,
    ax: plt.Axes,
    prepared_data: Dict[str, Any] # Removed max_context_texts
) -> Dict[str, Any]:
    artists_dict = {
        'background_scatter': ax.scatter([], [], s=1.0, color=DEBUG_ANIM_COLOR_BACKGROUND, alpha=0.5, zorder=1, label="Background LiDAR"),
        'context_points_scatter': ax.scatter([], [], s=15, color=DEBUG_ANIM_COLOR_CONTEXT_POINTS, alpha=0.8, zorder=5, label="Context Pts (Historical)"),
        'debug_point_scatter': ax.scatter([], [], s=30, color=DEBUG_ANIM_COLOR_DEBUG_POINT, edgecolor='black', linewidth=0.5, zorder=10, label="Debug Point (Target)"),
        'ego_vehicle_marker': ax.plot([], [], marker='^', markersize=10, color=DEBUG_ANIM_COLOR_EGO, zorder=15, label="Ego Vehicle")[0],
        'ego_vehicle_orientation': ax.plot([], [], color=DEBUG_ANIM_COLOR_EGO, linewidth=2, zorder=15)[0],
        'time_text': ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=20),
        'lidar_cache': {},
        'debug_point_depth_text': ax.text(0, 0, '', color='darkgreen', fontsize=7, ha='center', va='bottom', zorder=12, visible=False),
        'min_context_depth_text': ax.text(0, 0, '', color='blue', fontsize=7, ha='center', va='top', zorder=7, visible=False), # va='top' to place below point
        'max_context_depth_text': ax.text(0, 0, '', color='purple', fontsize=7, ha='center', va='top', zorder=7, visible=False) # va='top' to place below point
    }
    ax.legend(loc='upper right', fontsize='x-small')
    return artists_dict


def _update_debug_animation_frame(
    frame_idx: int,
    nusc: NuScenes,
    artists_dict: Dict[str, Any],
    prepared_data: Dict[str, Any],
    scene_name_for_title: str,
    point_downsample_anim: int
):
    current_sweep_anim_data = prepared_data['animation_sweeps_data_dicts'][frame_idx]
    current_sd_token = current_sweep_anim_data['lidar_sd_token']
    ax = artists_dict['background_scatter'].axes

    # 1. Update background LiDAR points (remains the same)
    # ... (identical to previous version) ...
    if current_sd_token not in artists_dict['lidar_cache']:
        current_points_global = transform_points_numpy(
            current_sweep_anim_data['points_sensor_frame'],
            current_sweep_anim_data['T_global_lidar']
        )
        if point_downsample_anim > 1:
            current_points_global = current_points_global[::point_downsample_anim]
        artists_dict['lidar_cache'][current_sd_token] = current_points_global[:, :2]
    artists_dict['background_scatter'].set_offsets(artists_dict['lidar_cache'][current_sd_token])


    # --- Manage visibility of all depth texts initially ---
    artists_dict['debug_point_depth_text'].set_visible(False)
    artists_dict['min_context_depth_text'].set_visible(False)
    artists_dict['max_context_depth_text'].set_visible(False)

    # 2. Update debug point and context points visibility and data
    target_sd_token = prepared_data['target_sweep_sd_token']
    historical_context_sd_token = prepared_data['historical_sweep_sd_token_for_context']
    text_offset_y_above = 0.4 # Small offset for text display above points
    text_offset_y_below = -0.4 # Small offset for text display below points

    # Debug point (only visible when showing the target sweep)
    if current_sd_token == target_sd_token and prepared_data['debug_point_global_coord'] is not None:
        debug_point_xy_bev = prepared_data['debug_point_global_coord'][:2]
        artists_dict['debug_point_scatter'].set_offsets(debug_point_xy_bev.reshape(1,2))
        artists_dict['debug_point_scatter'].set_visible(True)

        debug_depth_val = prepared_data.get('debug_point_depth_in_hist_frame')
        if debug_depth_val is not None:
            dt = artists_dict['debug_point_depth_text']
            dt.set_position((debug_point_xy_bev[0], debug_point_xy_bev[1] + text_offset_y_above))
            dt.set_text(f"DbgPt(d_curr): {debug_depth_val:.2f}m") # Clarified label
            dt.set_visible(True)
            dt.set_zorder(12) # Ensure it's on top
    else:
        artists_dict['debug_point_scatter'].set_visible(False)

    # Context points scatter (remains the same logic for visibility)
    if prepared_data['context_points_global'].shape[0] > 0:
        if current_sd_token == historical_context_sd_token or current_sd_token == target_sd_token:
            context_points_xy_bev = prepared_data['context_points_global'][:, :2]
            artists_dict['context_points_scatter'].set_offsets(context_points_xy_bev)
            artists_dict['context_points_scatter'].set_visible(True)
            alpha_val = 0.8
            zorder_val = 5
            if current_sd_token == target_sd_token:
                alpha_val = 0.4
                zorder_val = 4
            artists_dict['context_points_scatter'].set_alpha(alpha_val)
            artists_dict['context_points_scatter'].set_zorder(zorder_val)
        else:
            artists_dict['context_points_scatter'].set_visible(False)
    else:
        artists_dict['context_points_scatter'].set_visible(False)

    # Specific depth annotations for min/max context points (only on target frame for clarity)
    if current_sd_token == target_sd_token:
        min_ctx_depth_val = prepared_data.get('min_depth_context_value_in_hist_frame')
        min_ctx_xy_bev = prepared_data.get('min_depth_context_point_global_xy')
        if min_ctx_depth_val is not None and min_ctx_xy_bev is not None:
            mcdt = artists_dict['min_context_depth_text']
            mcdt.set_position((min_ctx_xy_bev[0], min_ctx_xy_bev[1] + text_offset_y_below))
            mcdt.set_text(f"CtxMinD: {min_ctx_depth_val:.2f}m")
            mcdt.set_visible(True)
            mcdt.set_zorder(11) # Slightly below debug point text

        max_ctx_depth_val = prepared_data.get('max_depth_context_value_in_hist_frame')
        max_ctx_xy_bev = prepared_data.get('max_depth_context_point_global_xy')
        if max_ctx_depth_val is not None and max_ctx_xy_bev is not None:
            mxcdt = artists_dict['max_context_depth_text']
            # Try to position max depth text differently if it's the same point as min depth
            pos_y_max = max_ctx_xy_bev[1] + text_offset_y_below
            if min_ctx_xy_bev is not None and np.allclose(min_ctx_xy_bev, max_ctx_xy_bev):
                 pos_y_max = max_ctx_xy_bev[1] + text_offset_y_above * 1.5 # Place it further above if same point
            mxcdt.set_position((max_ctx_xy_bev[0], pos_y_max))
            mxcdt.set_text(f"CtxMaxD: {max_ctx_depth_val:.2f}m")
            mxcdt.set_visible(True)
            mxcdt.set_zorder(11)


    # 3. Update Ego Vehicle Pose (remains the same)
    # ... (identical to previous version) ...
    ego_trans, ego_rot = prepared_data['ego_poses'].get(current_sd_token, (np.array([0,0,0]), Quaternion()))
    artists_dict['ego_vehicle_marker'].set_data([ego_trans[0]], [ego_trans[1]])
    ego_front_direction = ego_rot.rotate(np.array([2.0, 0, 0]))
    artists_dict['ego_vehicle_orientation'].set_data(
        [ego_trans[0], ego_trans[0] + ego_front_direction[0]],
        [ego_trans[1], ego_trans[1] + ego_front_direction[1]]
    )
    
    # 4. Update text and plot aesthetics (remains the same)
    # ... (identical to previous version) ...
    plot_lims = prepared_data['plot_lims']
    ax.set_xlim(plot_lims['xlim'])
    ax.set_ylim(plot_lims['ylim'])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    frame_type = "Target Sweep" if current_sd_token == target_sd_token else \
                 "Historical Context Sweep" if current_sd_token == historical_context_sd_token else \
                 "Historical Sweep"
    sweep_ts_us = current_sweep_anim_data['timestamp']
    start_ts_us = prepared_data['animation_sweeps_data_dicts'][0]['timestamp']
    relative_time_s = (sweep_ts_us - start_ts_us) / 1e6
    artists_dict['time_text'].set_text(
        f'Scene: {scene_name_for_title}\nFrame: {frame_idx+1}/{len(prepared_data["animation_sweeps_data_dicts"])} ({frame_type})\n'
        f'Time: {relative_time_s:.2f}s (Sweep: {current_sd_token[:6]}...)'
    )
    ax.set_title(f"Debug Point Animation: {scene_name_for_title} - {frame_type}")
    ax.grid(True, alpha=0.5)


    all_artists_to_return = [
        artists_dict['background_scatter'], artists_dict['context_points_scatter'],
        artists_dict['debug_point_scatter'], artists_dict['ego_vehicle_marker'],
        artists_dict['ego_vehicle_orientation'], artists_dict['time_text'],
        artists_dict['debug_point_depth_text'], artists_dict['min_context_depth_text'],
        artists_dict['max_context_depth_text']
    ]
    return all_artists_to_return


def create_point_debug_bev_animation(
    nusc: NuScenes,
    target_sweep_sd_token: str,     # SD token of the 'current_di_from_detector_logic'
    mdetector: MDetector,           # MDetector instance
    current_di_from_detector_logic: DepthImage, # The actual 'current_di' object
    pt_idx_in_current_di: int,      # The 'pt_idx' being debugged in current_di
    idx_of_current_di_in_lib: int,  # Library index of current_di_from_detector_logic
    scene_name: str = "Scene",
    num_past_sweeps_to_show: int = 1,
    point_downsample_anim: int = 5,
    interval_ms: int = 200,
    figsize_anim: tuple = (10, 10),
    save_path: Optional[str] = None,
    save_writer: Optional[str] = None,
    save_fps: int = 5,
    save_dpi: int = 150
):
    """
    Creates a BEV animation focusing on a specific debug point and its context.
    Uses the modified _prepare_debug_animation_data.
    """
    print(f"ANIMATION: Preparing data for point {pt_idx_in_current_di} in sweep {target_sweep_sd_token}...")
    prepared_data = _prepare_debug_animation_data(
        nusc,
        target_sweep_sd_token,
        mdetector,
        current_di_from_detector_logic,
        pt_idx_in_current_di,
        idx_of_current_di_in_lib,
        num_past_sweeps_to_show
    )

    if prepared_data is None or not prepared_data.get('animation_sweeps_data_dicts'): # Check if list is empty or key missing
        print("ANIMATION: Failed to prepare animation data or no sweeps to animate.")
        return None, None, None # Return None for projected_pixel as well

    fig, ax = plt.subplots(figsize=figsize_anim)
    artists_dict = _initialize_debug_plot_artists(fig, ax, prepared_data) # This remains the same

    update_wrapper_lambda = lambda frame_idx_lambda: _update_debug_animation_frame( # This remains the same
        frame_idx_lambda, nusc, artists_dict, prepared_data, scene_name, point_downsample_anim
    )

    num_frames_to_animate = len(prepared_data['animation_sweeps_data_dicts'])
    anim = FuncAnimation(fig, update_wrapper_lambda, frames=num_frames_to_animate,
                         interval=interval_ms, blit=False)

    html_output = None
    if save_path:
        # ... (save logic remains the same)
        actual_anim_save_fps = save_fps if save_fps is not None else (1000 / interval_ms)
        try:
            print(f"ANIMATION: Saving animation to {save_path} ({num_frames_to_animate} frames, FPS: {actual_anim_save_fps})...")
            anim.save(save_path, writer=save_writer, fps=actual_anim_save_fps, dpi=save_dpi)
            print(f"ANIMATION: Animation saved successfully to {save_path}")
        except Exception as e:
            print(f"ANIMATION: Error saving animation: {e}. Consider installing ffmpeg or specifying a writer.")
            print("ANIMATION: Attempting to generate HTML output instead...")
            html_output = HTML(anim.to_jshtml())
        finally:
            plt.close(fig)
    else:
        print(f"ANIMATION: Generating HTML for animation ({num_frames_to_animate} frames)...")
        html_output = HTML(anim.to_jshtml())
        plt.close(fig)

    # Return the projected pixel from the animation helper for comparison
    projected_pixel_from_anim = prepared_data.get('projected_pixel_in_hist_di')
    return html_output, anim, projected_pixel_from_anim