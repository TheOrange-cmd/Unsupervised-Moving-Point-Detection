# utils/refactor.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from pyquaternion import Quaternion
from scipy.spatial.transform import Slerp, Rotation as R
import json
import inspect
from PIL import Image
import k3d

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Box as NuScenesDataClassesBox
from nuscenes.utils.geometry_utils import transform_matrix, view_points

# --- Core Data Finding/Processing ---

def find_instances_in_scene(nusc, scene_token, min_annotations=1):
    """
    Finds all unique instances in a given scene that have at least a minimum number of annotations.
    Args:
        nusc: NuScenes API instance.
        scene_token (str): The token of the scene to process.
        min_annotations (int): The minimum number of annotations an instance must have.
    Returns:
        list: A list of instance tokens.
    """
    scene_record = nusc.get('scene', scene_token)
    unique_instance_tokens_in_scene = set()
    
    current_sample_token = scene_record['first_sample_token']
    while current_sample_token:
        sample_record = nusc.get('sample', current_sample_token)
        for annotation_token in sample_record['anns']:
            annotation_record = nusc.get('sample_annotation', annotation_token)
            unique_instance_tokens_in_scene.add(annotation_record['instance_token'])
        current_sample_token = sample_record['next']
        if not current_sample_token:
            break
            
    detailed_instances = []
    for instance_token in unique_instance_tokens_in_scene:
        instance_record = nusc.get('instance', instance_token)
        if instance_record['nbr_annotations'] >= min_annotations:
            detailed_instances.append(instance_token)
            
    return detailed_instances

def get_lidar_sweeps_for_interval(nusc, first_sample_token, last_sample_token, channel='LIDAR_TOP'):
    """
    Returns a list of LIDAR_TOP sample_data records for a given scene interval,
    ordered by timestamp. Each element in the list is a dictionary containing
    key information about the sweep.
    """
    sweeps = []
    first_sample_rec = nusc.get('sample', first_sample_token)
    # Get the scene_token from the first sample of the interval.
    # All sweeps in this interval will belong to this scene.
    scene_token_for_interval = first_sample_rec['scene_token']

    # last_sample_rec = nusc.get('sample', last_sample_token) # Not strictly needed for scene_token

    current_sd_token = first_sample_rec['data'].get(channel)
    if not current_sd_token:
        print(f"Warning: Channel {channel} not found in the first sample {first_sample_token}. Returning empty list.")
        return []
    
    processed_tokens = set() # To avoid infinite loops in case of corrupted data (rare)

    while current_sd_token != "" and current_sd_token not in processed_tokens:
        processed_tokens.add(current_sd_token)
        sd_rec = nusc.get('sample_data', current_sd_token)
        
        sweeps.append({
            'token': current_sd_token,
            'timestamp': sd_rec['timestamp'],
            'scene_token': scene_token_for_interval,  # <<< CORRECTED: Use scene_token from the interval's start
            'filename': sd_rec['filename'],
            'is_key_frame': sd_rec['is_key_frame'],
            'sample_token': sd_rec['sample_token'],
            'ego_pose_token': sd_rec['ego_pose_token'],
            'calibrated_sensor_token': sd_rec['calibrated_sensor_token']
        })
        
        current_sd_token = sd_rec['next']
    
    # Sort by timestamp just in case (though `next` should maintain order within a valid chain)
    sweeps.sort(key=lambda x: x['timestamp'])
    
    return sweeps

# --- Interpolation/Extrapolation Logic ---

def extrapolate_box_with_velocity(initial_box, initial_timestamp, target_timestamp):
    """
    Extrapolates a box to a target timestamp using its constant velocity.
    """
    delta_t_s = (target_timestamp - initial_timestamp) / 1e6
    new_center = initial_box.center + initial_box.velocity * delta_t_s
    
    return NuScenesDataClassesBox(
        center=new_center, size=initial_box.wlh, orientation=initial_box.orientation,
        velocity=initial_box.velocity, name=initial_box.name,
        token=f"extrap_{initial_box.token}_{target_timestamp}"
    )

def get_interpolated_extrapolated_boxes_for_instance(nusc, instance_token, target_lidar_sweeps):
    """
    For a given instance, finds its annotations (keyframes) and then interpolates
    or extrapolates its bounding box to the timestamps of all target_lidar_sweeps.
    Ensures that for keyframe sweeps, the original annotation is used directly.
    Returned Box objects include name and velocity.

    Args:
        nusc: NuScenes API instance.
        instance_token (str): The token of the instance to track.
        target_lidar_sweeps (list): A list of dictionaries, where each dictionary
                                    represents a LiDAR sweep and must contain at least
                                    'timestamp', 'is_key_frame', and 'sample_token' keys.

    Returns:
        tuple: (
            list: List of NuScenes Box objects (or None) for each target_lidar_sweep.
            list: List of annotation tokens (or None) corresponding to the box (original if keyframe).
            list: List of booleans indicating if the box was interpolated.
            list: List of booleans indicating if the box was extrapolated.
        )
    """
    instance_rec = nusc.get('instance', instance_token)
    first_ann_token = instance_rec['first_annotation_token']
    last_ann_token = instance_rec['last_annotation_token']

    keyframe_annotations = []
    current_ann_token = first_ann_token
    while current_ann_token:
        ann_rec = nusc.get('sample_annotation', current_ann_token)
        sample_rec = nusc.get('sample', ann_rec['sample_token'])
        # Calculate velocity between this annotation and the previous one for this instance
        # This is a simplified velocity; NuScenes annotations have their own velocity if available from trackers.
        # For simplicity here, we'll primarily use the annotation's inherent properties.
        # The NuScenes Box class can derive velocity from its state if needed for some operations,
        # but the raw annotations sometimes have explicit velocity fields (though not directly in ann_rec).
        # We will use a placeholder or simple diff if needed for extrapolation.
        
        keyframe_annotations.append({
            'token': ann_rec['token'],
            'sample_token': ann_rec['sample_token'],
            'instance_token': ann_rec['instance_token'],
            'timestamp': sample_rec['timestamp'],
            'category_name': ann_rec['category_name'],
            'translation': np.array(ann_rec['translation']),
            'size': np.array(ann_rec['size']),
            'rotation': Quaternion(ann_rec['rotation']),
            # NuScenes sample_annotation records do not directly store velocity.
            # Velocity is typically derived or comes from a tracker.
            # For now, we'll handle velocity during extrapolation.
        })
        if current_ann_token == last_ann_token:
            break
        current_ann_token = ann_rec['next']
    
    if not keyframe_annotations:
        return [None] * len(target_lidar_sweeps), [None] * len(target_lidar_sweeps), \
               [False] * len(target_lidar_sweeps), [False] * len(target_lidar_sweeps)

    keyframe_annotations.sort(key=lambda x: x['timestamp'])
    kf_timestamps = np.array([ann['timestamp'] for ann in keyframe_annotations])
    
    output_boxes_at_sweeps = [None] * len(target_lidar_sweeps)
    output_ann_tokens = [None] * len(target_lidar_sweeps)
    output_is_interpolated = [False] * len(target_lidar_sweeps)
    output_is_extrapolated = [False] * len(target_lidar_sweeps)

    target_sweep_timestamps = np.array([sweep['timestamp'] for sweep in target_lidar_sweeps])

    KEYFRAME_INTERVAL_US = 500000
    MAX_EXTRAPOLATION_TIME_US = KEYFRAME_INTERVAL_US * 1.5 # Max 0.75s extrapolation

    for i, target_sweep_info in enumerate(target_lidar_sweeps):
        target_ts = target_sweep_info['timestamp']
        box_velocity = np.array([0.0, 0.0, 0.0]) # Default velocity

        # --- Priority 1: Direct keyframe match using sample_token ---
        if target_sweep_info['is_key_frame']:
            target_sample_token = target_sweep_info['sample_token']
            found_keyframe_ann = next((kf_ann for kf_ann in keyframe_annotations if kf_ann['sample_token'] == target_sample_token), None)
            
            if found_keyframe_ann:
                ann = found_keyframe_ann
                # For velocity of a GT annotation, nusc.box_velocity() can be used if needed,
                # but it requires the sample_annotation_token.
                # For now, our Box object will have default zero velocity unless extrapolated.
                output_boxes_at_sweeps[i] = NuScenesDataClassesBox(ann['translation'], ann['size'], ann['rotation'], 
                                                name=ann['category_name'], token=f"{ann['token']}",
                                                velocity=box_velocity) # Use default for GT
                output_ann_tokens[i] = ann['token']
                # output_is_interpolated and output_is_extrapolated remain False (default)
                continue 
        
        # --- Priority 2: Exact timestamp match (fallback or for non-keyframe sweeps that align perfectly) ---
        idx_after = np.searchsorted(kf_timestamps, target_ts, side='left')
        if idx_after > 0 and kf_timestamps[idx_after - 1] == target_ts:
            ann = keyframe_annotations[idx_after - 1]
            output_boxes_at_sweeps[i] = NuScenesDataClassesBox(ann['translation'], ann['size'], ann['rotation'], 
                                            name=ann['category_name'], token=f"{ann['token']}",
                                            velocity=box_velocity) # Use default for GT
            output_ann_tokens[i] = ann['token']
            continue

        # --- Case 3: Interpolation ---
        if idx_after > 0 and idx_after < len(kf_timestamps):
            prev_ann = keyframe_annotations[idx_after - 1]
            next_ann = keyframe_annotations[idx_after]
            
            if prev_ann['timestamp'] < target_ts < next_ann['timestamp']: # Strictly between
                ratio = (target_ts - prev_ann['timestamp']) / (next_ann['timestamp'] - prev_ann['timestamp'])
                interp_translation = prev_ann['translation'] + ratio * (next_ann['translation'] - prev_ann['translation'])
                interp_size = prev_ann['size'] 
                interp_rotation = Quaternion.slerp(prev_ann['rotation'], next_ann['rotation'], ratio)
                
                # Estimate velocity for interpolated box
                dt_interp_sec = (next_ann['timestamp'] - prev_ann['timestamp']) / 1e6
                if dt_interp_sec > 1e-3:
                    box_velocity = (next_ann['translation'] - prev_ann['translation']) / dt_interp_sec
                
                output_boxes_at_sweeps[i] = NuScenesDataClassesBox(interp_translation, interp_size, interp_rotation, 
                                                name=prev_ann['category_name'], # Name from prev_ann
                                                token=f"interp_{prev_ann['token']}_{next_ann['token']}_{i}",
                                                velocity=box_velocity)
                output_is_interpolated[i] = True
                continue

        # --- Case 4: Extrapolation ---
        extrap_ann_ref = None
        time_diff_us = 0
        is_forward_extrap = False

        if target_ts < kf_timestamps[0]:
            extrap_ann_ref = keyframe_annotations[0]
            time_diff_us = target_ts - extrap_ann_ref['timestamp']
        elif target_ts > kf_timestamps[-1]:
            extrap_ann_ref = keyframe_annotations[-1]
            time_diff_us = target_ts - extrap_ann_ref['timestamp']
            is_forward_extrap = True
        
        if extrap_ann_ref and abs(time_diff_us) <= MAX_EXTRAPOLATION_TIME_US:
            extrap_translation = np.copy(extrap_ann_ref['translation'])
            
            if len(keyframe_annotations) > 1:
                if not is_forward_extrap: # Extrapolating backwards from first annotation
                    ref1, ref2 = keyframe_annotations[0], keyframe_annotations[1]
                else: # Extrapolating forwards from last annotation
                    ref1, ref2 = keyframe_annotations[-2], keyframe_annotations[-1]
                
                dt_ref_sec = (ref2['timestamp'] - ref1['timestamp']) / 1e6
                if dt_ref_sec > 1e-3:
                    velocity_ref = (ref2['translation'] - ref1['translation']) / dt_ref_sec
                    extrap_translation = extrap_ann_ref['translation'] + velocity_ref * (time_diff_us / 1e6)
                    box_velocity = velocity_ref # Use this calculated velocity
            
            extrap_size = extrap_ann_ref['size']
            extrap_rotation = extrap_ann_ref['rotation'] # Keep rotation constant for extrapolation

            output_boxes_at_sweeps[i] = NuScenesDataClassesBox(extrap_translation, extrap_size, extrap_rotation,
                                            name=extrap_ann_ref['category_name'],
                                            token=f"extrap_{extrap_ann_ref['token']}_{i}",
                                            velocity=box_velocity)
            output_is_extrapolated[i] = True
            
    return output_boxes_at_sweeps, output_ann_tokens, output_is_interpolated, output_is_extrapolated

# --- LiDAR Point Loading ---
def load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1):
    lidar_sd_rec = nusc.get('sample_data', lidar_sd_token)
    pcl_path = os.path.join(nusc.dataroot, lidar_sd_rec['filename'])
    if not os.path.exists(pcl_path): return np.zeros((0, 3))
    pc = LidarPointCloud.from_file(pcl_path)
    points_sensor_frame = pc.points[:3, :] 
    cs_rec = nusc.get('calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
    sensor_to_ego_tf = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']))
    ego_pose_rec = nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])
    ego_to_global_tf = transform_matrix(ego_pose_rec['translation'], Quaternion(ego_pose_rec['rotation']))
    points_global = ego_to_global_tf @ sensor_to_ego_tf @ np.vstack((points_sensor_frame, np.ones(points_sensor_frame.shape[1])))
    points_global = points_global[:3, :]
    if downsample_factor > 1: points_global = points_global[:, ::downsample_factor]
    return points_global.T

# --- Animation Component Functions ---

def _prepare_animation_data(nusc, instance_tokens, all_lidar_sweeps):
    instance_animation_data = {}
    min_x_all, max_x_all = float('inf'), float('-inf')
    min_y_all, max_y_all = float('inf'), float('-inf')
    
    any_instance_has_boxes = False
    for token in instance_tokens:
        boxes_at_sweeps,orig_b, _,  _ = get_interpolated_extrapolated_boxes_for_instance(
            nusc, token, all_lidar_sweeps
        )
        
        instance_category = orig_b[0] if orig_b else "N/A"
        
        instance_animation_data[token] = {
            'boxes_at_sweeps': boxes_at_sweeps, 
            'category': instance_category,
        }

        for box in boxes_at_sweeps:
            if box is not None:
                any_instance_has_boxes = True
                corners_xy = box.bottom_corners()[:2, :]
                min_x_all = min(min_x_all, np.min(corners_xy[0, :]))
                max_x_all = max(max_x_all, np.max(corners_xy[0, :]))
                min_y_all = min(min_y_all, np.min(corners_xy[1, :]))
                max_y_all = max(max_y_all, np.max(corners_xy[1, :]))

    if not any_instance_has_boxes: # If no boxes at all were found across all instances and sweeps
        # print("Warning: No valid boxes found for any instance to determine plot limits. Using defaults.")
        plot_lims = {'xlim': (-50, 50), 'ylim': (-50, 50)}
        if not instance_animation_data: # No instances were even processed
             return None, [], {}
    else:
        padding = 15.0
        plot_lims = {
            'xlim': (min_x_all - padding, max_x_all + padding),
            'ylim': (min_y_all - padding, max_y_all + padding)
        }
    
    return instance_animation_data, all_lidar_sweeps, plot_lims


def _initialize_plot_artists(fig, ax, instance_animation_data, instance_colors, animation_lidar_sweeps):
    artists_dict = {
        'box_polys': {}, 'orientation_lines': {}, 'lidar_scatter': None,
        'time_text': None, 'legend_elements': [], 'lidar_cache': {}
    }

    for token, data in instance_animation_data.items():
        color = instance_colors.get(token, plt.cm.tab10.colors[0])
        edgecolor = tuple(0.7 * np.array(color[:3]))
        poly = Polygon(np.zeros((4, 2)), closed=True, facecolor=color, edgecolor=edgecolor,
                       alpha=0.6, linewidth=1.5, zorder=10, visible=False)
        ax.add_patch(poly)
        artists_dict['box_polys'][token] = poly
        line, = ax.plot([], [], color=edgecolor, linewidth=2, zorder=11, visible=False)
        artists_dict['orientation_lines'][token] = line
        artists_dict['legend_elements'].append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=data['category'])
        )

    artists_dict['lidar_scatter'] = ax.scatter([], [], s=1.5, color='dimgray', alpha=0.6, zorder=1)
    artists_dict['time_text'] = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                                        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=15)
    
    if animation_lidar_sweeps and instance_animation_data:
        for token, data in instance_animation_data.items():
            if data['boxes_at_sweeps']: 
                initial_box = data['boxes_at_sweeps'][0] 
                if initial_box is not None:
                    poly = artists_dict['box_polys'][token]
                    poly.set_xy(initial_box.bottom_corners()[:2, :].T)
                    poly.set_visible(True)
                    line = artists_dict['orientation_lines'][token]
                    front_vec_local = np.array([initial_box.wlh[1] / 2.0, 0, 0])
                    front_vec_global = initial_box.orientation.rotation_matrix @ front_vec_local
                    line.set_data(
                        [initial_box.center[0], initial_box.center[0] + front_vec_global[0]],
                        [initial_box.center[1], initial_box.center[1] + front_vec_global[1]]
                    )
                    line.set_visible(True)
    return artists_dict

def _update_animation_frame(frame_idx, nusc, artists_dict, instance_animation_data, 
                            animation_lidar_sweeps, plot_lims, scene_name, point_downsample):
    current_sweep = animation_lidar_sweeps[frame_idx]
    returned_artists = []
    visible_instances_count = 0

    for token, data in instance_animation_data.items():
        box = data['boxes_at_sweeps'][frame_idx]
        poly = artists_dict['box_polys'][token]
        line = artists_dict['orientation_lines'][token]

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

    lidar_sd_token = current_sweep['token']
    if lidar_sd_token not in artists_dict['lidar_cache']:
        points_xy = load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=point_downsample)[:, :2]
        artists_dict['lidar_cache'][lidar_sd_token] = points_xy
    artists_dict['lidar_scatter'].set_offsets(artists_dict['lidar_cache'].get(lidar_sd_token, np.zeros((0,2))))
    returned_artists.append(artists_dict['lidar_scatter'])
    
    start_ts = animation_lidar_sweeps[0]['timestamp']
    relative_time_s = (current_sweep['timestamp'] - start_ts) / 1e6
    # artists_dict['time_text'].set_text(
    #     f'Frame: {frame_idx}/{len(animation_lidar_sweeps)-1}\nTime: {relative_time_s:.3f}s\n'
    #     f'Visible: {visible_instances_count}/{len(instance_animation_data)}'
    # )
    returned_artists.append(artists_dict['time_text'])

    ax = artists_dict['lidar_scatter'].axes 
    ax.set_xlim(plot_lims['xlim']); ax.set_ylim(plot_lims['ylim'])
    ax.set_aspect('equal', adjustable='box'); ax.set_xlabel("Global X (m)"); ax.set_ylabel("Global Y (m)")
    ax.set_title(f"NuScenes: {scene_name} - Interpolated/Extrapolated Annotations"); ax.grid(True, alpha=0.6)
    # if not ax.get_legend(): ax.legend(handles=artists_dict['legend_elements'], loc='lower right', fontsize='small')
    return returned_artists

# --- Main Animation Function ---
def create_synchronized_animation(nusc, instance_tokens, all_lidar_sweeps, scene_name="Scene",
                                  interval_ms=50, figsize=(10, 10), point_downsample=1,
                                  save_path=None, save_writer=None, save_fps=20, save_dpi=200):
    # print(f"Preparing animation data for {len(instance_tokens)} instances and {len(all_lidar_sweeps)} LiDAR sweeps...")
    prepared_data = _prepare_animation_data(nusc, instance_tokens, all_lidar_sweeps)
    
    if prepared_data is None or not prepared_data[0]: # Check if instance_animation_data is empty or None
        print("Animation preparation failed or no instance data. No data to animate.")
        return None, None
        
    instance_animation_data, animation_lidar_sweeps, plot_lims = prepared_data

    if not animation_lidar_sweeps: # Should be caught by all_lidar_sweeps check earlier, but good for safety
        print("No LiDAR sweeps to animate.")
        return None, None
    
    # print(f"Data prepared. Animating {len(animation_lidar_sweeps)} frames.")

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors 
    instance_colors = { token: colors[i % len(colors)] for i, token in enumerate(instance_animation_data.keys()) }

    # print("Initializing plot artists...")
    artists_dict = _initialize_plot_artists(fig, ax, instance_animation_data, instance_colors, animation_lidar_sweeps)

    update_wrapper = lambda frame_idx: _update_animation_frame(
        frame_idx, nusc, artists_dict, instance_animation_data,
        animation_lidar_sweeps, plot_lims, scene_name, point_downsample
    )

    # print("Creating animation object...")
    anim = FuncAnimation(fig, update_wrapper, frames=len(animation_lidar_sweeps), 
                         interval=interval_ms, blit=False)

    if save_path:
        actual_save_fps = save_fps if save_fps is not None else (1000 / interval_ms)
        # print(f"Saving animation to {save_path} at {actual_save_fps} FPS, DPI {save_dpi}...")
        try:
            anim.save(save_path, writer=save_writer, fps=actual_save_fps, dpi=save_dpi)
            # print(f"Animation saved successfully to {save_path}")
            plt.close(fig); return None, anim
        except Exception as e:
            print(f"Error saving animation: {e}. Generating HTML output instead.")
            html_output = HTML(anim.to_jshtml()); plt.close(fig); return html_output, anim
    else:
        # print("Generating HTML for animation display...")
        html_output = HTML(anim.to_jshtml()); plt.close(fig); return html_output, anim

POINT_LABEL_DTYPE = np.dtype([
    ('instance_token', 'S32'),  # 32-byte string for instance token
    ('category_name', 'S64'),   # 64-byte string for category name (e.g., 'vehicle.car')
    ('velocity_x', np.float32), # Global velocity x of the object the point belongs to
    ('velocity_y', np.float32), # Global velocity y
    ('velocity_z', np.float32)  # Global velocity z (will often be 0 for our 2D velocity source)
])

def _get_points_in_box_mask_global_coords(points_global, box_global, debug_instance_token=None, max_points_to_print=5):
    """
    Alternative check for points in an OBB using projections onto box axes.
    Includes debug printing and corrected WLH to axis mapping.

    Args:
        points_global (np.ndarray): Array of points, shape (N, 3) in global XYZ.
        box_global (NuScenesDataClassesBox): The box object, in global coordinates.
        debug_instance_token (str, optional): Token of the instance for targeted debugging prints.
        max_points_to_print (int): Max number of points to print details for.
    Returns:
        np.ndarray: A boolean array of shape (N,) indicating for each point
                    if it's inside the box.
    """
    if points_global.shape[0] == 0:
        return np.array([], dtype=bool)

    print_debug_for_this_box = (debug_instance_token is not None and
                                box_global.token is not None and
                                debug_instance_token in box_global.token)

    if print_debug_for_this_box:
        print(f"\n--- Debugging Projections for Box (Token Fragment: {box_global.token[-10:]}, Name: {box_global.name}) ---")
        print(f"Box Center (Global): {box_global.center}")
        print(f"Box WLH (Width, Length, Height): {box_global.wlh}") # NuScenes: Width, Length, Height
        print(f"Box Orientation (Quaternion w,x,y,z): {box_global.orientation.elements}")
        # print(f"Box Rotation Matrix (Local to Global):\n{box_global.rotation_matrix}") # Can be verbose

    vec_center_to_points = points_global - box_global.center

    # NuScenes Box: Local X is along Length, Local Y is along Width, Local Z is along Height
    # rotation_matrix column 0 is local X-axis (Length direction)
    # rotation_matrix column 1 is local Y-axis (Width direction)
    # rotation_matrix column 2 is local Z-axis (Height direction)
    box_axis_len_global = box_global.rotation_matrix[:, 0] # Axis for Length
    box_axis_wid_global = box_global.rotation_matrix[:, 1] # Axis for Width
    box_axis_hgt_global = box_global.rotation_matrix[:, 2] # Axis for Height

    if print_debug_for_this_box:
        print(f"Box Local Length-axis (Global Coords, from rot_mat col 0): {box_axis_len_global}")
        print(f"Box Local Width-axis (Global Coords, from rot_mat col 1): {box_axis_wid_global}")
        print(f"Box Local Height-axis (Global Coords, from rot_mat col 2): {box_axis_hgt_global}")
        # Orthogonality and Norm checks (can be commented out if consistently good)
        # print(f"  Len.Wid dot: {np.dot(box_axis_len_global, box_axis_wid_global):.4f}")
        # print(f"  Len.Hgt dot: {np.dot(box_axis_len_global, box_axis_hgt_global):.4f}")
        # print(f"  Wid.Hgt dot: {np.dot(box_axis_wid_global, box_axis_hgt_global):.4f}")
        # print(f"  Len norm: {np.linalg.norm(box_axis_len_global):.4f}")
        # print(f"  Wid norm: {np.linalg.norm(box_axis_wid_global):.4f}")
        # print(f"  Hgt norm: {np.linalg.norm(box_axis_hgt_global):.4f}")

    # Projections onto these axes
    dist_along_len_axis = vec_center_to_points @ box_axis_len_global
    dist_along_wid_axis = vec_center_to_points @ box_axis_wid_global
    dist_along_hgt_axis = vec_center_to_points @ box_axis_hgt_global

    # Half dimensions from WLH array: [width, length, height]
    half_width  = box_global.wlh[0] / 2.0
    half_length = box_global.wlh[1] / 2.0
    half_height = box_global.wlh[2] / 2.0

    # Compare projection along Length-axis with half_length
    # Compare projection along Width-axis with half_width
    # Compare projection along Height-axis with half_height
    mask_len = np.abs(dist_along_len_axis) < half_length
    mask_wid = np.abs(dist_along_wid_axis) < half_width
    mask_hgt = np.abs(dist_along_hgt_axis) < half_height
    
    in_box_mask = mask_len & mask_wid & mask_hgt

    if print_debug_for_this_box:
        num_printed = 0
        for i in range(points_global.shape[0]):
            if num_printed >= max_points_to_print:
                break
            
            print(f"  Point {i} (Global): {points_global[i]}")
            # print(f"    Vec Center-to-Point: {vec_center_to_points[i]}") # Can be verbose
            print(f"    Proj Length-Axis: {dist_along_len_axis[i]:.3f} (Limit: +/-{half_length:.3f}) -> In: {mask_len[i]}")
            print(f"    Proj Width-Axis:  {dist_along_wid_axis[i]:.3f} (Limit: +/-{half_width:.3f}) -> In: {mask_wid[i]}")
            print(f"    Proj Height-Axis: {dist_along_hgt_axis[i]:.3f} (Limit: +/-{half_height:.3f}) -> In: {mask_hgt[i]}")
            print(f"    Overall In Box: {in_box_mask[i]}")
            num_printed += 1
        print(f"--- End Debugging Projections for Box ---")

    return in_box_mask


def generate_point_labels_for_scene(nusc, scene_record, output_base_dir, verbose=True):
    """
    Generates point-level label files for all LiDAR sweeps in a given scene.
    Optimized with broad-phase culling.
    """
    scene_token = scene_record['token']
    scene_name = scene_record['name']
    
    scene_label_dir = os.path.join(output_base_dir, scene_name)
    os.makedirs(scene_label_dir, exist_ok=True)
    
    if verbose:
        print(f"Processing scene for point labels: {scene_name} ({scene_token})")
        print(f"Point labels will be saved to: {scene_label_dir}")

    all_lidar_sweeps = get_lidar_sweeps_for_interval(
        nusc, scene_record['first_sample_token'], scene_record['last_sample_token']
    )
    if not all_lidar_sweeps:
        if verbose: print(f"No LiDAR sweeps found for scene {scene_name}. Skipping.")
        return
    if verbose: print(f"Found {len(all_lidar_sweeps)} LiDAR sweeps for scene {scene_name}.")

    instance_tokens = find_instances_in_scene(nusc, scene_token, min_annotations=1)
    if not instance_tokens:
        if verbose: print(f"No instances found in scene {scene_name}. Skipping label generation.")
        return
    if verbose: print(f"Found {len(instance_tokens)} instances to process.")

    all_instances_boxes_at_sweeps = {}
    for i, inst_token in enumerate(instance_tokens):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Pre-calculating box states for instance {i+1}/{len(instance_tokens)} ({inst_token[:6]}...)")
        
        boxes_at_sweeps, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
            nusc, inst_token, all_lidar_sweeps
        )
        all_instances_boxes_at_sweeps[inst_token] = boxes_at_sweeps
    
    if verbose: print("All instance box states generated. Now creating point label files per sweep...")

    num_point_label_files_generated = 0
    for sweep_idx, sweep_data in enumerate(all_lidar_sweeps):
        if verbose and (sweep_idx + 1) % 5 == 0 : # Print more frequently for long processes
             print(f"  Processing sweep {sweep_idx + 1}/{len(all_lidar_sweeps)} ({sweep_data['token']})")

        lidar_sd_token = sweep_data['token']
        # points_global is (NumPoints, 3)
        points_global = load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1) 
        
        if points_global.shape[0] == 0:
            if verbose: print(f"    No points in LiDAR sweep {lidar_sd_token}. Skipping.")
            continue

        # Initialize point labels for all original points
        point_labels = np.zeros(points_global.shape[0], dtype=POINT_LABEL_DTYPE)
        point_labels['instance_token'] = b''
        point_labels['category_name'] = b''

        for inst_token, list_of_boxes_for_instance in all_instances_boxes_at_sweeps.items():
            box_object = list_of_boxes_for_instance[sweep_idx] # This is a NuScenesDataClassesBox
            
            if box_object is not None:
                try:
                    # --- Broad-phase culling ---
                    # 1. Get the 8 corners of the OBB in global coordinates
                    obb_corners_global = box_object.corners().T # corners() returns (3,8), so transpose to (8,3)
                    
                    # 2. Calculate the AABB of the OBB in global coordinates
                    aabb_min_global = np.min(obb_corners_global, axis=0)
                    aabb_max_global = np.max(obb_corners_global, axis=0)

                    # 3. Filter points_global to get candidate points within this AABB
                    # Create mask for points within the AABB
                    mask_in_aabb_x = (points_global[:, 0] >= aabb_min_global[0]) & (points_global[:, 0] <= aabb_max_global[0])
                    mask_in_aabb_y = (points_global[:, 1] >= aabb_min_global[1]) & (points_global[:, 1] <= aabb_max_global[1])
                    mask_in_aabb_z = (points_global[:, 2] >= aabb_min_global[2]) & (points_global[:, 2] <= aabb_max_global[2])
                    
                    mask_in_aabb = mask_in_aabb_x & mask_in_aabb_y & mask_in_aabb_z
                    
                    # Get original indices of points that are candidates
                    candidate_indices = np.where(mask_in_aabb)[0]
                    
                    if candidate_indices.size == 0: # No points even in the AABB
                        continue 
                        
                    candidate_points_global = points_global[candidate_indices]
                    # --- End Broad-phase culling ---

                    # --- Narrow-phase (precise OBB check) only on candidate points ---
                    if candidate_points_global.shape[0] > 0:
                        mask_of_candidates_in_obb = _get_points_in_box_mask_global_coords(
                            candidate_points_global, box_object
                        )
                        
                        # Get the original indices of points that are truly inside the OBB
                        final_indices_in_obb = candidate_indices[mask_of_candidates_in_obb]

                        if final_indices_in_obb.size > 0:
                            point_labels['instance_token'][final_indices_in_obb] = inst_token.encode('utf-8')
                            point_labels['category_name'][final_indices_in_obb] = box_object.name.encode('utf-8')
                            point_labels['velocity_x'][final_indices_in_obb] = box_object.velocity[0]
                            point_labels['velocity_y'][final_indices_in_obb] = box_object.velocity[1]
                            point_labels['velocity_z'][final_indices_in_obb] = box_object.velocity[2]
                
                except Exception as e_gen:
                    if verbose:
                        print(f"\n--- DEBUG: Error during point-in-box processing for instance {inst_token[:6]} ---")
                        print(f"    Sweep: {lidar_sd_token}, Error Type: {type(e_gen).__name__}, Error: {e_gen}")
                        if isinstance(box_object, NuScenesDataClassesBox):
                             print(f"    Box Center: {box_object.center}, WLH: {box_object.wlh}")
                        print(f"--- END DEBUG ---\n")
                    continue
        
        label_filename = f"{lidar_sd_token}_pointlabels.npy"
        label_filepath = os.path.join(scene_label_dir, label_filename)
        try:
            np.save(label_filepath, point_labels)
            num_point_label_files_generated += 1
        except IOError as e:
            if verbose: print(f"    Error writing point label file {label_filepath}: {e}")
        except Exception as e_gen:
            if verbose: print(f"    An unexpected error occurred saving {label_filepath}: {e_gen}")

    if verbose:
        print(f"Finished processing scene {scene_name} for point labels.")
        print(f"Generated {num_point_label_files_generated} point label files in {scene_label_dir}.")

def visualize_point_labels_for_sweep(
    nusc,
    scene_name,
    sweep_index,
    point_labels_base_dir,
    downsample_visualization_points=1, # Downsample points for k3d performance if needed
    point_size=0.05
):
    """
    Visualizes LiDAR points, their generated labels, and the corresponding
    interpolated/extrapolated 3D bounding boxes for a specific sweep in a scene
    using k3d.

    Args:
        nusc: NuScenes API instance.
        scene_name (str): The name of the scene (e.g., "scene-0103").
        sweep_index (int): The 0-based index of the LiDAR sweep within the scene to visualize.
        point_labels_base_dir (str): The base directory where point label .npy files are stored.
        downsample_visualization_points (int): Factor by which to downsample points for visualization.
        point_size (float): Size of points in k3d plot.
    """
    # --- 1. Find the scene record ---
    scene_record = None
    for rec in nusc.scene:
        if rec['name'] == scene_name:
            scene_record = rec
            break
    if scene_record is None:
        print(f"Error: Scene '{scene_name}' not found.")
        return None
    scene_token = scene_record['token']

    # --- 2. Get all LiDAR sweeps for the scene ---
    all_scene_lidar_sweeps = get_lidar_sweeps_for_interval(
        nusc, scene_record['first_sample_token'], scene_record['last_sample_token']
    )
    if not all_scene_lidar_sweeps:
        print(f"Error: No LiDAR sweeps found for scene '{scene_name}'.")
        return None
    if not (0 <= sweep_index < len(all_scene_lidar_sweeps)):
        print(f"Error: sweep_index {sweep_index} is out of range for scene '{scene_name}' "
              f"(0 to {len(all_scene_lidar_sweeps) - 1}).")
        return None
    
    target_sweep_data = all_scene_lidar_sweeps[sweep_index]
    lidar_sd_token = target_sweep_data['token']
    print(f"Visualizing scene: '{scene_name}', Sweep Index: {sweep_index}, LiDAR Token: {lidar_sd_token}")

    # --- 3. Load LiDAR points for the target sweep ---
    # points_global will be (N, 3) for XYZ
    points_global = load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1)
    if points_global.shape[0] == 0:
        print(f"Error: No points in LiDAR sweep {lidar_sd_token}.")
        return None

    # --- 4. Construct label file path and load labels ---
    label_filename = f"{lidar_sd_token}_pointlabels.npy"
    # Path construction based on your modification: os.path.join(output_base_dir, scene_name)
    label_filepath = os.path.join(point_labels_base_dir, scene_name, label_filename)

    try:
        loaded_point_labels = np.load(label_filepath)
        if loaded_point_labels.shape[0] != points_global.shape[0]:
            print(f"Error: Mismatch in point count between LiDAR data ({points_global.shape[0]}) "
                  f"and label file ({loaded_point_labels.shape[0]}) for {label_filepath}.")
            return None
        print(f"Successfully loaded point labels from: {label_filepath}")
    except FileNotFoundError:
        print(f"Error: Point label file not found: {label_filepath}")
        return None
    except Exception as e:
        print(f"Error loading point label file {label_filepath}: {e}")
        return None

    # --- 5. Prepare data for K3D plot ---
    plot = k3d.plot(name=f"{scene_name} - Sweep {sweep_index}", grid_visible=True)
    
    # Define colors
    color_map_instances = plt.cm.get_cmap('viridis', 20) # Colormap for different instances
    background_color_k3d = 0xAAAAAA  # Light grey for background points

    # --- 6. Plot all LiDAR points (background first) ---
    # Downsample for visualization if requested
    vis_points_global = points_global[::downsample_visualization_points]
    vis_loaded_point_labels = loaded_point_labels[::downsample_visualization_points]

    # Initially, color all visualized points as background
    point_colors_k3d = np.full(vis_points_global.shape[0], background_color_k3d, dtype=np.uint32)

    # --- 7. Identify unique instances and plot their boxes and points ---
    unique_instance_tokens_in_labels = np.unique(vis_loaded_point_labels['instance_token'])
    # Filter out the empty token (background)
    unique_instance_tokens_in_labels = [tok for tok in unique_instance_tokens_in_labels if tok != b'']

    print(f"Found {len(unique_instance_tokens_in_labels)} unique labeled instances in this sweep.")

    instance_color_idx = 0
    for inst_token_bytes in unique_instance_tokens_in_labels:
        inst_token_str = inst_token_bytes.decode('utf-8')

        # Get the interpolated/extrapolated box for this instance at this specific sweep
        # We need to pass all_scene_lidar_sweeps to this function.
        # It returns a list of boxes, one for each sweep in all_scene_lidar_sweeps.
        boxes_for_instance_all_sweeps, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
            nusc, inst_token_str, all_scene_lidar_sweeps
        )
        
        # Select the box for the current target_sweep_index
        current_box_object = boxes_for_instance_all_sweeps[sweep_index]

        if current_box_object is not None:
            # Assign a color to this instance
            instance_color_rgb = color_map_instances(instance_color_idx / max(1, len(unique_instance_tokens_in_labels)-1))[:3]
            instance_color_hex = int(instance_color_rgb[0]*255)<<16 | int(instance_color_rgb[1]*255)<<8 | int(instance_color_rgb[2]*255)
            instance_color_idx += 1

            # Update colors for points belonging to this instance
            mask_instance_points = (vis_loaded_point_labels['instance_token'] == inst_token_bytes)
            point_colors_k3d[mask_instance_points] = instance_color_hex
            
            # Get box corners (8 corners, (3,8) array)
            corners = current_box_object.corners() # Already in global frame
            
            # Define lines for the box edges
            # Bottom face: 0-1, 1-2, 2-3, 3-0
            # Top face:    4-5, 5-6, 6-7, 7-4
            # Vertical:    0-4, 1-5, 2-6, 3-7
            lines_vertices = np.concatenate([
                corners[:, [0, 1, 2, 3, 0]].T, # Bottom face (0-1-2-3-0)
                corners[:, [4, 5, 6, 7, 4]].T, # Top face    (4-5-6-7-4)
                corners[:, [0, 4]].T,          # Vert 0-4
                corners[:, [1, 5]].T,          # Vert 1-5
                corners[:, [2, 6]].T,          # Vert 2-6
                corners[:, [3, 7]].T           # Vert 3-7
            ])
            
            # K3D lines requires vertices and indices defining start/end of each line segment
            # For the way lines_vertices is constructed above, it's a bit tricky.
            # Easier: define explicit segments
            box_edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
                (4, 5), (5, 6), (6, 7), (7, 4),  # top
                (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
            ]
            
            # Prepare vertices and indices for k3d.lines
            # Vertices are the 8 corners
            k3d_box_vertices = corners.T.astype(np.float32) # Shape (8, 3)
            
            # Indices for the 12 lines
            k3d_box_indices = np.array(box_edges, dtype=np.uint32).flatten() # Not quite, k3d expects pairs
            
            # Create line segments for k3d
            for start_idx, end_idx in box_edges:
                segment_vertices = corners[:, [start_idx, end_idx]].T.astype(np.float32)
                plot += k3d.line(segment_vertices, 
                                 shader='simple', 
                                 color=instance_color_hex, 
                                 width=0.05, # Thinner lines for boxes
                                 name=f'Box_{inst_token_str[:6]}')
            
    # Add all points to the plot with their assigned colors
    plot += k3d.points(positions=vis_points_global.astype(np.float32),
                       colors=point_colors_k3d, # k3d expects an array of uint32 colors
                       point_size=point_size,
                       shader='3d', # '3d' or 'mesh' for better lighting, 'simple' for basic
                       name='LiDAR_Points')
    
    # Set camera auto fit to see all objects
    plot.camera_auto_fit = True
    return plot
