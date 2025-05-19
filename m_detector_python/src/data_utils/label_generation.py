# src/data_utils/label_generation.py
import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesDataClassesBox
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import List, Dict, Optional
import h5py

# Import from M-Detector's codebase
from .nuscenes_helper import get_scene_sweep_data_sequence 
from ..utils.transformations import transform_points_numpy
from ..core.constants import POINT_LABEL_DTYPE

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
        # Initialize box_velocity, but it will be overridden if possible
        box_velocity = np.array([0.0, 0.0, 0.0]) 

        # --- Priority 1: Direct keyframe match using sample_token ---
        if target_sweep_info['is_key_frame']:
            target_sample_token = target_sweep_info['sample_token']
            found_keyframe_ann = next((kf_ann for kf_ann in keyframe_annotations if kf_ann['sample_token'] == target_sample_token), None)
            
            if found_keyframe_ann:
                ann = found_keyframe_ann
                try:
                    # Attempt to get velocity using nusc.box_velocity
                    # This returns velocity in global frame (vx, vy, vz)
                    velocity_from_sdk = nusc.box_velocity(ann['token']) # ann['token'] is the sample_annotation_token
                    if not np.any(np.isnan(velocity_from_sdk)): # Check for NaN if instance has only one ann
                        box_velocity = velocity_from_sdk[:3] # Take vx, vy, vz
                except AssertionError: # Handles cases like single annotation for instance
                    # Fallback: if nusc.box_velocity fails (e.g. single ann), try to estimate if possible
                    # This part is tricky if it's truly a single annotation.
                    # For now, if nusc.box_velocity fails, it might remain default zero or you can implement
                    # a simpler diff if there are at least two keyframe_annotations for the instance.
                    if len(keyframe_annotations) > 1:
                        # Try to find this ann in the list to get prev/next for manual diff
                        current_kf_idx = -1
                        for kf_idx, kf in enumerate(keyframe_annotations):
                            if kf['token'] == ann['token']:
                                current_kf_idx = kf_idx
                                break
                        if current_kf_idx != -1:
                            if current_kf_idx > 0: # Has a previous
                                prev_kf = keyframe_annotations[current_kf_idx - 1]
                                dt = (ann['timestamp'] - prev_kf['timestamp']) / 1e6
                                if dt > 1e-3:
                                    box_velocity = (ann['translation'] - prev_kf['translation']) / dt
                            elif current_kf_idx < len(keyframe_annotations) - 1: # Has a next
                                next_kf = keyframe_annotations[current_kf_idx + 1]
                                dt = (next_kf['timestamp'] - ann['timestamp']) / 1e6
                                if dt > 1e-3:
                                    box_velocity = (next_kf['translation'] - ann['translation']) / dt
                    # If still zero, it means it's likely a single annotation instance or other edge case
                    pass # box_velocity remains default or as calculated by fallback

                output_boxes_at_sweeps[i] = NuScenesDataClassesBox(ann['translation'], ann['size'], ann['rotation'], 
                                                name=ann['category_name'], token=f"{ann['token']}",
                                                velocity=box_velocity) # USE THE CALCULATED/SDK VELOCITY
                output_ann_tokens[i] = ann['token']
                continue 
        
        # --- Priority 2: Exact timestamp match (fallback or for non-keyframe sweeps that align perfectly) ---
        idx_after = np.searchsorted(kf_timestamps, target_ts, side='left')
        if idx_after > 0 and kf_timestamps[idx_after - 1] == target_ts:
            ann = keyframe_annotations[idx_after - 1]
            try:
                velocity_from_sdk = nusc.box_velocity(ann['token'])
                if not np.any(np.isnan(velocity_from_sdk)):
                    box_velocity = velocity_from_sdk[:3]
            except AssertionError:
                 # Similar fallback as above if needed
                pass # box_velocity remains default
            
            output_boxes_at_sweeps[i] = NuScenesDataClassesBox(ann['translation'], ann['size'], ann['rotation'], 
                                            name=ann['category_name'], token=f"{ann['token']}",
                                            velocity=box_velocity) # USE THE CALCULATED/SDK VELOCITY
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



def _get_points_in_box_mask_global_coords(
    points_global: np.ndarray, 
    box_global: NuScenesDataClassesBox, 
    debug_instance_token: Optional[str] = None, 
    max_points_to_print_debug: int = 5
) -> np.ndarray:
    """
    Checks which points are inside a 3D Oriented Bounding Box (OBB) in global coordinates.
    Uses an Axis-Aligned Bounding Box (AABB) pre-filter for efficiency.

    Args:
        points_global (np.ndarray): Array of points, shape (N, 3) in global XYZ.
        box_global (NuScenesDataClassesBox): The box object, in global coordinates.
        debug_instance_token (str, optional): If the box_global.token contains this string,
                                             detailed debug prints will be activated.
        max_points_to_print_debug (int): Max number of points to print details for during debug.

    Returns:
        np.ndarray: A boolean array of shape (N,) indicating for each original point
                    if it's inside the box.
    """
    num_original_points = points_global.shape[0]
    if num_original_points == 0:
        return np.array([], dtype=bool)

    # Initialize the final mask for all original points as False
    final_in_box_mask = np.zeros(num_original_points, dtype=bool)

    # --- 1. Broad-phase: AABB Culling ---
    obb_corners_global = box_global.corners() # Shape (3, 8)
    aabb_min_global = np.min(obb_corners_global, axis=1) # Min for X, Y, Z
    aabb_max_global = np.max(obb_corners_global, axis=1) # Max for X, Y, Z

    # Create a mask for points within the AABB of the box
    mask_in_aabb = (
        (points_global[:, 0] >= aabb_min_global[0]) & (points_global[:, 0] <= aabb_max_global[0]) &
        (points_global[:, 1] >= aabb_min_global[1]) & (points_global[:, 1] <= aabb_max_global[1]) &
        (points_global[:, 2] >= aabb_min_global[2]) & (points_global[:, 2] <= aabb_max_global[2])
    )
    
    candidate_indices = np.where(mask_in_aabb)[0]
    if candidate_indices.size == 0:
        return final_in_box_mask # No points even in AABB, so none in OBB

    candidate_points_global = points_global[candidate_indices]

    # --- 2. Narrow-phase: OBB Check (Separating Axis Theorem based on projections) ---
    # This part is applied only to candidate_points_global

    print_debug_for_this_box = (debug_instance_token is not None and
                                box_global.token is not None and
                                debug_instance_token in box_global.token) # Check if token exists

    if print_debug_for_this_box:
        print(f"\n--- Debugging OBB Projections for Box (Token Fragment: {box_global.token[-10:] if box_global.token else 'N/A'}, Name: {box_global.name}) ---")
        print(f"  AABB Culling: {num_original_points} total -> {candidate_indices.size} candidates.")
        print(f"  Box Center (Global): {box_global.center}")
        print(f"  Box WLH (W,L,H): {box_global.wlh}")
        print(f"  Box Orientation (Quat w,x,y,z): {box_global.orientation.elements}")

    vec_center_to_candidate_points = candidate_points_global - box_global.center

    # Box axes in global coordinates (from box's rotation matrix)
    # NuScenes Box: Local X is along Length, Local Y is along Width, Local Z is along Height
    box_axis_len_global = box_global.rotation_matrix[:, 0] # Local X -> Length
    box_axis_wid_global = box_global.rotation_matrix[:, 1] # Local Y -> Width
    box_axis_hgt_global = box_global.rotation_matrix[:, 2] # Local Z -> Height

    # Projections of (candidate_points - center) onto box axes
    dist_along_len_axis = vec_center_to_candidate_points @ box_axis_len_global
    dist_along_wid_axis = vec_center_to_candidate_points @ box_axis_wid_global
    dist_along_hgt_axis = vec_center_to_candidate_points @ box_axis_hgt_global

    # Half dimensions from WLH array: [width, length, height]
    half_width  = box_global.wlh[0] / 2.0
    half_length = box_global.wlh[1] / 2.0
    half_height = box_global.wlh[2] / 2.0

    # Check if projections are within half dimensions
    mask_candidates_len = np.abs(dist_along_len_axis) <= half_length
    mask_candidates_wid = np.abs(dist_along_wid_axis) <= half_width
    mask_candidates_hgt = np.abs(dist_along_hgt_axis) <= half_height
    
    mask_candidates_in_obb = mask_candidates_len & mask_candidates_wid & mask_candidates_hgt

    # Map the results from candidate points back to the original point indices
    final_indices_in_obb = candidate_indices[mask_candidates_in_obb]
    final_in_box_mask[final_indices_in_obb] = True

    if print_debug_for_this_box:
        num_printed = 0
        for i_cand in range(candidate_points_global.shape[0]): # Iterate over candidate indices
            if num_printed >= max_points_to_print_debug:
                break
            original_pt_idx = candidate_indices[i_cand]
            
            print(f"  Candidate Point {i_cand} (Original Index: {original_pt_idx}, Global: {candidate_points_global[i_cand]})")
            print(f"    Proj Length-Axis: {dist_along_len_axis[i_cand]:.3f} (Limit: +/-{half_length:.3f}) -> In: {mask_candidates_len[i_cand]}")
            print(f"    Proj Width-Axis:  {dist_along_wid_axis[i_cand]:.3f} (Limit: +/-{half_width:.3f}) -> In: {mask_candidates_wid[i_cand]}")
            print(f"    Proj Height-Axis: {dist_along_hgt_axis[i_cand]:.3f} (Limit: +/-{half_height:.3f}) -> In: {mask_candidates_hgt[i_cand]}")
            print(f"    Candidate In OBB: {mask_candidates_in_obb[i_cand]} -> Final Mask at Original Idx: {final_in_box_mask[original_pt_idx]}")
            if mask_candidates_in_obb[i_cand]:
                 num_printed +=1 # Only count if it's an "in" point for debug brevity
        print(f"  OBB Check: {candidate_indices.size} candidates -> {np.sum(mask_candidates_in_obb)} in OBB.")
        print(f"--- End Debugging OBB Projections for Box ---")

    return final_in_box_mask



    
def generate_and_save_point_labels_for_scene_hdf5( # Renamed function
    nusc: NuScenes,
    scene_token: str,
    output_hdf5_dir: str,
    verbose: bool = True
) -> Optional[str]: # Returns path to HDF5 file or None
    """
    Generates point labels for all points in all LiDAR sweeps of a given scene
    using interpolated/extrapolated GT boxes.
    Saves all results for the scene into a single HDF5 file.
    """
    scene_record = nusc.get('scene', scene_token)
    scene_name = scene_record['name']

    os.makedirs(output_hdf5_dir, exist_ok=True)
    output_hdf5_filename = f"gt_point_labels_{scene_name}.h5" # Changed extension
    output_hdf5_filepath = os.path.join(output_hdf5_dir, output_hdf5_filename)

    if verbose:
        tqdm.write(f"Processing scene for GT point labels: {scene_name} ({scene_token})")
        tqdm.write(f"Output will be saved to: {output_hdf5_filepath}")

    # --- Data gathering  ---
    all_sweep_data_dicts = list(get_scene_sweep_data_sequence(nusc, scene_token))
    if not all_sweep_data_dicts:
        if verbose: tqdm.write(f"No LiDAR sweeps found for scene {scene_name}. Skipping.")
        return None
    if verbose: tqdm.write(f"Found {len(all_sweep_data_dicts)} LiDAR sweeps for scene {scene_name}.")

    instance_tokens = find_instances_in_scene(nusc, scene_token, min_annotations=1)
    if not instance_tokens:
        if verbose: tqdm.write(f"No instances with annotations found in scene {scene_name}. Skipping.")
        return None
    if verbose: tqdm.write(f"Found {len(instance_tokens)} instances to process for GT labels.")

    all_instances_boxes_at_sweeps: Dict[str, List[Optional[NuScenesDataClassesBox]]] = {}
    for i, inst_token in enumerate(instance_tokens):
        if verbose and (i + 1) % 10 == 0:
            tqdm.write(f"  Pre-calculating GT box states for instance {i+1}/{len(instance_tokens)} ({inst_token[:6]}...)")
        boxes_at_sweeps, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
            nusc, inst_token, all_sweep_data_dicts
        )
        all_instances_boxes_at_sweeps[inst_token] = boxes_at_sweeps
    if verbose: tqdm.write("All instance GT box states generated. Now creating point labels per sweep...")

    hdf5_sweep_lidar_sd_tokens: List[bytes] = [] # Store as bytes for 'S36'
    hdf5_sweep_timestamps_us: List[int] = []
    hdf5_all_point_labels_list: List[np.ndarray] = []
    hdf5_point_labels_indices: List[int] = [0]

    sweep_iterator = tqdm(enumerate(all_sweep_data_dicts), total=len(all_sweep_data_dicts), desc="  Generating GT Labels") if verbose else enumerate(all_sweep_data_dicts)

    for sweep_idx, sweep_data in sweep_iterator:
        lidar_sd_token_str = sweep_data['lidar_sd_token']
        # Ensure lidar_sd_token is bytes if it's coming as str from dummy or real function
        if isinstance(lidar_sd_token_str, str):
            lidar_sd_token_bytes = lidar_sd_token_str.encode('utf-8')
        else: # Assuming it's already bytes
            lidar_sd_token_bytes = lidar_sd_token_str

        points_sensor_frame = sweep_data['points_sensor_frame']
        T_global_lidar = sweep_data['T_global_lidar']
        timestamp = sweep_data['timestamp']

        hdf5_sweep_lidar_sd_tokens.append(lidar_sd_token_bytes)
        hdf5_sweep_timestamps_us.append(timestamp)

        if points_sensor_frame.shape[0] == 0:
            current_sweep_point_labels = np.zeros(0, dtype=POINT_LABEL_DTYPE)
            hdf5_all_point_labels_list.append(current_sweep_point_labels)
            hdf5_point_labels_indices.append(hdf5_point_labels_indices[-1])
            if verbose: tqdm.write(f"    No points in LiDAR sweep {lidar_sd_token_str}. Recorded empty labels.")
            continue

        points_global = transform_points_numpy(points_sensor_frame, T_global_lidar)

        if points_global.shape[0] == 0:
            current_sweep_point_labels = np.zeros(0, dtype=POINT_LABEL_DTYPE)
            hdf5_all_point_labels_list.append(current_sweep_point_labels)
            hdf5_point_labels_indices.append(hdf5_point_labels_indices[-1])
            if verbose: tqdm.write(f"    No points after transformation for {lidar_sd_token_str}. Recorded empty labels.")
            continue

        current_sweep_point_labels = np.zeros(points_global.shape[0], dtype=POINT_LABEL_DTYPE)
        current_sweep_point_labels['instance_token'] = b''
        current_sweep_point_labels['category_name'] = b''
        if points_global.shape[0] > 0:
            current_sweep_point_labels['x'] = points_global[:, 0]
            current_sweep_point_labels['y'] = points_global[:, 1]
            current_sweep_point_labels['z'] = points_global[:, 2]
            current_sweep_point_labels['x_sensor'] = points_sensor_frame[:, 0]
            current_sweep_point_labels['y_sensor'] = points_sensor_frame[:, 1]
            current_sweep_point_labels['z_sensor'] = points_sensor_frame[:, 2]

        for inst_token, list_of_boxes_for_instance in all_instances_boxes_at_sweeps.items():
            box_object = list_of_boxes_for_instance[sweep_idx]
            if box_object is not None:
                try:
                    mask_points_in_obb = _get_points_in_box_mask_global_coords(
                        points_global, box_object
                    )
                    if np.any(mask_points_in_obb):
                        unassigned_mask_for_current_obb_points = (current_sweep_point_labels['instance_token'][mask_points_in_obb] == b'')
                        indices_in_obb_and_unassigned = np.where(mask_points_in_obb)[0][unassigned_mask_for_current_obb_points]
                        if indices_in_obb_and_unassigned.size > 0:
                            current_sweep_point_labels['instance_token'][indices_in_obb_and_unassigned] = inst_token.encode('utf-8')
                            current_sweep_point_labels['category_name'][indices_in_obb_and_unassigned] = box_object.name.encode('utf-8')
                            current_sweep_point_labels['velocity_x'][indices_in_obb_and_unassigned] = box_object.velocity[0]
                            current_sweep_point_labels['velocity_y'][indices_in_obb_and_unassigned] = box_object.velocity[1]
                            current_sweep_point_labels['velocity_z'][indices_in_obb_and_unassigned] = box_object.velocity[2]
                except Exception as e_gen:
                    if verbose: tqdm.write(f"Error processing instance {inst_token[:6]} in sweep {lidar_sd_token_str}: {e_gen}")
                    continue

        hdf5_all_point_labels_list.append(current_sweep_point_labels)
        hdf5_point_labels_indices.append(hdf5_point_labels_indices[-1] + len(current_sweep_point_labels))

    if not hdf5_sweep_lidar_sd_tokens:
        if verbose: tqdm.write(f"No sweep data was processed to generate labels for scene {scene_name}. No HDF5 file created.")
        return None

    final_all_point_labels = np.concatenate(hdf5_all_point_labels_list, axis=0) if hdf5_all_point_labels_list else np.empty(0, dtype=POINT_LABEL_DTYPE)

    # --- Save to HDF5 ---
    try:
        with h5py.File(output_hdf5_filepath, 'w') as hf:
            # Store scene token as a scalar dataset of fixed-length string
            # h5py can create scalar datasets from 0-dim numpy arrays or directly from python strings/bytes
            hf.create_dataset('scene_token', data=scene_token.encode('utf-8')) # Save as bytes

            hf.create_dataset('sweep_lidar_sd_tokens', data=np.array(hdf5_sweep_lidar_sd_tokens, dtype='S36'))
            hf.create_dataset('sweep_timestamps_us', data=np.array(hdf5_sweep_timestamps_us, dtype=np.int64))
            
            # For the structured array, h5py handles it directly.
            # Check if final_all_point_labels is truly empty (shape (0,) or (0, num_fields))
            if final_all_point_labels.size == 0 and final_all_point_labels.shape[0] == 0 :
                 # Create an empty dataset with the correct dtype if the array is empty.
                 # This is important if the dtype has fields, h5py needs to know them.
                 hf.create_dataset('all_gt_point_labels', shape=(0,), dtype=POINT_LABEL_DTYPE)
            else:
                 hf.create_dataset('all_gt_point_labels', data=final_all_point_labels)

            hf.create_dataset('gt_point_labels_indices', data=np.array(hdf5_point_labels_indices, dtype=np.int64))

        if verbose: tqdm.write(f"Successfully saved GT point labels for scene {scene_name} to {output_hdf5_filepath}")
        return output_hdf5_filepath
    except Exception as e:
        if verbose: tqdm.write(f"Error saving GT point labels HDF5 file {output_hdf5_filepath}: {e}")
        return None


