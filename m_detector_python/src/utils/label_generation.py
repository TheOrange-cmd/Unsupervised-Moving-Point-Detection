# src/data_utils/label_generation.py
import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box as NuScenesDataClassesBox
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import List, Dict, Optional
import h5py
import torch 

from .transformations import transform_points_numpy

from nuscenes.utils.data_classes import LidarPointCloud 
from pyquaternion import Quaternion 

def get_scene_sweep_data_sequence(nusc: NuScenes, scene_token: str, lidar_name: str = "LIDAR_TOP"):
    scene_rec = nusc.get('scene', scene_token)
    first_sample_token = scene_rec['first_sample_token']
    current_sd_token = nusc.get('sample', first_sample_token)['data'].get(lidar_name)
    if not current_sd_token:
        sample_token = first_sample_token
        while sample_token:
            sample_rec = nusc.get('sample', sample_token)
            if lidar_name in sample_rec['data']:
                current_sd_token = sample_rec['data'][lidar_name]
                break
            sample_token = sample_rec['next']
        if not current_sd_token: return

    while True:
        sd_rec = nusc.get('sample_data', current_sd_token)
        if sd_rec['prev']: current_sd_token = sd_rec['prev']
        else: break
            
    while current_sd_token:
        sweep_rec = nusc.get('sample_data', current_sd_token)
        cs_rec = nusc.get('calibrated_sensor', sweep_rec['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', sweep_rec['ego_pose_token'])
        pc_filepath = os.path.join(nusc.dataroot, sweep_rec['filename'])

        if not os.path.exists(pc_filepath):
            points_sensor_frame = np.empty((0, 5), dtype=np.float32)
        else:
            pc = LidarPointCloud.from_file(pc_filepath)
            points_sensor_frame = pc.points.T

        sens_to_ego_rot = Quaternion(cs_rec['rotation']).rotation_matrix
        sens_to_ego_trans = np.array(cs_rec['translation'])
        T_sensor_ego = np.eye(4, dtype=np.float32)
        T_sensor_ego[:3, :3] = sens_to_ego_rot
        T_sensor_ego[:3, 3] = sens_to_ego_trans

        ego_to_glob_rot = Quaternion(pose_rec['rotation']).rotation_matrix
        ego_to_glob_trans = np.array(pose_rec['translation'])
        T_ego_global = np.eye(4, dtype=np.float32)
        T_ego_global[:3, :3] = ego_to_glob_rot
        T_ego_global[:3, 3] = ego_to_glob_trans

        T_global_sensor = T_ego_global @ T_sensor_ego

        yield {
            'points_sensor_frame': points_sensor_frame, # Yield the raw (N, 5) point cloud
            'T_global_lidar': T_global_sensor,
            'timestamp': sweep_rec['timestamp'],
            'calibrated_sensor_token': sweep_rec['calibrated_sensor_token'],
            'lidar_sd_token': current_sd_token,
            'is_key_frame': sweep_rec['is_key_frame'],
            'sample_token': sweep_rec['sample_token'],
        }
        
        current_sd_token = sweep_rec['next']


def generate_and_save_point_labels_for_scene_pytorch(
    nusc: NuScenes,
    scene_token: str,
    output_dir: str,
    gt_velocity_threshold: float,
    verbose: bool = True
) -> Optional[str]:
    """
    Generates sparse ground truth labels for a scene and saves them to a lean .pt file.
    The file contains only the *original indices* of dynamic points for each sweep.

    Args:
        nusc: NuScenes API instance.
        scene_token (str): The token of the scene to process.
        output_dir (str): The directory to save the output .pt file.
        gt_velocity_threshold (float): The speed (m/s) above which a point is considered dynamic.
        verbose (bool): Whether to print progress messages.

    Returns:
        Optional[str]: The path to the generated .pt file, or None on failure.
    """
    scene_record = nusc.get('scene', scene_token)
    scene_name = scene_record['name']

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"gt_sparse_labels_{scene_name}_v{gt_velocity_threshold}.pt"
    output_filepath = os.path.join(output_dir, output_filename)

    if verbose:
        tqdm.write(f"Processing scene for sparse GT labels: {scene_name} (Vel Threshold: {gt_velocity_threshold} m/s)")

    all_sweep_data_dicts = list(get_scene_sweep_data_sequence(nusc, scene_token))
    if not all_sweep_data_dicts:
        if verbose: tqdm.write(f"No LiDAR sweeps found for scene {scene_name}. Skipping.")
        return None

    instance_tokens = find_instances_in_scene(nusc, scene_token, min_annotations=1)
    if not instance_tokens:
        if verbose: tqdm.write(f"No instances found in scene {scene_name}. Skipping.")
        return None

    all_instances_boxes_at_sweeps: Dict[str, List[Optional[NuScenesDataClassesBox]]] = {}
    for inst_token in instance_tokens:
        boxes_at_sweeps, _, _, _ = get_interpolated_extrapolated_boxes_for_instance(
            nusc, inst_token, all_sweep_data_dicts
        )
        all_instances_boxes_at_sweeps[inst_token] = boxes_at_sweeps
    
    # --- Processing and Data Structuring ---
    all_dynamic_indices_list = []
    sweep_boundary_indices = [0] # Marks the start of dynamic indices for each sweep

    sweep_iterator = tqdm(enumerate(all_sweep_data_dicts), total=len(all_sweep_data_dicts), desc="  Generating Sparse GT Labels") if verbose else enumerate(all_sweep_data_dicts)

    for sweep_idx, sweep_data in sweep_iterator:
        # Load the raw, unfiltered point cloud
        points_sensor_raw = sweep_data['points_sensor_frame'][:, :3] # Use only XYZ
        
        if points_sensor_raw.shape[0] == 0:
            all_dynamic_indices_list.append(np.array([], dtype=np.int32))
            sweep_boundary_indices.append(sweep_boundary_indices[-1])
            continue

        T_global_lidar = sweep_data['T_global_lidar']
        points_global_raw = transform_points_numpy(points_sensor_raw, T_global_lidar)
        
        # This mask will accumulate all dynamic points for the current sweep
        dynamic_mask_for_sweep = np.zeros(points_global_raw.shape[0], dtype=bool)

        for inst_token, list_of_boxes_for_instance in all_instances_boxes_at_sweeps.items():
            box_object = list_of_boxes_for_instance[sweep_idx]
            if box_object is not None:
                # Check if the object itself is dynamic
                speed_sq = box_object.velocity[0]**2 + box_object.velocity[1]**2
                if speed_sq >= gt_velocity_threshold**2:
                    # If the box is dynamic, find which raw points fall inside it
                    mask_points_in_box = _get_points_in_box_mask_global_coords(points_global_raw, box_object)
                    dynamic_mask_for_sweep[mask_points_in_box] = True

        # Find the original indices of the dynamic points for this sweep
        dynamic_indices_this_sweep = np.where(dynamic_mask_for_sweep)[0].astype(np.int32)
        
        all_dynamic_indices_list.append(dynamic_indices_this_sweep)
        sweep_boundary_indices.append(sweep_boundary_indices[-1] + len(dynamic_indices_this_sweep))

    # --- Final Assembly and Saving ---
    final_all_dynamic_indices = np.concatenate(all_dynamic_indices_list) if all_dynamic_indices_list else np.array([], dtype=np.int32)

    data_to_save = {
        'dynamic_point_indices': final_all_dynamic_indices,
        'sweep_boundary_indices': np.array(sweep_boundary_indices, dtype=np.int64)
    }

    try:
        torch.save(data_to_save, output_filepath)
        if verbose: tqdm.write(f"Successfully saved sparse GT data to {output_filepath}")
        return output_filepath
    except Exception as e:
        if verbose: tqdm.write(f"Error saving PyTorch file {output_filepath}: {e}")
        return None


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
        
        keyframe_annotations.append({
            'token': ann_rec['token'],
            'sample_token': ann_rec['sample_token'],
            'instance_token': ann_rec['instance_token'],
            'timestamp': sample_rec['timestamp'],
            'category_name': ann_rec['category_name'],
            'translation': np.array(ann_rec['translation']),
            'size': np.array(ann_rec['size']),
            'rotation': Quaternion(ann_rec['rotation']),
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



    