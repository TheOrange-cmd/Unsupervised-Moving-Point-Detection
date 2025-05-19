# src/utils/notebook_helpers.py
import os
import numpy as np
import h5py
import k3d
from typing import Dict, Tuple, Optional, List, Any

from nuscenes.nuscenes import NuScenes

from src.core.constants import OcclusionResult, POINT_LABEL_DTYPE

def load_gt_labels_for_sweep_from_hdf5(
    scene_name: str,
    sweep_lidar_sd_token: str,
    gt_hdf5_base_dir: str,
    velocity_threshold_gt: float # Added for consistency with your notebook
) -> Optional[Dict[str, np.ndarray]]:
    """
    Loads GT dynamic/static/unlabeled points for a specific sweep directly from 
    the scene's HDF5 GT file.
    This version returns points as stored in the GT HDF5, not necessarily matched
    to a specific DepthImage's points.
    """
    gt_scene_hdf5_filename = f"gt_point_labels_{scene_name}.h5"
    gt_labels_scene_hdf5_filepath = os.path.join(gt_hdf5_base_dir, gt_scene_hdf5_filename)

    if not os.path.exists(gt_labels_scene_hdf5_filepath):
        print(f"GT HDF5 file not found: {gt_labels_scene_hdf5_filepath}")
        return None
    
    points_from_gt_file = {}
    try:
        with h5py.File(gt_labels_scene_hdf5_filepath, 'r') as hf:
            all_tokens_bytes = hf['sweep_lidar_sd_tokens'][:]
            target_token_bytes = sweep_lidar_sd_token.encode('utf-8')
            
            matches = np.where(all_tokens_bytes == target_token_bytes)[0]
            if not matches.size > 0:
                # Attempt decoding if direct byte match failed (robustness)
                all_tokens_str = [t.decode('utf-8', 'ignore') for t in all_tokens_bytes]
                matches = np.where(np.array(all_tokens_str) == sweep_lidar_sd_token)[0]
                if not matches.size > 0:
                    print(f"Sweep token {sweep_lidar_sd_token} not found in GT HDF5 {gt_labels_scene_hdf5_filepath}")
                    return None
            
            idx_in_token_list = matches[0]

            indices_array = hf['gt_point_labels_indices'][:]
            start_idx = indices_array[idx_in_token_list]
            
            if idx_in_token_list + 1 >= len(indices_array):
                 print(f"Warning: Index issue for token {sweep_lidar_sd_token}. "
                       f"idx_in_token_list+1 ({idx_in_token_list + 1}) is out of bounds for "
                       f"indices_array (len: {len(indices_array)}).")
                 return None
            end_idx = indices_array[idx_in_token_list + 1]
            
            point_labels_for_sweep_structured = hf['all_gt_point_labels'][start_idx:end_idx]

        if point_labels_for_sweep_structured.shape[0] > 0:
            # Ensure dtype consistency if there are subtle differences
            if point_labels_for_sweep_structured.dtype != POINT_LABEL_DTYPE:
                try:
                    point_labels_for_sweep_structured = point_labels_for_sweep_structured.astype(POINT_LABEL_DTYPE)
                except ValueError as e:
                    print(f"Warning: Dtype mismatch for GT labels and cast failed for sweep {sweep_lidar_sd_token}. Error: {e}. Skipping this sweep's GT.")
                    return None # Or return empty dicts

            gt_points_global = np.stack((point_labels_for_sweep_structured['x'],
                                         point_labels_for_sweep_structured['y'],
                                         point_labels_for_sweep_structured['z']), axis=-1)
            
            gt_speed_sq = point_labels_for_sweep_structured['velocity_x']**2 + \
                          point_labels_for_sweep_structured['velocity_y']**2
            is_valid_instance_mask = (point_labels_for_sweep_structured['instance_token'] != b'')
            
            dynamic_mask = (gt_speed_sq >= velocity_threshold_gt**2) & is_valid_instance_mask
            static_mask = (~dynamic_mask) & is_valid_instance_mask
            unlabeled_mask = ~is_valid_instance_mask # Points with no instance token

            points_from_gt_file['gt_dynamic_pts'] = gt_points_global[dynamic_mask]
            points_from_gt_file['gt_static_pts'] = gt_points_global[static_mask]
            points_from_gt_file['gt_unlabeled_pts'] = gt_points_global[unlabeled_mask]
            points_from_gt_file['gt_all_labeled_pts_structured'] = point_labels_for_sweep_structured
            points_from_gt_file['gt_all_pts_global'] = gt_points_global # All points from GT file for this sweep
        else:
            points_from_gt_file = {
                'gt_dynamic_pts': np.empty((0,3)), 'gt_static_pts': np.empty((0,3)),
                'gt_unlabeled_pts': np.empty((0,3)), 
                'gt_all_labeled_pts_structured': np.empty(0, dtype=POINT_LABEL_DTYPE),
                'gt_all_pts_global': np.empty((0,3))
            }
    except Exception as e:
        print(f"Error reading GT HDF5 file {gt_labels_scene_hdf5_filepath} for sweep {sweep_lidar_sd_token}: {e}")
        return None
        
    return points_from_gt_file

def load_mdet_results_for_sweep_from_hdf5(
    mdet_scene_hdf5_filepath: str,
    target_sweep_token: str
) -> Optional[Dict[str, np.ndarray]]:
    """Loads M-Detector classified points for a specific sweep from its HDF5 results file."""
    if not os.path.exists(mdet_scene_hdf5_filepath):
        print(f"M-Detector HDF5 file not found: {mdet_scene_hdf5_filepath}")
        return None

    try:
        with h5py.File(mdet_scene_hdf5_filepath, 'r') as hf:
            # Use the internal helper from k3d_visualizer if it's robust enough
            # or replicate its logic here for directness.
            # For now, replicating a simplified version:
            all_tokens_bytes = hf['sweep_lidar_sd_tokens'][:]
            target_token_bytes = target_sweep_token.encode('utf-8')
            
            matches = np.where(all_tokens_bytes == target_token_bytes)[0]
            if not matches.size > 0:
                all_tokens_str = [t.decode('utf-8', 'ignore') for t in all_tokens_bytes]
                matches = np.where(np.array(all_tokens_str) == target_sweep_token)[0]
                if not matches.size > 0:
                    print(f"Sweep token {target_sweep_token} not found in MDet HDF5 {mdet_scene_hdf5_filepath}")
                    return None

            idx_in_token_list = matches[0]

            indices_array = hf['points_predictions_indices'][:]
            start_idx = indices_array[idx_in_token_list]
            if idx_in_token_list + 1 >= len(indices_array):
                 print(f"Warning: Index issue for token {target_sweep_token} in MDet HDF5. "
                       f"idx_in_token_list+1 ({idx_in_token_list + 1}) is out of bounds for "
                       f"indices_array (len: {len(indices_array)}).")
                 return None
            end_idx = indices_array[idx_in_token_list + 1]
            
            preds_structured = hf['all_points_predictions'][start_idx:end_idx]

        if preds_structured.shape[0] > 0:
            points_global = np.stack((preds_structured['x'], preds_structured['y'], preds_structured['z']), axis=-1)
            labels = preds_structured['mdet_label']
            
            mdet_results = {
                'mdet_dynamic_pts': points_global[labels == OcclusionResult.OCCLUDING_IMAGE.value],
                'mdet_occluded_pts': points_global[labels == OcclusionResult.OCCLUDED_BY_IMAGE.value],
                'mdet_empty_pts': points_global[labels == OcclusionResult.EMPTY_IN_IMAGE.value],
                'mdet_undetermined_pts': points_global[labels == OcclusionResult.UNDETERMINED.value],
                'mdet_all_pts_global': points_global,
                'mdet_all_labels': labels,
                'mdet_all_structured_preds': preds_structured # For access to scores etc.
            }
        else:
            mdet_results = {
                'mdet_dynamic_pts': np.empty((0,3)), 'mdet_occluded_pts': np.empty((0,3)),
                'mdet_empty_pts': np.empty((0,3)), 'mdet_undetermined_pts': np.empty((0,3)),
                'mdet_all_pts_global': np.empty((0,3)), 'mdet_all_labels': np.empty(0, dtype=int),
                'mdet_all_structured_preds': np.empty(0, dtype=preds_structured.dtype if preds_structured is not None and preds_structured.dtype is not None else 'f4') # Fallback dtype
            }
        return mdet_results
    except Exception as e:
        print(f"Error reading MDet HDF5 file {mdet_scene_hdf5_filepath} for sweep {target_sweep_token}: {e}")
        return None

def create_k3d_plot_with_points(points_dict: Dict[str, Tuple[np.ndarray, int, float]], 
                                plot_title: str = "LiDAR Sweep",
                                background_color=0xAAAAAA, 
                                grid_visible=True, camera_auto_fit=True,
                                initial_camera_settings: Optional[List[float]] = None) -> Optional[k3d.Plot]:
    """
    Creates a K3D plot with multiple sets of points.
    points_dict: {'label': (points_array_Nx3, color_hex, point_size)}
    initial_camera_settings: [eye_x, eye_y, eye_z, look_at_x, look_at_y, look_at_z, up_x, up_y, up_z]
    """
    try:
        plot = k3d.plot(name=plot_title, grid_visible=grid_visible, 
                        camera_auto_fit=camera_auto_fit, background_color=background_color)
        for label, data_tuple in points_dict.items():
            if len(data_tuple) == 3:
                points, color, size = data_tuple
                shader = '3d' 
            elif len(data_tuple) == 4: 
                points, color, size, shader = data_tuple
            else:
                print(f"Warning: Incorrect data tuple length for '{label}'. Skipping.")
                continue

            if points is not None and points.shape[0] > 0 and points.ndim == 2 and points.shape[1] == 3:
                plot += k3d.points(positions=points.astype(np.float32), 
                                   color=color, point_size=size, shader=shader, name=label)
            elif points is not None and points.shape[0] > 0:
                 print(f"Warning: Points for '{label}' have incorrect shape {points.shape}. Expected Nx3. Skipping.")
            # else: points are None or empty, so skip silently

        if initial_camera_settings and len(initial_camera_settings) == 9:
            plot.camera = initial_camera_settings
            plot.camera_auto_fit = False 

        return plot
    except Exception as e:
        print(f"Error creating K3D plot '{plot_title}': {e}")
        return None