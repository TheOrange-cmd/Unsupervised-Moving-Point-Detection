# src/tuning/ray_actors.py

import ray
import os
import torch
import numpy as np
from pyquaternion import Quaternion
from typing import Optional


# Import project-specific classes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from ..config_loader import MDetectorConfigAccessor 

# --- Standalone Helper Function for Loading ---
@ray.remote
def load_all_gt_data_in_background(config_path: str, device_str: str = 'cpu'):
    """
    A single, standalone remote task that loads all SPARSE GT files and returns them
    in a dictionary. The data is already lean, so we just load it directly.
    """
    accessor = MDetectorConfigAccessor(config_path)
    nuscenes_cfg = accessor.get_nuscenes_params()
    # --- We now need the velocity threshold to find the correct files ---
    validation_cfg = accessor.get_validation_params()
    gt_vel_thresh = validation_cfg['gt_velocity_threshold']
    
    gt_labels_dir = nuscenes_cfg['label_path']
    
    nusc = NuScenes(version=nuscenes_cfg['version'], dataroot=nuscenes_cfg['dataroot'], verbose=False)
    
    gt_cache = {}
    print(f"Background loader: Searching for sparse GT files with v_thresh={gt_vel_thresh}...")
    for scene_rec in nusc.scene:
        scene_name = scene_rec['name']
        gt_filename = f"gt_sparse_labels_{scene_name}_v{gt_vel_thresh}.pt"
        gt_path = os.path.join(gt_labels_dir, gt_filename)
        
        if os.path.exists(gt_path):
            # The loaded data is already the lean dictionary we want
            gt_cache[scene_rec['token']] = torch.load(gt_path, map_location=device_str, weights_only=False)
            
    if not gt_cache:
        print(f"[bold red]WARNING:[/bold red] Background loader did not find any sparse GT files for v_thresh={gt_vel_thresh} in {gt_labels_dir}.")
    else:
        print(f"Background loading complete. Loaded {len(gt_cache)} sparse GT files.")
    return gt_cache

@ray.remote
class NuScenesDataActor:
    def __init__(self, version: str, dataroot: str, config_path: str):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.gt_cache = {}
        print("NuScenesDataActor initialized (empty).")

    def set_cache(self, gt_cache: dict):
        """A simple method to set the cache from the outside."""
        self.gt_cache = gt_cache
        print(f"Actor cache populated with {len(self.gt_cache)} entries.")

    def get_ground_truth_slice(self, scene_token: str, sweep_index: int) -> Optional[np.ndarray]:
        """
        Returns a slice of the sparse dynamic point indices for a specific sweep.
        """
        if scene_token not in self.gt_cache:
            return None
            
        gt_data = self.gt_cache[scene_token]
        # These are the two arrays we saved in our new .pt files
        gt_dynamic_indices_all = gt_data['dynamic_point_indices']
        gt_sweep_boundaries = gt_data['sweep_boundary_indices']
        
        if sweep_index >= len(gt_sweep_boundaries) - 1:
            return None
            
        # Use the boundaries to slice the correct portion of the dynamic indices array
        start_idx = gt_sweep_boundaries[sweep_index]
        end_idx = gt_sweep_boundaries[sweep_index + 1]
        
        return gt_dynamic_indices_all[start_idx:end_idx]

    # All other get_* methods are also simple and direct
    def get_nusc_handle(self):
        return self.nusc
    
    def get_scene_record(self, scene_index: int) -> dict:
        return self.nusc.scene[scene_index]
    
    def get_scene_count(self) -> int:
        return len(self.nusc.scene)

    def get_sweep_data_by_token(self, lidar_sd_token: str) -> dict:
        sweep_rec = self.nusc.get('sample_data', lidar_sd_token)
        cs_rec = self.nusc.get('calibrated_sensor', sweep_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sweep_rec['ego_pose_token'])
        pc_filepath = os.path.join(self.nusc.dataroot, sweep_rec['filename'])

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

        return {
            # Return the raw point cloud so the M-Detector can do its own filtering
            'points_sensor_frame': points_sensor_frame[:, :3],
            'T_global_lidar': T_global_sensor,
            'timestamp': sweep_rec['timestamp'],
        }

    def get_scene_sweep_tokens(self, scene_token: str) -> list[str]:
        lidar_name = 'LIDAR_TOP'
        scene_rec = self.nusc.get('scene', scene_token)
        first_sample_token = scene_rec['first_sample_token']
        current_sd_token = self.nusc.get('sample', first_sample_token)['data'].get(lidar_name)
        if not current_sd_token:
            sample_token = first_sample_token
            while sample_token:
                sample_rec = self.nusc.get('sample', sample_token)
                if lidar_name in sample_rec['data']:
                    current_sd_token = sample_rec['data'][lidar_name]
                    break
                sample_token = sample_rec['next']
            if not current_sd_token: return []
        
        while True:
            sd_rec = self.nusc.get('sample_data', current_sd_token)
            if sd_rec['prev']:
                current_sd_token = sd_rec['prev']
            else:
                break
        
        all_tokens = []
        while current_sd_token:
            all_tokens.append(current_sd_token)
            current_sd_token = self.nusc.get('sample_data', current_sd_token)['next']
            
        return all_tokens
