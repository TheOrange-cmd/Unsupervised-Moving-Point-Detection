# src/ray_scripts/ray_actors.py

import ray
import os
import logging
import time
import torch
import traceback
import numpy as np
from pyquaternion import Quaternion
from typing import Optional
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

# Profiling
import cProfile
import pstats
import io

# Import project-specific classes
from src.core.m_detector.base import MDetector
from src.data_utils.nuscenes_helper import NuScenesProcessor
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from ..config_loader import MDetectorConfigAccessor 
from ..data_utils.seeding_utils import set_seed  

@ray.remote
def _load_gt_file(path: str, device: str) -> tuple[str, dict]:
    scene_name = os.path.basename(path).split('_r')[0].replace('gt_point_labels_', '')
    data = torch.load(path, map_location=device, weights_only=False)
    return scene_name, data

# --- Standalone Helper Function for Loading ---
@ray.remote
def load_all_gt_data_in_background(config_path: str, device_str: str = 'cpu'):
    """
    A single, standalone remote task that loads all GT files and returns them
    in a dictionary. This is completely separate from the actor.
    """
    accessor = MDetectorConfigAccessor(config_path)
    nuscenes_cfg = accessor.get_nuscenes_params()
    filt_cfg = accessor.get_point_pre_filtering_params()
    min_r, max_r = filt_cfg['min_range_meters'], filt_cfg['max_range_meters']
    gt_labels_dir = nuscenes_cfg['label_path']
    
    # We need a mini-nusc object here just to iterate through scenes
    nusc = NuScenes(version=nuscenes_cfg['version'], dataroot=nuscenes_cfg['dataroot'], verbose=False)
    
    gt_cache = {}
    for scene_rec in nusc.scene:
        scene_name = scene_rec['name']
        gt_filename = f"gt_point_labels_{scene_name}_r{min_r}-{max_r}.pt"
        gt_path = os.path.join(gt_labels_dir, gt_filename)
        if os.path.exists(gt_path):
            gt_cache[scene_rec['token']] = torch.load(gt_path, map_location=device_str, weights_only=False)
            
    print(f"Background loading complete. Loaded {len(gt_cache)} GT files.")
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
        # No waiting logic is needed anymore. The cache is either there or not.
        if scene_token not in self.gt_cache:
            return None
            
        gt_data = self.gt_cache[scene_token]
        gt_labels_all = gt_data['point_labels']
        gt_sweep_indices = gt_data['sweep_indices']
        if sweep_index >= len(gt_sweep_indices) - 1:
            return None
        start_idx = gt_sweep_indices[sweep_index]
        end_idx = gt_sweep_indices[sweep_index + 1]
        return gt_labels_all[start_idx:end_idx]

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
            'points_sensor_frame': points_sensor_frame[:, :3],
            'T_global_lidar': T_global_sensor,
            'timestamp': sweep_rec['timestamp'],
        }

    def get_scene_sweep_tokens(self, scene_token: str, lidar_name: str) -> list[str]:
        scene_rec = self.nusc.get('scene', scene_token)
        current_sd_token = self.nusc.get('sample', scene_rec['first_sample_token'])['data'][lidar_name]
        
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
