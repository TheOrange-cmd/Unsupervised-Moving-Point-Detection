# scripts/generate_labels.py

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config_loader import MDetectorConfigAccessor
from src.utils.point_cloud_utils import filter_points, transform_points_numpy
from nuscenes.utils.data_classes import Box as NuScenesDataClassesBox, LidarPointCloud

# ==============================================================================
# Data Classes and Constants
# ==============================================================================

@dataclass
class SweepData:
    """Container for LiDAR sweep information."""
    points_sensor_frame: np.ndarray
    T_global_lidar: np.ndarray
    timestamp: int
    calibrated_sensor_token: str
    lidar_sd_token: str
    is_key_frame: bool
    sample_token: str

@dataclass
class AnnotationData:
    """Container for keyframe annotation information."""
    token: str
    sample_token: str
    timestamp: int
    translation: np.ndarray
    size: np.ndarray
    rotation: Quaternion
    category_name: str

class Constants:
    """Configuration constants."""
    KEYFRAME_INTERVAL_US = 500000
    MAX_EXTRAPOLATION_TIME_US = KEYFRAME_INTERVAL_US * 1.5  # 0.75s max extrapolation
    MIN_TIME_DIFF_SEC = 1e-3

# ==============================================================================
# Box Interpolation/Extrapolation Logic
# ==============================================================================

class BoxInterpolator:
    """Handles box interpolation and extrapolation for instance tracking."""
    
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
    
    def get_keyframe_annotations(self, instance_token: str) -> List[AnnotationData]:
        """Fetches and sorts all keyframe annotations for a given instance."""
        instance_rec = self.nusc.get('instance', instance_token)
        annotations = []
        
        current_token = instance_rec['first_annotation_token']
        while current_token:
            ann_rec = self.nusc.get('sample_annotation', current_token)
            sample_rec = self.nusc.get('sample', ann_rec['sample_token'])
            
            annotations.append(AnnotationData(
                token=ann_rec['token'],
                sample_token=ann_rec['sample_token'],
                timestamp=sample_rec['timestamp'],
                translation=np.array(ann_rec['translation']),
                size=np.array(ann_rec['size']),
                rotation=Quaternion(ann_rec['rotation']),
                category_name=ann_rec['category_name']
            ))
            
            if current_token == instance_rec['last_annotation_token']:
                break
            current_token = ann_rec['next']
        
        return sorted(annotations, key=lambda x: x.timestamp)
    
    def get_velocity(self, annotation: AnnotationData, all_annotations: List[AnnotationData]) -> np.ndarray:
        """Calculate velocity for an annotation using SDK or fallback methods."""
        try:
            velocity_sdk = self.nusc.box_velocity(annotation.token)
            if not np.any(np.isnan(velocity_sdk)):
                return velocity_sdk[:3]
        except AssertionError:
            pass
        
        # Fallback: manual calculation
        return self._calculate_velocity_fallback(annotation, all_annotations)
    
    def _calculate_velocity_fallback(self, annotation: AnnotationData, all_annotations: List[AnnotationData]) -> np.ndarray:
        """Calculate velocity using neighboring annotations when SDK fails."""
        if len(all_annotations) <= 1:
            return np.array([0.0, 0.0, 0.0])
        
        # Find current annotation index
        current_idx = next((i for i, ann in enumerate(all_annotations) if ann.token == annotation.token), -1)
        if current_idx == -1:
            return np.array([0.0, 0.0, 0.0])
        
        # Use previous annotation if available
        if current_idx > 0:
            prev_ann = all_annotations[current_idx - 1]
            dt_sec = (annotation.timestamp - prev_ann.timestamp) / 1e6
            if dt_sec > Constants.MIN_TIME_DIFF_SEC:
                return (annotation.translation - prev_ann.translation) / dt_sec
        
        # Use next annotation if available
        if current_idx < len(all_annotations) - 1:
            next_ann = all_annotations[current_idx + 1]
            dt_sec = (next_ann.timestamp - annotation.timestamp) / 1e6
            if dt_sec > Constants.MIN_TIME_DIFF_SEC:
                return (next_ann.translation - annotation.translation) / dt_sec
        
        return np.array([0.0, 0.0, 0.0])
    
    def create_box_from_annotation(self, annotation: AnnotationData, velocity: np.ndarray, token_suffix: str = "") -> NuScenesDataClassesBox:
        """Create a NuScenes box from annotation data."""
        token = f"{annotation.token}{token_suffix}" if token_suffix else annotation.token
        return NuScenesDataClassesBox(
            annotation.translation, 
            annotation.size, 
            annotation.rotation,
            name=annotation.category_name, 
            token=token,
            velocity=velocity
        )
    
    def interpolate_box(self, prev_ann: AnnotationData, next_ann: AnnotationData, target_timestamp: int, sweep_idx: int) -> NuScenesDataClassesBox:
        """Interpolate a box between two keyframe annotations."""
        ratio = (target_timestamp - prev_ann.timestamp) / (next_ann.timestamp - prev_ann.timestamp)
        
        interp_translation = prev_ann.translation + ratio * (next_ann.translation - prev_ann.translation)
        interp_rotation = Quaternion.slerp(prev_ann.rotation, next_ann.rotation, ratio)
        
        # Calculate velocity from the two reference annotations
        dt_sec = (next_ann.timestamp - prev_ann.timestamp) / 1e6
        velocity = np.array([0.0, 0.0, 0.0])
        if dt_sec > Constants.MIN_TIME_DIFF_SEC:
            velocity = (next_ann.translation - prev_ann.translation) / dt_sec
        
        return NuScenesDataClassesBox(
            interp_translation, 
            prev_ann.size, 
            interp_rotation,
            name=prev_ann.category_name,
            token=f"interp_{prev_ann.token}_{next_ann.token}_{sweep_idx}",
            velocity=velocity
        )
    
    def extrapolate_box(self, annotations: List[AnnotationData], target_timestamp: int, sweep_idx: int) -> Optional[NuScenesDataClassesBox]:
        """Extrapolate a box beyond the keyframe range."""
        if len(annotations) < 1:
            return None
        
        is_forward = target_timestamp > annotations[-1].timestamp
        ref_ann = annotations[-1] if is_forward else annotations[0]
        time_diff_us = target_timestamp - ref_ann.timestamp
        
        # Check if within extrapolation limit
        if abs(time_diff_us) > Constants.MAX_EXTRAPOLATION_TIME_US:
            return None
        
        extrap_translation = ref_ann.translation.copy()
        velocity = np.array([0.0, 0.0, 0.0])
        
        # Calculate velocity from neighboring annotations if available
        if len(annotations) > 1:
            ref1, ref2 = (annotations[-2], annotations[-1]) if is_forward else (annotations[0], annotations[1])
            dt_sec = (ref2.timestamp - ref1.timestamp) / 1e6
            
            if dt_sec > Constants.MIN_TIME_DIFF_SEC:
                velocity = (ref2.translation - ref1.translation) / dt_sec
                extrap_translation = ref_ann.translation + velocity * (time_diff_us / 1e6)
        
        return NuScenesDataClassesBox(
            extrap_translation,
            ref_ann.size,
            ref_ann.rotation,
            name=ref_ann.category_name,
            token=f"extrap_{ref_ann.token}_{sweep_idx}",
            velocity=velocity
        )
    
    def get_boxes_for_sweeps(self, instance_token: str, sweeps: List[SweepData]) -> List[Optional[NuScenesDataClassesBox]]:
        """Get interpolated/extrapolated boxes for all sweeps of an instance."""
        annotations = self.get_keyframe_annotations(instance_token)
        if not annotations:
            return [None] * len(sweeps)
        
        timestamps = np.array([ann.timestamp for ann in annotations])
        boxes = [None] * len(sweeps)
        
        for i, sweep in enumerate(sweeps):
            target_ts = sweep.timestamp
            
            # Priority 1: Direct keyframe match
            if sweep.is_key_frame:
                match = next((ann for ann in annotations if ann.sample_token == sweep.sample_token), None)
                if match:
                    velocity = self.get_velocity(match, annotations)
                    boxes[i] = self.create_box_from_annotation(match, velocity)
                    continue
            
            # Priority 2: Exact timestamp match
            exact_match_idx = np.where(timestamps == target_ts)[0]
            if len(exact_match_idx) > 0:
                match = annotations[exact_match_idx[0]]
                velocity = self.get_velocity(match, annotations)
                boxes[i] = self.create_box_from_annotation(match, velocity)
                continue
            
            # Priority 3: Interpolation
            idx_after = np.searchsorted(timestamps, target_ts, side='left')
            if 0 < idx_after < len(timestamps):
                prev_ann, next_ann = annotations[idx_after - 1], annotations[idx_after]
                if prev_ann.timestamp < target_ts < next_ann.timestamp:
                    boxes[i] = self.interpolate_box(prev_ann, next_ann, target_ts, i)
                    continue
            
            # Priority 4: Extrapolation
            extrap_box = self.extrapolate_box(annotations, target_ts, i)
            if extrap_box:
                boxes[i] = extrap_box
        
        return boxes

# ==============================================================================
# Point-in-Box Detection
# ==============================================================================

class PointInBoxDetector:
    """Handles efficient point-in-box detection using AABB culling and OBB checks."""
    
    def __init__(self, debug_instance_token: Optional[str] = None, max_debug_points: int = 5):
        self.debug_instance_token = debug_instance_token
        self.max_debug_points = max_debug_points
    
    def get_points_in_box_mask(self, points_global: np.ndarray, box_global: NuScenesDataClassesBox) -> np.ndarray:
        """Check which points are inside a 3D oriented bounding box."""
        if points_global.shape[0] == 0:
            return np.array([], dtype=bool)
        
        # Initialize result mask
        final_mask = np.zeros(points_global.shape[0], dtype=bool)
        
        # Phase 1: AABB culling for efficiency
        aabb_candidates = self._get_aabb_candidates(points_global, box_global)
        if aabb_candidates.size == 0:
            return final_mask
        
        # Phase 2: Precise OBB check on candidates
        obb_mask = self._check_obb_candidates(
            points_global[aabb_candidates], box_global, aabb_candidates
        )
        
        final_mask[aabb_candidates[obb_mask]] = True
        return final_mask
    
    def _get_aabb_candidates(self, points: np.ndarray, box: NuScenesDataClassesBox) -> np.ndarray:
        """Get candidate points using Axis-Aligned Bounding Box culling."""
        corners = box.corners()  # Shape (3, 8)
        aabb_min = np.min(corners, axis=1)
        aabb_max = np.max(corners, axis=1)
        
        mask = (
            (points[:, 0] >= aabb_min[0]) & (points[:, 0] <= aabb_max[0]) &
            (points[:, 1] >= aabb_min[1]) & (points[:, 1] <= aabb_max[1]) &
            (points[:, 2] >= aabb_min[2]) & (points[:, 2] <= aabb_max[2])
        )
        
        return np.where(mask)[0]
    
    def _check_obb_candidates(self, candidate_points: np.ndarray, box: NuScenesDataClassesBox, candidate_indices: np.ndarray) -> np.ndarray:
        """Check candidates against oriented bounding box using separating axis theorem."""
        if candidate_points.shape[0] == 0:
            return np.array([], dtype=bool)
        
        # Vector from box center to each candidate point
        center_to_points = candidate_points - box.center
        
        # Box axes in global coordinates
        axes = box.rotation_matrix  # Each column is an axis
        length_axis = axes[:, 0]    # Local X -> Length
        width_axis = axes[:, 1]     # Local Y -> Width  
        height_axis = axes[:, 2]    # Local Z -> Height
        
        # Project points onto each box axis
        proj_length = center_to_points @ length_axis
        proj_width = center_to_points @ width_axis
        proj_height = center_to_points @ height_axis
        
        # Check if projections are within box half-dimensions
        half_length, half_width, half_height = box.wlh[1]/2, box.wlh[0]/2, box.wlh[2]/2
        
        length_mask = np.abs(proj_length) <= half_length
        width_mask = np.abs(proj_width) <= half_width
        height_mask = np.abs(proj_height) <= half_height
        
        obb_mask = length_mask & width_mask & height_mask
        
        # Debug output if requested
        self._debug_obb_check(box, candidate_points, candidate_indices, 
                            proj_length, proj_width, proj_height,
                            half_length, half_width, half_height,
                            length_mask, width_mask, height_mask, obb_mask)
        
        return obb_mask
    
    def _debug_obb_check(self, box, candidate_points, candidate_indices, 
                        proj_length, proj_width, proj_height,
                        half_length, half_width, half_height,
                        length_mask, width_mask, height_mask, obb_mask):
        """Print debug information for OBB checks if debug token matches."""
        should_debug = (
            self.debug_instance_token is not None and 
            box.token is not None and 
            self.debug_instance_token in box.token
        )
        
        if not should_debug:
            return
        
        print(f"\n--- Debug OBB for Box (Token: {box.token[-10:] if box.token else 'N/A'}, Name: {box.name}) ---")
        print(f"  Candidates: {len(candidate_indices)} -> {np.sum(obb_mask)} in OBB")
        print(f"  Box Center: {box.center}, WLH: {box.wlh}")
        
        points_printed = 0
        for i, (is_in_obb, orig_idx) in enumerate(zip(obb_mask, candidate_indices)):
            if points_printed >= self.max_debug_points:
                break
            
            if is_in_obb:  # Only print points that are inside
                print(f"  Point {i} (Orig: {orig_idx}, Pos: {candidate_points[i]})")
                print(f"    Length: {proj_length[i]:.3f} ≤ {half_length:.3f} -> {length_mask[i]}")
                print(f"    Width:  {proj_width[i]:.3f} ≤ {half_width:.3f} -> {width_mask[i]}")
                print(f"    Height: {proj_height[i]:.3f} ≤ {half_height:.3f} -> {height_mask[i]}")
                points_printed += 1

# ==============================================================================
# Scene Processing
# ==============================================================================

class SceneProcessor:
    """Handles processing of individual scenes."""
    
    def __init__(self, nusc: NuScenes, config_accessor: MDetectorConfigAccessor):
        self.nusc = nusc
        self.config = config_accessor
        self.interpolator = BoxInterpolator(nusc)
        self.point_detector = PointInBoxDetector()
    
    def get_scene_instances(self, scene_token: str, min_annotations: int = 1) -> List[str]:
        """Find all instances in a scene with minimum annotation count."""
        scene_rec = self.nusc.get('scene', scene_token)
        instance_tokens = set()
        
        current_sample_token = scene_rec['first_sample_token']
        while current_sample_token:
            sample_rec = self.nusc.get('sample', current_sample_token)
            for ann_token in sample_rec['anns']:
                ann_rec = self.nusc.get('sample_annotation', ann_token)
                instance_tokens.add(ann_rec['instance_token'])
            current_sample_token = sample_rec['next']
        
        # Filter by annotation count
        valid_instances = []
        for instance_token in instance_tokens:
            instance_rec = self.nusc.get('instance', instance_token)
            if instance_rec['nbr_annotations'] >= min_annotations:
                valid_instances.append(instance_token)
        
        return valid_instances
    
    def get_scene_sweeps(self, scene_token: str, lidar_name: str = "LIDAR_TOP") -> List[SweepData]:
        """Get all LiDAR sweeps for a scene in chronological order."""
        scene_rec = self.nusc.get('scene', scene_token)
        
        # Find first sweep
        first_sample_token = scene_rec['first_sample_token']
        current_sd_token = self._find_first_sweep_token(first_sample_token, lidar_name)
        if not current_sd_token:
            return []
        
        # Collect all sweeps
        sweeps = []
        while current_sd_token:
            sweep_data = self._create_sweep_data(current_sd_token)
            if sweep_data:
                sweeps.append(sweep_data)
            
            sweep_rec = self.nusc.get('sample_data', current_sd_token)
            current_sd_token = sweep_rec['next']
        
        return sweeps
    
    def _find_first_sweep_token(self, first_sample_token: str, lidar_name: str) -> Optional[str]:
        """Find the first sweep token for the given LiDAR sensor."""
        # Get initial sweep token
        current_sd_token = self.nusc.get('sample', first_sample_token)['data'].get(lidar_name)
        if not current_sd_token:
            # Search through samples to find one with the LiDAR data
            sample_token = first_sample_token
            while sample_token:
                sample_rec = self.nusc.get('sample', sample_token)
                if lidar_name in sample_rec['data']:
                    current_sd_token = sample_rec['data'][lidar_name]
                    break
                sample_token = sample_rec['next']
        
        if not current_sd_token:
            return None
        
        # Go back to the very first sweep
        while True:
            sweep_rec = self.nusc.get('sample_data', current_sd_token)
            if sweep_rec['prev']:
                current_sd_token = sweep_rec['prev']
            else:
                break
        
        return current_sd_token
    
    def _create_sweep_data(self, sd_token: str) -> Optional[SweepData]:
        """Create sweep data from sample_data token."""
        sweep_rec = self.nusc.get('sample_data', sd_token)
        cs_rec = self.nusc.get('calibrated_sensor', sweep_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sweep_rec['ego_pose_token'])
        
        # Load point cloud
        pc_filepath = os.path.join(self.nusc.dataroot, sweep_rec['filename'])
        if not os.path.exists(pc_filepath):
            points = np.empty((0, 5), dtype=np.float32)
        else:
            pc = LidarPointCloud.from_file(pc_filepath)
            points = pc.points.T
        
        # Calculate transformation matrices
        T_sensor_ego = self._create_transform_matrix(cs_rec['rotation'], cs_rec['translation'])
        T_ego_global = self._create_transform_matrix(pose_rec['rotation'], pose_rec['translation'])
        T_global_sensor = T_ego_global @ T_sensor_ego
        
        return SweepData(
            points_sensor_frame=points,
            T_global_lidar=T_global_sensor,
            timestamp=sweep_rec['timestamp'],
            calibrated_sensor_token=sweep_rec['calibrated_sensor_token'],
            lidar_sd_token=sd_token,
            is_key_frame=sweep_rec['is_key_frame'],
            sample_token=sweep_rec['sample_token']
        )
    
    def _create_transform_matrix(self, rotation: List[float], translation: List[float]) -> np.ndarray:
        """Create 4x4 transformation matrix from rotation and translation."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Quaternion(rotation).rotation_matrix
        T[:3, 3] = np.array(translation)
        return T
    
    def generate_labels(self, scene_rec: Dict) -> None:
        """Generate and save sparse GT labels for a scene."""
        scene_token, scene_name = scene_rec['token'], scene_rec['name']
        
        # Setup paths and parameters
        output_dir = Path(self.config.get_nuscenes_params()['label_path'])
        gt_vel_thresh = self.config.get_validation_params()['gt_velocity_threshold']
        filter_params = self.config.get_point_pre_filtering_params()
        output_file = output_dir / f"gt_sparse_labels_{scene_name}_v{gt_vel_thresh}.pt"
        
        tqdm.write(f"Processing scene '{scene_name}'...")
        
        # Get scene data
        sweeps = self.get_scene_sweeps(scene_token)
        if not sweeps:
            return
        
        instance_tokens = self.get_scene_instances(scene_token, min_annotations=1)
        
        # Get boxes for all instances across all sweeps
        tqdm.write(f"  Interpolating boxes for {len(instance_tokens)} instances...")
        instance_boxes = {}
        for token in tqdm(instance_tokens, desc="  Interpolating", leave=False):
            instance_boxes[token] = self.interpolator.get_boxes_for_sweeps(token, sweeps)
        
        # Process each sweep to find dynamic points
        all_dynamic_indices = []
        sweep_boundaries = [0]
        
        for i, sweep in enumerate(tqdm(sweeps, desc="  Processing sweeps", leave=False)):
            dynamic_indices = self._process_sweep(sweep, instance_boxes, i, filter_params, gt_vel_thresh)
            all_dynamic_indices.append(dynamic_indices)
            sweep_boundaries.append(sweep_boundaries[-1] + len(dynamic_indices))
        
        # Save results
        data_to_save = {
            'dynamic_point_indices': np.concatenate(all_dynamic_indices) if all_dynamic_indices else np.array([], dtype=np.int32),
            'sweep_boundary_indices': np.array(sweep_boundaries, dtype=np.int64)
        }
        torch.save(data_to_save, output_file)
        tqdm.write(f"  -> Saved to {output_file}")
    
    def _process_sweep(self, sweep: SweepData, instance_boxes: Dict[str, List], sweep_idx: int, 
                      filter_params: Dict, gt_vel_thresh: float) -> np.ndarray:
        """Process a single sweep to find dynamic point indices."""
        points_raw = sweep.points_sensor_frame[:, :3]
        if points_raw.shape[0] == 0:
            return np.array([], dtype=np.int32)
        
        # Apply point filtering
        keep_mask = filter_points(points_raw, filter_params)
        if not np.any(keep_mask):
            return np.array([], dtype=np.int32)
        
        original_indices = np.arange(points_raw.shape[0])
        points_global = transform_points_numpy(points_raw[keep_mask], sweep.T_global_lidar)
        
        # Find dynamic points
        dynamic_mask = np.zeros(points_global.shape[0], dtype=bool)
        for instance_token, boxes in instance_boxes.items():
            box = boxes[sweep_idx]
            if box and np.linalg.norm(box.velocity[:2]) >= gt_vel_thresh:
                dynamic_mask |= self.point_detector.get_points_in_box_mask(points_global, box)
        
        # Map back to original indices
        dynamic_original_indices = original_indices[keep_mask][dynamic_mask]
        return dynamic_original_indices.astype(np.int32)

# ==============================================================================
# Main Application
# ==============================================================================

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Generate sparse ground truth point labels for NuScenes.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main YAML configuration file.')
    parser.add_argument('--scenes', type=str, default='all', help="Comma-separated list of scene indices or 'all'.")
    parser.add_argument('--clean', action='store_true', help="Remove existing label files before processing.")
    args = parser.parse_args()

    # Initialize components
    config_accessor = MDetectorConfigAccessor(args.config)
    nusc = NuScenes(
        version=config_accessor.get_nuscenes_params()['version'], 
        dataroot=config_accessor.get_nuscenes_params()['dataroot'], 
        verbose=False
    )
    processor = SceneProcessor(nusc, config_accessor)
    
    # Setup output directory
    output_dir = Path(config_accessor.get_nuscenes_params()['label_path'])
    if args.clean and output_dir.exists():
        print(f"Cleaning output directory: {output_dir}")
        for f in output_dir.glob('*.pt'):
            f.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine scenes to process
    if args.scenes.lower() == 'all':
        scene_indices = list(range(len(nusc.scene)))
    else:
        scene_indices = [int(i.strip()) for i in args.scenes.split(',')]
    
    # Process scenes
    print(f"Processing {len(scene_indices)} scenes...")
    for scene_idx in scene_indices:
        scene_record = nusc.scene[scene_idx]
        processor.generate_labels(scene_record)
    
    print("\nLabel generation complete.")

if __name__ == '__main__':
    main()