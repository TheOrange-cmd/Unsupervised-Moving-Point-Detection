# scripts/visualize_gt.py

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Core Project Imports ---
from src.config_loader import MDetectorConfigAccessor
from src.utils.visualization import ColorMapper, BoxDrawer, VideoRenderer
from src.utils.point_cloud_utils import filter_points, transform_points_numpy
from nuscenes.nuscenes import NuScenes
from scripts.generate_labels import SceneProcessor, SweepData

def create_gt_visualization_geometries(
    sweep_data: SweepData, 
    gt_boxes_for_sweep: list, 
    gt_dynamic_indices: np.ndarray,
    filter_params: dict,
    color_mapper: ColorMapper,
    box_drawer: BoxDrawer
) -> dict:
    """Creates a dictionary of Open3D geometries for a single frame of GT visualization."""
    geometries = {}
    points_sensor_raw = sweep_data.points_sensor_frame[:, :3]
    points_global_raw = transform_points_numpy(points_sensor_raw, sweep_data.T_global_lidar)
    keep_mask = filter_points(points_sensor_raw, filter_params)
    
    colors = np.full((len(points_global_raw), 3), color_mapper.FILTERED_OUT)
    
    if np.any(keep_mask):
        # 1. Get the original indices of the points that survived filtering 
        original_indices_of_kept_points = np.where(keep_mask)[0]
        
        # 2. Find which of these kept points are also in the list of dynamic GT indices.
        is_dynamic_mask_for_kept = np.isin(original_indices_of_kept_points, gt_dynamic_indices)
        
        # 3. Get the corresponding blue/grey colors for the kept points.
        colors_for_kept_points = color_mapper.get_gt_colors(is_dynamic_mask_for_kept)
        
        # 4. Apply these colors to the main color array.
        colors[keep_mask] = colors_for_kept_points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_global_raw)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if len(pcd.points) > 0:
        pcd.estimate_normals()
    geometries['pcd'] = pcd
    
    for i, box in enumerate(gt_boxes_for_sweep):
        geometries[f"gt_box_{i}"] = box_drawer.get_box_obb_wireframe(box, color=color_mapper.GT_DYNAMIC)
            
    return geometries

def main():
    """Main entry point for the ground truth visualization script."""
    parser = argparse.ArgumentParser(description="Generate a video visualizing NuScenes ground truth with filtering.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main YAML configuration file.')
    parser.add_argument('--scene-index', type=int, required=True, help='The index of the scene to visualize.')
    parser.add_argument('--output-path', type=str, default='gt_visualization.mp4', help='Path to save the output video file.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the output video.')
    args = parser.parse_args()

    config_accessor = MDetectorConfigAccessor(args.config)
    nusc_params = config_accessor.get_nuscenes_params()
    nusc = NuScenes(version=nusc_params['version'], dataroot=nusc_params['dataroot'], verbose=False)
    
    color_mapper = ColorMapper(); box_drawer = BoxDrawer()
    
    scene_rec = nusc.scene[args.scene_index]
    scene_token = scene_rec['token']
    
    print(f"Loading data for scene: {scene_rec['name']}...")
    
    #Use the centralized processor for all data loading ---
    scene_processor = SceneProcessor(nusc, config_accessor)
    all_sweeps = scene_processor.get_scene_sweeps(scene_token)
    if not all_sweeps:
        print(f"No sweeps found for scene {scene_rec['name']}. Exiting.")
        return
    
    # Re-interpolate boxes on the fly using the shared logic
    instance_tokens = scene_processor.get_scene_instances(scene_token, min_annotations=1)
    instance_boxes = {
        token: scene_processor.interpolator.get_boxes_for_sweeps(token, all_sweeps)
        for token in tqdm(instance_tokens, desc="Interpolating GT boxes")
    }
        
    gt_vel_thresh = config_accessor.get_validation_params()['gt_velocity_threshold']
    gt_file_path = Path(nusc_params['label_path']) / f"gt_sparse_labels_{scene_rec['name']}_v{gt_vel_thresh}.pt"
    if not gt_file_path.exists():
        print(f"Ground truth file not found at {gt_file_path}. Please generate labels first.")
        return
        
    print(f"Loading sparse GT indices from {gt_file_path}...")
    gt_data = torch.load(gt_file_path, weights_only=False, map_location='cpu')
    
    gt_all_indices = gt_data['dynamic_point_indices']
    gt_boundaries = gt_data['sweep_boundary_indices']
    
    filter_params = config_accessor.get_point_pre_filtering_params()
    print(f"Rendering {len(all_sweeps)} frames to {args.output_path}...")
    
    renderer = VideoRenderer(output_path=args.output_path, fps=args.fps)
    
    for i, sweep in enumerate(tqdm(all_sweeps, desc="Rendering Frames")):
        # Get the GT boxes for this sweep (dynamic only)
        gt_boxes_for_sweep = [
            boxes[i] for boxes in instance_boxes.values() 
            if boxes[i] and np.linalg.norm(boxes[i].velocity[:2]) >= gt_vel_thresh
        ]
        
        start_idx, end_idx = gt_boundaries[i], gt_boundaries[i+1]
        gt_dynamic_indices_for_sweep = gt_all_indices[start_idx:end_idx]
        
        geometries = create_gt_visualization_geometries(
            sweep, gt_boxes_for_sweep, gt_dynamic_indices_for_sweep,
            filter_params, color_mapper, box_drawer
        )
        
        ego_position = sweep.T_global_lidar[:3, 3]
        camera_params = {
            'center': ego_position,
            'eye': ego_position + np.array([-40, -40, 20]), 
            'up': np.array([0, 0, 1])
        }
            
        renderer.render_frame(geometries, camera_params)
        
    renderer.close()

if __name__ == '__main__':
    main()