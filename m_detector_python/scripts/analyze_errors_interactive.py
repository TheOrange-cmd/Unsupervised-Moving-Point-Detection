# scripts/analyze_errors_interactive.py

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import open3d as o3d

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Core Project Imports ---
from src.config_loader import MDetectorConfigAccessor
from src.utils.visualization import ColorMapper, BoxDrawer, InteractiveVisualizer
from src.core.constants import OcclusionResult
from scripts.generate_labels import SceneProcessor

def prepare_visualization_frames(
    processed_data: list,
    gt_data: dict,
    instance_boxes: dict,
    color_mapper: ColorMapper,
    box_drawer: BoxDrawer,
    gt_vel_thresh: float
):
    """Prepares frames for the interactive visualizer with precise error coloring."""
    cached_frames = []
    gt_all_indices = gt_data['dynamic_point_indices']
    gt_boundaries = gt_data['sweep_boundary_indices']
    
    for i, frame_data in enumerate(tqdm(processed_data, desc="Preparing visualization frames")):
        geometries = {}
        
        if frame_data['points'].shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(frame_data['points'])
            
            # --- START OF CHANGE ---
            # Use the newly saved original_indices for precise error analysis
            pred_is_dyn = (frame_data['labels'] == OcclusionResult.OCCLUDING_IMAGE.value)
            
            # 1. Get the GT dynamic indices for the complete, raw sweep
            start_idx, end_idx = gt_boundaries[i], gt_boundaries[i+1]
            gt_dynamic_indices_for_raw_sweep = gt_all_indices[start_idx:end_idx]
            
            # 2. Get the original indices of the points the detector actually processed
            original_indices_processed = frame_data['original_indices']
            
            # 3. Create a boolean mask for GT labels that is perfectly aligned with the processed points
            #    `np.isin` checks which of our processed points' original indices are in the GT list.
            gt_is_dyn = np.isin(original_indices_processed, gt_dynamic_indices_for_raw_sweep)
            
            # 4. Generate precise TP/FP/FN/TN colors
            error_colors = color_mapper.get_error_analysis_colors(pred_is_dyn, gt_is_dyn)
            pcd.colors = o3d.utility.Vector3dVector(error_colors)
            # --- END OF CHANGE ---
            
            geometries['pcd'] = pcd
        else:
            geometries['pcd'] = o3d.geometry.PointCloud()

        # Add ground truth bounding boxes for context
        gt_boxes_for_sweep = [
            boxes[i] for boxes in instance_boxes.values() 
            if boxes[i] and np.linalg.norm(boxes[i].velocity[:2]) >= gt_vel_thresh
        ]
        for j, box in enumerate(gt_boxes_for_sweep):
            geometries[f"gt_box_{j}"] = box_drawer.get_box_obb_wireframe(box, color=color_mapper.GT_DYNAMIC)
        
        cached_frames.append({'geometries': geometries})
        
    return cached_frames

def main():
    parser = argparse.ArgumentParser(description="Interactively analyze processed M-Detector results.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main YAML config file.')
    parser.add_argument('--scene-index', type=int, required=True, help='The index of the scene to analyze.')
    parser.add_argument('--processed-file', type=str, required=True, help='Path to the pre-processed .pt results file.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the interactive animation.')
    args = parser.parse_args()

    config_accessor = MDetectorConfigAccessor(args.config)
    color_mapper = ColorMapper()
    box_drawer = BoxDrawer()
    
    nusc_params = config_accessor.get_nuscenes_params()
    nusc = NuScenes(version=nusc_params['version'], dataroot=nusc_params['dataroot'], verbose=False)
    scene_processor = SceneProcessor(nusc=nusc, config_accessor=config_accessor)
    
    scene_rec = nusc.scene[args.scene_index]
    all_sweeps = scene_processor.get_scene_sweeps(scene_rec['token'])

    gt_vel_thresh = config_accessor.get_validation_params()['gt_velocity_threshold']
    gt_file_path = Path(nusc_params['label_path']) / f"gt_sparse_labels_{scene_rec['name']}_v{gt_vel_thresh}.pt"
    if not gt_file_path.exists():
        print(f"FATAL: Ground truth file not found at {gt_file_path}. Cannot perform error analysis.")
        return
    gt_data = torch.load(gt_file_path, map_location='cpu', weights_only=False)

    instance_tokens = scene_processor.get_scene_instances(scene_rec['token'], min_annotations=1)
    instance_boxes = {token: scene_processor.interpolator.get_boxes_for_sweeps(token, all_sweeps) for token in instance_tokens}

    print(f"Loading processed data from {args.processed_file}...")
    processed_data = torch.load(args.processed_file, weights_only=False)
    
    cached_frames = prepare_visualization_frames(processed_data, gt_data, instance_boxes, color_mapper, box_drawer, gt_vel_thresh)

    visualizer = InteractiveVisualizer(window_name="M-Detector Interactive Error Analysis", fps=args.fps)
    visualizer.run_animation_loop(cached_frames)

if __name__ == '__main__':
    main()