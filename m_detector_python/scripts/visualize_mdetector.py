# scripts/visualize_mdetector.py (Refactored)

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
from src.utils.visualization import ColorMapper, VideoRenderer
from scripts.generate_labels import SceneProcessor

def main():
    """Main entry point for visualizing pre-processed M-Detector results."""
    parser = argparse.ArgumentParser(description="Generate a video from processed M-Detector results.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main YAML config file (for data loading).')
    parser.add_argument('--scene-index', type=int, required=True, help='The index of the scene to visualize.')
    parser.add_argument('--processed-file', type=str, required=True, help='Path to the pre-processed .pt results file.')
    parser.add_argument('--output-path', type=str, default='mdetector_output.mp4', help='Path to save the output video.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for the output video.')
    args = parser.parse_args()

    # --- Initialization ---
    config_accessor = MDetectorConfigAccessor(args.config)
    color_mapper = ColorMapper()
    renderer = VideoRenderer(output_path=args.output_path, fps=args.fps)
    
    nusc_params = config_accessor.get_nuscenes_params()
    nusc = NuScenes(version=nusc_params['version'], dataroot=nusc_params['dataroot'], verbose=False)
    scene_processor = SceneProcessor(nusc=nusc, config_accessor=config_accessor)
    
    scene_rec = nusc.scene[args.scene_index]
    all_sweeps = scene_processor.get_scene_sweeps(scene_rec['token'])
    
    print(f"Loading processed data from {args.processed_file}...")
    processed_data = torch.load(args.processed_file, weights_only=False)

    # --- Main Rendering Loop ---
    for i, frame_data in enumerate(tqdm(processed_data, desc="Rendering Frames")):
        pcd = o3d.geometry.PointCloud()
        if frame_data['points'].shape[0] > 0:
            pcd.points = o3d.utility.Vector3dVector(frame_data['points'])
            pcd.colors = o3d.utility.Vector3dVector(color_mapper.get_detector_output_colors(frame_data['labels']))
        
        geometries = {'pcd': pcd}
        ego_position = all_sweeps[i].T_global_lidar[:3, 3]
        camera_params = {'center': ego_position, 'eye': ego_position + np.array([-40, -40, 20]), 'up': np.array([0, 0, 1])}
        
        renderer.render_frame(geometries, camera_params)
        
    renderer.close()

if __name__ == '__main__':
    main()