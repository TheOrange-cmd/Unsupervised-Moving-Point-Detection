# scripts/process_scene.py

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# --- Path Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Core Project Imports ---
from src.config_loader import MDetectorConfigAccessor
from src.core.m_detector.base import MDetector
from scripts.generate_labels import SceneProcessor

def deep_update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def main():
    """Main entry point for the scene processing script."""
    parser = argparse.ArgumentParser(description="Run M-Detector on a scene and save the results.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main YAML configuration file.')
    parser.add_argument('--scene-index', type=int, required=True, help='The index of the scene to process.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the output .pt file.')
    parser.add_argument('--params', type=str, help='Optional path to a YAML file with override parameters.')
    args = parser.parse_args()

    # --- Configuration and Initialization ---
    if args.params:
        with open(args.config, 'r') as f: base_config = yaml.safe_load(f)
        with open(args.params, 'r') as f: override_params = yaml.safe_load(f)
        final_config = deep_update_dict(base_config, override_params)
        config_accessor = MDetectorConfigAccessor(config_dict=final_config)
    else:
        config_accessor = MDetectorConfigAccessor(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = MDetector(config_accessor=config_accessor, device=device)
    
    nusc_params = config_accessor.get_nuscenes_params()
    nusc = NuScenes(version=nusc_params['version'], dataroot=nusc_params['dataroot'], verbose=False)
    scene_processor = SceneProcessor(nusc=nusc, config_accessor=config_accessor)
    
    scene_rec = nusc.scene[args.scene_index]
    print(f"Loading data for scene: {scene_rec['name']}...")
    all_sweeps = scene_processor.get_scene_sweeps(scene_rec['token'])

    # --- Main Processing Loop ---
    results_per_frame = []
    for sweep in tqdm(all_sweeps, desc="Processing Scene"):
        points_sensor_raw = sweep.points_sensor_frame
        points_global_raw = (sweep.T_global_lidar[:3, :3] @ points_sensor_raw[:, :3].T).T + sweep.T_global_lidar[:3, 3]
        
        detector.add_sweep(
            points_global_raw=points_global_raw.astype(np.float32),
            points_sensor_raw=points_sensor_raw.astype(np.float32),
            pose_global=sweep.T_global_lidar.astype(np.float32),
            timestamp=float(sweep.timestamp)
        )
        mdet_result = detector.process_latest_sweep()
        
        # --- START OF CHANGE ---
        # Initialize with an extra key for the original indices
        frame_result = {'points': np.array([]), 'labels': np.array([]), 'original_indices': np.array([])}
        
        if mdet_result and mdet_result.get('success'):
            processed_di = mdet_result.get('processed_di')
            if processed_di and processed_di.num_points > 0:
                frame_result['points'] = processed_di.get_original_points_global()
                frame_result['labels'] = processed_di.get_all_point_labels()
                # Save the original indices of the points that the detector processed
                frame_result['original_indices'] = processed_di.original_indices_of_filtered_points.cpu().numpy()
        # --- END OF CHANGE ---
        
        results_per_frame.append(frame_result)

    # --- Save Results ---
    # Ensure the output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(results_per_frame)} frames of results to {args.output_path}...")
    torch.save(results_per_frame, args.output_path)
    print("Processing complete.")

if __name__ == '__main__':
    main()