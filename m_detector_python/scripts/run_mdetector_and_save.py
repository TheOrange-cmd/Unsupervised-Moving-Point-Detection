# scripts/run_mdetector_and_save.py
import yaml
import os
import json
import numpy as np # For NumpyEncoder if needed, though tolist() should handle it
from nuscenes.nuscenes import NuScenes
import sys
from tqdm import tqdm

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    # Use tqdm.write instead of print to avoid interfering with progress bar
    tqdm.write(f"Added project root to sys.path: {PROJECT_ROOT}")

from src.core.m_detector.base import MDetector
from src.data_utils.nuscenes_helper import NuScenesProcessor

# Optional: A more robust JSON encoder for numpy types if tolist() isn't used everywhere
class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpySafeEncoder, self).default(obj)

def main():
    config_path = 'config/m_detector_config.yaml'
    if not os.path.isabs(config_path) and PROJECT_ROOT:
        config_path = os.path.join(PROJECT_ROOT, config_path)
        
    tqdm.write(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mdet_output_cfg = config.get('mdetector_output', {})
    output_base_dir = mdet_output_cfg.get('save_path', 'output/mdetector_results')
    if not os.path.isabs(output_base_dir) and PROJECT_ROOT:
        output_base_dir = os.path.join(PROJECT_ROOT, output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    tqdm.write("Initializing NuScenes...")
    nusc = NuScenes(
        version=config['nuscenes']['version'],
        dataroot=config['nuscenes']['dataroot'],
        verbose=config.get('nuscenes',{}).get('verbose_load', False) # Usually false for batch runs
    )
    
    tqdm.write("Initializing MDetector...")
    detector = MDetector(config)
    
    processor = NuScenesProcessor(nusc, config)

    scene_indices_to_process = mdet_output_cfg.get('scene_indices_to_run', [0]) # e.g., [0, 1, 5] or 'all'
    if scene_indices_to_process == 'all':
        scene_indices_to_process = list(range(len(nusc.scene)))

    for scene_idx in scene_indices_to_process:
        if scene_idx < 0 or scene_idx >= len(nusc.scene):
            tqdm.write(f"Scene index {scene_idx} is out of bounds. Skipping.")
            continue
        
        # Reset detector state for each new scene (if MDetector has state)
        # detector.reset_state() # Implement this if MDetector accumulates state across scenes

        scene_name = nusc.scene[scene_idx]['name']
        
        dict_of_arrays_for_scene = processor.process_scene(
            scene_index=scene_idx,
            detector=detector,
            with_progress=True
        )

        config_json_str = json.dumps(config, sort_keys=True, indent=4)
        dict_of_arrays_for_scene['_config_json_str'] = np.array(config_json_str)
        
        if dict_of_arrays_for_scene is None:
            tqdm.write(f"No data returned from M-Detector processing for scene '{scene_name}'. Skipping save.")
            continue

        output_filename = f"mdet_results_{scene_name}.npz" # Save as .npz
        output_filepath = os.path.join(output_base_dir, output_filename)
        
        tqdm.write(f"Saving M-Detector results for scene '{scene_name}' to {output_filepath}...")
        try:
            # Use np.savez_compressed for smaller files, or np.savez for faster save if space is no issue
            # np.savez_compressed(output_filepath, **dict_of_arrays_for_scene)
            np.savez(output_filepath, **dict_of_arrays_for_scene)
            tqdm.write(f"Successfully saved results for {len(dict_of_arrays_for_scene['sweep_lidar_sd_tokens'])} frames.")
        except Exception as e:
            tqdm.write(f"Error saving M-Detector results to NPZ for scene '{scene_name}': {e}")
            # Consider printing traceback for detailed error: import traceback; traceback.print_exc()

    tqdm.write("M-Detector processing and NPZ saving complete.")

if __name__ == '__main__':
    main()