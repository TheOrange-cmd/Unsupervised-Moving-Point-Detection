# scripts/run_mdetector_and_save.py
import yaml
import os
import json
import numpy as np # For NumpyEncoder if needed, though tolist() should handle it
from nuscenes.nuscenes import NuScenes
import sys
from tqdm import tqdm
import logging
import h5py

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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

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
        verbose=config.get('nuscenes',{}).get('verbose_load', False)
    )
    
    tqdm.write("Initializing MDetector...")
    detector = MDetector(config)
    
    # NuScenesProcessor remains the same as it returns a dict of NumPy arrays
    processor = NuScenesProcessor(nusc, config)

    scene_indices_to_process = mdet_output_cfg.get('scene_indices_to_run', [0])
    if scene_indices_to_process == 'all':
        scene_indices_to_process = list(range(len(nusc.scene)))

    for scene_idx in scene_indices_to_process:
        if scene_idx < 0 or scene_idx >= len(nusc.scene):
            tqdm.write(f"Scene index {scene_idx} is out of bounds. Skipping.")
            continue
        
        scene_name = nusc.scene[scene_idx]['name']
        
        # dict_of_arrays_for_scene is the output from NuScenesProcessor.process_scene
        # It should contain NumPy arrays as values.
        dict_of_arrays_for_scene = processor.process_scene(
            scene_index=scene_idx,
            detector=detector,
            with_progress=True
        )
        
        if dict_of_arrays_for_scene is None: # Check if processor returned None
            tqdm.write(f"No data returned from M-Detector processing for scene '{scene_name}'. Skipping save.")
            continue

        # Ensure _config_json_str is present (same logic as before)
        if '_config_json_str' not in dict_of_arrays_for_scene:
            tqdm.write("Warning: '_config_json_str' not found in data from NuScenesProcessor. Adding it now.")
            config_json_str = json.dumps(config, sort_keys=True, indent=4, cls=NumpySafeEncoder)
            # The processor expects dict_of_arrays_for_scene to contain np.array for config
            # However, for HDF5 saving, we can handle the string directly.
            # For consistency with how it might have been structured for NPZ (as 0-d array),
            # we can keep it as such, or simplify if HDF5 saving handles Python strings better.
            dict_of_arrays_for_scene['_config_json_str'] = np.array(config_json_str)


        # --- CHANGE 1: Output filename extension ---
        output_filename = f"mdet_results_{scene_name}.h5" # Save as .h5
        output_filepath = os.path.join(output_base_dir, output_filename)
        
        tqdm.write(f"Saving M-Detector results for scene '{scene_name}' to {output_filepath} (HDF5)...")
        
        # --- CHANGE 2: Saving logic using h5py ---
        try:
            with h5py.File(output_filepath, 'w') as hf:
                for key, array_data in dict_of_arrays_for_scene.items():
                    # Ensure data is in a format h5py can handle directly
                    # (NumPy arrays, Python strings/bytes, scalars)
                    
                    if isinstance(array_data, np.ndarray):
                        if array_data.ndim == 0:
                            # For 0-d arrays (like _config_json_str), save the item
                            item_to_save = array_data.item()
                            hf.create_dataset(key, data=item_to_save) # No compression
                        else:
                            hf.create_dataset(key, data=array_data) # No compression
                    elif isinstance(array_data, (str, bytes, int, float)):
                        # If some items in dict_of_arrays_for_scene are already Python scalars/strings
                        hf.create_dataset(key, data=array_data) # No compression
                    else:
                        # Attempt to convert to NumPy array if it's some other list-like structure
                        # This might not be necessary if NuScenesProcessor always returns np.ndarrays
                        try:
                            converted_array = np.array(array_data)
                            hf.create_dataset(key, data=converted_array) # No compression
                            tqdm.write(f"  Info: Converted data for key '{key}' to NumPy array before saving to HDF5.")
                        except Exception as e_conv:
                            tqdm.write(f"  Warning: Could not convert data for key '{key}' (type: {type(array_data)}) to NumPy array or save directly. Skipping this key. Error: {e_conv}")
                            continue
            
            # Get a count of sweeps if 'sweep_lidar_sd_tokens' exists for the log message
            num_frames_saved = 0
            if 'sweep_lidar_sd_tokens' in dict_of_arrays_for_scene and \
               hasattr(dict_of_arrays_for_scene['sweep_lidar_sd_tokens'], '__len__'):
                num_frames_saved = len(dict_of_arrays_for_scene['sweep_lidar_sd_tokens'])
            tqdm.write(f"Successfully saved HDF5 results for {num_frames_saved} frames.")

        except Exception as e:
            tqdm.write(f"Error saving M-Detector results to HDF5 for scene '{scene_name}': {e}")
            import traceback # For debugging
            traceback.print_exc() # For debugging

    tqdm.write("M-Detector processing and HDF5 saving complete.")

if __name__ == '__main__':
    main()