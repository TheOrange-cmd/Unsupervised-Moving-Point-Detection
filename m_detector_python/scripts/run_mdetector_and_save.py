# scripts/run_mdetector_and_save.py
import yaml
import os
import json
import numpy as np
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
    tqdm.write(f"Added project root to sys.path: {PROJECT_ROOT}")

from src.core.m_detector.base import MDetector
from src.data_utils.nuscenes_helper import NuScenesProcessor
# --- NEW IMPORT ---
from src.config_loader import MDetectorConfigAccessor

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

    logging.basicConfig(level=logging.INFO) # General level
    logging.getLogger('src.core.m_detector.base').setLevel(logging.INFO)
    logging.getLogger('src.core.m_detector.processing').setLevel(logging.INFO)
    logging.getLogger('src.core.m_detector.map_consistency').setLevel(logging.INFO) # MCC logs
    logging.getLogger('src.core.m_detector.interpolation_utils').setLevel(logging.INFO) # Interpolation logs

    config_file_path_relative = 'config/m_detector_config.yaml' # Relative to project root
    config_file_path_absolute = config_file_path_relative
    if not os.path.isabs(config_file_path_relative) and PROJECT_ROOT:
        config_file_path_absolute = os.path.join(PROJECT_ROOT, config_file_path_relative)
        
    tqdm.write(f"Loading config from: {config_file_path_absolute}")
    
    # --- CHANGE 1: Initialize MDetectorConfigAccessor ---
    try:
        config_accessor = MDetectorConfigAccessor(config_file_path_absolute)
    except Exception as e:
        tqdm.write(f"FATAL: Error initializing MDetectorConfigAccessor: {e}")
        return

    # --- CHANGE 2: Get specific config sections using the accessor ---
    mdet_output_cfg = config_accessor.get_mdetector_output_paths()
    nuscenes_cfg = config_accessor.get_nuscenes_params()

    output_base_dir = mdet_output_cfg.get('save_path', 'output/mdetector_results')
    if not os.path.isabs(output_base_dir) and PROJECT_ROOT:
        output_base_dir = os.path.join(PROJECT_ROOT, output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    tqdm.write("Initializing NuScenes...")
    try:
        nusc = NuScenes(
            version=nuscenes_cfg.get('version'), # Use accessor
            dataroot=nuscenes_cfg.get('dataroot'), # Use accessor
            verbose=nuscenes_cfg.get('verbose_load', False) # Use accessor
        )
    except Exception as e:
        tqdm.write(f"FATAL: Error initializing NuScenes: {e}")
        tqdm.write(f"  Please check 'version' ({nuscenes_cfg.get('version')}) and 'dataroot' ({nuscenes_cfg.get('dataroot')}) in your config.")
        return
    
    tqdm.write("Initializing MDetector...")
    try:
        detector = MDetector(config_accessor=config_accessor)
    except Exception as e:
        tqdm.write(f"FATAL: Error initializing MDetector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    tqdm.write("Initializing NuScenesProcessor...")
    try:
        processor = NuScenesProcessor(nusc, config_accessor=config_accessor)
    except Exception as e:
        tqdm.write(f"FATAL: Error initializing NuScenesProcessor: {e}")
        import traceback
        traceback.print_exc()
        return

    scene_indices_to_process_config = mdet_output_cfg.get('scene_indices_to_run', [0])
    scene_indices_to_process_list = [] 
    if isinstance(scene_indices_to_process_config, str) and scene_indices_to_process_config.lower() == 'all':
        scene_indices_to_process_list = list(range(len(nusc.scene)))
    elif isinstance(scene_indices_to_process_config, list):
        scene_indices_to_process_list = scene_indices_to_process_config
    else:
        tqdm.write(f"Warning: 'scene_indices_to_run' format is invalid ({scene_indices_to_process_config}). Defaulting to scene 0.")
        scene_indices_to_process_list = [0]


    for scene_idx in scene_indices_to_process_list:
        if not isinstance(scene_idx, int) or scene_idx < 0 or scene_idx >= len(nusc.scene):
            tqdm.write(f"Scene index {scene_idx} is invalid or out of bounds. Skipping.")
            continue
        
        scene_record = nusc.scene[scene_idx] 
        scene_name = scene_record['name']
        
        tqdm.write(f"\nProcessing Scene {scene_idx}: {scene_name} ({scene_record['token']})")
        
        dict_of_arrays_for_scene = processor.process_scene(
            scene_index=scene_idx,
            detector=detector,
            with_progress=True
        )
        
        if dict_of_arrays_for_scene is None:
            tqdm.write(f"No data returned from M-Detector processing for scene '{scene_name}'. Skipping save.")
            continue

        # Include the config we used for this run in the saved results file 
        try:
            with open(config_file_path_absolute, 'r') as f_raw_cfg:
                raw_config_for_saving = yaml.safe_load(f_raw_cfg)
            config_json_str = json.dumps(raw_config_for_saving, sort_keys=True, indent=4, cls=NumpySafeEncoder)
            dict_of_arrays_for_scene['_config_json_str'] = np.array(config_json_str) # Store as 0-d array
        except Exception as e_cfg_save:
            tqdm.write(f"Warning: Could not read/process raw config for saving: {e_cfg_save}")
            dict_of_arrays_for_scene['_config_json_str'] = np.array("Error saving config string")


        output_filename = f"mdet_results_{scene_name}.h5"
        output_filepath = os.path.join(output_base_dir, output_filename)
        
        tqdm.write(f"Saving M-Detector results for scene '{scene_name}' to {output_filepath} (HDF5)...")
        
        try:
            with h5py.File(output_filepath, 'w') as hf:
                for key, array_data in dict_of_arrays_for_scene.items():
                    if isinstance(array_data, np.ndarray):
                        # Special handling for 0-dim array (like the config string)
                        if array_data.ndim == 0:
                            hf.create_dataset(key, data=array_data.item())
                        else:
                            hf.create_dataset(key, data=array_data)
                    elif isinstance(array_data, (str, bytes, int, float, bool)): 
                        hf.create_dataset(key, data=array_data)
                    else:
                        try:
                            converted_array = np.array(array_data)
                            hf.create_dataset(key, data=converted_array)
                            tqdm.write(f"  Info: Converted data for key '{key}' to NumPy array for HDF5.")
                        except Exception as e_conv:
                            tqdm.write(f"  Warning: Could not save key '{key}' (type: {type(array_data)}). Error: {e_conv}")
            
            num_frames_saved = 0
            if 'sweep_lidar_sd_tokens' in dict_of_arrays_for_scene and \
               hasattr(dict_of_arrays_for_scene['sweep_lidar_sd_tokens'], '__len__'):
                num_frames_saved = len(dict_of_arrays_for_scene['sweep_lidar_sd_tokens'])
            tqdm.write(f"Successfully saved HDF5 results for {num_frames_saved} frames for scene '{scene_name}'.")

        except Exception as e:
            tqdm.write(f"Error saving M-Detector results to HDF5 for scene '{scene_name}': {e}")
            import traceback
            traceback.print_exc()

    tqdm.write("\nMain M-Detector processing and HDF5 saving complete for all selected scenes.")

if __name__ == '__main__':
    main()