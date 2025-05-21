# scripts/generate_video.py
import yaml
import os
from nuscenes.nuscenes import NuScenes
import sys
from tqdm import tqdm
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config_loader import MDetectorConfigAccessor
from src.visualization.video_generator import generate_video_from_hdf5_list
from src.visualization.frame_composers import compose_gt_vs_mdet_frame
# --- NEW IMPORT ---
from src.utils.validation_utils import load_config_from_hdf5 # To load config from MDet HDF5

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    config_file_path_relative = 'config/m_detector_config.yaml'
    config_file_path_absolute = config_file_path_relative
    if not os.path.isabs(config_file_path_relative) and PROJECT_ROOT:
        config_file_path_absolute = os.path.join(PROJECT_ROOT, config_file_path_relative)

    tqdm.write(f"Loading current M-Detector config from: {config_file_path_absolute}")
    try:
        current_config_accessor = MDetectorConfigAccessor(config_file_path_absolute)
    except Exception as e:
        tqdm.write(f"FATAL: Error initializing current MDetectorConfigAccessor: {e}")
        return

    nuscenes_params = current_config_accessor.get_nuscenes_params()
    video_gen_params = current_config_accessor.get_video_generation_params()
    mdet_output_paths = current_config_accessor.get_mdetector_output_paths()

    tqdm.write("Initializing NuScenes...")
    try:
        nusc = NuScenes(
            version=nuscenes_params.get('version'),
            dataroot=nuscenes_params.get('dataroot'),
            verbose=nuscenes_params.get('verbose_load', False)
        )
    except Exception as e:
        tqdm.write(f"FATAL: Error initializing NuScenes: {e}")
        return

    gt_label_path_config = nuscenes_params.get('label_path', '')
    if not gt_label_path_config or not os.path.isdir(gt_label_path_config):
        tqdm.write(f"Warning: nuscenes_params 'label_path' ('{gt_label_path_config}') is not valid.")

    scene_index_to_process = video_gen_params.get('default_scene_index', 0)
    if not isinstance(scene_index_to_process, int) or \
       scene_index_to_process < 0 or scene_index_to_process >= len(nusc.scene):
        tqdm.write(f"Error: default_scene_index ({scene_index_to_process}) is invalid.")
        return
        
    scene_rec = nusc.scene[scene_index_to_process]
    scene_name = scene_rec['name']
    scene_token = scene_rec['token']

    mdet_results_dir = mdet_output_paths.get('save_path')
    if not mdet_results_dir:
        tqdm.write(f"Error: 'mdetector_output_paths.save_path' not found in config.")
        return
    if not os.path.isabs(mdet_results_dir) and PROJECT_ROOT:
        mdet_results_dir = os.path.join(PROJECT_ROOT, mdet_results_dir)
    
    mdet_results_filename = f"mdet_results_{scene_name}.h5" 
    mdetector_results_hdf5_filepath = os.path.join(mdet_results_dir, mdet_results_filename)

    if not os.path.exists(mdetector_results_hdf5_filepath):
        tqdm.write(f"Error: M-Detector results HDF5 file not found: {mdetector_results_hdf5_filepath}")
        return

    list_of_hdf5_paths_for_video = [mdetector_results_hdf5_filepath]
    tqdm.write(f"Using M-Detector results from: {mdetector_results_hdf5_filepath}")

    # --- Load the config that WAS USED for the M-Detector run being visualized ---
    run_specific_mdet_config_dict = load_config_from_hdf5(mdetector_results_hdf5_filepath)
    if run_specific_mdet_config_dict:
        tqdm.write(f"Successfully loaded run-specific M-Detector config from {mdetector_results_hdf5_filepath} for filtering consistency.")
    else:
        tqdm.write(f"Warning: Could not load run-specific config from {mdetector_results_hdf5_filepath}. "
                   f"Video generation will use filtering parameters from the current script's config ('{config_file_path_absolute}'), "
                   f"which might lead to point count mismatches if it differs from the M-Detector run config.")
    # --- End loading run-specific config ---

    output_video_filename_template = video_gen_params.get('output_filename_template', "scene_{scene_name}_comparison_video.mp4")
    output_video_filename = output_video_filename_template.format(scene_name=scene_name)
    output_dir = video_gen_params.get('output_directory', 'output/videos')
    if not os.path.isabs(output_dir) and PROJECT_ROOT:
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_video_filename)

    tqdm.write(f"Starting video generation for scene: {scene_name} ({scene_token})")
    tqdm.write(f"Output will be saved to: {output_video_path}")

    selected_frame_composer = compose_gt_vs_mdet_frame

    video_results = generate_video_from_hdf5_list( 
        nusc=nusc,
        scene_token=scene_token,
        list_of_mdetector_hdf5_paths=list_of_hdf5_paths_for_video, 
        output_video_path=output_video_path,
        frame_composer_function=selected_frame_composer,
        config_accessor=current_config_accessor, # For general video/style params
        # --- PASS THE LOADED RUN-SPECIFIC CONFIG ---
        run_specific_mdet_config=run_specific_mdet_config_dict 
    )

    if video_results and video_results.get('frames_rendered', 0) > 0:
        tqdm.write(f"Video generation successful: {video_results.get('video_path')}")
    else:
        tqdm.write(f"Video generation failed or rendered 0 frames.")

if __name__ == '__main__':
    main()