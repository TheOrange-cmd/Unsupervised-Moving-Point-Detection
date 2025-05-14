# scripts/generate_video.py
import yaml
import os
# import numpy as np # Not directly needed here for loading NPZ anymore
from nuscenes.nuscenes import NuScenes
import sys
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the HDF5 version of the generator
from src.visualization.video_generator import generate_video_from_hdf5_list 
from src.visualization.frame_composers import compose_gt_vs_mdet_frame

def main():
    config_path = 'config/m_detector_config.yaml'
    if not os.path.isabs(config_path) and PROJECT_ROOT:
        config_path = os.path.join(PROJECT_ROOT, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    nusc = NuScenes(
        version=config['nuscenes']['version'],
        dataroot=config['nuscenes']['dataroot'],
        verbose=config.get('nuscenes',{}).get('verbose', False)
    )

    # viz_cfg = config.get('visualization', {})
    video_gen_cfg = config.get('video_generation', {})

    # Ensure the GT label path in the config points to the HDF5 GT labels directory
    # This is used by compose_gt_vs_mdet_frame -> get_gt_dynamic_points_for_sweep
    gt_label_path_config = config.get('nuscenes', {}).get('label_path', '')
    if not gt_label_path_config or not os.path.isdir(gt_label_path_config):
        tqdm.write(f"Warning: `config['nuscenes']['label_path']` ('{gt_label_path_config}') "
                   f"is not set or not a valid directory. "
                   f"This path should point to your HDF5 GT label files for video generation.")
    # Example check for one GT file (optional, but good for user feedback)
    # scene_for_gt_check = nusc.scene[0]['name'] # Check for first scene as an example
    # expected_gt_hdf5 = os.path.join(gt_label_path_config, f"gt_point_labels_{scene_for_gt_check}.h5")
    # if not os.path.exists(expected_gt_hdf5) and gt_label_path_config:
    #     tqdm.write(f"  Potentially missing GT HDF5 file: {expected_gt_hdf5}")


    scene_index_to_process = video_gen_cfg.get('default_scene_index', 0)
    if scene_index_to_process < 0 or scene_index_to_process >= len(nusc.scene):
        tqdm.write(f"Error: default_scene_index ({scene_index_to_process}) is out of bounds (0-{len(nusc.scene)-1}).")
        return
    scene_rec = nusc.scene[scene_index_to_process]
    scene_name = scene_rec['name']
    scene_token = scene_rec['token']

    # --- HDF5 File Path(s) for M-Detector Results ---
    # Update config keys if necessary (e.g., from npz_dir to hdf5_dir)
    mdet_results_dir = config.get('mdetector_output').get('save_path')
    if not os.path.isabs(mdet_results_dir) and PROJECT_ROOT:
        mdet_results_dir = os.path.join(PROJECT_ROOT, mdet_results_dir)
    hdf5_filename_template = f"mdet_results_{scene_name}.h5"

    mdet_results_filename = hdf5_filename_template.format(scene_name=scene_name)
    mdetector_results_hdf5_filepath = os.path.join(mdet_results_dir, mdet_results_filename)

    if not os.path.exists(mdetector_results_hdf5_filepath):
        tqdm.write(f"Error: M-Detector results HDF5 file not found: {mdetector_results_hdf5_filepath}")
        tqdm.write(f"Please ensure 'mdetector_results_directory' and 'hdf5_filename_template' (or 'npz_filename_template' adapted to .h5) in your config are correct.")
        return

    # `generate_video_from_hdf5_list` expects a list of file paths
    list_of_hdf5_paths_for_video = [mdetector_results_hdf5_filepath]
    tqdm.write(f"Using M-Detector results from: {mdetector_results_hdf5_filepath}")

    # --- Output Video Path ---
    output_video_filename_template = "scene_{scene_name}_comparison_video.mp4"
    output_video_filename = output_video_filename_template.format(scene_name=scene_name)

    output_dir = video_gen_cfg.get('output_directory', 'output/videos')
    if not os.path.isabs(output_dir) and PROJECT_ROOT:
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_video_filename)

    tqdm.write(f"Starting video generation for scene: {scene_name}")
    tqdm.write(f"Output will be saved to: {output_video_path}")

    selected_frame_composer = compose_gt_vs_mdet_frame

    video_results = generate_video_from_hdf5_list( 
        nusc=nusc,
        scene_token=scene_token,
        list_of_mdetector_hdf5_paths=list_of_hdf5_paths_for_video, 
        output_video_path=output_video_path,
        frame_composer_function=selected_frame_composer,
        config=config
    )

    if video_results and video_results.get('frames_rendered', 0) > 0:
        tqdm.write(f"Video generation successful: {video_results.get('video_path')}")
    else:
        tqdm.write(f"Video generation failed or rendered 0 frames.")

if __name__ == '__main__':
    main()