# scripts/generate_video.py
import yaml
import os
import numpy as np
from nuscenes.nuscenes import NuScenes
import sys
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import the main generator and a specific composer
from src.visualization.video_generator import generate_video_from_npz_list
from src.visualization.frame_composers import compose_gt_vs_mdet_frame 

def main():
    config_path = 'config/m_detector_config.yaml' # Ensure this path is correct
    if not os.path.isabs(config_path) and PROJECT_ROOT:
        config_path = os.path.join(PROJECT_ROOT, config_path)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    nusc = NuScenes(
        version=config['nuscenes']['version'],
        dataroot=config['nuscenes']['dataroot'],
        verbose=config.get('nuscenes',{}).get('verbose', False) # Less verbose for script
    )
    
    viz_cfg = config.get('visualization', {})
    video_gen_cfg = viz_cfg.get('video_generation', {})

    scene_index_to_process = video_gen_cfg.get('scene_index_for_video', 0)
    scene_rec = nusc.scene[scene_index_to_process]
    scene_name = scene_rec['name']
    scene_token = scene_rec['token']

    # --- NPZ File Path(s) ---
    # For now, expects one NPZ file. Future: could be a list.
    npz_dir = video_gen_cfg.get('mdetector_results_directory', 'output/mdetector_results')
    if not os.path.isabs(npz_dir) and PROJECT_ROOT:
        npz_dir = os.path.join(PROJECT_ROOT, npz_dir)

    npz_filename_template = video_gen_cfg.get('npz_filename_template', "mdetector_results_scene_{scene_name}.npz")
    npz_filename = npz_filename_template.format(scene_name=scene_name)
    mdetector_results_npz_filepath = os.path.join(npz_dir, npz_filename)

    if not os.path.exists(mdetector_results_npz_filepath):
        tqdm.write(f"Error: M-Detector results NPZ file not found: {mdetector_results_npz_filepath}")
        return

    tqdm.write(f"Loading M-Detector results from: {mdetector_results_npz_filepath}")
    try:
        mdetector_npz_data = np.load(mdetector_results_npz_filepath, allow_pickle=True)
    except Exception as e:
        tqdm.write(f"Error loading NPZ file: {e}")
        return
    
    list_of_npz_for_video = [mdetector_npz_data] # For now, one NPZ

    # --- Output Video Path ---
    output_video_filename_template = video_gen_cfg.get('output_video_filename_template', 
                                           "scene_{scene_name}_comparison_video.mp4")
    output_video_filename = output_video_filename_template.format(scene_name=scene_name)
    
    output_dir = video_gen_cfg.get('output_directory', 'output/videos')
    if not os.path.isabs(output_dir) and PROJECT_ROOT:
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_video_filename)
    
    tqdm.write(f"Starting video generation for scene: {scene_name}")
    tqdm.write(f"Output will be saved to: {output_video_path}")

    # --- Choose Frame Composer ---
    # For now, we hardcode to GT vs MDet. Future: could be config-driven.
    selected_frame_composer = compose_gt_vs_mdet_frame

    video_results = generate_video_from_npz_list(
        nusc=nusc,
        scene_token=scene_token,
        list_of_mdetector_npz_data=list_of_npz_for_video,
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