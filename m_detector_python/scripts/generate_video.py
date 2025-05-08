# scripts/generate_video.py

import yaml
import os
from nuscenes.nuscenes import NuScenes
import sys
import os 
from tqdm import tqdm

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    # Use tqdm.write instead of print to avoid interfering with progress bar
    tqdm.write(f"Added project root to sys.path: {PROJECT_ROOT}")

from src.core.m_detector import MDetector
from src.visualization.video_helpers import generate_video

def main():
    # Load config
    with open('config/m_detector_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize NuScenes
    nusc = NuScenes(
        version=config['nuscenes']['version'],
        dataroot=config['nuscenes']['dataroot'],
        verbose=True
    )
    
    # Initialize MDetector
    detector = MDetector(config)
    
    # Generate video for a scene
    scene_index = 1
    output_path = 'output/scene_{:03d}.mp4'.format(scene_index)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = generate_video(
        nusc=nusc,
        scene_index=scene_index,
        detector=detector,
        output_path=output_path,
        config=config
    )
    
    print(f"Video generated: {output_path}")
    print(f"Processed {results['frames_processed']} frames")

if __name__ == '__main__':
    main()