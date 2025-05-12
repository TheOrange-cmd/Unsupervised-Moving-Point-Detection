# src/visualization/video_generator.py
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from typing import Dict, Any, List, Callable, Optional

from ..data_utils.nuscenes_helper import get_scene_sweep_data_sequence # For original sweep data
from .frame_composers import compose_gt_vs_mdet_frame 

import logging
logger = logging.getLogger(__name__) # Module-level logger

def _get_mdet_data_for_token_from_npz(
    npz_file_data: np.lib.npyio.NpzFile, 
    target_sweep_token: str
) -> Optional[Dict[str, np.ndarray]]:
    """
    Helper to find and extract M-Detector data for a specific sweep token from a single NPZ file.
    Returns a dictionary of point arrays or None if the token is not found.
    """
    try:
        all_tokens_in_this_npz = npz_file_data['sweep_lidar_sd_tokens'].astype(str)
        # Find the index of the target token
        matching_indices = np.where(all_tokens_in_this_npz == target_sweep_token)[0]
        
        if not matching_indices.size > 0:
            tqdm.write(f"No matching indices found between detector output files!")
            # tqdm.write(f"Debug: Token {target_sweep_token} not found in this NPZ.")
            return None # Token not found in this NPZ
            

        if len(matching_indices) > 1:
            tqdm.write(f"Multiple matching indices found between detector output files! Taking the first match ...")
        idx_in_npz = matching_indices[0] # Use the first match if multiple (should not happen for tokens)

        # Check bounds for indices arrays (which are start/end pairs)
        if not (idx_in_npz + 1 < len(npz_file_data['dynamic_points_indices']) and \
                idx_in_npz + 1 < len(npz_file_data['occluded_points_indices']) and \
                idx_in_npz + 1 < len(npz_file_data['undetermined_points_indices'])):
            tqdm.write(f"Warning: Index {idx_in_npz} for token {target_sweep_token} leads to out-of-bounds "
                       f"access in point indices arrays for this NPZ. Returning no data for this NPZ.")
            return None
            
        mdet_data = {
            'dynamic': npz_file_data['all_dynamic_points'][npz_file_data['dynamic_points_indices'][idx_in_npz]:npz_file_data['dynamic_points_indices'][idx_in_npz+1]],
            'occluded_by_mdet': npz_file_data['all_occluded_points'][npz_file_data['occluded_points_indices'][idx_in_npz]:npz_file_data['occluded_points_indices'][idx_in_npz+1]],
            'undetermined_by_mdet': npz_file_data['all_undetermined_points'][npz_file_data['undetermined_points_indices'][idx_in_npz]:npz_file_data['undetermined_points_indices'][idx_in_npz+1]],
            # Optionally add other per-sweep scalar data from NPZ if needed by composer
            # 'success_flag': npz_file_data['mdet_success_flags'][idx_in_npz],
        }
        return mdet_data
    except KeyError as e:
        tqdm.write(f"Warning: Missing key '{e}' in NPZ file while looking for token {target_sweep_token}. Returning no data for this NPZ.")
        return None
    except Exception as e:
        tqdm.write(f"Error extracting data for token {target_sweep_token} from NPZ: {e}. Returning no data for this NPZ.")
        return None


def generate_video_from_npz_list(
    nusc: NuScenes,
    scene_token: str,
    list_of_mdetector_npz_data: List[np.lib.npyio.NpzFile], # List of loaded NPZ objects
    output_video_path: str,
    frame_composer_function: Callable, # e.g., compose_gt_vs_mdet_frame
    config: Dict
):
    """
    Generates a video by iterating through scene sweeps, fetching corresponding data
    from a list of NPZ files (by matching sweep tokens), and using a
    frame_composer_function to render each frame.
    """
    if not list_of_mdetector_npz_data:
        tqdm.write("Error: No M-Detector NPZ data provided.")
        return {'video_path': output_video_path, 'frames_rendered': 0}

    video_cfg = config.get('visualization', {}).get('video_generation', {})
    processing_cfg = config.get('processing', {}) # For skip/max frames used during NPZ generation
    
    fps = video_cfg.get('fps', 10)
    
    # --- Iterate through the original scene sweeps to define the video sequence ---
    # This ensures the video covers the scene structure, and we then look up NPZ data.
    # Apply skip_frames and max_frames from config if specified, to match NPZ generation if desired.
    # These apply to the iteration over the *original scene sweeps*.
    skip_initial_sweeps = processing_cfg.get('skip_frames', 0)
    max_sweeps_to_process_for_video = processing_cfg.get('max_frames', None) 

    scene_sweep_iterator = get_scene_sweep_data_sequence(nusc, scene_token)
    
    # Buffer all relevant scene sweeps first to easily apply skip/max and get a definitive list
    all_scene_sweeps_for_video: List[Dict] = []
    temp_sweep_counter = 0
    for i, sweep_data_dict in enumerate(scene_sweep_iterator):
        if i < skip_initial_sweeps:
            continue
        all_scene_sweeps_for_video.append(sweep_data_dict)
        temp_sweep_counter += 1
        if max_sweeps_to_process_for_video is not None and temp_sweep_counter >= max_sweeps_to_process_for_video:
            break
            
    if not all_scene_sweeps_for_video:
        tqdm.write("No scene sweeps available for video generation after applying skip/max frames.")
        return {'video_path': output_video_path, 'frames_rendered': 0}

    # --- Initialize Video Writer (get dimensions from the first frame) ---
    first_original_sweep_data = all_scene_sweeps_for_video[0]
    first_sweep_token = first_original_sweep_data['lidar_sd_token']
    
    mdet_data_for_first_frame_list: List[Optional[Dict[str, np.ndarray]]] = []
    for npz_idx, npz_file in enumerate(list_of_mdetector_npz_data):
        data = _get_mdet_data_for_token_from_npz(npz_file, first_sweep_token)
        if data is None: # Handle case where first sweep might be missing in some NPZs
             tqdm.write(f"Info: First sweep token {first_sweep_token} not found in NPZ run {npz_idx}. Using empty data for this run in first frame.")
             data = { # Provide empty structure for composer
                'dynamic': np.empty((0,3)), 'occluded_by_mdet': np.empty((0,3)), 
                'undetermined_by_mdet': np.empty((0,3)), 'source_name': f"MDet_Run_{npz_idx}_(NoData)"
            }
        else:
            data['source_name'] = video_cfg.get('mdetector_run_names', [f"MDet_Run_{j}" for j in range(len(list_of_mdetector_npz_data))])[npz_idx]
        mdet_data_for_first_frame_list.append(data)

    # Call composer for the first frame to get dimensions.
    # The composer needs to be robust to the structure of mdet_data_for_first_frame_list.
    # If frame_composer_function is compose_gt_vs_mdet_frame, it expects a single mdet_result dict.
    if frame_composer_function == compose_gt_vs_mdet_frame:
        if not mdet_data_for_first_frame_list or mdet_data_for_first_frame_list[0] is None:
             tqdm.write(f"Error: Cannot initialize video dimensions. M-Detector data for the first sweep token ({first_sweep_token}) is missing in the primary NPZ.")
             return {'video_path': output_video_path, 'frames_rendered': 0}
        temp_frame_bgr = frame_composer_function(
            nusc, first_original_sweep_data, mdet_data_for_first_frame_list[0], config, 0
        )
    else:
        # For a composer designed for multiple runs:
        temp_frame_bgr = frame_composer_function(
            nusc, first_original_sweep_data, mdet_data_for_first_frame_list, config, 0
        )
        
    height, width, _ = temp_frame_bgr.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # --- Iterate through selected scene sweeps and Generate Frames ---
    frames_rendered_count = 0
    scene_name = nusc.get('scene', scene_token)['name']
    desc = f"Generating video for scene {scene_name}"

    for frame_idx, original_sweep_data_dict in enumerate(tqdm(all_scene_sweeps_for_video, desc=desc)):
        current_sweep_token = original_sweep_data_dict['lidar_sd_token']
        
        # Gather M-Detector data for this sweep from ALL provided NPZ files
        current_mdet_results_from_all_runs: List[Optional[Dict[str, np.ndarray]]] = []
        
        for npz_run_idx, npz_file_data in enumerate(list_of_mdetector_npz_data):
            mdet_data_for_this_run = _get_mdet_data_for_token_from_npz(npz_file_data, current_sweep_token)
            
            if mdet_data_for_this_run is None:
                # If token not found in this NPZ, append a placeholder or empty data structure
                # The composer function must handle this.
                tqdm.write(f"Info: Token {current_sweep_token} (frame {frame_idx}) not found in NPZ run {npz_run_idx}. Composer will receive empty data for this run.")
                mdet_data_for_this_run = {
                    'dynamic': np.empty((0,3)), 'occluded_by_mdet': np.empty((0,3)), 
                    'undetermined_by_mdet': np.empty((0,3)), 
                    'source_name': video_cfg.get('mdetector_run_names', [f"MDet_Run_{j}" for j in range(len(list_of_mdetector_npz_data))])[npz_run_idx] + " (NoData)"
                }
            else:
                 mdet_data_for_this_run['source_name'] = video_cfg.get('mdetector_run_names', [f"MDet_Run_{j}" for j in range(len(list_of_mdetector_npz_data))])[npz_run_idx]

            current_mdet_results_from_all_runs.append(mdet_data_for_this_run)
            
        # Call the composer function.
        if frame_composer_function == compose_gt_vs_mdet_frame:
            # This composer expects a single MDet result. We use the first one.
            # If you want GT vs. a specific run, ensure list_of_mdetector_npz_data[0] is that run.
            if not current_mdet_results_from_all_runs or current_mdet_results_from_all_runs[0] is None:
                 tqdm.write(f"Warning: M-Detector data for sweep {current_sweep_token} (frame {frame_idx}) is missing in the primary NPZ. Skipping video frame.")
                 continue
            bgr_frame = frame_composer_function(
                nusc, original_sweep_data_dict, current_mdet_results_from_all_runs[0], config, frames_rendered_count
            )
        else:
            # For a composer designed for multiple runs:
            bgr_frame = frame_composer_function(
                nusc, original_sweep_data_dict, current_mdet_results_from_all_runs, config, frames_rendered_count
            )
            
        video_writer.write(bgr_frame)
        frames_rendered_count += 1
        
    video_writer.release()
    tqdm.write(f"Video generation complete. Rendered {frames_rendered_count} frames.")
    return {
        'video_path': output_video_path,
        'frames_rendered': frames_rendered_count,
    }
