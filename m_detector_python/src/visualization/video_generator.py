# src/visualization/video_generator.py
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from typing import Dict, Any, List, Callable, Optional
import h5py

from ..data_utils.nuscenes_helper import get_scene_sweep_data_sequence # For original sweep data
from .frame_composers import compose_gt_vs_mdet_frame 
from ..core.constants import OcclusionResult

import logging
logger = logging.getLogger(__name__) # Module-level logger


def _get_mdet_data_for_token_from_hdf5(
    h5_file_handle: h5py.File,
    target_sweep_token: str
) -> Optional[Dict[str, np.ndarray]]:
    # print(f"    DEBUG (_get_mdet_data): Attempting to get MDet data for token: {target_sweep_token}") # DEBUG
    try:
        all_tokens_bytes = h5_file_handle['sweep_lidar_sd_tokens'][:]
        target_token_bytes = target_sweep_token.encode('utf-8') if isinstance(target_sweep_token, str) else target_sweep_token
        all_tokens_str = [t.decode('utf-8', 'ignore') for t in all_tokens_bytes] # Added ignore

        matching_indices = np.where(np.array(all_tokens_str) == target_sweep_token)[0]

        if not matching_indices.size > 0:
            # print(f"    DEBUG (_get_mdet_data): Token {target_sweep_token} NOT FOUND in HDF5 'sweep_lidar_sd_tokens'.") # DEBUG
            return None
        idx_in_h5 = matching_indices[0]
        # print(f"    DEBUG (_get_mdet_data): Token {target_sweep_token} found at HDF5 index {idx_in_h5}.") # DEBUG


        indices_array = h5_file_handle['points_predictions_indices'][:]
        start_idx = indices_array[idx_in_h5]
        if idx_in_h5 + 1 >= len(indices_array):
            tqdm.write(f"    DEBUG (_get_mdet_data): Index {idx_in_h5} for token {target_sweep_token} leads to out-of-bounds "
                       f"access in 'points_predictions_indices' (len: {len(indices_array)}).") # DEBUG
            return None
        end_idx = indices_array[idx_in_h5 + 1]
        # print(f"    DEBUG (_get_mdet_data): For token {target_sweep_token}, MDet point indices: {start_idx} to {end_idx}") # DEBUG


        all_preds_for_sweep = h5_file_handle['all_points_predictions'][start_idx:end_idx]
        # print(f"    DEBUG (_get_mdet_data): Loaded 'all_points_predictions' for token {target_sweep_token}, shape: {all_preds_for_sweep.shape}") # DEBUG


        if all_preds_for_sweep.shape[0] == 0:
            # print(f"    DEBUG (_get_mdet_data): No MDet points for token {target_sweep_token} (empty slice).") # DEBUG
            return {
                'dynamic': np.empty((0, 3)),
                'occluded_by_mdet': np.empty((0, 3)),
                'undetermined_by_mdet': np.empty((0, 3))
            }

        dynamic_label_val = OcclusionResult.OCCLUDING_IMAGE.value
        occluded_label_val = OcclusionResult.OCCLUDED_BY_IMAGE.value
        undetermined_label_val = OcclusionResult.UNDETERMINED.value

        dynamic_mask = (all_preds_for_sweep['mdet_label'] == dynamic_label_val)
        occluded_mask = (all_preds_for_sweep['mdet_label'] == occluded_label_val)
        undetermined_mask = (all_preds_for_sweep['mdet_label'] == undetermined_label_val)

        mdet_data = {
            'dynamic': np.stack((all_preds_for_sweep['x'][dynamic_mask],
                                 all_preds_for_sweep['y'][dynamic_mask],
                                 all_preds_for_sweep['z'][dynamic_mask]), axis=-1),
            'occluded_by_mdet': np.stack((all_preds_for_sweep['x'][occluded_mask],
                                          all_preds_for_sweep['y'][occluded_mask],
                                          all_preds_for_sweep['z'][occluded_mask]), axis=-1),
            'undetermined_by_mdet': np.stack((all_preds_for_sweep['x'][undetermined_mask],
                                              all_preds_for_sweep['y'][undetermined_mask],
                                              all_preds_for_sweep['z'][undetermined_mask]), axis=-1),
        }
        # print(f"    DEBUG (_get_mdet_data): Token {target_sweep_token} - Dynamic pts: {mdet_data['dynamic'].shape[0]}, Occluded pts: {mdet_data['occluded_by_mdet'].shape[0]}") # DEBUG
        return mdet_data
    except KeyError as e:
        tqdm.write(f"    DEBUG (_get_mdet_data): KeyError '{e}' in HDF5 for token {target_sweep_token}.") # DEBUG
        return None
    except Exception as e_gen:
        tqdm.write(f"    DEBUG (_get_mdet_data): Error extracting data for token {target_sweep_token} from HDF5: {e_gen}.") # DEBUG
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        return None


def generate_video_from_hdf5_list(
    nusc: NuScenes,
    scene_token: str,
    list_of_mdetector_hdf5_paths: List[str],
    output_video_path: str,
    frame_composer_function: Callable,
    config: Dict
):
    if not list_of_mdetector_hdf5_paths:
        tqdm.write("DEBUG (generate_video): No M-Detector HDF5 paths provided.") # DEBUG
        return {'video_path': output_video_path, 'frames_rendered': 0}

    video_cfg = config.get('visualization', {}).get('video_generation', {})
    processing_cfg = config.get('processing', {})
    fps = video_cfg.get('fps', 10)

    mdet_h5_file_handles: List[Optional[h5py.File]] = []
    for hdf5_path in list_of_mdetector_hdf5_paths:
        try:
            handle = h5py.File(hdf5_path, 'r')
            mdet_h5_file_handles.append(handle)
            # print(f"  DEBUG (generate_video): Successfully opened MDet HDF5: {hdf5_path}") # DEBUG
        except Exception as e_open:
            tqdm.write(f"  DEBUG (generate_video): Could not open MDet HDF5 {hdf5_path}: {e_open}. Skipping.") # DEBUG
            mdet_h5_file_handles.append(None)

    if not any(mdet_h5_file_handles):
        tqdm.write("DEBUG (generate_video): None of the MDet HDF5 files could be opened.") # DEBUG
        return {'video_path': output_video_path, 'frames_rendered': 0}

    skip_initial_sweeps = processing_cfg.get('skip_frames', 0)
    max_sweeps_to_process_for_video = processing_cfg.get('max_frames', None)
    # print(f"  DEBUG (generate_video): Skip initial: {skip_initial_sweeps}, Max sweeps for video: {max_sweeps_to_process_for_video}") # DEBUG

    scene_sweep_iterator = get_scene_sweep_data_sequence(nusc, scene_token)
    all_scene_sweeps_for_video: List[Dict] = []
    temp_sweep_counter = 0
    for i, sweep_data_dict_item in enumerate(scene_sweep_iterator): # Renamed to avoid conflict
        if i < skip_initial_sweeps: continue
        all_scene_sweeps_for_video.append(sweep_data_dict_item)
        temp_sweep_counter += 1
        if max_sweeps_to_process_for_video is not None and temp_sweep_counter >= max_sweeps_to_process_for_video: break

    if not all_scene_sweeps_for_video:
        tqdm.write("DEBUG (generate_video): No scene sweeps available for video after skip/max frames.") # DEBUG
        for hf_handle in mdet_h5_file_handles:
            if hf_handle: hf_handle.close()
        return {'video_path': output_video_path, 'frames_rendered': 0}
    # print(f"  DEBUG (generate_video): Total sweeps to process for video: {len(all_scene_sweeps_for_video)}") # DEBUG

    first_original_sweep_data = all_scene_sweeps_for_video[0]
    first_sweep_token = first_original_sweep_data['lidar_sd_token']
    # print(f"  DEBUG (generate_video): First sweep token for video dimensions: {first_sweep_token}") # DEBUG

    mdet_data_for_first_frame_list: List[Optional[Dict[str, np.ndarray]]] = []
    mdet_run_names = video_cfg.get('mdetector_run_names', [f"MDet_Run_{j}" for j in range(len(mdet_h5_file_handles))])

    for hdf5_idx, h5_file_handle_item in enumerate(mdet_h5_file_handles): # Renamed
        data = None
        if h5_file_handle_item:
            data = _get_mdet_data_for_token_from_hdf5(h5_file_handle_item, first_sweep_token)

        if data is None:
            # print(f"  DEBUG (generate_video): MDet data for first token {first_sweep_token} from HDF5 run {hdf5_idx} is None. Using placeholder.") # DEBUG
            data = {'dynamic': np.empty((0,3)), 'occluded_by_mdet': np.empty((0,3)),
                    'undetermined_by_mdet': np.empty((0,3)), 'source_name': mdet_run_names[hdf5_idx] + " (NoData)"}
        else:
            data['source_name'] = mdet_run_names[hdf5_idx]
        mdet_data_for_first_frame_list.append(data)
    
    # print(f"  DEBUG (generate_video): MDet data for first frame (token {first_sweep_token}):") # DEBUG
    # if mdet_data_for_first_frame_list: # Check if list is not empty
    #     for i, d in enumerate(mdet_data_for_first_frame_list):
    #         print(f"    Run {i}: Dynamic shape: {d['dynamic'].shape if 'dynamic' in d else 'N/A'}, Occluded shape: {d['occluded_by_mdet'].shape if 'occluded_by_mdet' in d else 'N/A'}") # DEBUG
    # else:
    #     print("    mdet_data_for_first_frame_list is empty.")


    # Condition for early exit if first frame MDet data is problematic
    early_exit_condition_met = False
    if not mdet_data_for_first_frame_list or mdet_data_for_first_frame_list[0] is None:
        early_exit_condition_met = True
    elif 'dynamic' not in mdet_data_for_first_frame_list[0] or \
         'occluded_by_mdet' not in mdet_data_for_first_frame_list[0]: # Check keys exist
        early_exit_condition_met = True
        # print(f"  DEBUG (generate_video): Key 'dynamic' or 'occluded_by_mdet' missing in first frame MDet data.") # DEBUG
    elif mdet_data_for_first_frame_list[0]['dynamic'].size == 0 and \
         mdet_data_for_first_frame_list[0]['occluded_by_mdet'].size == 0:
        # This condition might be too strict if a valid first frame truly has no dynamic/occluded points
        # print(f"  DEBUG (generate_video): First frame MDet data has 0 dynamic AND 0 occluded points. This might be okay or indicate an issue.") # DEBUG
        # Consider if this should always lead to an early exit. For now, let it proceed to composer.
        pass


    if early_exit_condition_met and frame_composer_function == compose_gt_vs_mdet_frame:
         tqdm.write(f"DEBUG (generate_video): Cannot initialize video dimensions. M-Detector data for the first sweep token ({first_sweep_token}) is missing or malformed in the primary HDF5.") # DEBUG
         for hf_handle in mdet_h5_file_handles:
             if hf_handle: hf_handle.close()
         return {'video_path': output_video_path, 'frames_rendered': 0}

    try:
        if frame_composer_function == compose_gt_vs_mdet_frame:
            temp_frame_bgr = frame_composer_function(
                nusc, first_original_sweep_data, mdet_data_for_first_frame_list[0], config, 0
            )
        else:
            temp_frame_bgr = frame_composer_function(
                nusc, first_original_sweep_data, mdet_data_for_first_frame_list, config, 0
            )
    except Exception as e_compose_first:
        tqdm.write(f"DEBUG (generate_video): Error during first frame composition for dimensions: {e_compose_first}") # DEBUG
        import traceback
        traceback.print_exc()
        for hf_handle in mdet_h5_file_handles:
            if hf_handle: hf_handle.close()
        return {'video_path': output_video_path, 'frames_rendered': 0}


    height, width, _ = temp_frame_bgr.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames_rendered_count = 0
    scene_name_for_log = nusc.get('scene', scene_token)['name'] # Renamed
    desc = f"Generating video for scene {scene_name_for_log} (HDF5)"

    for frame_idx, original_sweep_data_dict_loop in enumerate(tqdm(all_scene_sweeps_for_video, desc=desc)): # Renamed
        current_sweep_token_loop = original_sweep_data_dict_loop['lidar_sd_token'] # Renamed
        current_mdet_results_from_all_runs: List[Optional[Dict[str, np.ndarray]]] = []

        for hdf5_run_idx, h5_file_handle_loop in enumerate(mdet_h5_file_handles): # Renamed
            mdet_data_for_this_run = None
            if h5_file_handle_loop:
                mdet_data_for_this_run = _get_mdet_data_for_token_from_hdf5(h5_file_handle_loop, current_sweep_token_loop)

            if mdet_data_for_this_run is None:
                mdet_data_for_this_run = {
                    'dynamic': np.empty((0,3)), 'occluded_by_mdet': np.empty((0,3)),
                    'undetermined_by_mdet': np.empty((0,3)),
                    'source_name': mdet_run_names[hdf5_run_idx] + " (NoData)"}
            else:
                 mdet_data_for_this_run['source_name'] = mdet_run_names[hdf5_run_idx]
            current_mdet_results_from_all_runs.append(mdet_data_for_this_run)
        
        try:
            if frame_composer_function == compose_gt_vs_mdet_frame:
                if not current_mdet_results_from_all_runs or current_mdet_results_from_all_runs[0] is None:
                    # tqdm.write(f"  DEBUG (generate_video): MDet data for sweep {current_sweep_token_loop} (frame {frame_idx}) missing in primary HDF5. Skipping video frame.") # DEBUG
                    continue
                bgr_frame = frame_composer_function(
                    nusc, original_sweep_data_dict_loop, current_mdet_results_from_all_runs[0], config, frames_rendered_count
                )
            else:
                bgr_frame = frame_composer_function(
                    nusc, original_sweep_data_dict_loop, current_mdet_results_from_all_runs, config, frames_rendered_count
                )
            video_writer.write(bgr_frame)
            frames_rendered_count += 1
        except Exception as e_compose_loop:
            tqdm.write(f"DEBUG (generate_video): Error during frame composition for sweep {current_sweep_token_loop} (frame_idx {frame_idx}): {e_compose_loop}") # DEBUG
            # import traceback # Already imported
            # traceback.print_exc() # Optionally print for every error, can be noisy
            # Decide if you want to continue to the next frame or stop
            # For now, let's skip this frame and continue
            continue


    video_writer.release()
    for hf_handle in mdet_h5_file_handles:
        if hf_handle:
            try:
                hf_handle.close()
            except Exception as e_close:
                tqdm.write(f"  DEBUG (generate_video): Warning - Error closing HDF5 file: {e_close}")

    tqdm.write(f"Video generation attempt complete. Rendered {frames_rendered_count} frames.") # Changed message
    return {'video_path': output_video_path, 'frames_rendered': frames_rendered_count}
