# src/visualization/video_generator.py
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from typing import Dict, Any, List, Callable, Optional
import h5py

from ..data_utils.nuscenes_helper import get_scene_sweep_data_sequence
from ..core.constants import OcclusionResult
from ..config_loader import MDetectorConfigAccessor

import logging
logger = logging.getLogger(__name__) # Module-level logger


def _get_mdet_data_for_token_from_hdf5(
    h5_file_handle: h5py.File,
    target_sweep_token: str
) -> Optional[Dict[str, np.ndarray]]:
    # (This helper function remains unchanged from your provided code)
    try:
        all_tokens_bytes = h5_file_handle['sweep_lidar_sd_tokens'][:]
        target_token_bytes = target_sweep_token.encode('utf-8') if isinstance(target_sweep_token, str) else target_sweep_token
        all_tokens_str = [t.decode('utf-8', 'ignore') for t in all_tokens_bytes]
        matching_indices = np.where(np.array(all_tokens_str) == target_sweep_token)[0]
        if not matching_indices.size > 0: return None
        idx_in_h5 = matching_indices[0]
        indices_array = h5_file_handle['points_predictions_indices'][:]
        start_idx = indices_array[idx_in_h5]
        if idx_in_h5 + 1 >= len(indices_array): return None
        end_idx = indices_array[idx_in_h5 + 1]
        all_preds_for_sweep = h5_file_handle['all_points_predictions'][start_idx:end_idx]
        if all_preds_for_sweep.shape[0] == 0:
            return {'dynamic': np.empty((0, 3)), 'occluded_by_mdet': np.empty((0, 3)),
                    'undetermined_by_mdet': np.empty((0, 3))}
        dynamic_label_val = OcclusionResult.OCCLUDING_IMAGE.value
        occluded_label_val = OcclusionResult.OCCLUDED_BY_IMAGE.value
        undetermined_label_val = OcclusionResult.UNDETERMINED.value
        dynamic_mask = (all_preds_for_sweep['mdet_label'] == dynamic_label_val)
        occluded_mask = (all_preds_for_sweep['mdet_label'] == occluded_label_val)
        undetermined_mask = (all_preds_for_sweep['mdet_label'] == undetermined_label_val)
        return {
            'dynamic': np.stack((all_preds_for_sweep['x'][dynamic_mask], all_preds_for_sweep['y'][dynamic_mask], all_preds_for_sweep['z'][dynamic_mask]), axis=-1),
            'occluded_by_mdet': np.stack((all_preds_for_sweep['x'][occluded_mask], all_preds_for_sweep['y'][occluded_mask], all_preds_for_sweep['z'][occluded_mask]), axis=-1),
            'undetermined_by_mdet': np.stack((all_preds_for_sweep['x'][undetermined_mask], all_preds_for_sweep['y'][undetermined_mask], all_preds_for_sweep['z'][undetermined_mask]), axis=-1),
        }
    except KeyError: return None
    except Exception: return None



def generate_video_from_hdf5_list(
    nusc: NuScenes,
    scene_token: str,
    list_of_mdetector_hdf5_paths: List[str],
    output_video_path: str,
    frame_composer_function: Callable,
    config_accessor: MDetectorConfigAccessor,
    # --- NEW PARAMETER ---
    run_specific_mdet_config: Optional[Dict] = None 
):
    if not list_of_mdetector_hdf5_paths:
        tqdm.write("DEBUG (generate_video): No M-Detector HDF5 paths provided.")
        return {'video_path': output_video_path, 'frames_rendered': 0}

    video_gen_cfg = config_accessor.get_video_generation_params()
    processing_settings = config_accessor.get_processing_settings()
    nuscenes_params = config_accessor.get_nuscenes_params()
    fps = video_gen_cfg['fps']
    mdet_run_names = video_gen_cfg.get('mdetector_run_names', 
                                     [f"MDet_Run_{j}" for j in range(len(list_of_mdetector_hdf5_paths))])

    mdet_h5_file_handles: List[Optional[h5py.File]] = []
    for hdf5_path in list_of_mdetector_hdf5_paths:
        try:
            handle = h5py.File(hdf5_path, 'r')
            mdet_h5_file_handles.append(handle)
        except Exception as e_open:
            tqdm.write(f"  DEBUG (generate_video): Could not open MDet HDF5 {hdf5_path}: {e_open}. Skipping.")
            mdet_h5_file_handles.append(None)

    if not any(mdet_h5_file_handles):
        tqdm.write("DEBUG (generate_video): None of the MDet HDF5 files could be opened.")
        return {'video_path': output_video_path, 'frames_rendered': 0}

    skip_initial_sweeps = processing_settings['skip_frames']
    max_sweeps_to_process_for_video = processing_settings['max_frames']
    lidar_name_for_sweeps = nuscenes_params['lidar_sensor_name']

    # --- Determine filtering parameters to pass to frame_composer ---
    # Priority: run_specific_mdet_config, then current config_accessor
    mdet_min_range_for_composer_override: Optional[float] = None
    mdet_max_range_for_composer_override: Optional[float] = None

    if run_specific_mdet_config:
        try:
            m_detector_cfg_loaded = run_specific_mdet_config['m_detector']
            filtering_config_loaded = m_detector_cfg_loaded['point_pre_filtering']
            min_r = filtering_config_loaded.get('min_range_meters')
            max_r = filtering_config_loaded.get('max_range_meters')
            if min_r is not None and max_r is not None:
                mdet_min_range_for_composer_override = float(min_r)
                mdet_max_range_for_composer_override = float(max_r)
                tqdm.write(f"INFO (generate_video): Using MDet range from run-specific config: "
                           f"{mdet_min_range_for_composer_override:.1f}-{mdet_max_range_for_composer_override:.1f}m for consistency.")
        except Exception as e_parse_run_cfg:
            tqdm.write(f"Warning (generate_video): Could not parse min/max range from run_specific_mdet_config: {e_parse_run_cfg}. "
                       "Will use current script's config for filtering.")
    # If overrides are still None, compose_gt_vs_mdet_frame will use its own config_accessor fallback.
    # --- End determining filtering parameters ---

    scene_sweep_iterator = get_scene_sweep_data_sequence(nusc, scene_token, lidar_name=lidar_name_for_sweeps)
    all_scene_sweeps_for_video: List[Dict] = []
    temp_sweep_counter = 0
    for i, sweep_data_dict_item in enumerate(scene_sweep_iterator):
        if i < skip_initial_sweeps: continue
        all_scene_sweeps_for_video.append(sweep_data_dict_item)
        temp_sweep_counter += 1
        if max_sweeps_to_process_for_video is not None and temp_sweep_counter >= max_sweeps_to_process_for_video: break

    if not all_scene_sweeps_for_video:
        tqdm.write("DEBUG (generate_video): No scene sweeps available for video after skip/max frames.")
        # ... (close HDF5 handles) ...
        return {'video_path': output_video_path, 'frames_rendered': 0}

    first_original_sweep_data = all_scene_sweeps_for_video[0]
    first_sweep_token = first_original_sweep_data['lidar_sd_token']
    
    mdet_data_for_first_frame_list: List[Optional[Dict[str, np.ndarray]]] = []
    # ... (populate mdet_data_for_first_frame_list as before) ...
    for hdf5_idx, h5_file_handle_item in enumerate(mdet_h5_file_handles):
        data = None
        if h5_file_handle_item:
            data = _get_mdet_data_for_token_from_hdf5(h5_file_handle_item, first_sweep_token)
        if data is None:
            data = {'dynamic': np.empty((0,3)), 'occluded_by_mdet': np.empty((0,3)),
                    'undetermined_by_mdet': np.empty((0,3)), 'source_name': mdet_run_names[hdf5_idx] + " (NoData)"}
        else:
            data['source_name'] = mdet_run_names[hdf5_idx]
        mdet_data_for_first_frame_list.append(data)


    if not mdet_data_for_first_frame_list or mdet_data_for_first_frame_list[0] is None or \
       ('dynamic' not in mdet_data_for_first_frame_list[0]):
        tqdm.write(f"DEBUG (generate_video): Cannot initialize video dimensions. Primary M-Detector data for first sweep ({first_sweep_token}) is missing or malformed.")
        # ... (close HDF5 handles) ...
        return {'video_path': output_video_path, 'frames_rendered': 0}

    try:
        temp_frame_bgr = frame_composer_function(
            nusc, first_original_sweep_data, 
            mdet_data_for_first_frame_list[0] if frame_composer_function.__name__ == 'compose_gt_vs_mdet_frame' else mdet_data_for_first_frame_list, 
            config_accessor, 0,
            # --- PASS OVERRIDES ---
            mdet_min_range_override=mdet_min_range_for_composer_override,
            mdet_max_range_override=mdet_max_range_for_composer_override
        )
    except Exception as e_compose_first:
        tqdm.write(f"DEBUG (generate_video): Error during first frame composition for dimensions: {e_compose_first}")
        # ... (close HDF5 handles, traceback) ...
        return {'video_path': output_video_path, 'frames_rendered': 0}

    height, width, _ = temp_frame_bgr.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frames_rendered_count = 0
    scene_name_for_log = nusc.get('scene', scene_token)['name']
    desc = f"Generating video for scene {scene_name_for_log} (HDF5)"

    for frame_idx, original_sweep_data_dict_loop in enumerate(tqdm(all_scene_sweeps_for_video, desc=desc)):
        current_sweep_token_loop = original_sweep_data_dict_loop['lidar_sd_token']
        current_mdet_results_from_all_runs: List[Optional[Dict[str, np.ndarray]]] = []
        # ... (populate current_mdet_results_from_all_runs as before) ...
        for hdf5_run_idx, h5_file_handle_loop in enumerate(mdet_h5_file_handles):
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
            bgr_frame = frame_composer_function(
                nusc, original_sweep_data_dict_loop, 
                current_mdet_results_from_all_runs[0] if frame_composer_function.__name__ == 'compose_gt_vs_mdet_frame' else current_mdet_results_from_all_runs, 
                config_accessor, frames_rendered_count,
                # --- PASS OVERRIDES ---
                mdet_min_range_override=mdet_min_range_for_composer_override,
                mdet_max_range_override=mdet_max_range_for_composer_override
            )
            video_writer.write(bgr_frame)
            frames_rendered_count += 1
        except Exception as e_compose_loop:
            tqdm.write(f"DEBUG (generate_video): Error during frame composition for sweep {current_sweep_token_loop} (frame_idx {frame_idx}): {e_compose_loop}")
            continue

    video_writer.release()
    for hf_handle in mdet_h5_file_handles:
        if hf_handle:
            try: hf_handle.close()
            except Exception as e_close: tqdm.write(f"  DEBUG (generate_video): Warning - Error closing HDF5 file: {e_close}")

    tqdm.write(f"Video generation attempt complete. Rendered {frames_rendered_count} frames.")
    return {'video_path': output_video_path, 'frames_rendered': frames_rendered_count}
