# src/visualization/frame_composers.py
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from typing import Dict, Any, Optional
from nuscenes.nuscenes import NuScenes # For type hinting
import os
from tqdm import tqdm

from .plot_primitives import render_bev_points_on_ax # Use the primitive
from ..visualization.visualization_utils import mpl_fig_to_opencv_bgr 
from ..utils.validation_utils import get_gt_dynamic_points_for_sweep 
from ..utils.transformations import transform_points_numpy

def compose_gt_vs_mdet_frame(
    nusc: NuScenes,
    current_sweep_data: Dict,
    mdetector_points_for_sweep: Dict[str, np.ndarray],
    config: Dict,
    frame_number_in_video: int
) -> np.ndarray:
    # print(f"  DEBUG (compose_frame {frame_number_in_video}): MDet input - Dynamic shape: {mdetector_points_for_sweep.get('dynamic', np.empty((0,0))).shape}, Occluded shape: {mdetector_points_for_sweep.get('occluded_by_mdet', np.empty((0,0))).shape}") # DEBUG

    viz_cfg = config.get('visualization', {})
    video_cfg = viz_cfg.get('video_generation', {})
    gt_cfg = viz_cfg.get('ground_truth_style', {})
    mdet_cfg = viz_cfg.get('mdetector_style', {})
    common_bev_cfg = viz_cfg.get('common_bev_style', {})

    fig_size = tuple(video_cfg.get('figure_size_inches_side_by_side', (20, 10)))
    # IMPORTANT: Add try-except around plt operations if they might fail with empty data
    try:
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=fig_size)
    except Exception as e_plt_subplots:
        tqdm.write(f"  DEBUG (compose_frame {frame_number_in_video}): Error in plt.subplots: {e_plt_subplots}")
        # Return a dummy black frame or raise error
        return np.zeros((int(fig_size[1]*100), int(fig_size[0]*100), 3), dtype=np.uint8) # Assuming 100 DPI for dummy

    gt_vel_thresh = config.get('validation', {}).get('vel_threshold', 0.5)
    fig_title = video_cfg.get('figure_title_template', 'GT (vel>={vel_thresh}m/s) vs. M-Detector - Frame {frame_num}')
    fig.suptitle(fig_title.format(vel_thresh=gt_vel_thresh, frame_num=frame_number_in_video), fontsize=video_cfg.get('figure_title_fontsize', 14))

    points_sensor_frame = current_sweep_data['points_sensor_frame']
    T_global_lidar = current_sweep_data['T_global_lidar']
    all_points_global_background = transform_points_numpy(points_sensor_frame, T_global_lidar)
    
    # If all_points_global_background is empty, BEV plots might be strange or fail.
    if all_points_global_background.shape[0] == 0:
        # print(f"  DEBUG (compose_frame {frame_number_in_video}): No background points for this sweep ({current_sweep_data.get('lidar_sd_token', 'Unknown token')}). Plots might be empty.")
        # Still proceed, render_bev_points_on_ax should handle empty background_points_global
        pass


    cs_rec = nusc.get('calibrated_sensor', current_sweep_data['calibrated_sensor_token'])
    T_vehicle_lidar_np = np.eye(4)
    T_vehicle_lidar_np[:3,:3] = Quaternion(cs_rec['rotation']).rotation_matrix
    T_vehicle_lidar_np[:3,3] = np.array(cs_rec['translation'])
    T_global_vehicle = T_global_lidar @ np.linalg.inv(T_vehicle_lidar_np)
    ego_translation = T_global_vehicle[:3, 3]
    ego_rotation = Quaternion(matrix=T_global_vehicle)

    gt_label_base_dir = config.get('nuscenes',{}).get('label_path')
    # scene_rec = nusc.get('scene', current_sweep_data['scene_token']) # scene_token might not be in current_sweep_data
    # Need to get scene_token. If it's not in current_sweep_data, we might need to find it.
    # Assuming current_sweep_data has 'sample_token' from which we can get 'scene_token'
    sample_token = current_sweep_data.get('sample_token')
    scene_token_for_gt = None
    if sample_token:
        sample_rec = nusc.get('sample', sample_token)
        scene_token_for_gt = sample_rec['scene_token']
    
    if not scene_token_for_gt: # Fallback if sample_token wasn't there
        # This is a bit risky, assumes the scene_token from the video generator context is the right one
        # It's better if current_sweep_data always includes 'scene_token'
        tqdm.write("Warning (compose_frame): 'scene_token' not directly in current_sweep_data, attempting to use context. This might be unreliable.")
        # This part is tricky if compose_gt_vs_mdet_frame is called without scene_token in current_sweep_data
        # For now, let's assume the video_generator passes a scene_token that can be used
        # or the config['nuscenes']['label_path'] needs to be structured differently.
        # The `get_gt_dynamic_points_for_sweep` expects the full path to the scene's HDF5 file.
        # The main video script `generate_video.py` knows the scene_name and scene_token.
        # This implies `gt_labels_scene_hdf5_filepath` should be constructed *before* calling this composer,
        # or this composer needs the scene_token.
        # Let's assume for now `config` might contain the direct path if pre-calculated by caller.
        # This is a potential refactoring point.
        # For now, we rely on the path construction done in the previous response for this function.
        scene_rec_for_gt = nusc.get('scene', current_sweep_data['scene_token']) # Requires 'scene_token' in current_sweep_data
        scene_name_for_gt = scene_rec_for_gt['name']
        gt_scene_hdf5_filename = f"gt_point_labels_{scene_name_for_gt}.h5"
        gt_labels_scene_hdf5_filepath = os.path.join(gt_label_base_dir, gt_scene_hdf5_filename)
    else:
        scene_rec_for_gt = nusc.get('scene', scene_token_for_gt)
        scene_name_for_gt = scene_rec_for_gt['name']
        gt_scene_hdf5_filename = f"gt_point_labels_{scene_name_for_gt}.h5"
        gt_labels_scene_hdf5_filepath = os.path.join(gt_label_base_dir, gt_scene_hdf5_filename)


    gt_point_categories = get_gt_dynamic_points_for_sweep(
        nusc, current_sweep_data, all_points_global_background,
        gt_labels_scene_hdf5_filepath, # Pass the full HDF5 file path
        gt_vel_thresh
    )
    # print(f"  DEBUG (compose_frame {frame_number_in_video}): GT data - Dynamic shape: {gt_point_categories.get('dynamic', np.empty((0,0))).shape}, Static shape: {gt_point_categories.get('static', np.empty((0,0))).shape}") # DEBUG

    # ... (rest of the function, including render_bev_points_on_ax calls) ...
    # Add try-except around render_bev_points_on_ax and mpl_fig_to_opencv_bgr if needed
    try:
        # GT Plot
        gt_points_to_plot = {'GT_Dynamic': gt_point_categories.get('dynamic', np.empty((0,3)))}
        if gt_cfg.get('show_static_points', True):
           gt_points_to_plot['GT_Static_Labeled'] = gt_point_categories.get('static', np.empty((0,3)))
        if gt_cfg.get('show_unlabeled_points_as_background_detail', False):
            gt_points_to_plot['GT_Unlabeled'] = gt_point_categories.get('unlabeled', np.empty((0,3)))

        gt_highlight_styles = {
            'GT_Dynamic': gt_cfg.get('dynamic_points_style', {'color': 'blue', 's': 2.5, 'legend_label': 'GT Dynamic', 'zorder': 6}),
            'GT_Static_Labeled': gt_cfg.get('static_points_style', {'color': 'cyan', 's': 1.0, 'legend_label': 'GT Static (Labeled)', 'zorder': 3}),
            'GT_Unlabeled': {'color': 'lightgrey', 's': 0.1, 'legend_label': '_nolegend_', 'zorder': 0},
        }
        gt_plot_specific_cfg = {**common_bev_cfg, **gt_cfg.get('subplot_style', {})}
        gt_plot_specific_cfg['subplot_title'] = gt_cfg.get('subplot_title', f'Ground Truth (vel>={gt_vel_thresh}m/s)')
        render_bev_points_on_ax(
            ax_gt, all_points_global_background, gt_points_to_plot, gt_highlight_styles,
            ego_translation, ego_rotation, gt_plot_specific_cfg, is_right_subplot=False
        )

        # M-Detector Plot
        mdet_points_to_plot = {}
        if mdet_cfg.get('show_dynamic_points', True):
            mdet_points_to_plot['MDet_Dynamic'] = mdetector_points_for_sweep.get('dynamic', np.empty((0,3)))
        if mdet_cfg.get('show_occluded_points', False): # From your original config
            mdet_points_to_plot['MDet_Occluded'] = mdetector_points_for_sweep.get('occluded_by_mdet', np.empty((0,3)))
        # Add undetermined if you have it in mdetector_points_for_sweep and want to plot
        # if mdet_cfg.get('show_undetermined_points', False):
        #    mdet_points_to_plot['MDet_Undetermined'] = mdetector_points_for_sweep.get('undetermined_by_mdet', np.empty((0,3)))


        mdet_highlight_styles = {
            'MDet_Dynamic': mdet_cfg.get('dynamic_points_style', {'color': 'green', 's': 2.0, 'legend_label': 'MDet: Dynamic', 'zorder': 5}),
            'MDet_Occluded': mdet_cfg.get('occluded_points_style', {'color': 'orange', 's': 1.5, 'legend_label': 'MDet: Occluded', 'zorder': 4}),
            # 'MDet_Undetermined': mdet_cfg.get('undetermined_points_style', {'color': 'yellow', 's': 1.0, 'legend_label': 'MDet: Undetermined', 'zorder': 3}),
        }
        mdet_plot_specific_cfg = {**common_bev_cfg, **mdet_cfg.get('subplot_style', {})}
        mdet_plot_specific_cfg['subplot_title'] = mdet_cfg.get('subplot_title', 'M-Detector Predictions')
        render_bev_points_on_ax(
            ax_pred, all_points_global_background, mdet_points_to_plot, mdet_highlight_styles,
            ego_translation, ego_rotation, mdet_plot_specific_cfg, is_right_subplot=True
        )
        
        fig.tight_layout(rect=[0, 0.03, 1, video_cfg.get('tight_layout_rect_top_factor', 0.95)])
        frame_bgr = mpl_fig_to_opencv_bgr(fig)
    except Exception as e_render:
        tqdm.write(f"  DEBUG (compose_frame {frame_number_in_video}): Error during rendering frame: {e_render}")
        import traceback
        traceback.print_exc()
        frame_bgr = np.zeros((int(fig_size[1]*100), int(fig_size[0]*100), 3), dtype=np.uint8) # Return black frame
    finally:
        plt.close(fig) # IMPORTANT to free memory
    return frame_bgr

# Future:
# def compose_multirun_mdet_frame(nusc, current_sweep_data, list_of_mdet_results, config, frame_num):
#    fig, axes = plt.subplots(1, len(list_of_mdet_results), ...)
#    for i, mdet_result in enumerate(list_of_mdet_results):
#        render_bev_points_on_ax(axes[i], ..., mdet_result, ...)
#    return mpl_fig_to_opencv_bgr(fig)

# def compose_multi_run_comparison_frame(
#     nusc: NuScenes,
#     current_sweep_data: Dict,
#     list_of_mdet_results_for_sweep: List[Optional[Dict[str, np.ndarray]]], # List of results, one per run
#     config: Dict,
#     frame_number_in_video: int
# ) -> np.ndarray:
#     num_runs = len(list_of_mdet_results_for_sweep)
#     fig, axes = plt.subplots(1, num_runs, figsize=(num_runs * 10, 10)) # Example: 1 row, N columns
#     if num_runs == 1: # Handle single run case if plt.subplots returns a single Axes
#         axes = [axes] 

#     fig.suptitle(f"M-Detector Multi-Run Comparison - Frame {frame_number_in_video}", ...)

#     # ... common data prep (ego pose, background points) ...

#     for i, mdet_result_this_run in enumerate(list_of_mdet_results_for_sweep):
#         ax = axes[i]
#         # subplot_title = f"Run: {mdet_result_this_run.get('source_name', f'Run {i}')}"
#         # plot_config_for_this_ax = { ... 'subplot_title': subplot_title ... }
        
#         points_to_plot_this_run = {}
#         if mdet_result_this_run: # Check if data exists for this run for this sweep
#             # Populate points_to_plot_this_run from mdet_result_this_run
#             # e.g., points_to_plot_this_run['Dynamic'] = mdet_result_this_run.get('dynamic')
#             pass # Add your logic here

#         # render_bev_points_on_ax(ax, background_points, points_to_plot_this_run, ...)
    
#     # ... fig.tight_layout(), mpl_fig_to_opencv_bgr(fig), plt.close(fig) ...
#     return np.zeros((100,100,3), dtype=np.uint8) # Placeholder