# src/visualization/frame_composers.py
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from typing import Dict, Any, Optional
from nuscenes.nuscenes import NuScenes # For type hinting
import os
from tqdm import tqdm

from .plot_primitives import render_bev_points_on_ax
from ..visualization.visualization_utils import mpl_fig_to_opencv_bgr 
from ..utils.validation_utils import get_gt_dynamic_points_for_sweep 
from ..utils.transformations import transform_points_numpy
from ..config_loader import MDetectorConfigAccessor

def compose_gt_vs_mdet_frame(
    nusc: NuScenes,
    current_sweep_data: Dict,
    mdetector_points_for_sweep: Dict[str, np.ndarray],
    config_accessor: MDetectorConfigAccessor,
    frame_number_in_video: int,
    mdet_min_range_override: Optional[float] = None,
    mdet_max_range_override: Optional[float] = None
) -> np.ndarray:
    
    viz_params = config_accessor.get_visualization_params()
    video_gen_cfg = viz_params.get('video_generation', {})
    gt_style_cfg = viz_params.get('ground_truth_style', {})
    mdet_style_cfg = viz_params.get('mdetector_style', {})
    common_bev_cfg = viz_params.get('common_bev_style', {})
    validation_params = config_accessor.get_validation_params()
    nuscenes_params = config_accessor.get_nuscenes_params()

    # --- Determine M-Detector's range filtering parameters to use ---
    # Priority: override, then current config_accessor's point_pre_filtering
    mdet_min_range_to_use: float
    mdet_max_range_to_use: float

    if mdet_min_range_override is not None and mdet_max_range_override is not None:
        mdet_min_range_to_use = mdet_min_range_override
        mdet_max_range_to_use = mdet_max_range_override
        # Optional: Log that override is being used
        # logger.debug(f"compose_frame: Using OVERRIDE MDet range: {mdet_min_range_to_use}-{mdet_max_range_to_use}m")
    else:
        point_pre_filtering_params = config_accessor.get_point_pre_filtering_params()
        mdet_min_range_to_use = point_pre_filtering_params.get('min_range_meters', 1.0)
        mdet_max_range_to_use = point_pre_filtering_params.get('max_range_meters', 80.0)
        # logger.debug(f"compose_frame: Using CURRENT SCRIPT's MDet range: {mdet_min_range_to_use}-{mdet_max_range_to_use}m")
    # --- End determining filtering parameters ---

    fig_size = tuple(video_gen_cfg.get('figure_size_inches_side_by_side', (20, 10)))
    try:
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=fig_size)
    except Exception as e_plt_subplots:
        tqdm.write(f"  DEBUG (compose_frame {frame_number_in_video}): Error in plt.subplots: {e_plt_subplots}")
        return np.zeros((int(fig_size[1]*100), int(fig_size[0]*100), 3), dtype=np.uint8)

    gt_vel_thresh = validation_params.get('gt_velocity_threshold', 0.5)
    fig_title_template = video_gen_cfg.get('figure_title_template', 'GT (vel>={vel_thresh}m/s) vs. M-Detector - Frame {frame_num}')
    fig.suptitle(fig_title_template.format(vel_thresh=gt_vel_thresh, frame_num=frame_number_in_video), 
                 fontsize=video_gen_cfg.get('figure_title_fontsize', 14))

    points_sensor_frame = current_sweep_data['points_sensor_frame']
    T_global_lidar = current_sweep_data['T_global_lidar']

    # --- Apply range filtering to points_sensor_frame BEFORE transforming to global ---
    if points_sensor_frame.shape[0] > 0:
        ranges_in_sensor_frame = np.linalg.norm(points_sensor_frame[:, :3], axis=1)
        range_mask = (ranges_in_sensor_frame >= mdet_min_range_to_use) & \
                     (ranges_in_sensor_frame <= mdet_max_range_to_use)
        
        filtered_points_sensor_frame = points_sensor_frame[range_mask]
        all_points_global_background = transform_points_numpy(filtered_points_sensor_frame, T_global_lidar)
    else:
        all_points_global_background = np.empty((0,3))
    # --- End applying range filtering for background ---
    
    # (Ego pose calculation remains the same)
    cs_rec = nusc.get('calibrated_sensor', current_sweep_data['calibrated_sensor_token'])
    T_vehicle_lidar_np = np.eye(4)
    T_vehicle_lidar_np[:3,:3] = Quaternion(cs_rec['rotation']).rotation_matrix
    T_vehicle_lidar_np[:3,3] = np.array(cs_rec['translation'])
    T_global_vehicle = T_global_lidar @ np.linalg.inv(T_vehicle_lidar_np)
    ego_translation = T_global_vehicle[:3, 3]
    ego_rotation = Quaternion(matrix=T_global_vehicle)

    # (GT HDF5 path construction remains the same)
    gt_label_base_dir = nuscenes_params.get('label_path')
    sample_token = current_sweep_data.get('sample_token')
    scene_token_for_gt = None
    if sample_token:
        sample_rec = nusc.get('sample', sample_token)
        scene_token_for_gt = sample_rec['scene_token']
    if not scene_token_for_gt:
        gt_labels_scene_hdf5_filepath = "PATH_NOT_FOUND_DUE_TO_MISSING_SCENE_TOKEN"
    else:
        scene_rec_for_gt = nusc.get('scene', scene_token_for_gt)
        scene_name_for_gt = scene_rec_for_gt['name']
        gt_scene_hdf5_filename = f"gt_point_labels_{scene_name_for_gt}.h5"
        gt_labels_scene_hdf5_filepath = os.path.join(gt_label_base_dir, gt_scene_hdf5_filename)

    # Pass the now consistently filtered all_points_global_background
    # and the mdet_min_range_to_use, mdet_max_range_to_use
    gt_indices_result = get_gt_dynamic_points_for_sweep(
        nusc, current_sweep_data, 
        all_points_global_background, # This is now range-filtered
        gt_labels_scene_hdf5_filepath,
        gt_vel_thresh,
        mdet_min_range_to_use, # Pass the determined min range for GT filtering
        mdet_max_range_to_use  # Pass the determined max range for GT filtering
    )
    
    gt_point_categories = {
        'dynamic': all_points_global_background[gt_indices_result['gt_dynamic_indices']] 
                     if gt_indices_result['gt_dynamic_indices'].size > 0 else np.empty((0,3)),
        'static': all_points_global_background[gt_indices_result['gt_static_indices']]
                    if gt_indices_result['gt_static_indices'].size > 0 else np.empty((0,3)),
        'unlabeled': all_points_global_background[gt_indices_result['unlabeled_indices']]
                       if gt_indices_result['unlabeled_indices'].size > 0 else np.empty((0,3)),
    }
    if gt_indices_result.get('error_msg'):
        tqdm.write(f"  Warning (compose_frame {frame_number_in_video}): Error getting GT points: {gt_indices_result['error_msg']}")

    try:
        # (GT Plotting logic remains the same, using gt_style_cfg, common_bev_cfg)
        gt_points_to_plot = {'GT_Dynamic': gt_point_categories.get('dynamic', np.empty((0,3)))}
        if gt_style_cfg.get('show_static_points', True):
           gt_points_to_plot['GT_Static_Labeled'] = gt_point_categories.get('static', np.empty((0,3)))
        if gt_style_cfg.get('show_unlabeled_points_as_background_detail', False):
            gt_points_to_plot['GT_Unlabeled'] = gt_point_categories.get('unlabeled', np.empty((0,3)))
        gt_highlight_styles = {
            'GT_Dynamic': gt_style_cfg.get('dynamic_points_style', {'color': 'blue', 's': 2.5, 'legend_label': 'GT Dynamic', 'zorder': 6}),
            'GT_Static_Labeled': gt_style_cfg.get('static_points_style', {'color': 'cyan', 's': 1.0, 'legend_label': 'GT Static (Labeled)', 'zorder': 3}),
            'GT_Unlabeled': gt_style_cfg.get('unlabeled_points_style',{'color': 'lightgrey', 's': 0.1, 'legend_label': '_nolegend_', 'zorder': 0}),
        }
        gt_plot_specific_cfg = {**common_bev_cfg, **gt_style_cfg.get('subplot_style', {})}
        gt_plot_specific_cfg['subplot_title'] = gt_style_cfg.get('subplot_title_template', 'Ground Truth (vel>={vel_thresh}m/s)').format(vel_thresh=gt_vel_thresh)
        render_bev_points_on_ax(
            ax_gt, all_points_global_background, gt_points_to_plot, gt_highlight_styles,
            ego_translation, ego_rotation, gt_plot_specific_cfg, is_right_subplot=False
        )

        # (M-Detector Plotting logic remains the same, using mdet_style_cfg, common_bev_cfg)
        mdet_points_to_plot = {}
        if mdet_style_cfg.get('show_dynamic_points', True):
            mdet_points_to_plot['MDet_Dynamic'] = mdetector_points_for_sweep.get('dynamic', np.empty((0,3)))
        if mdet_style_cfg.get('show_occluded_points', False):
            mdet_points_to_plot['MDet_Occluded'] = mdetector_points_for_sweep.get('occluded_by_mdet', np.empty((0,3)))
        if mdet_style_cfg.get('show_undetermined_points', False):
           mdet_points_to_plot['MDet_Undetermined'] = mdetector_points_for_sweep.get('undetermined_by_mdet', np.empty((0,3)))
        mdet_highlight_styles = {
            'MDet_Dynamic': mdet_style_cfg.get('dynamic_points_style', {'color': 'green', 's': 2.0, 'legend_label': 'MDet: Dynamic', 'zorder': 5}),
            'MDet_Occluded': mdet_style_cfg.get('occluded_points_style', {'color': 'orange', 's': 1.5, 'legend_label': 'MDet: Occluded', 'zorder': 4}),
            'MDet_Undetermined': mdet_style_cfg.get('undetermined_points_style', {'color': 'yellow', 's': 1.0, 'legend_label': 'MDet: Undetermined', 'zorder': 3}),
        }
        mdet_plot_specific_cfg = {**common_bev_cfg, **mdet_style_cfg.get('subplot_style', {})}
        mdet_plot_specific_cfg['subplot_title'] = mdet_style_cfg.get('subplot_title', 'M-Detector Predictions')
        render_bev_points_on_ax(
            ax_pred, all_points_global_background, mdet_points_to_plot, mdet_highlight_styles,
            ego_translation, ego_rotation, mdet_plot_specific_cfg, is_right_subplot=True
        )
        
        fig.tight_layout(rect=[0, 0.03, 1, video_gen_cfg.get('tight_layout_rect_top_factor', 0.95)])
        frame_bgr = mpl_fig_to_opencv_bgr(fig)
    except Exception as e_render:
        tqdm.write(f"  DEBUG (compose_frame {frame_number_in_video}): Error during rendering frame: {e_render}")
        import traceback
        traceback.print_exc()
        frame_bgr = np.zeros((int(fig_size[1]*100), int(fig_size[0]*100), 3), dtype=np.uint8)
    finally:
        plt.close(fig)
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