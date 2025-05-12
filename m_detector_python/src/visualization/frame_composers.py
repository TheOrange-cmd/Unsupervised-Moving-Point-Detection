# src/visualization/frame_composers.py
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from typing import Dict, Any, Optional
from nuscenes.nuscenes import NuScenes # For type hinting

from .plot_primitives import render_bev_points_on_ax # Use the primitive
from ..visualization.visualization_utils import mpl_fig_to_opencv_bgr 
from ..utils.validation_utils import get_gt_dynamic_points_for_sweep 
from ..utils.transformations import transform_points_numpy

def compose_gt_vs_mdet_frame(
    nusc: NuScenes,
    current_sweep_data: Dict, # Original sweep data (points_sensor_frame, T_global_lidar, etc.)
    mdetector_points_for_sweep: Dict[str, np.ndarray], # {'dynamic': ..., 'occluded_by_mdet': ...}
    config: Dict,
    frame_number_in_video: int
) -> np.ndarray:
    """
    Composes a side-by-side BEV frame: Ground Truth vs. M-Detector.
    Returns an OpenCV BGR image.
    """
    viz_cfg = config.get('visualization', {}) # Main visualization config section
    video_cfg = viz_cfg.get('video_generation', {}) # Specific to video output
    gt_cfg = viz_cfg.get('ground_truth_style', {})
    mdet_cfg = viz_cfg.get('mdetector_style', {})
    common_bev_cfg = viz_cfg.get('common_bev_style', {})

    fig_size = tuple(video_cfg.get('figure_size_inches_side_by_side', (20, 10)))
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=fig_size)
    
    gt_vel_thresh = config.get('validation', {}).get('vel_threshold', 0.5) # From main config
    fig_title = video_cfg.get('figure_title_template', 
                              'GT (vel>={vel_thresh}m/s) vs. M-Detector - Frame {frame_num}')
    fig.suptitle(fig_title.format(vel_thresh=gt_vel_thresh, frame_num=frame_number_in_video), 
                 fontsize=video_cfg.get('figure_title_fontsize', 14))

    # --- Common Data for Both Subplots ---
    points_sensor_frame = current_sweep_data['points_sensor_frame']
    T_global_lidar = current_sweep_data['T_global_lidar']
    all_points_global_background = transform_points_numpy(points_sensor_frame, T_global_lidar)
    
    cs_rec = nusc.get('calibrated_sensor', current_sweep_data['calibrated_sensor_token'])
    T_vehicle_lidar_np = np.eye(4)
    T_vehicle_lidar_np[:3,:3] = Quaternion(cs_rec['rotation']).rotation_matrix
    T_vehicle_lidar_np[:3,3] = np.array(cs_rec['translation'])
    T_global_vehicle = T_global_lidar @ np.linalg.inv(T_vehicle_lidar_np)
    
    ego_translation = T_global_vehicle[:3, 3]
    ego_rotation = Quaternion(matrix=T_global_vehicle)

    # --- 1. Ground Truth Plot (Left Subplot: ax_gt) ---
    gt_point_labels = get_gt_dynamic_points_for_sweep( # This should return a dict like {'dynamic': pts, 'static': pts}
        nusc, current_sweep_data, all_points_global_background,
        config.get('nuscenes',{}).get('label_path'), gt_vel_thresh
    )
    
    gt_points_to_plot = {'GT_Dynamic': gt_point_labels.get('dynamic', np.empty((0,3)))}
    # Example: if you want to show static GT points with a different style
    # if gt_cfg.get('show_static_points', False):
    #    gt_points_to_plot['GT_Static'] = gt_point_labels.get('static', np.empty((0,3)))

    # Define highlight_configs for GT plot from main config
    gt_highlight_styles = {
        'GT_Dynamic': gt_cfg.get('dynamic_points_style', {'color': 'blue', 's': 2.5, 'legend_label': 'GT Dynamic', 'zorder': 6}),
        # 'GT_Static': gt_cfg.get('static_points_style', {'color': 'cyan', 's': 1.0, 'legend_label': 'GT Static', 'zorder': 3}),
    }
    
    gt_plot_specific_cfg = {**common_bev_cfg, **gt_cfg.get('subplot_style', {})} # Merge common with GT specific
    gt_plot_specific_cfg['subplot_title'] = gt_cfg.get('subplot_title', f'Ground Truth (vel>={gt_vel_thresh}m/s)')
    
    render_bev_points_on_ax(
        ax_gt, all_points_global_background, gt_points_to_plot, gt_highlight_styles,
        ego_translation, ego_rotation, gt_plot_specific_cfg, is_right_subplot=False
    )

    # --- 2. M-Detector Predictions Plot (Right Subplot: ax_pred) ---
    mdet_points_to_plot = {}
    if mdet_cfg.get('show_dynamic_points', True):
        mdet_points_to_plot['MDet_Dynamic'] = mdetector_points_for_sweep.get('dynamic', np.empty((0,3)))
    if mdet_cfg.get('show_occluded_points', False):
        mdet_points_to_plot['MDet_Occluded'] = mdetector_points_for_sweep.get('occluded_by_mdet', np.empty((0,3)))
    if mdet_cfg.get('show_undetermined_points', False):
        mdet_points_to_plot['MDet_Undetermined'] = mdetector_points_for_sweep.get('undetermined_by_mdet', np.empty((0,3)))

    mdet_highlight_styles = {
        'MDet_Dynamic': mdet_cfg.get('dynamic_points_style', {'color': 'green', 's': 2.0, 'legend_label': 'MDet: Dynamic', 'zorder': 5}),
        'MDet_Occluded': mdet_cfg.get('occluded_points_style', {'color': 'orange', 's': 1.5, 'legend_label': 'MDet: Occluded', 'zorder': 4}),
        'MDet_Undetermined': mdet_cfg.get('undetermined_points_style', {'color': 'yellow', 's': 1.0, 'legend_label': 'MDet: Undetermined', 'zorder': 3}),
    }

    mdet_plot_specific_cfg = {**common_bev_cfg, **mdet_cfg.get('subplot_style', {})}
    mdet_plot_specific_cfg['subplot_title'] = mdet_cfg.get('subplot_title', 'M-Detector Predictions')

    render_bev_points_on_ax(
        ax_pred, all_points_global_background, mdet_points_to_plot, mdet_highlight_styles,
        ego_translation, ego_rotation, mdet_plot_specific_cfg, is_right_subplot=True
    )
    
    fig.tight_layout(rect=[0, 0.03, 1, video_cfg.get('tight_layout_rect_top_factor', 0.95)]) # Adjust for suptitle
    frame_bgr = mpl_fig_to_opencv_bgr(fig) # Assumes this utility exists
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