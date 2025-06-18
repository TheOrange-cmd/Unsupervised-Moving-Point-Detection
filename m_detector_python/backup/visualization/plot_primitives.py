# src/visualization/plot_primitives.py
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from typing import Dict, Any, List, Optional

def render_bev_points_on_ax(
    ax: plt.Axes,
    background_points_global: Optional[np.ndarray], # e.g., all LiDAR points
    points_to_highlight: Dict[str, np.ndarray], # {'label1': points1_global, 'label2': points2_global}
    highlight_configs: Dict[str, Dict[str, Any]], # {'label1': {'color':'red', 's':5, 'legend_label':'Dynamic'}}
    ego_translation_global: np.ndarray,
    ego_rotation_global: Quaternion,
    plot_config: Dict[str, Any], # General settings like bev_range, titles, ego style
    is_right_subplot: bool = False # For minor layout adjustments like y-labels
):
    """
    Renders a Bird's-Eye View (BEV) plot onto a given Matplotlib Axes object.
    Plots background points and multiple sets of highlighted points.
    """
    bev_range = plot_config['bev_range_meters']
    
    # 1. Plot background LiDAR points
    if background_points_global is not None and background_points_global.shape[0] > 0:
        bg_cfg = plot_config['background_points_style']
        ax.scatter(
            background_points_global[:, 0], background_points_global[:, 1],
            s=bg_cfg['s'], 
            color=bg_cfg['color'], 
            alpha=bg_cfg['alpha'], 
            zorder=bg_cfg['zorder'],
            label='_nolegend_'
        )

    # 2. Plot highlighted points
    legend_handles = []
    for label_key, points_data in points_to_highlight.items():
        if points_data is not None and points_data.shape[0] > 0:
            style_cfg = highlight_configs.get(label_key, {}) # Get specific style for this label set
            handle = ax.scatter(
                points_data[:, 0], points_data[:, 1],
                s=style_cfg['s'],
                color=style_cfg['color'],
                alpha=style_cfg['alpha'],
                label=style_cfg['legend_label'] + f" ({points_data.shape[0]})",
                zorder=style_cfg['zorder']
            )
            legend_handles.append(handle)

    # 3. Plot Ego Vehicle
    ego_style = plot_config['ego_vehicle_style']
    ax.plot(
        ego_translation_global[0], ego_translation_global[1],
        marker=ego_style['marker'],
        markersize=ego_style['markersize'],
        color=ego_style['color'],
        label='_nolegend_'
    )
    ego_front_direction = ego_rotation_global.rotate(np.array([ego_style['arrow_length'], 0, 0]))
    ax.arrow(
        ego_translation_global[0], ego_translation_global[1],
        ego_front_direction[0], ego_front_direction[1],
        head_width=ego_style['arrow_head_width'],
        head_length=ego_style['arrow_head_length'],
        fc=ego_style['arrow_fc'],
        ec=ego_style['arrow_ec'],
        zorder=ego_style['arrow_zorder']
    )

    # 4. Aesthetics
    ax.set_xlim(ego_translation_global[0] - bev_range, ego_translation_global[0] + bev_range)
    ax.set_ylim(ego_translation_global[1] - bev_range, ego_translation_global[1] + bev_range)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(plot_config["xlabel"], fontsize=plot_config["label_fontsize"])
    
    if not is_right_subplot:
        ax.set_ylabel(plot_config["ylabel"], fontsize=plot_config["label_fontsize"])
    else:
        ax.set_yticklabels([])
        
    ax.set_title(plot_config['subplot_title'], fontsize=plot_config["title_fontsize"])
    ax.grid(plot_config["grid_visible"], 
            linestyle=plot_config.get("grid_linestyle", '--'), 
            alpha=plot_config["grid_alpha"])
    
    if legend_handles and plot_config["legend_visible"]:
        ax.legend(handles=legend_handles, 
                  loc=plot_config["legend_loc"], 
                  fontsize=plot_config.get("legend_fontsize", 'x-small'))

# Maybe add other primitives here later, e.g.:
# def render_camera_image_on_ax(ax: plt.Axes, image_bgr: np.ndarray, ...):
# def render_boxes_on_ax(ax: plt.Axes, boxes: List[NuScenesDataClassesBox], ...):