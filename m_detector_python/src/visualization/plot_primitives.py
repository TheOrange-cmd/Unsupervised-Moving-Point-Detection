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
    bev_range = plot_config.get('bev_range_meters', 50)
    
    # 1. Plot background LiDAR points
    if background_points_global is not None and background_points_global.shape[0] > 0:
        bg_cfg = plot_config.get('background_points_style', {})
        ax.scatter(
            background_points_global[:, 0], background_points_global[:, 1],
            s=bg_cfg.get('s', 0.2), 
            color=bg_cfg.get('color', 'lightgrey'), 
            alpha=bg_cfg.get('alpha', 0.5), 
            zorder=bg_cfg.get('zorder', 1),
            label='_nolegend_'
        )

    # 2. Plot highlighted points
    legend_handles = []
    for label_key, points_data in points_to_highlight.items():
        if points_data is not None and points_data.shape[0] > 0:
            style_cfg = highlight_configs.get(label_key, {}) # Get specific style for this label set
            handle = ax.scatter(
                points_data[:, 0], points_data[:, 1],
                s=style_cfg.get('s', 2.0),
                color=style_cfg.get('color', 'red'),
                alpha=style_cfg.get('alpha', 1.0),
                label=style_cfg.get('legend_label', label_key) + f" ({points_data.shape[0]})",
                zorder=style_cfg.get('zorder', 5)
            )
            legend_handles.append(handle)

    # 3. Plot Ego Vehicle
    ego_style = plot_config.get('ego_vehicle_style', {})
    ax.plot(
        ego_translation_global[0], ego_translation_global[1],
        marker=ego_style.get('marker', 'o'),
        markersize=ego_style.get('markersize', 8),
        color=ego_style.get('color', 'blue'),
        label='_nolegend_'
    )
    ego_front_direction = ego_rotation_global.rotate(np.array([ego_style.get('arrow_length', 2.0), 0, 0]))
    ax.arrow(
        ego_translation_global[0], ego_translation_global[1],
        ego_front_direction[0], ego_front_direction[1],
        head_width=ego_style.get('arrow_head_width', 0.8),
        head_length=ego_style.get('arrow_head_length', 1.0),
        fc=ego_style.get('arrow_fc', 'blue'),
        ec=ego_style.get('arrow_ec', 'blue'),
        zorder=ego_style.get('arrow_zorder', 10)
    )

    # 4. Aesthetics
    ax.set_xlim(ego_translation_global[0] - bev_range, ego_translation_global[0] + bev_range)
    ax.set_ylim(ego_translation_global[1] - bev_range, ego_translation_global[1] + bev_range)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(plot_config.get("xlabel", "Global X (m)"), fontsize=plot_config.get("label_fontsize", "small"))
    
    if not is_right_subplot:
        ax.set_ylabel(plot_config.get("ylabel", "Global Y (m)"), fontsize=plot_config.get("label_fontsize", "small"))
    else:
        ax.set_yticklabels([])
        
    ax.set_title(plot_config.get('subplot_title', 'BEV Plot'), fontsize=plot_config.get("title_fontsize", 10))
    ax.grid(plot_config.get("grid_visible", True), 
            linestyle=plot_config.get("grid_linestyle", '--'), 
            alpha=plot_config.get("grid_alpha", 0.4))
    
    if legend_handles and plot_config.get("legend_visible", True):
        ax.legend(handles=legend_handles, 
                  loc=plot_config.get("legend_loc", 'upper right'), 
                  fontsize=plot_config.get("legend_fontsize", 'x-small'))

# Maybe add other primitives here later, e.g.:
# def render_camera_image_on_ax(ax: plt.Axes, image_bgr: np.ndarray, ...):
# def render_boxes_on_ax(ax: plt.Axes, boxes: List[NuScenesDataClassesBox], ...):