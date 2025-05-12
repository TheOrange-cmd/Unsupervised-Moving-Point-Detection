# src/utils/visualization.py
import k3d
import numpy as np
import torch # Your original code uses torch, keeping it for k3d compatibility if needed
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon
import cv2 # For mpl_fig_to_opencv_bgr

# Assuming get_lidar_sweep_data is in data_utils.nuscenes_helper
from src.data_utils.nuscenes_helper import get_lidar_sweep_data
from src.core.depth_image import DepthImage # For type hinting
from src.core.m_detector.base import MDetector, OcclusionResult
from src.core.constants import OcclusionResult


def plot_axes_k3d(T_plotorigin_target:np.ndarray=np.eye(4), length=1.0, name='axes'):
    """
    Creates k3d axes representation.
    Args:
        T_plotorigin_target (np.ndarray) : Homogeneous transformation matrix.
    Return:
        pose_axes (k3d.vectors) : Object representing axes.
    """
    origins = np.stack([T_plotorigin_target[:3,3]]*3)
    vectors = T_plotorigin_target[:3,:3] @ (np.eye(3) * length)
    colors_k3d = [0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF] # Red X, Green Y, Blue Z
    
    # k3d.vectors expects (N,3) origins and (N,3) vectors, where each pair defines a line segment
    # For axes, we draw from origin to end of each axis vector
    plot_origins = np.tile(T_plotorigin_target[:3,3], (3,1))
    plot_vectors = vectors.T # Each row is an axis vector from origin
    
    return k3d.vectors(
        origins=plot_origins,
        vectors=plot_vectors, # These are interpreted as (end_point - origin_point)
        colors=colors_k3d,
        name=name
    )

# --- Matplotlib BEV Plotting Helpers (from the video script in the notebook) ---
def draw_bev_box_on_ax(ax, box: Box, color='red', linewidth=1):
    corners_3d = box.corners()
    points_for_polygon = np.transpose(corners_3d[:2, [0, 1, 2, 3]])
    ax.add_patch(MatplotlibPolygon(points_for_polygon, closed=True, fill=False, edgecolor=color, linewidth=linewidth))
    center_bottom_face = np.mean(corners_3d[:, [0,1,2,3]], axis=1)
    orientation_vec = box.orientation.rotate(np.array([box.wlh[0] / 2.0, 0, 0]))
    arrow_start = center_bottom_face[:2]
    arrow_end = center_bottom_face[:2] + orientation_vec[:2]
    ax.arrow(arrow_start[0], arrow_start[1],
             arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[0],
             head_width=0.5, head_length=0.7, fc=color, ec=color, linewidth=linewidth*0.5)

def mpl_fig_to_opencv_bgr(fig):
    """Converts a Matplotlib figure to an OpenCV BGR image."""
    fig.canvas.draw()
    img_np_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_np_rgb = img_np_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
    return img_np_bgr

# --- K3D Plotting Helpers ---
def plot_axes_k3d(T_plotorigin_target:np.ndarray=np.eye(4), length=1.0, name='axes'):
    # ... (your existing plot_axes_k3d implementation)
    plot_origins = np.tile(T_plotorigin_target[:3,3], (3,1))
    plot_vectors = (T_plotorigin_target[:3,:3] @ (np.eye(3) * length)).T
    colors_k3d = [0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF] # Red X, Green Y, Blue Z
    return k3d.vectors(
        origins=plot_origins,
        vectors=plot_vectors,
        colors=colors_k3d,
        name=name
    )

def plot_predictions_k3d(
    depth_image_with_labels: DepthImage,
    plot_title: Optional[str] = None,
    point_size: float = 0.07,
    color_map_override: Optional[Dict[Any, int]] = None
) -> Optional[k3d.Plot]:
    """
    Visualizes a single DepthImage's points, colored by their pre-computed labels
    stored within the DepthImage's point_info structures.

    Args:
        depth_image_with_labels (DepthImage): The DepthImage object to plot,
                                              assumed to have its point labels already processed
                                              and stored in pt_info['label'].
        plot_title (Optional[str]): Title for the K3D plot. If None, a default is generated.
        point_size (float): Size of the points in the K3D plot.
        color_map_override (Optional[Dict[Any, int]]): Optional custom color map.
                                                       Keys can be OcclusionResult enums or string labels.
                                                       Values are K3D color integers (e.g., 0xRRGGBB).

    Returns:
        Optional[k3d.Plot]: The K3D plot object, or None if DI is invalid.
    """
    if not isinstance(depth_image_with_labels, DepthImage):
        print("Error: Invalid DepthImage object provided.")
        return None

    di_to_plot = depth_image_with_labels

    if plot_title is None:
        plot_title = f"M-Detector Labeled Points for DI @ {di_to_plot.timestamp/1e6:.2f}s"

    plot = k3d.plot(name=plot_title, grid_visible=True, camera_auto_fit=False, menu_visibility=True)
    
    # Add axes for the plotted DI's sensor pose
    plot += plot_axes_k3d(di_to_plot.image_pose_global, length=1.5, name="Sensor Pose (DI to Plot)")

    all_points_global_coords = []
    point_colors_int = [] # K3D expects uint32 for colors array

    # Default color map
    default_color_map = {
        OcclusionResult.OCCLUDING_IMAGE: 0x00FF00,   # Green
        OcclusionResult.OCCLUDED_BY_IMAGE: 0xFF0000, # Red
        OcclusionResult.EMPTY_IN_IMAGE: 0xFFFF00,    # Yellow
        OcclusionResult.UNDETERMINED: 0x808080,      # Grey
        "PENDING_CLASSIFICATION": 0xFF00FF,          # Magenta (initial label before processing)
        "NON_EVENT": 0xCCCCCC,                       # Light Grey (initial label for non-event points)
        # Add more default labels as your M-Detector evolves
    }
    current_color_map = default_color_map.copy()
    if color_map_override:
        current_color_map.update(color_map_override)
    
    fallback_color = 0x000000 # Black for labels not in map

    label_summary = {}

    # Collect all points using the new data structure
    for pixel_key, points_list in di_to_plot.pixel_points.items():
        for pt_info in points_list:
            all_points_global_coords.append(pt_info['global_pt'])
            
            # Retrieve the pre-computed label
            label = pt_info.get('label', OcclusionResult.UNDETERMINED)
            
            point_colors_int.append(current_color_map.get(label, fallback_color))
            
            # For summary
            label_name = label.name if isinstance(label, Enum) else str(label)
            label_summary[label_name] = label_summary.get(label_name, 0) + 1

    print(f"Plotting {len(all_points_global_coords)} points from DI (Timestamp: {di_to_plot.timestamp/1e6:.2f}s).")
    if label_summary:
        print("Label Summary:")
        for name, count in sorted(label_summary.items()):
            print(f"  - {name}: {count}")
    else:
        print("No point labels found in the DepthImage.")

    if all_points_global_coords:
        positions_np = np.array(all_points_global_coords).astype(np.float32)
        colors_np = np.array(point_colors_int).astype(np.uint32)
        
        plot += k3d.points(
            positions=positions_np,
            colors=colors_np, 
            point_size=point_size,
            shader='simple', 
            name="Labeled Points"
        )
    
    # Set camera to view from the sensor's perspective
    cam_eye = di_to_plot.image_pose_global[:3, 3]
    look_at_offset_local = np.array([10.0, 0.0, 0.0, 1.0]) 
    cam_look_at = (di_to_plot.image_pose_global @ look_at_offset_local)[:3]
    cam_up = di_to_plot.image_pose_global[:3, 2] 
    offset_backward_local = np.array([-3.0, 0.0, 1.5, 1.0]) 
    cam_eye_adjusted = (di_to_plot.image_pose_global @ offset_backward_local)[:3]

    plot.camera = [
        cam_eye_adjusted[0], cam_eye_adjusted[1], cam_eye_adjusted[2],
        cam_look_at[0], cam_look_at[1], cam_look_at[2],
        cam_up[0], cam_up[1], cam_up[2]
    ]
    
    return plot

def plot_with_directionality(depth_image, plot_title=None):
    """Visualize with colors indicating which temporal direction contributed most."""
    
    # Custom color map showing temporal contribution
    color_map = {
        "OCCLUDING_PAST": 0x00AA00,    # Dark green (past)
        "OCCLUDING_FUTURE": 0x00FF00,  # Bright green (future)
        "OCCLUDED_PAST": 0xAA0000,     # Dark red (past)
        "OCCLUDED_FUTURE": 0xFF0000,   # Bright red (future)
    }
    
    return plot_predictions_k3d(depth_image, plot_title, color_map_override=color_map)