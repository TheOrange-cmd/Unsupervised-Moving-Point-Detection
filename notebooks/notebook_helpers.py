import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- NuScenes Specific Imports 
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

# --- mpy_detector Import (for DynObjLabel enum) ---
# This assumes mpy_detector is installable/importable in the notebook environment
import mpy_detector as mdet

log = logging.getLogger(__name__)

# --- Constants ---
DYNAMIC_CATEGORIES = [
    'vehicle.car', 'vehicle.truck', 'vehicle.bus', 'vehicle.trailer',
    'vehicle.construction', 'vehicle.motorcycle', 'vehicle.bicycle',
    'human.pedestrian', 'human.police_officer', 'human.construction_worker',
    'animal'
]

# Define a color map for DynObjLabel
# Ensure this order/length matches your DynObjLabel enum values
LABEL_COLORS = [
    'gray',      # 0: INVALID
    'blue',      # 1: STATIC
    'red',       # 2: APPEARING
    'orange',    # 3: OCCLUDING
    'purple',    # 4: SELF
    'green',     # 5: UNCERTAIN
    'yellow',    # 6: INVALID
]
# Create a ListedColormap based on the number of known labels
# Adjust max_label_value if your enum has more/fewer values
try:
    max_label_value = mdet.DynObjLabel.get_max_value()
except AttributeError: # Fallback if mdet or get_max_value isn't fully defined
    max_label_value = 7
custom_cmap = ListedColormap(LABEL_COLORS[:max_label_value + 1])


# --- Helper Functions (Adapted from your test_nuscenes.py) ---

def get_sensor_global_pose(nusc, lidar_data_token, current_seq_id=-1, debug_seq_id=-1):
    """
    Calculates the global pose (Rotation Matrix, Translation Vector) of the LIDAR sensor.
    """
    try:
        lidar_data = nusc.get('sample_data', lidar_data_token)
        cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        ep_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        sens_to_ego_trans = np.array(cs_record['translation'])
        sens_to_ego_rot = Quaternion(cs_record['rotation'])
        sens_to_ego_matrix = np.eye(4)
        sens_to_ego_matrix[:3, :3] = sens_to_ego_rot.rotation_matrix
        sens_to_ego_matrix[:3, 3] = sens_to_ego_trans

        ego_to_glob_trans = np.array(ep_record['translation'])
        ego_to_glob_rot = Quaternion(ep_record['rotation'])
        ego_to_glob_matrix = np.eye(4)
        ego_to_glob_matrix[:3, :3] = ego_to_glob_rot.rotation_matrix
        ego_to_glob_matrix[:3, 3] = ego_to_glob_trans

        sens_to_world_matrix = ego_to_glob_matrix @ sens_to_ego_matrix
        rotation_matrix = sens_to_world_matrix[:3, :3].astype(np.float64)
        position_vector = sens_to_world_matrix[:3, 3].astype(np.float64)

        if current_seq_id == debug_seq_id and debug_seq_id != -1:
            log.info(f"--- get_sensor_global_pose DEBUG (Seq ID: {current_seq_id}, Token: {lidar_data_token}) ---")
            # Add more detailed logging here if needed for the notebook
            log.info(f"  FINAL position_vector: {position_vector.tolist()}")
            log.info(f"--- END DEBUG ---")

        return rotation_matrix, position_vector
    except Exception as e:
        log.error(f"Error getting sensor global pose for token {lidar_data_token}: {e}", exc_info=True)
        raise

def get_gt_dynamic_mask(nusc, sample_token, lidar_token, points_sensor_frame):
    """
    Generates a boolean mask indicating which points belong to dynamic objects.
    """
    if points_sensor_frame is None or points_sensor_frame.shape[0] == 0:
        return np.array([], dtype=bool)

    try:
        lidar_data = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_rec = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        dynamic_point_mask = np.zeros(points_sensor_frame.shape[0], dtype=bool)
        annotation_tokens = nusc.get('sample', sample_token)['anns']

        for ann_token in annotation_tokens:
            annotation = nusc.get('sample_annotation', ann_token)
            category_name = annotation['category_name']
            is_dynamic = any(category_name.startswith(dyn_cat) for dyn_cat in DYNAMIC_CATEGORIES)

            if is_dynamic:
                box = Box(annotation['translation'], annotation['size'], Quaternion(annotation['rotation']))
                box.translate(-np.array(pose_rec['translation']))
                box.rotate(Quaternion(pose_rec['rotation']).inverse)
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)
                indices_in_box = points_in_box(box, points_sensor_frame[:, :3].T, wlh_factor=1.0)
                dynamic_point_mask[indices_in_box] = True
        return dynamic_point_mask
    except Exception as e:
        log.error(f"Error generating GT dynamic mask for sample {sample_token}: {e}", exc_info=True)
        return np.zeros(points_sensor_frame.shape[0], dtype=bool)

def align_filter_labels(processed_info_list, num_input_points, invalid_label_value):
    """
    Aligns labels from processed_info_list to an array of size num_input_points.
    """
    if not processed_info_list and num_input_points > 0:
        log.warning(f"Filter returned 0 points for non-empty input ({num_input_points}). Assuming all INVALID.")
        return np.full(num_input_points, invalid_label_value, dtype=int)
    if not processed_info_list and num_input_points == 0:
        return np.array([], dtype=int)

    num_points_processed = len(processed_info_list)
    filter_labels = np.full(num_input_points, invalid_label_value, dtype=int) # Default to invalid

    if num_points_processed == num_input_points and not hasattr(processed_info_list[0], 'original_index'):
        # Assuming direct correspondence if no original_index and counts match
        log.info("Label alignment: Processed count matches input, assuming direct correspondence.")
        filter_labels = np.array([info.label for info in processed_info_list], dtype=int)
    elif hasattr(processed_info_list[0], 'original_index'):
        log.info(f"Label alignment: Using 'original_index' for {num_points_processed} processed points -> {num_input_points} input slots.")
        valid_indices_mapped = 0
        for info in processed_info_list:
            if 0 <= info.original_index < num_input_points:
                filter_labels[info.original_index] = info.label
                valid_indices_mapped +=1
            else:
                log.warning(f"  Invalid original_index {info.original_index} received (max input: {num_input_points-1}).")
        if valid_indices_mapped != num_points_processed:
            log.warning(f"  Alignment mismatch: {valid_indices_mapped} labels mapped vs {num_points_processed} processed.")
    else:
        log.warning(f"Label alignment: Processed ({num_points_processed}) != Input ({num_input_points}) and no 'original_index'. Some labels might be lost or misaligned.")
        # Attempt a naive mapping if counts differ but no original_index
        # This is risky and might be incorrect
        min_len = min(num_points_processed, num_input_points)
        for i in range(min_len):
            filter_labels[i] = processed_info_list[i].label

    return filter_labels


# --- New Visualization Helper ---
def plot_point_clouds_2d(plot_data_list, title_prefix="", x_lims=None, y_lims=None, point_size=1):
    """
    Plots multiple point clouds side-by-side in 2D (Bird's Eye View: X vs Y).

    Args:
        plot_data_list (list of dict): Each dict should have:
            'points': NxK numpy array (K>=2, using first two cols for X, Y).
            'labels': N numpy array of integer labels, or None for single color.
            'title': Subplot title.
        title_prefix (str): Prefix for the main figure title.
        x_lims (tuple, optional): (min, max) for x-axis.
        y_lims (tuple, optional): (min, max) for y-axis.
        point_size (int): Size of the points in the scatter plot.
    """
    num_plots = len(plot_data_list)
    if num_plots == 0:
        log.warning("No data provided to plot_point_clouds_2d.")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes] # Make it iterable

    fig.suptitle(f"{title_prefix} (Bird's Eye View: X-Y)", fontsize=16)

    for i, data in enumerate(plot_data_list):
        ax = axes[i]
        points = data.get('points')
        labels = data.get('labels') # Integer labels
        plot_title = data.get('title', f'Plot {i+1}')

        if points is None or points.shape[0] == 0:
            ax.text(0.5, 0.5, 'No points to display', ha='center', va='center')
            ax.set_title(plot_title)
            ax.set_xlabel("X (sensor frame)")
            ax.set_ylabel("Y (sensor frame)")
            if x_lims: ax.set_xlim(x_lims)
            if y_lims: ax.set_ylim(y_lims)
            continue

        x_coords = points[:, 0]
        y_coords = points[:, 1]

        if labels is not None:
            # Ensure labels are integers for cmap indexing
            scatter = ax.scatter(x_coords, y_coords, c=labels.astype(int), cmap=custom_cmap, s=point_size,
                                 vmin=0, vmax=max_label_value) # Use vmin/vmax for consistent coloring
            # Create a legend for labels
            handles = []
            unique_labels_present = np.unique(labels)
            for label_val in range(max_label_value + 1): # Iterate all possible labels for consistent legend
                if label_val in unique_labels_present:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=custom_cmap(label_val / max_label_value if max_label_value > 0 else 0), # Normalize
                                              markersize=5,
                                              label=f'{mdet.DynObjLabel(label_val).name if hasattr(mdet, "DynObjLabel") else f"Label {label_val}"} ({label_val})'))
            if handles:
                ax.legend(handles=handles, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        else:
            scatter = ax.scatter(x_coords, y_coords, s=point_size) # Default color

        ax.set_title(plot_title)
        ax.set_xlabel("X (sensor frame)")
        ax.set_ylabel("Y (sensor frame)")
        ax.set_aspect('equal', adjustable='box')
        if x_lims: ax.set_xlim(x_lims)
        if y_lims: ax.set_ylim(y_lims)
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust layout to make space for suptitle and legend
    # plt.show() # Typically called from the notebook after this function returns