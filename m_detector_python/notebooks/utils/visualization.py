import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.animation as animation
from .nuscenes_utils import load_lidar_points_global
import k3d

def plot_box_bev(ax, box, color='skyblue', edgecolor='royalblue', alpha=0.7, show_orientation=True, zorder=10):
    """
    Plot a NuScenes Box object in Bird's Eye View.
    
    Args:
        ax: Matplotlib axis
        box: NuScenes Box object
        color (str): Face color
        edgecolor (str): Edge color
        alpha (float): Transparency
        show_orientation (bool): Whether to show orientation arrow
        zorder (int): Z-order for plotting
        
    Returns:
        tuple: (polygon_patch, orientation_line or None)
    """
    # Get box corners for bottom face (BEV)
    corners_bev = box.bottom_corners()[:2, :].T  # (4, 2) array of (x,y) corners
    
    # Create and add polygon
    poly = Polygon(corners_bev, closed=True, 
                   facecolor=color, edgecolor=edgecolor, 
                   alpha=alpha, linewidth=1.5, zorder=zorder)
    ax.add_patch(poly)
    
    # Add orientation arrow (front direction)
    orientation_line = None
    if show_orientation:
        # Local x-axis (front) vector scaled by half length, transformed to global
        front_vec_local = np.array([box.wlh[1] / 2.0, 0, 0])  # [length/2, 0, 0]
        front_vec_global = box.orientation.rotation_matrix @ front_vec_local
        
        orientation_line, = ax.plot(
            [box.center[0], box.center[0] + front_vec_global[0]],
            [box.center[1], box.center[1] + front_vec_global[1]],
            color='red', linewidth=2, zorder=zorder+1
        )
    
    return poly, orientation_line

def create_animation_with_lidar(boxes, timestamps, nusc, lidar_sweeps, 
                                interval_ms=100, figsize=(10, 10), point_downsample=20):
    """
    Create a matplotlib animation with interpolated boxes and LiDAR context.
    
    Args:
        boxes (list): List of Box objects
        timestamps (list): List of timestamps (microseconds)
        nusc: NuScenes instance
        lidar_sweeps (list): List of LiDAR sample data info dicts
        interval_ms (int): Animation interval in milliseconds
        figsize (tuple): Figure size
        point_downsample (int): Downsample factor for LiDAR points
        
    Returns:
        HTML: Animation for display in notebook
    """
    # Determine plot bounds
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for box in boxes:
        corners_xy = box.bottom_corners()[:2, :]
        min_x = min(min_x, np.min(corners_xy[0, :]))
        max_x = max(max_x, np.max(corners_xy[0, :]))
        min_y = min(min_y, np.min(corners_xy[1, :]))
        max_y = max(max_y, np.max(corners_xy[1, :]))
    
    # Add padding
    padding = 15.0
    plot_xlim = (min_x - padding, max_x + padding)
    plot_ylim = (min_y - padding, max_y + padding)
    
    # Create figure and initial objects
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create initial polygon and orientation line
    box_poly, orientation_line = plot_box_bev(ax, boxes[0])
    
    # Create scatter plot for LiDAR points (will be updated)
    lidar_scatter = ax.scatter([], [], s=1.5, c='dimgray', alpha=0.6, zorder=1)
    
    # Text annotation
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=12)
    
    # LiDAR cache to avoid reloading
    lidar_cache = {}
    
    # Helper function to find closest LiDAR sweep
    def get_closest_lidar_sweep(target_ts):
        return min(lidar_sweeps, key=lambda sd: abs(sd['timestamp_us'] - target_ts))
    
    # Animation update function
    def update(frame_num):
        # Get current box and timestamp
        current_box = boxes[frame_num]
        current_ts = timestamps[frame_num]
        
        # Update box polygon
        corners_bev = current_box.bottom_corners()[:2, :].T
        box_poly.set_xy(corners_bev)
        
        # Update orientation line
        front_vec_local = np.array([current_box.wlh[1] / 2.0, 0, 0])
        front_vec_global = current_box.orientation.rotation_matrix @ front_vec_local
        orientation_line.set_data(
            [current_box.center[0], current_box.center[0] + front_vec_global[0]],
            [current_box.center[1], current_box.center[1] + front_vec_global[1]]
        )
        
        # Get closest LiDAR sweep
        closest_lidar = get_closest_lidar_sweep(current_ts)
        lidar_token = closest_lidar['token']
        
        # Load LiDAR points if not in cache
        if lidar_token not in lidar_cache:
            points_global = load_lidar_points_global(nusc, lidar_token, downsample_factor=point_downsample)
            lidar_cache[lidar_token] = points_global[:, :2]  # Keep only x,y for BEV
        
        # Update scatter plot with points
        lidar_scatter.set_offsets(lidar_cache[lidar_token])
        
        # Update time text
        relative_time_s = (current_ts - timestamps[0]) / 1e6
        time_text.set_text(f'Frame: {frame_num}\nTime: {relative_time_s:+.3f}s\n'
                          f'LiDAR TS: {closest_lidar["timestamp_us"]/1e6:.3f}s')
        
        # Set axis properties
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect('equal', adjustable='box')
        
        return box_poly, orientation_line, lidar_scatter, time_text
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(boxes), 
                       interval=interval_ms, blit=True)
    
    # Close the figure to prevent display
    plt.close(fig)
    
    # Return animation as HTML
    return HTML(anim.to_jshtml())

def create_k3d_box_visualization(box, points_sensor_frame=None, colormap=None):
    """
    Create a k3d visualization of a box and optionally point cloud.
    
    Args:
        box: NuScenes Box object (in sensor frame)
        points_sensor_frame: Optional point cloud in sensor frame
        colormap: NuScenes colormap for categories
        
    Returns:
        k3d.plot: Plot object
    """
    # Define mesh indices for box faces
    BOX_MESH_INDICES = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [2, 3, 7], [2, 7, 6],  # Back face
        [1, 2, 6], [1, 6, 5],  # Right face
        [0, 3, 7], [0, 7, 4]   # Left face
    ], dtype=np.uint32)
    
    # Create k3d plot
    plot = k3d.plot(camera_auto_fit=False, grid_visible=False, menu_visibility=True, axes_helper=0.1)
    
    # Get box corners
    corners = box.corners().T.astype(np.float32)  # 8x3 array of vertices
    
    # Get color for the box
    if colormap is not None and hasattr(box, 'name'):
        try:
            category_color_rgb = colormap[box.name]  # This is (R,G,B) in 0-255
            box_color = (category_color_rgb[0] << 16) + (category_color_rgb[1] << 8) + category_color_rgb[2]
        except KeyError:
            box_color = 0xff0000  # Default to red
    else:
        box_color = 0xff0000  # Default to red
    
    # Add box as mesh
    plot += k3d.mesh(
        vertices=corners,
        indices=BOX_MESH_INDICES,
        color=box_color,
        opacity=0.5,
        name=f'Annotation: {box.name if hasattr(box, "name") else "Unknown"}'
    )
    
    # Add point cloud if provided
    if points_sensor_frame is not None and points_sensor_frame.shape[0] > 0:
        # Downsample if necessary
        points_to_plot = points_sensor_frame
        if points_sensor_frame.shape[0] > 70000:
            choice_indices = np.random.choice(points_sensor_frame.shape[0], 70000, replace=False)
            points_to_plot = points_sensor_frame[choice_indices, :]
            
        plot += k3d.points(
            positions=points_to_plot.astype(np.float32),
            point_size=0.05,
            color=0xaaaaaa,  # Light grey
            name='LiDAR Points'
        )
    
    # Set default camera position
    plot.camera = [14.32, -28.31, 5.94, 10.16, -0.15, -3.74, 0.03, 0.27, 0.95]
    
    return plot

def create_animation_with_lidar_synchronized(boxes, timestamps, nusc, lidar_sweeps, 
                                          interval_ms=100, figsize=(10, 10), point_downsample=20):
    """
    Create a matplotlib animation with boxes synchronized to LiDAR sweeps.
    
    This version ensures each box matches exactly one LiDAR sweep.
    
    Args:
        boxes (list): List of Box objects
        timestamps (list): List of timestamps (microseconds)
        nusc: NuScenes instance
        lidar_sweeps (list): List of LiDAR sample data info dicts
        interval_ms (int): Animation interval in milliseconds
        figsize (tuple): Figure size
        point_downsample (int): Downsample factor for LiDAR points
        
    Returns:
        HTML: Animation for display in notebook
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    # First, let's print the timestamp ranges to understand what we're working with
    box_time_range = (min(timestamps), max(timestamps))
    lidar_time_range = (lidar_sweeps[0]['timestamp_us'], lidar_sweeps[-1]['timestamp_us'])
    
    print(f"Box timestamps range: {box_time_range[0]/1e6:.2f}s to {box_time_range[1]/1e6:.2f}s")
    print(f"LiDAR timestamps range: {lidar_time_range[0]/1e6:.2f}s to {lidar_time_range[1]/1e6:.2f}s")
    
    # Create synchronized pairs of (box, lidar_sweep)
    # Each LiDAR sweep should have a matching interpolated box
    synchronized_frames = []
    
    # For each LiDAR sweep, find the closest box by timestamp
    for lidar_sweep in lidar_sweeps:
        lidar_ts = lidar_sweep['timestamp_us']
        
        # Skip LiDAR frames outside our box timestamp range
        if lidar_ts < box_time_range[0] or lidar_ts > box_time_range[1]:
            continue
            
        # Find the closest box by timestamp
        closest_box_idx = min(range(len(timestamps)), 
                             key=lambda i: abs(timestamps[i] - lidar_ts))
        
        # Only add if the timestamps are reasonably close (within 0.05s)
        time_diff = abs(timestamps[closest_box_idx] - lidar_ts) / 1e6
        if time_diff <= 0.05:  # 50ms threshold, adjust if needed
            synchronized_frames.append({
                'box': boxes[closest_box_idx],
                'lidar_sweep': lidar_sweep,
                'box_ts': timestamps[closest_box_idx],
                'lidar_ts': lidar_ts
            })
    
    print(f"Created {len(synchronized_frames)} synchronized frames")
    
    if not synchronized_frames:
        print("Error: No synchronized frames could be created.")
        return None
    
    # Function to load LiDAR points in global frame
    def load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1):
        lidar_sd_rec = nusc.get('sample_data', lidar_sd_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_sd_rec['filename'])
        
        if not os.path.exists(pcl_path):
            print(f"LiDAR file not found: {pcl_path}")
            return np.zeros((0, 3))
        
        # Load points (sensor frame)
        pc = LidarPointCloud.from_file(pcl_path)
        points_sensor_frame = pc.points[:3, :]  # Shape (3, N)
        
        # Get sensor pose relative to ego
        cs_rec = nusc.get('calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
        sensor_to_ego_tf = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']))
        
        # Get ego pose relative to global
        ego_pose_rec = nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])
        ego_to_global_tf = transform_matrix(ego_pose_rec['translation'], Quaternion(ego_pose_rec['rotation']))
        
        # Transform points: sensor -> ego -> global
        points_sensor_homogeneous = np.vstack((points_sensor_frame, np.ones(points_sensor_frame.shape[1])))
        points_global_homogeneous = ego_to_global_tf @ sensor_to_ego_tf @ points_sensor_homogeneous
        points_global = points_global_homogeneous[:3, :]
        
        # Downsample if requested
        if downsample_factor > 1:
            points_global = points_global[:, ::downsample_factor]
        
        return points_global.T  # Return as (N, 3)
    
    # Plot box in BEV
    def plot_box_bev(ax, box, color='skyblue', edgecolor='royalblue', alpha=0.7, zorder=10):
        # Get box corners for bottom face (BEV)
        corners_bev = box.bottom_corners()[:2, :].T  # (4, 2) array of (x,y) corners
        
        # Create and add polygon
        poly = Polygon(corners_bev, closed=True, 
                      facecolor=color, edgecolor=edgecolor, 
                      alpha=alpha, linewidth=1.5, zorder=zorder)
        ax.add_patch(poly)
        
        # Add orientation arrow (front direction)
        front_vec_local = np.array([box.wlh[1] / 2.0, 0, 0])  # [length/2, 0, 0]
        front_vec_global = box.orientation.rotation_matrix @ front_vec_local
        
        orientation_line, = ax.plot(
            [box.center[0], box.center[0] + front_vec_global[0]],
            [box.center[1], box.center[1] + front_vec_global[1]],
            color='red', linewidth=2, zorder=zorder+1
        )
        
        return poly, orientation_line
    
    # Determine plot bounds
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for frame in synchronized_frames:
        box = frame['box']
        corners_xy = box.bottom_corners()[:2, :]
        min_x = min(min_x, np.min(corners_xy[0, :]))
        max_x = max(max_x, np.max(corners_xy[0, :]))
        min_y = min(min_y, np.min(corners_xy[1, :]))
        max_y = max(max_y, np.max(corners_xy[1, :]))
    
    # Add padding
    padding = 15.0
    plot_xlim = (min_x - padding, max_x + padding)
    plot_ylim = (min_y - padding, max_y + padding)
    
    # Create figure and initial objects
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create initial box and orientation line
    initial_box = synchronized_frames[0]['box']
    box_poly, orientation_line = plot_box_bev(ax, initial_box)
    
    # Create scatter plot for LiDAR points
    lidar_scatter = ax.scatter([], [], s=1.5, c='dimgray', alpha=0.6, zorder=1)
    
    # Text annotation
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=12)
    
    # LiDAR cache
    lidar_cache = {}
    
    # Animation update function
    def update(frame_num):
        # Get current frame data
        frame_data = synchronized_frames[frame_num]
        current_box = frame_data['box']
        lidar_sweep = frame_data['lidar_sweep']
        lidar_token = lidar_sweep['token']
        
        # Update box polygon
        corners_bev = current_box.bottom_corners()[:2, :].T
        box_poly.set_xy(corners_bev)
        
        # Update orientation line
        front_vec_local = np.array([current_box.wlh[1] / 2.0, 0, 0])
        front_vec_global = current_box.orientation.rotation_matrix @ front_vec_local
        orientation_line.set_data(
            [current_box.center[0], current_box.center[0] + front_vec_global[0]],
            [current_box.center[1], current_box.center[1] + front_vec_global[1]]
        )
        
        # Load LiDAR points if not in cache
        if lidar_token not in lidar_cache:
            points_global = load_lidar_points_global(nusc, lidar_token, downsample_factor=point_downsample)
            lidar_cache[lidar_token] = points_global[:, :2]  # Keep only x,y for BEV
        
        # Update scatter plot with points
        lidar_scatter.set_offsets(lidar_cache[lidar_token])
        
        # Update time text
        relative_time_s = (frame_data['lidar_ts'] - synchronized_frames[0]['lidar_ts']) / 1e6
        box_lidar_diff_ms = (frame_data['box_ts'] - frame_data['lidar_ts']) / 1e3  # milliseconds
        
        time_text.set_text(f'Frame: {frame_num}/{len(synchronized_frames)-1}\n'
                          f'Time: {relative_time_s:.3f}s\n'
                          f'Box-LiDAR diff: {box_lidar_diff_ms:.1f}ms')
        
        # Set axis properties
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Global X (meters)")
        ax.set_ylabel("Global Y (meters)")
        ax.set_title(f"BEV Visualization (Synchronized Box and LiDAR)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return box_poly, orientation_line, lidar_scatter, time_text
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(synchronized_frames), 
                        interval=interval_ms, blit=True)
    
    # Close the figure to prevent display
    plt.close(fig)
    
    # Return animation as HTML
    return HTML(anim.to_jshtml())

def create_multi_box_animation(box_sequences, nusc, lidar_sweeps, 
                             interval_ms=100, figsize=(10, 10), point_downsample=20):
    """
    Create an animation with multiple annotation boxes synchronized to LiDAR sweeps.
    
    Args:
        box_sequences (dict): Dictionary where keys are instance tokens and values are
                             tuples of (boxes, timestamps) for each instance
        nusc: NuScenes instance
        lidar_sweeps (list): List of LiDAR sample data info dicts
        interval_ms (int): Animation interval in milliseconds
        figsize (tuple): Figure size
        point_downsample (int): Downsample factor for LiDAR points
        
    Returns:
        HTML: Animation for display in notebook
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    # First, get the overall time range across all box sequences
    min_timestamp = float('inf')
    max_timestamp = float('-inf')
    
    for instance_token, (boxes, timestamps) in box_sequences.items():
        min_timestamp = min(min_timestamp, min(timestamps))
        max_timestamp = max(max_timestamp, max(timestamps))
    
    box_time_range = (min_timestamp, max_timestamp)
    lidar_time_range = (lidar_sweeps[0]['timestamp_us'], lidar_sweeps[-1]['timestamp_us'])
    
    print(f"All boxes timestamps range: {box_time_range[0]/1e6:.2f}s to {box_time_range[1]/1e6:.2f}s")
    print(f"LiDAR timestamps range: {lidar_time_range[0]/1e6:.2f}s to {lidar_time_range[1]/1e6:.2f}s")
    
    # Get LiDAR sweeps within the box time range
    valid_lidar_sweeps = [
        sweep for sweep in lidar_sweeps
        if sweep['timestamp_us'] >= box_time_range[0] and sweep['timestamp_us'] <= box_time_range[1]
    ]
    
    print(f"Found {len(valid_lidar_sweeps)} LiDAR sweeps within the box time range")
    
    if not valid_lidar_sweeps:
        print("Error: No valid LiDAR sweeps for the box time range")
        return None
    
    # Function to find the closest box for an instance at a given timestamp
    def find_closest_box(boxes, timestamps, target_ts):
        closest_idx = min(range(len(timestamps)), 
                         key=lambda i: abs(timestamps[i] - target_ts))
        return boxes[closest_idx], timestamps[closest_idx]
    
    # Create synchronized frames
    synchronized_frames = []
    
    for lidar_sweep in valid_lidar_sweeps:
        lidar_ts = lidar_sweep['timestamp_us']
        
        # Find closest box for each instance
        frame_boxes = {}
        
        for instance_token, (boxes, timestamps) in box_sequences.items():
            # Skip if the lidar timestamp is outside this instance's range
            if lidar_ts < min(timestamps) or lidar_ts > max(timestamps):
                continue
                
            closest_box, box_ts = find_closest_box(boxes, timestamps, lidar_ts)
            
            # Only include if timestamps are reasonably close
            time_diff = abs(box_ts - lidar_ts) / 1e6
            if time_diff <= 0.05:  # 50ms threshold
                frame_boxes[instance_token] = {
                    'box': closest_box,
                    'timestamp': box_ts
                }
        
        # Only add frames where we have at least one box
        if frame_boxes:
            synchronized_frames.append({
                'lidar_sweep': lidar_sweep,
                'lidar_ts': lidar_ts,
                'boxes': frame_boxes
            })
    
    print(f"Created {len(synchronized_frames)} synchronized frames")
    
    if not synchronized_frames:
        print("Error: No synchronized frames could be created")
        return None
    
    # Function to load LiDAR points
    def load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1):
        lidar_sd_rec = nusc.get('sample_data', lidar_sd_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_sd_rec['filename'])
        
        if not os.path.exists(pcl_path):
            print(f"LiDAR file not found: {pcl_path}")
            return np.zeros((0, 3))
        
        # Load points (sensor frame)
        pc = LidarPointCloud.from_file(pcl_path)
        points_sensor_frame = pc.points[:3, :]  # Shape (3, N)
        
        # Get sensor pose relative to ego
        cs_rec = nusc.get('calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
        sensor_to_ego_tf = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']))
        
        # Get ego pose relative to global
        ego_pose_rec = nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])
        ego_to_global_tf = transform_matrix(ego_pose_rec['translation'], Quaternion(ego_pose_rec['rotation']))
        
        # Transform points: sensor -> ego -> global
        points_sensor_homogeneous = np.vstack((points_sensor_frame, np.ones(points_sensor_frame.shape[1])))
        points_global_homogeneous = ego_to_global_tf @ sensor_to_ego_tf @ points_sensor_homogeneous
        points_global = points_global_homogeneous[:3, :]
        
        # Downsample if requested
        if downsample_factor > 1:
            points_global = points_global[:, ::downsample_factor]
        
        return points_global.T  # Return as (N, 3)
    
    # Determine plot bounds
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for frame in synchronized_frames:
        for instance_token, box_data in frame['boxes'].items():
            box = box_data['box']
            corners_xy = box.bottom_corners()[:2, :]
            min_x = min(min_x, np.min(corners_xy[0, :]))
            max_x = max(max_x, np.max(corners_xy[0, :]))
            min_y = min(min_y, np.min(corners_xy[1, :]))
            max_y = max(max_y, np.max(corners_xy[1, :]))
    
    # Add padding
    padding = 15.0
    plot_xlim = (min_x - padding, max_x + padding)
    plot_ylim = (min_y - padding, max_y + padding)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors for different instances
    colors = plt.cm.tab10.colors  # 10 distinct colors
    instance_tokens = list(box_sequences.keys())
    instance_colors = {token: colors[i % len(colors)] for i, token in enumerate(instance_tokens)}
    
    # Create initial box polygons and orientation lines
    box_polys = {}
    orientation_lines = {}
    
    first_frame_boxes = synchronized_frames[0]['boxes']
    for instance_token, box_data in first_frame_boxes.items():
        color = instance_colors[instance_token]
        edgecolor = tuple(0.7*np.array(color))  # Darker edge
        poly = Polygon(box_data['box'].bottom_corners()[:2, :].T, closed=True,
                      facecolor=color, edgecolor=edgecolor,
                      alpha=0.6, linewidth=1.5, zorder=10)
        ax.add_patch(poly)
        box_polys[instance_token] = poly
        
        # Orientation line
        front_vec_local = np.array([box_data['box'].wlh[1] / 2.0, 0, 0])
        front_vec_global = box_data['box'].orientation.rotation_matrix @ front_vec_local
        line, = ax.plot(
            [box_data['box'].center[0], box_data['box'].center[0] + front_vec_global[0]],
            [box_data['box'].center[1], box_data['box'].center[1] + front_vec_global[1]],
            color=edgecolor, linewidth=2, zorder=11
        )
        orientation_lines[instance_token] = line
    
    # Create scatter plot for LiDAR points
    lidar_scatter = ax.scatter([], [], s=1.5, c='dimgray', alpha=0.6, zorder=1)
    
    # Text annotation
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=12)
    
    # Legend for instance colors
    legend_elements = []
    for instance_token in instance_tokens:
        if instance_token in first_frame_boxes:
            instance = nusc.get('instance', instance_token)
            category = nusc.get('category', instance['category_token'])
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=instance_colors[instance_token],
                                            markersize=10, label=category['name']))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')
    
    # LiDAR cache
    lidar_cache = {}
    
    # Animation update function
    def update(frame_num):
        artists = []
        
        # Get current frame data
        frame_data = synchronized_frames[frame_num]
        lidar_sweep = frame_data['lidar_sweep']
        lidar_ts = frame_data['lidar_ts']
        lidar_token = lidar_sweep['token']
        
        # Update each box
        for instance_token, box_data in frame_data['boxes'].items():
            box = box_data['box']
            box_ts = box_data['timestamp']
            
            # Update box polygon if it exists (it might not be in every frame)
            if instance_token in box_polys:
                poly = box_polys[instance_token]
                poly.set_xy(box.bottom_corners()[:2, :].T)
                artists.append(poly)
                
                # Update orientation line
                line = orientation_lines[instance_token]
                front_vec_local = np.array([box.wlh[1] / 2.0, 0, 0])
                front_vec_global = box.orientation.rotation_matrix @ front_vec_local
                line.set_data(
                    [box.center[0], box.center[0] + front_vec_global[0]],
                    [box.center[1], box.center[1] + front_vec_global[1]]
                )
                artists.append(line)
            else:
                # This instance wasn't in the first frame but appears later
                # Create new polygon and line
                color = instance_colors[instance_token]
                edgecolor = tuple(0.7*np.array(color))
                poly = Polygon(box.bottom_corners()[:2, :].T, closed=True,
                              facecolor=color, edgecolor=edgecolor,
                              alpha=0.6, linewidth=1.5, zorder=10)
                ax.add_patch(poly)
                box_polys[instance_token] = poly
                artists.append(poly)
                
                front_vec_local = np.array([box.wlh[1] / 2.0, 0, 0])
                front_vec_global = box.orientation.rotation_matrix @ front_vec_local
                line, = ax.plot(
                    [box.center[0], box.center[0] + front_vec_global[0]],
                    [box.center[1], box.center[1] + front_vec_global[1]],
                    color=edgecolor, linewidth=2, zorder=11
                )
                orientation_lines[instance_token] = line
                artists.append(line)
        
        # Handle instances that were in previous frames but not this one
        for instance_token in list(box_polys.keys()):
            if instance_token not in frame_data['boxes']:
                # Make the box invisible by setting empty vertices
                box_polys[instance_token].set_xy(np.zeros((0, 2)))
                orientation_lines[instance_token].set_data([], [])
        
        # Load LiDAR points if not in cache
        if lidar_token not in lidar_cache:
            points_global = load_lidar_points_global(nusc, lidar_token, downsample_factor=point_downsample)
            lidar_cache[lidar_token] = points_global[:, :2]  # Keep only x,y for BEV
        
        # Update scatter plot with points
        lidar_scatter.set_offsets(lidar_cache[lidar_token])
        artists.append(lidar_scatter)
        
        # Update time text
        relative_time_s = (lidar_ts - synchronized_frames[0]['lidar_ts']) / 1e6
        num_boxes = len(frame_data['boxes'])
        
        time_text.set_text(f'Frame: {frame_num}/{len(synchronized_frames)-1}\n'
                          f'Time: {relative_time_s:.3f}s\n'
                          f'Boxes visible: {num_boxes}')
        artists.append(time_text)
        
        # Set axis properties
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Global X (meters)")
        ax.set_ylabel("Global Y (meters)")
        ax.set_title(f"Multi-Object BEV Visualization")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return artists
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(synchronized_frames), 
                        interval=interval_ms, blit=True)
    
    # Close the figure to prevent display
    plt.close(fig)
    
    # Return animation as HTML
    return HTML(anim.to_jshtml())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def interpolate_boxes_to_lidar_sweeps(nusc, instance_token, lidar_sweeps):
    """
    Interpolate boxes to match exactly with LiDAR sweep timestamps.
    
    This function:
    1. Gets all annotations for an instance
    2. Gets all LiDAR sweeps between the first and last annotation
    3. Interpolates boxes to exactly match LiDAR sweep timestamps
    
    Args:
        nusc: NuScenes instance
        instance_token (str): Instance token
        lidar_sweeps (list): List of LiDAR sample data dicts with timestamps
        
    Returns:
        tuple: (interpolated_boxes, interpolated_timestamps, original_boxes, original_timestamps)
    """
    from nuscenes.utils.data_classes import Box
    import numpy as np
    from pyquaternion import Quaternion
    
    # Get all annotations for the instance
    annotations = []
    ann_token = nusc.get('instance', instance_token)['first_annotation_token']
    
    while ann_token:
        ann_rec = nusc.get('sample_annotation', ann_token)
        sample = nusc.get('sample', ann_rec['sample_token'])
        
        annotations.append({
            'token': ann_token,
            'translation': ann_rec['translation'],
            'size': ann_rec['size'],
            'rotation': ann_rec['rotation'],
            'timestamp': sample['timestamp'],
            'prev': ann_rec['prev'],
            'next': ann_rec['next']
        })
        
        ann_token = ann_rec['next']
    
    if len(annotations) < 2:
        print(f"Not enough annotations for instance {instance_token}")
        return None, None, None, None
    
    # Sort annotations by timestamp
    annotations.sort(key=lambda x: x['timestamp'])
    
    # Create original boxes and timestamps
    original_boxes = []
    original_timestamps = []
    
    for ann in annotations:
        # Convert annotation to Box object
        box = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation']),
            velocity=np.zeros(3),  # No velocity in annotations
            name=None,
            token=ann['token']
        )
        
        original_boxes.append(box)
        original_timestamps.append(ann['timestamp'])
    
    # Identify LiDAR sweeps between first and last annotation timestamp
    first_ts = original_timestamps[0]
    last_ts = original_timestamps[-1]
    
    # Filter LiDAR sweeps to those within annotation time range
    relevant_sweeps = [
        sweep for sweep in lidar_sweeps
        if first_ts <= sweep['timestamp_us'] <= last_ts
    ]
    
    if not relevant_sweeps:
        print(f"No LiDAR sweeps found between annotations for instance {instance_token}")
        return None, None, original_boxes, original_timestamps
    
    # Sort sweeps by timestamp
    relevant_sweeps.sort(key=lambda x: x['timestamp_us'])
    
    # Get sweep timestamps
    sweep_timestamps = [sweep['timestamp_us'] for sweep in relevant_sweeps]
    
    # Now interpolate boxes to exactly match LiDAR sweep timestamps
    interpolated_boxes = []
    
    for target_ts in sweep_timestamps:
        # Find the two annotations that bracket this timestamp
        idx_next = next((i for i, ts in enumerate(original_timestamps) if ts > target_ts), len(original_timestamps))
        idx_prev = idx_next - 1
        
        # Handle edge cases
        if idx_prev < 0:
            # Before first annotation, use the first
            interpolated_boxes.append(original_boxes[0])
            continue
        if idx_next >= len(original_timestamps):
            # After last annotation, use the last
            interpolated_boxes.append(original_boxes[-1])
            continue
            
        # Get the bracketing timestamps and boxes
        ts_prev = original_timestamps[idx_prev]
        ts_next = original_timestamps[idx_next]
        box_prev = original_boxes[idx_prev]
        box_next = original_boxes[idx_next]
        
        # Interpolation factor (0 at prev, 1 at next)
        t = (target_ts - ts_prev) / (ts_next - ts_prev)
        
        # Interpolate
        interp_box = interpolate_box(box_prev, box_next, t)
        interpolated_boxes.append(interp_box)
    
    return interpolated_boxes, sweep_timestamps, original_boxes, original_timestamps


def interpolate_box(box1, box2, t):
    """
    Interpolate between two boxes with factor t (0 to 1).
    
    Args:
        box1: First box
        box2: Second box
        t: Interpolation factor (0 = first box, 1 = second box)
        
    Returns:
        Box: Interpolated box
    """
    from nuscenes.utils.data_classes import Box
    from scipy.spatial.transform import Slerp
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    from pyquaternion import Quaternion
    
    # Interpolate center position (linear)
    center = box1.center * (1 - t) + box2.center * t
    
    # Interpolate size (linear)
    size = box1.wlh * (1 - t) + box2.wlh * t
    
    # Interpolate orientation (spherical)
    key_rots = R.from_quat(np.stack([
        box1.orientation.elements[[1, 2, 3, 0]],  # Convert to scipy format (x,y,z,w)
        box2.orientation.elements[[1, 2, 3, 0]]
    ]))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([t])[0]
    
    # Convert back to pyquaternion (w,x,y,z)
    q = np.roll(interp_rot.as_quat(), 1)  # Convert from (x,y,z,w) to (w,x,y,z)
    orientation = Quaternion(q)  # Create new Quaternion directly
    
    # Create interpolated box
    return Box(
        center=center,
        size=size,
        orientation=orientation,
        velocity=np.zeros(3),
        name=None,
        token=None
    )


def create_exact_synchronized_animation(nusc, instance_token, lidar_sweeps, 
                                      interval_ms=100, figsize=(10, 10), point_downsample=20):
    """
    Create an animation with boxes precisely synchronized to LiDAR sweeps.
    
    Args:
        nusc: NuScenes instance
        instance_token (str): Instance token
        lidar_sweeps (list): List of LiDAR sample data info dicts
        interval_ms (int): Animation interval in milliseconds
        figsize (tuple): Figure size
        point_downsample (int): Downsample factor for LiDAR points
        
    Returns:
        HTML: Animation for display in notebook
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    # Interpolate boxes to match LiDAR sweeps exactly
    boxes, timestamps, orig_boxes, orig_timestamps = interpolate_boxes_to_lidar_sweeps(
        nusc, instance_token, lidar_sweeps
    )
    
    if not boxes or not timestamps:
        print("Failed to generate interpolated boxes")
        return None
    
    print(f"Created {len(boxes)} interpolated boxes exactly matching LiDAR sweeps")
    print(f"Original annotations: {len(orig_boxes)}")
    
    # Determine plot bounds
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for box in boxes:
        corners_xy = box.bottom_corners()[:2, :]
        min_x = min(min_x, np.min(corners_xy[0, :]))
        max_x = max(max_x, np.max(corners_xy[0, :]))
        min_y = min(min_y, np.min(corners_xy[1, :]))
        max_y = max(max_y, np.max(corners_xy[1, :]))
    
    # Add padding
    padding = 15.0
    plot_xlim = (min_x - padding, max_x + padding)
    plot_ylim = (min_y - padding, max_y + padding)
    
    # Create figure and initial objects
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create initial polygon and orientation line
    initial_box = boxes[0]
    initial_corners_bev = initial_box.bottom_corners()[:2, :].T
    box_poly = Polygon(initial_corners_bev, closed=True, 
                      edgecolor='royalblue', facecolor='skyblue', 
                      alpha=0.7, linewidth=1.5, zorder=10)
    ax.add_patch(box_poly)
    
    front_vec_local = np.array([initial_box.wlh[1] / 2.0, 0, 0])
    front_vec_global = initial_box.orientation.rotation_matrix @ front_vec_local
    orientation_line, = ax.plot(
        [initial_box.center[0], initial_box.center[0] + front_vec_global[0]],
        [initial_box.center[1], initial_box.center[1] + front_vec_global[1]],
        color='red', linewidth=2, zorder=11
    )
    
    # Create scatter plot for LiDAR points
    lidar_scatter = ax.scatter([], [], s=1.5, c='dimgray', alpha=0.6, zorder=1)
    
    # Create scatter for original keyframes
    keyframe_scatter = ax.scatter([], [], s=40, marker='*', c='gold', 
                                edgecolor='black', linewidth=1, zorder=12, alpha=0)
    
    # Text annotation
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=15)
    
    # Create a mapping from timestamp to lidar token
    lidar_token_map = {sweep['timestamp_us']: sweep['token'] for sweep in lidar_sweeps}
    
    # Create a mapping from original timestamps to box centers (for keyframe markers)
    keyframe_centers = {ts: box.center[:2] for ts, box in zip(orig_timestamps, orig_boxes)}
    
    # Function to load LiDAR points
    def load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1):
        lidar_sd_rec = nusc.get('sample_data', lidar_sd_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_sd_rec['filename'])
        
        if not os.path.exists(pcl_path):
            print(f"LiDAR file not found: {pcl_path}")
            return np.zeros((0, 3))
        
        # Load points (sensor frame)
        pc = LidarPointCloud.from_file(pcl_path)
        points_sensor_frame = pc.points[:3, :]  # Shape (3, N)
        
        # Get sensor pose relative to ego
        cs_rec = nusc.get('calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
        sensor_to_ego_tf = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']))
        
        # Get ego pose relative to global
        ego_pose_rec = nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])
        ego_to_global_tf = transform_matrix(ego_pose_rec['translation'], Quaternion(ego_pose_rec['rotation']))
        
        # Transform points: sensor -> ego -> global
        points_sensor_homogeneous = np.vstack((points_sensor_frame, np.ones(points_sensor_frame.shape[1])))
        points_global_homogeneous = ego_to_global_tf @ sensor_to_ego_tf @ points_sensor_homogeneous
        points_global = points_global_homogeneous[:3, :]
        
        # Downsample if requested
        if downsample_factor > 1:
            points_global = points_global[:, ::downsample_factor]
        
        return points_global.T  # Return as (N, 3)
    
    # LiDAR cache
    lidar_cache = {}
    
    # Animation update function
    def update(frame_num):
        # Get current box and timestamp
        current_box = boxes[frame_num]
        current_ts = timestamps[frame_num]
        
        # Update box polygon
        corners_bev = current_box.bottom_corners()[:2, :].T
        box_poly.set_xy(corners_bev)
        
        # Update orientation line
        front_vec_local = np.array([current_box.wlh[1] / 2.0, 0, 0])
        front_vec_global = current_box.orientation.rotation_matrix @ front_vec_local
        orientation_line.set_data(
            [current_box.center[0], current_box.center[0] + front_vec_global[0]],
            [current_box.center[1], current_box.center[1] + front_vec_global[1]]
        )
        
        # Get LiDAR token for this timestamp
        lidar_token = lidar_token_map.get(current_ts)
        
        # Load LiDAR points if not in cache
        if lidar_token and lidar_token not in lidar_cache:
            points_global = load_lidar_points_global(nusc, lidar_token, downsample_factor=point_downsample)
            lidar_cache[lidar_token] = points_global[:, :2]  # Keep only x,y for BEV
        
        # Update scatter plot with points
        if lidar_token and lidar_token in lidar_cache:
            lidar_scatter.set_offsets(lidar_cache[lidar_token])
        else:
            lidar_scatter.set_offsets(np.zeros((0, 2)))
        
        # Check if current frame corresponds to an original keyframe
        is_keyframe = current_ts in orig_timestamps
        
        # Show keyframe markers
        if is_keyframe:
            keyframe_scatter.set_offsets([current_box.center[:2]])
            keyframe_scatter.set_alpha(1.0)
        else:
            keyframe_scatter.set_alpha(0)
        
        # Update time text
        relative_time_s = (current_ts - timestamps[0]) / 1e6
        time_text.set_text(f'Frame: {frame_num}/{len(timestamps)-1}\n'
                          f'Time: {relative_time_s:.3f}s\n'
                          f'{"KEYFRAME" if is_keyframe else "Interpolated"}')
        
        # Set axis properties
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Global X (meters)")
        ax.set_ylabel("Global Y (meters)")
        ax.set_title(f"Exactly Synchronized Box and LiDAR")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return box_poly, orientation_line, lidar_scatter, keyframe_scatter, time_text
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(timestamps), 
                        interval=interval_ms, blit=True)
    
    # Close the figure to prevent display
    plt.close(fig)
    
    # Return animation as HTML
    return HTML(anim.to_jshtml())

def create_multi_box_exact_synchronized_animation(nusc, instance_tokens, lidar_sweeps, 
                                               interval_ms=100, figsize=(8, 8), point_downsample=20,
    save_path=None,       # Path to save the animation (e.g., "animation.mp4" or "animation.gif")
    save_writer=None,     # Writer to use (e.g., 'ffmpeg', 'pillow')
    save_fps=None,        # FPS for the saved animation. If None, calculated from interval_ms.
    save_dpi=150          # DPI for the saved animation.
    ):
    """
    Create an animation with multiple boxes exactly synchronized to LiDAR sweeps.
    Can optionally save the animation to a file.
    
    Args:
        nusc: NuScenes instance
        instance_tokens (list): List of instance tokens to animate
        lidar_sweeps (list): List of LiDAR sample data info dicts
        interval_ms (int): Animation interval in milliseconds for display
        figsize (tuple): Figure size
        point_downsample (int): Downsample factor for LiDAR points
        save_path (str, optional): File path to save the animation. If None, returns HTML.
        save_writer (str, optional): Writer for saving (e.g., 'ffmpeg', 'pillow').
                                     Matplotlib will try to infer if not set and save_path is given.
        save_fps (int, optional): FPS for the saved file. Defaults to 1000/interval_ms.
        save_dpi (int, optional): DPI for the saved file.
        
    Returns:
        HTML: Animation for display in notebook (if save_path is None).
        None: If animation is saved to a file.
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    # First, sort LiDAR sweeps by timestamp
    lidar_sweeps = sorted(lidar_sweeps, key=lambda x: x['timestamp_us'])
    lidar_timestamps = [sweep['timestamp_us'] for sweep in lidar_sweeps]
    
    # Process each instance to get interpolated boxes for all LiDAR timestamps
    instance_boxes = {}
    
    # Track the min and max timestamps across all instances
    min_ts_all = float('inf')
    max_ts_all = float('-inf')
    
    for instance_token in instance_tokens:
        boxes, timestamps, orig_boxes, orig_timestamps = interpolate_boxes_to_lidar_sweeps(
            nusc, instance_token, lidar_sweeps
        )
        
        if boxes and timestamps:
            instance = nusc.get('instance', instance_token)
            category = nusc.get('category', instance['category_token'])
            
            print(f"Instance {instance_token[:6]} ({category['name']}): "
                 f"{len(boxes)} interpolated boxes from {len(orig_boxes)} keyframes")
            
            # Update overall min and max timestamps
            if timestamps:
                min_ts_all = min(min_ts_all, min(timestamps))
                max_ts_all = max(max_ts_all, max(timestamps))
            
            instance_boxes[instance_token] = {
                'boxes': boxes,
                'timestamps': timestamps,
                'orig_boxes': orig_boxes,
                'orig_timestamps': orig_timestamps,
                'category': category['name']
            }
    
    if not instance_boxes:
        print("No valid instance data found")
        return None
    
    # Get all LiDAR timestamps between min_ts_all and max_ts_all
    animation_timestamps = [ts for ts in lidar_timestamps 
                          if min_ts_all <= ts <= max_ts_all]
    
    print(f"Animation will use {len(animation_timestamps)} timestamps from {min_ts_all/1e6:.2f}s to {max_ts_all/1e6:.2f}s")
    
    # Create a mapping from timestamp to frame index for each instance
    instance_frame_indices = {}
    for instance_token, data in instance_boxes.items():
        # Create mapping: timestamp -> index in the instance's timestamp array
        ts_to_idx = {ts: i for i, ts in enumerate(data['timestamps'])}
        instance_frame_indices[instance_token] = ts_to_idx
    
    # Create a mapping from timestamp to lidar token
    lidar_token_map = {sweep['timestamp_us']: sweep['token'] for sweep in lidar_sweeps}
    
    # Determine plot bounds
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for instance_data in instance_boxes.values():
        for box in instance_data['boxes']:
            corners_xy = box.bottom_corners()[:2, :]
            min_x = min(min_x, np.min(corners_xy[0, :]))
            max_x = max(max_x, np.max(corners_xy[0, :]))
            min_y = min(min_y, np.min(corners_xy[1, :]))
            max_y = max(max_y, np.max(corners_xy[1, :]))
    
    # Add padding
    padding = 15.0
    plot_xlim = (min_x - padding, max_x + padding)
    plot_ylim = (min_y - padding, max_y + padding)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors for different instances
    colors = plt.cm.tab10.colors  # 10 distinct colors
    instance_colors = {token: colors[i % len(colors)] for i, token in enumerate(instance_tokens)}
    
    # Create initial box polygons and orientation lines
    box_polys = {}
    orientation_lines = {}
    keyframe_markers = {}
    
    # Initialize box visualization for each instance
    for instance_token, data in instance_boxes.items():
        if animation_timestamps:
            # Find the box for the first animation timestamp
            first_ts = animation_timestamps[0]
            first_idx = instance_frame_indices[instance_token].get(first_ts)
            
            if first_idx is not None:
                initial_box = data['boxes'][first_idx]
                
                # Create box polygon
                color = instance_colors[instance_token]
                edgecolor = tuple(0.7*np.array(color))  # Darker edge
                
                poly = Polygon(initial_box.bottom_corners()[:2, :].T, closed=True,
                              facecolor=color, edgecolor=edgecolor,
                              alpha=0.6, linewidth=1.5, zorder=10)
                ax.add_patch(poly)
                box_polys[instance_token] = poly
                
                # Create orientation line
                front_vec_local = np.array([initial_box.wlh[1] / 2.0, 0, 0])
                front_vec_global = initial_box.orientation.rotation_matrix @ front_vec_local
                line, = ax.plot(
                    [initial_box.center[0], initial_box.center[0] + front_vec_global[0]],
                    [initial_box.center[1], initial_box.center[1] + front_vec_global[1]],
                    color=edgecolor, linewidth=2, zorder=11
                )
                orientation_lines[instance_token] = line
                
                # Create keyframe marker - explicitly set color as a string or rgba tuple
                marker = ax.scatter([], [], s=40, marker='*', color=color, 
                                  edgecolor='black', linewidth=1, zorder=12, alpha=0)
                keyframe_markers[instance_token] = marker
    
    # Create scatter plot for LiDAR points - explicitly set color
    lidar_scatter = ax.scatter([], [], s=1.5, color='dimgray', alpha=0.6, zorder=1)
    
    # Text annotation
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7), zorder=15)
    
    # Add legend
    legend_elements = []
    for instance_token, data in instance_boxes.items():
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                         markerfacecolor=instance_colors[instance_token],
                                         markersize=10, label=data['category']))
    
    # if legend_elements:
    #     ax.legend(handles=legend_elements, loc='upper right')
    
    # Function to load LiDAR points
    def load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1):
        lidar_sd_rec = nusc.get('sample_data', lidar_sd_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_sd_rec['filename'])
        
        if not os.path.exists(pcl_path):
            print(f"LiDAR file not found: {pcl_path}")
            return np.zeros((0, 3))
        
        # Load points (sensor frame)
        pc = LidarPointCloud.from_file(pcl_path)
        points_sensor_frame = pc.points[:3, :]  # Shape (3, N)
        
        # Get sensor pose relative to ego
        cs_rec = nusc.get('calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
        sensor_to_ego_tf = transform_matrix(cs_rec['translation'], Quaternion(cs_rec['rotation']))
        
        # Get ego pose relative to global
        ego_pose_rec = nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])
        ego_to_global_tf = transform_matrix(ego_pose_rec['translation'], Quaternion(ego_pose_rec['rotation']))
        
        # Transform points: sensor -> ego -> global
        points_sensor_homogeneous = np.vstack((points_sensor_frame, np.ones(points_sensor_frame.shape[1])))
        points_global_homogeneous = ego_to_global_tf @ sensor_to_ego_tf @ points_sensor_homogeneous
        points_global = points_global_homogeneous[:3, :]
        
        # Downsample if requested
        if downsample_factor > 1:
            points_global = points_global[:, ::downsample_factor]
        
        return points_global.T  # Return as (N, 3)
    
    # LiDAR cache
    lidar_cache = {}
    
    # Animation update function
    def update(frame_idx):
        artists = []
        
        # Get the current timestamp
        current_ts = animation_timestamps[frame_idx]
        
        # Update each instance's box
        visible_instances = 0
        
        for instance_token, data in instance_boxes.items():
            # Get box index for this timestamp
            idx = instance_frame_indices[instance_token].get(current_ts)
            
            if idx is not None:
                box = data['boxes'][idx]
                visible_instances += 1
                
                # Update box polygon
                poly = box_polys[instance_token]
                poly.set_xy(box.bottom_corners()[:2, :].T)
                poly.set_visible(True)
                artists.append(poly)
                
                # Update orientation line
                line = orientation_lines[instance_token]
                front_vec_local = np.array([box.wlh[1] / 2.0, 0, 0])
                front_vec_global = box.orientation.rotation_matrix @ front_vec_local
                line.set_data(
                    [box.center[0], box.center[0] + front_vec_global[0]],
                    [box.center[1], box.center[1] + front_vec_global[1]]
                )
                line.set_visible(True)
                artists.append(line)
                
                # Check if this is a keyframe
                is_keyframe = current_ts in data['orig_timestamps']
                marker = keyframe_markers[instance_token]
                
                if is_keyframe:
                    marker.set_offsets([box.center[:2]])
                    marker.set_alpha(1.0)
                else:
                    marker.set_alpha(0)
                    
                artists.append(marker)
            else:
                # Hide this instance's visuals if no data for current timestamp
                if instance_token in box_polys:
                    box_polys[instance_token].set_visible(False)
                    orientation_lines[instance_token].set_visible(False)
                    keyframe_markers[instance_token].set_alpha(0)
        
        # Get LiDAR token for this timestamp
        lidar_token = lidar_token_map.get(current_ts)
        
        # Load LiDAR points if not in cache
        if lidar_token and lidar_token not in lidar_cache:
            points_global = load_lidar_points_global(nusc, lidar_token, downsample_factor=point_downsample)
            lidar_cache[lidar_token] = points_global[:, :2]  # Keep only x,y for BEV
        
        # Update scatter plot with points
        if lidar_token and lidar_token in lidar_cache:
            lidar_scatter.set_offsets(lidar_cache[lidar_token])
        else:
            lidar_scatter.set_offsets(np.zeros((0, 2)))
            
        artists.append(lidar_scatter)
        
        # Update time text
        relative_time_s = (current_ts - animation_timestamps[0]) / 1e6
        
        time_text.set_text(f'Frame: {frame_idx}/{len(animation_timestamps)-1}\n'
                          f'Time: {relative_time_s:.3f}s\n'
                          f'Visible objects: {visible_instances}/{len(instance_boxes)}')
        artists.append(time_text)
        
        # Set axis properties
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Global X (meters)")
        ax.set_ylabel("Global Y (meters)")
        ax.set_title(f"Nuscenes scene 1: Interpolated annotations")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return artists
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(animation_timestamps), 
                        interval=interval_ms, blit=False)
    

    # Default behavior: return HTML for notebook display
    html_output = HTML(anim.to_jshtml())
    plt.close(fig) # Close the figure to prevent static display before HTML
    return html_output