import numpy as np
import os
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from .interpolation import interpolate_sequence

def load_nuscenes_instance(nusc, instance_token, min_annotations=2):
    """
    Load all annotations for a specific instance across a scene.
    
    Args:
        nusc: NuScenes instance
        instance_token (str): Instance token
        min_annotations (int): Minimum number of annotations required
        
    Returns:
        dict: Dictionary with instance information and annotations
    """
    instance_record = nusc.get('instance', instance_token)
    
    if instance_record['nbr_annotations'] < min_annotations:
        return None
    
    # Get category information
    category_token = instance_record['category_token']
    category_record = nusc.get('category', category_token)
    category_name = category_record['name']
    
    # Collect all annotations for this instance
    annotations = []
    current_ann_token = instance_record['first_annotation_token']
    
    for _ in range(instance_record['nbr_annotations']):
        ann_record = nusc.get('sample_annotation', current_ann_token)
        annotations.append(ann_record)
        if not ann_record['next']:
            break
        current_ann_token = ann_record['next']
    
    # Sort annotations by timestamp
    annotations.sort(key=lambda ann: nusc.get('sample', ann['sample_token'])['timestamp'])
    
    return {
        'instance_token': instance_token,
        'category_name': category_name,
        'nbr_annotations': instance_record['nbr_annotations'],
        'annotations': annotations
    }

def find_instances_in_scene(nusc, scene_token, min_annotations=1):
    """
    Finds all unique instances in a given scene that have at least a minimum number of annotations.

    Args:
        nusc: NuScenes API instance.
        scene_token (str): The token of the scene to process.
        min_annotations (int): The minimum number of annotations an instance must have
                               to be included.

    Returns:
        list: A list of dictionaries, where each dictionary contains information
              about an instance (token, category, num_annotations).
    """
    scene_record = nusc.get('scene', scene_token)
    
    # Use a set to store unique instance tokens found in this scene
    unique_instance_tokens_in_scene = set()
    
    # Iterate through all samples in the scene
    current_sample_token = scene_record['first_sample_token']
    while current_sample_token:
        sample_record = nusc.get('sample', current_sample_token)
        for annotation_token in sample_record['anns']:
            annotation_record = nusc.get('sample_annotation', annotation_token)
            unique_instance_tokens_in_scene.add(annotation_record['instance_token'])
        
        current_sample_token = sample_record['next'] # Move to the next sample
        if not current_sample_token: # Break if it's the last sample
            break
            
    # Now, filter these instances by min_annotations and gather details
    detailed_instances = []
    for instance_token in unique_instance_tokens_in_scene:
        instance_record = nusc.get('instance', instance_token)
        if instance_record['nbr_annotations'] >= min_annotations:
            detailed_instances.append(instance_token)
            
    return detailed_instances

def annotation_to_box(nusc, annotation):
    """
    Convert a NuScenes annotation to a Box object.
    
    Args:
        nusc: NuScenes instance
        annotation (dict): NuScenes annotation record
        
    Returns:
        Box: NuScenes Box object
    """
    # Get velocity if available
    velocity = nusc.box_velocity(annotation['token'])
    if velocity is None:
        velocity = np.zeros(3)
    
    # Create Box object
    box = Box(
        center=annotation['translation'],
        size=annotation['size'],
        orientation=Quaternion(annotation['rotation']),
        velocity=velocity
    )
    
    # Add name attribute from category
    instance = nusc.get('instance', annotation['instance_token'])
    category = nusc.get('category', instance['category_token'])
    box.name = category['name']
    
    # Add token for reference
    box.token = annotation['token']
    
    return box

def get_annotations_with_timestamps(nusc, instance_data):
    """
    Extract annotation boxes and timestamps for an instance.
    
    Args:
        nusc: NuScenes instance
        instance_data (dict): Instance data from load_nuscenes_instance
        
    Returns:
        tuple: (boxes, timestamps)
    """
    boxes = []
    timestamps = []
    
    for ann in instance_data['annotations']:
        box = annotation_to_box(nusc, ann)
        sample = nusc.get('sample', ann['sample_token'])
        timestamp = sample['timestamp']
        
        boxes.append(box)
        timestamps.append(timestamp)
    
    return boxes, timestamps

def get_lidar_sweeps_for_interval(nusc, start_sample_token, end_sample_token):
    """
    Get all LiDAR sweeps between two sample tokens (inclusive).
    
    Args:
        nusc: NuScenes instance
        start_sample_token (str): Starting sample token
        end_sample_token (str): Ending sample token
        
    Returns:
        list: List of dictionaries with LiDAR sample data information
    """
    # Get sample records
    start_sample = nusc.get('sample', start_sample_token)
    end_sample = nusc.get('sample', end_sample_token)
    
    # Get timestamps to check duration
    start_timestamp = start_sample['timestamp']
    end_timestamp = end_sample['timestamp']
    
    duration_s = (end_timestamp - start_timestamp) / 1e6
    print(f"Scene duration: {duration_s:.2f} seconds")
    
    # Expected number of LiDAR sweeps (Nuscenes LiDAR is at ~20Hz)
    expected_sweeps = int(duration_s * 20)
    print(f"Expected ~{expected_sweeps} LiDAR sweeps at 20Hz")
    
    # Start with the first LIDAR_TOP sample data
    sd_token = start_sample['data']['LIDAR_TOP']
    lidar_sweeps = []
    max_sweeps = 2000  # Increased limit to handle long scenes
    
    count = 0
    while sd_token and count < max_sweeps:
        # Get sample data record
        sd_rec = nusc.get('sample_data', sd_token)
        
        # Store this sweep
        lidar_sweeps.append({
            'token': sd_token,
            'timestamp_us': sd_rec['timestamp'],
            'filename': sd_rec['filename'],
            'ego_pose_token': sd_rec['ego_pose_token'],
            'calibrated_sensor_token': sd_rec['calibrated_sensor_token']
        })
        
        count += 1
        
        # Check if we've reached the end sample
        if sd_rec['timestamp'] >= end_timestamp:
            print(f"Reached end timestamp after {count} sweeps")
            break
            
        # Move to next sample data
        if not sd_rec['next']:
            print(f"Reached end of sequence after {count} sweeps")
            break
            
        sd_token = sd_rec['next']
    
    print(f"Collected {len(lidar_sweeps)} LiDAR sweeps for the interval")
    
    # Verify we're getting all expected sweeps
    if len(lidar_sweeps) < expected_sweeps * 0.9:  # Allow 10% margin
        print(f"WARNING: Got fewer sweeps than expected! Check data access.")
    
    # Sort by timestamp to ensure proper ordering
    lidar_sweeps.sort(key=lambda x: x['timestamp_us'])
    
    return lidar_sweeps

def load_lidar_points_global(nusc, lidar_sd_token, downsample_factor=1):
    """
    Load LiDAR points and transform to global coordinates.
    
    Args:
        nusc: NuScenes instance
        lidar_sd_token (str): LiDAR sample data token
        downsample_factor (int): Factor to downsample points
        
    Returns:
        numpy.ndarray: Points in global coordinates (N x 3)
    """
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

def process_scene_with_multiple_instances(nusc, scene_token, min_annotations=5, target_frequency=10):
    """
    Process all dynamic instances in a scene and prepare them for visualization.
    
    Args:
        nusc: NuScenes instance
        scene_token (str): Scene token
        min_annotations (int): Minimum number of annotations required for an instance
        target_frequency (int): Target frequency in Hz for interpolation
        
    Returns:
        dict: Contains box_sequences and lidar_sweeps
    """
    # Get scene
    scene = nusc.get('scene', scene_token)
    print(f"Processing scene: {scene['name']}")
    
    # Find dynamic instances
    dynamic_instances = find_dynamic_instances_in_scene(nusc, scene_token, min_annotations)
    print(f"Found {len(dynamic_instances)} dynamic instances with at least {min_annotations} annotations")
    
    if not dynamic_instances:
        return None
    
    # Process each instance
    box_sequences = {}
    
    for instance_token in dynamic_instances:
        # Load instance data
        instance_data = load_nuscenes_instance(nusc, instance_token)
        
        if not instance_data or len(instance_data['annotations']) < 2:
            continue
        
        # Get annotation boxes and timestamps
        keyframe_boxes, keyframe_timestamps = get_annotations_with_timestamps(nusc, instance_data)
        
        # Interpolate the sequence
        interpolated_boxes, interpolated_timestamps = interpolate_sequence(
            keyframe_boxes, keyframe_timestamps, target_frequency
        )
        
        # Add to sequences
        instance = nusc.get('instance', instance_token)
        category = nusc.get('category', instance['category_token'])
        
        print(f"  Processed {category['name']} (instance {instance_token[:6]}) with "
              f"{len(interpolated_boxes)} interpolated boxes")
        
        box_sequences[instance_token] = (interpolated_boxes, interpolated_timestamps)
    
    # Get LiDAR sweeps for the entire scene
    first_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    
    lidar_sweeps = get_lidar_sweeps_for_interval(nusc, first_sample_token, last_sample_token)
    print(f"Collected {len(lidar_sweeps)} LiDAR sweeps for the scene")
    
    return {
        'box_sequences': box_sequences,
        'lidar_sweeps': lidar_sweeps
    }