import numpy as np
import os
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix

def transform_box_to_sensor_frame(box, sample_token, nusc):
    """
    Transform a box from global to sensor frame.
    
    Args:
        box: NuScenes Box object in global frame
        sample_token: Sample token for the frame
        nusc: NuScenes instance
        
    Returns:
        Box: Transformed box in sensor frame
    """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_record = nusc.get('sample_data', lidar_token)
    
    # Get ego pose at the time of the LiDAR sweep
    ego_pose = nusc.get('ego_pose', lidar_record['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    # Get sensor pose relative to ego
    cs_record = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
    sensor_translation = np.array(cs_record['translation'])
    sensor_rotation = Quaternion(cs_record['rotation'])
    
    # Make a copy of the box to avoid modifying the original
    box_sensor = Box(
        center=box.center.copy(),
        size=box.wlh.copy(),
        orientation=Quaternion(box.orientation.elements.copy()),
        velocity=box.velocity.copy() if box.velocity is not None else np.zeros(3)
    )
    
    # Global to ego
    box_sensor.translate(-ego_translation)
    box_sensor.rotate(ego_rotation.inverse)
    
    # Ego to sensor
    box_sensor.translate(-sensor_translation)
    box_sensor.rotate(sensor_rotation.inverse)
    
    # Copy name and token if available
    if hasattr(box, 'name'):
        box_sensor.name = box.name
    if hasattr(box, 'token'):
        box_sensor.token = box.token
    
    return box_sensor

def process_instance_sequence(nusc, instance_token, target_frequency=20):
    """
    Process a full sequence for an instance with interpolation.
    
    Args:
        nusc: NuScenes instance
        instance_token: Instance token
        target_frequency: Desired frequency in Hz for interpolated frames
        
    Returns:
        dict: Dictionary containing the processed sequence
    """
    # Load instance data
    instance_data = load_nuscenes_instance(nusc, instance_token)
    
    if not instance_data or len(instance_data['annotations']) < 2:
        print(f"Not enough annotations for instance {instance_token}")
        return None
    
    # Get annotation boxes and timestamps
    keyframe_boxes, keyframe_timestamps = get_annotations_with_timestamps(nusc, instance_data)
    
    # Calculate interpolated sequence
    interpolated_boxes, interpolated_timestamps = interpolate_sequence(
        keyframe_boxes, keyframe_timestamps, target_frequency
    )
    
    # Get LiDAR sweeps for the time interval
    first_sample_token = instance_data['annotations'][0]['sample_token']
    last_sample_token = instance_data['annotations'][-1]['sample_token']
    lidar_sweeps = get_lidar_sweeps_for_interval(nusc, first_sample_token, last_sample_token)
    
    return {
        'instance_token': instance_token,
        'category_name': instance_data['category_name'],
        'keyframe_boxes': keyframe_boxes,
        'keyframe_timestamps': keyframe_timestamps,
        'interpolated_boxes': interpolated_boxes,
        'interpolated_timestamps': interpolated_timestamps,
        'lidar_sweeps': lidar_sweeps
    }

def transform_sequence_to_sensor_frames(sequence_data, nusc):
    """
    Transform all boxes in a sequence to their respective sensor frames.
    
    Args:
        sequence_data: Output from process_instance_sequence
        nusc: NuScenes instance
        
    Returns:
        dict: Updated sequence data with sensor-frame boxes
    """
    # Add a new field for sensor-frame boxes
    sequence_data['sensor_frame_boxes'] = []
    
    # For each interpolated box, find the closest sample and transform
    for box, timestamp in zip(sequence_data['interpolated_boxes'], sequence_data['interpolated_timestamps']):
        # Find the closest sample (keyframe)
        closest_sample_token = None
        min_time_diff = float('inf')
        
        for ann in sequence_data['annotations']:
            sample_token = ann['sample_token']
            sample_timestamp = nusc.get('sample', sample_token)['timestamp']
            time_diff = abs(sample_timestamp - timestamp)
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_sample_token = sample_token
        
        # Transform the box to sensor frame
        if closest_sample_token:
            sensor_box = transform_box_to_sensor_frame(box, closest_sample_token, nusc)
            sequence_data['sensor_frame_boxes'].append(sensor_box)
    
    return sequence_data
