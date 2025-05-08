import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

def interpolate_box(box1: Box, box2: Box, ratio: float) -> Box:
    """
    Interpolates between two Box objects.
    - Linearly interpolates center, size (wlh), and velocity.
    - Spherically interpolates orientation (quaternion).

    Args:
        box1 (Box): The starting box.
        box2 (Box): The ending box.
        ratio (float): The interpolation ratio (0.0 for box1, 1.0 for box2).

    Returns:
        Box: A new Box object representing the interpolated state.
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("Interpolation ratio must be between 0.0 and 1.0.")

    # LERP for center
    interp_center = box1.center * (1 - ratio) + box2.center * ratio
    
    # LERP for size (wlh)
    interp_wlh = np.array(box1.wlh) * (1 - ratio) + np.array(box2.wlh) * ratio
    
    # SLERP for orientation
    interp_orientation = Quaternion.slerp(box1.orientation, box2.orientation, ratio)
    
    # LERP for velocity
    vel1 = np.array(box1.velocity)
    vel2 = np.array(box2.velocity)
    interp_velocity = vel1 * (1 - ratio) + vel2 * ratio
    
    # Create the new interpolated Box object
    interpolated_b = Box(center=interp_center.tolist(),
                         size=interp_wlh.tolist(), 
                         orientation=interp_orientation,
                         velocity=interp_velocity.tolist())
    
    # Carry over name if present
    if hasattr(box1, 'name'):
        interpolated_b.name = box1.name
    
    return interpolated_b

def interpolate_boxes_between_keyframes(box1, box2, timestamp1, timestamp2, intermediate_timestamps):
    """
    Interpolate boxes at specified intermediate timestamps between two keyframes.
    
    Args:
        box1 (Box): Box at first keyframe
        box2 (Box): Box at second keyframe
        timestamp1 (int): Timestamp of first keyframe (microseconds)
        timestamp2 (int): Timestamp of second keyframe (microseconds)
        intermediate_timestamps (list): List of intermediate timestamps for interpolation
        
    Returns:
        list: List of interpolated boxes at the requested timestamps
    """
    interpolated_boxes = []
    total_time_diff = timestamp2 - timestamp1
    
    if total_time_diff <= 0:
        raise ValueError("Second timestamp must be greater than first timestamp")
    
    for ts in intermediate_timestamps:
        if ts <= timestamp1 or ts >= timestamp2:
            continue  # Skip timestamps outside the interval
            
        ratio = (ts - timestamp1) / total_time_diff
        interpolated_box = interpolate_box(box1, box2, ratio)
        interpolated_boxes.append(interpolated_box)
        
    return interpolated_boxes

def interpolate_sequence(annotation_boxes, timestamps, target_frequency=20):
    """
    Generate a sequence of interpolated boxes at the target frequency.
    
    Args:
        annotation_boxes (list): List of Box objects at keyframes
        timestamps (list): List of timestamps (microseconds) for each keyframe box
        target_frequency (int): Target frequency in Hz for the interpolated sequence
        
    Returns:
        tuple: (interpolated_boxes, interpolated_timestamps)
    """
    if len(annotation_boxes) < 2 or len(annotation_boxes) != len(timestamps):
        raise ValueError("Need at least 2 boxes with matching timestamps")
        
    target_interval_us = int(1e6 / target_frequency)  # microseconds between frames
    
    all_boxes = []
    all_timestamps = []
    
    # Process each keyframe pair
    for i in range(len(annotation_boxes) - 1):
        start_box = annotation_boxes[i]
        end_box = annotation_boxes[i+1]
        start_ts = timestamps[i]
        end_ts = timestamps[i+1]
        
        # Add the start box
        all_boxes.append(start_box)
        all_timestamps.append(start_ts)
        
        # Generate intermediate timestamps
        current_ts = start_ts + target_interval_us
        intermediate_timestamps = []
        
        while current_ts < end_ts:
            intermediate_timestamps.append(current_ts)
            current_ts += target_interval_us
            
        # Get interpolated boxes
        if intermediate_timestamps:
            interp_boxes = interpolate_boxes_between_keyframes(
                start_box, end_box, start_ts, end_ts, intermediate_timestamps
            )
            
            all_boxes.extend(interp_boxes)
            all_timestamps.extend(intermediate_timestamps)
    
    # Add the final box
    all_boxes.append(annotation_boxes[-1])
    all_timestamps.append(timestamps[-1])
    
    return all_boxes, all_timestamps