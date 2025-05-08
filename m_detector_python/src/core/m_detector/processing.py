# src/core/m_detector/processing.py
# This file is imported into MDetector class

import numpy as np
from typing import Dict, Optional
from ..depth_image import DepthImage
from ..constants import OcclusionResult

def process_and_label_di(self,
                        current_di: DepthImage,
                        historical_di: Optional[DepthImage]) -> Dict:
    """
    Process points in current_di against historical_di (causal processing).
    
    Args:
        self: MDetector instance
        current_di: The DepthImage to process
        historical_di: The historical DepthImage to compare against
        
    Returns:
        dict: Processing statistics
    """
    if not isinstance(current_di, DepthImage):
        raise TypeError("current_di must be a DepthImage object.")

    # Initialize counters
    points_labeled_count = 0
    label_counts = {label: 0 for label in OcclusionResult}
    
    # Handle case where no valid historical DI is available
    if not historical_di or historical_di.timestamp >= current_di.timestamp:
        for v_idx in range(current_di.num_pixels_v):
            for h_idx in range(current_di.num_pixels_h):
                pixel_content = current_di.get_pixel_info(v_idx, h_idx)
                if pixel_content and pixel_content['points']:
                    for pt_info in pixel_content['points']:
                        pt_info['label'] = OcclusionResult.UNDETERMINED
                        label_counts[OcclusionResult.UNDETERMINED] += 1
                        points_labeled_count += 1
        
        return {
            'points_labeled': points_labeled_count,
            'label_counts': label_counts,
            'success': True,
            'timestamp': current_di.timestamp
        }

    # Batch processing for occlusion checks
    all_points_to_label_global = []
    point_info_references = [] # To map results back to pt_info dicts

    for v_idx in range(current_di.num_pixels_v):
        for h_idx in range(current_di.num_pixels_h):
            pixel_content = current_di.get_pixel_info(v_idx, h_idx)
            if pixel_content and pixel_content['points']:
                for pt_info in pixel_content['points']:
                    all_points_to_label_global.append(pt_info['global_pt'])
                    point_info_references.append(pt_info)
    
    if not all_points_to_label_global:
        return {
            'points_labeled': 0,
            'label_counts': label_counts,
            'success': True,
            'timestamp': current_di.timestamp
        }

    points_global_batch_np = np.array(all_points_to_label_global)
    
    # Perform batch occlusion check
    occlusion_results_batch = self.check_occlusion_batch(points_global_batch_np, historical_di)

    for i, initial_label in enumerate(occlusion_results_batch):
        pt_info_ref = point_info_references[i]
        final_label = initial_label

        # If initial label suggests a dynamic event, perform map consistency check
        if initial_label == OcclusionResult.OCCLUDING_IMAGE and self.map_consistency_enabled:
            if self.is_map_consistent(pt_info_ref['global_pt'], current_di.timestamp):
                # Point is consistent with map, so it's likely static despite occlusion
                final_label = OcclusionResult.UNDETERMINED
        
        pt_info_ref['label'] = final_label
        label_counts[final_label] += 1
        points_labeled_count += 1
        
    return {
        'points_labeled': points_labeled_count,
        'label_counts': label_counts,
        'success': True,
        'timestamp': current_di.timestamp
    }

def process_frame(self, frame_index: int) -> Dict:
    """
    Process a single frame by its index.
    This is the main entry point that decides between causal or bidirectional processing.
    
    Args:
        self: MDetector instance
        frame_index: Index of the frame to process
        
    Returns:
        dict: Processing results
    """
    if not self.is_ready_for_processing():
        return {'success': False, 'reason': 'Not enough data for processing'}
        
    if frame_index >= len(self.depth_image_library._images):
        return {'success': False, 'reason': f'Frame index {frame_index} out of range'}
    
    # Get the current depth image to process
    current_di = self.depth_image_library._images[frame_index]
    
    # Choose processing method based on configuration
    if self.use_bidirectional:
        # Use bidirectional processing
        return self.process_and_label_di_bidirectional(frame_index)
    else:
        # Use causal processing (only past frames)
        historical_di = None
        if frame_index > 0:
            historical_di = self.depth_image_library._images[frame_index - 1]
            
        return self.process_and_label_di(current_di, historical_di)