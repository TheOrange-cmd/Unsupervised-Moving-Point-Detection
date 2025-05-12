# src/core/m_detector/processing.py
# This file is imported into MDetector class

import numpy as np
from typing import Dict, Optional, Any
from ..depth_image import DepthImage
from ..constants import OcclusionResult
from tqdm import tqdm

def extract_mdetector_points(depth_image_output_from_mdetector: Optional[Any]) -> Dict[str, np.ndarray]: # Allow None
    """
    Extracts points by MDetector's label from its processed depth_image representation.
    Returns a dictionary like {'dynamic': points, 'occluded': points, ...}.
    """
    mdet_points = {
        'dynamic': [],
        'occluded_by_mdet': [],
        'undetermined_by_mdet': []
    }
    
    # Check if the input is not None AND has the expected structure
    if depth_image_output_from_mdetector is not None and \
       hasattr(depth_image_output_from_mdetector, 'pixel_points') and \
       isinstance(depth_image_output_from_mdetector.pixel_points, dict):
        for _, points_list_in_pixel in depth_image_output_from_mdetector.pixel_points.items():
            for pt_info in points_list_in_pixel:
                label = pt_info.get('label') 
                point_global_coords = pt_info.get('global_pt')

                if point_global_coords is None: continue

                if label == OcclusionResult.OCCLUDING_IMAGE:
                    mdet_points['dynamic'].append(point_global_coords)
                elif label == OcclusionResult.OCCLUDED_BY_IMAGE:
                    mdet_points['occluded_by_mdet'].append(point_global_coords)
                elif label == OcclusionResult.UNDETERMINED:
                    mdet_points['undetermined_by_mdet'].append(point_global_coords)
    elif depth_image_output_from_mdetector is not None: # Input was given, but not structured as expected
        # Only print warning if a non-None, but invalid, object was passed
        tqdm.write("Warning: MDetector output (depth_image.pixel_points) not found or not a dict.")

    return {k: (np.array(v) if v else np.empty((0,3))) for k, v in mdet_points.items()}

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


def _process_causal_di(self, di_to_process_idx: int) -> Dict:
    current_di = self.depth_image_library._images[di_to_process_idx]
    historical_di = None
    if di_to_process_idx > 0:
        historical_di = self.depth_image_library._images[di_to_process_idx - 1]
    
    # Calls the core logic of your original process_and_label_di
    # This is just a sketch of refactoring
    result = self.actual_causal_processing_logic(current_di, historical_di) 
    result['processed_frame_timestamp'] = current_di.timestamp
    result['frame_index'] = di_to_process_idx
    return result

def _process_bidirectional_di(self, di_to_process_idx: int) -> Dict:
    # This is essentially your existing process_and_label_di_bidirectional
    # It already takes center_index, which is di_to_process_idx here.
    result = self.process_and_label_di_bidirectional(di_to_process_idx) # Call existing func
    # Ensure it populates 'processed_frame_timestamp' and 'frame_index'
    if result.get('success'):
        result['processed_frame_timestamp'] = self.depth_image_library._images[di_to_process_idx].timestamp
        result['frame_index'] = di_to_process_idx # Already in your func
    return result