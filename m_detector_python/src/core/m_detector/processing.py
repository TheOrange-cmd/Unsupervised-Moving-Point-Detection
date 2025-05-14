# src/core/m_detector/processing.py
# This file is imported into MDetector class

import numpy as np
from typing import Dict, Optional, Any
from ..depth_image import DepthImage
from ..constants import OcclusionResult
from tqdm import tqdm

# def extract_mdetector_points(depth_image_output_from_mdetector: Optional[Any]) -> Dict[str, np.ndarray]: # Allow None
#     """
#     Extracts points by MDetector's label from its processed depth_image representation.
#     Returns a dictionary like {'dynamic': points, 'occluded': points, ...}.
#     """
#     mdet_points = {
#         'dynamic': [],
#         'occluded_by_mdet': [],
#         'undetermined_by_mdet': []
#     }
    
#     # Check if the input is not None AND has the expected structure
#     if depth_image_output_from_mdetector is not None and \
#        hasattr(depth_image_output_from_mdetector, 'pixel_points') and \
#        isinstance(depth_image_output_from_mdetector.pixel_points, dict):
#         for _, points_list_in_pixel in depth_image_output_from_mdetector.pixel_points.items():
#             for pt_info in points_list_in_pixel:
#                 label = pt_info.get('label') 
#                 point_global_coords = pt_info.get('global_pt')

#                 if point_global_coords is None: continue

#                 if label == OcclusionResult.OCCLUDING_IMAGE:
#                     mdet_points['dynamic'].append(point_global_coords)
#                 elif label == OcclusionResult.OCCLUDED_BY_IMAGE:
#                     mdet_points['occluded_by_mdet'].append(point_global_coords)
#                 elif label == OcclusionResult.UNDETERMINED:
#                     mdet_points['undetermined_by_mdet'].append(point_global_coords)
#     elif depth_image_output_from_mdetector is not None: # Input was given, but not structured as expected
#         # Only print warning if a non-None, but invalid, object was passed
#         tqdm.write("Warning: MDetector output (depth_image.pixel_points) not found or not a dict.")

#     return {k: (np.array(v) if v else np.empty((0,3))) for k, v in mdet_points.items()}

def extract_mdetector_points(depth_image_output_from_mdetector: Optional[Any]) -> Dict[str, np.ndarray]:
    """
    Extracts points by MDetector's label from its processed depth_image representation.
    NOTE: This function is based on the OLD DepthImage structure (pixel_points).
    It will not work correctly with the refactored DepthImage unless adapted.
    """
    mdet_points = {
        'dynamic': [],
        'occluded_by_mdet': [],
        'undetermined_by_mdet': []
    }
    if depth_image_output_from_mdetector is not None and \
       hasattr(depth_image_output_from_mdetector, 'pixel_points') and \
       isinstance(depth_image_output_from_mdetector.pixel_points, dict):
        # This part uses the old structure
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
    elif depth_image_output_from_mdetector is not None:
        # This warning might trigger if a new DI object is passed.
        # tqdm.write("Warning: extract_mdetector_points received an object not matching old DI.pixel_points structure.")
        pass # Avoid excessive warnings if this function is called with new DI

    return {k: (np.array(v) if v else np.empty((0,3))) for k, v in mdet_points.items()}

def process_and_label_di(self, # self is MDetector instance
                        current_di: DepthImage,
                        historical_di: Optional[DepthImage]) -> Dict:
    """
    Process points in current_di against historical_di (causal processing).
    ADAPTED FOR NEW DepthImage STRUCTURE.
    """
    if not isinstance(current_di, DepthImage):
        raise TypeError("current_di must be a DepthImage object.")

    points_labeled_count = 0
    label_counts = {label: 0 for label in OcclusionResult}

    # Ensure the main point arrays in current_di are initialized
    if current_di.original_points_global_coords is None:
        # This case should ideally be handled by DepthImage.add_points_batch ensuring arrays are created.
        # If it's None, it means no points were added, or add_points_batch wasn't called.
        self.logger.warning(f"Causal Processing: current_di (TS: {current_di.timestamp}) has no original_points_global_coords. Skipping.")
        return {
            'points_labeled': 0, 'label_counts': label_counts,
            'success': True, 'timestamp': current_di.timestamp,
            'reason': 'current_di has no points stored in main arrays'
        }

    num_points_in_current_di = current_di.original_points_global_coords.shape[0]

    if num_points_in_current_di == 0:
        # No points to process
        return {
            'points_labeled': 0, 'label_counts': label_counts,
            'success': True, 'timestamp': current_di.timestamp,
            'reason': 'current_di has 0 points'
        }

    # Case 1: No valid historical DI
    if not historical_di or historical_di.timestamp >= current_di.timestamp or \
       historical_di.original_points_global_coords is None or historical_di.original_points_global_coords.shape[0] == 0:
        # Label all points in current_di as UNDETERMINED
        current_di.mdet_labels_for_points.fill(OcclusionResult.UNDETERMINED.value)
        label_counts[OcclusionResult.UNDETERMINED] = num_points_in_current_di
        points_labeled_count = num_points_in_current_di
        
        # --- Add logging for this case ---
        if hasattr(self, 'logger'):
            logger_to_use = self.logger
        else: # Fallback logger
            import logging as py_logging
            logger_to_use = py_logging.getLogger('CAUSAL_PROCESSING_DEBUG') # Consistent name
            # Basic config for fallback logger if not already configured by main script
            if not logger_to_use.hasHandlers():
                handler = py_logging.StreamHandler()
                formatter = py_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger_to_use.addHandler(handler)
                logger_to_use.setLevel(py_logging.DEBUG)

        reason_no_hist = "No historical_di" if not historical_di else \
                         "Historical_di timestamp not older" if historical_di.timestamp >= current_di.timestamp else \
                         "Historical_di has no points"
        logger_to_use.debug(f"CAUSAL DEBUG (New DI): Frame TS {current_di.timestamp:.2f} ({reason_no_hist}) - Labels assigned:")
        logger_to_use.debug(f"  Label UNDETERMINED: {num_points_in_current_di} points")
        logger_to_use.debug(f"  (Total points processed: {points_labeled_count})")
        # --- End logging ---

        return {
            'points_labeled': points_labeled_count,
            'label_counts': label_counts,
            'success': True,
            'timestamp': current_di.timestamp
        }

    # Case 2: Valid historical DI exists, proceed with occlusion checks
    points_global_batch_np = current_di.original_points_global_coords
    
    # Perform batch occlusion check against historical_di
    # self.check_occlusion_batch returns a NumPy array of OcclusionResult enums
    occlusion_results_enums_batch = self.check_occlusion_batch(points_global_batch_np, historical_di)

    for i in range(num_points_in_current_di):
        initial_label_enum = occlusion_results_enums_batch[i]
        final_label_enum = initial_label_enum # Start with the raw occlusion result

        # If initial label suggests a dynamic event, perform map consistency check
        if initial_label_enum == OcclusionResult.OCCLUDING_IMAGE and self.map_consistency_enabled:
            current_point_global = points_global_batch_np[i]
            # is_map_consistent checks against past DIs relative to current_di.timestamp
            if self.is_map_consistent(current_point_global, current_di.timestamp, check_direction='past'):
                # Point is consistent with map, so it's likely static despite raw occlusion suggesting dynamic
                final_label_enum = OcclusionResult.UNDETERMINED 
                # Or: OcclusionResult.OCCLUDED_BY_IMAGE if config prefers to call it static
        
        # Store the final label's value in the main label array of current_di
        current_di.mdet_labels_for_points[i] = final_label_enum.value
        
        label_counts[final_label_enum] += 1
        points_labeled_count += 1
        
    # --- Add logging for regular processing case ---
    if hasattr(self, 'logger'):
        logger_to_use = self.logger
    else: # Fallback logger
        import logging as py_logging
        logger_to_use = py_logging.getLogger('CAUSAL_PROCESSING_DEBUG')
        if not logger_to_use.hasHandlers():
            handler = py_logging.StreamHandler(); handler.setFormatter(py_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')); logger_to_use.addHandler(handler); logger_to_use.setLevel(py_logging.DEBUG)

    logger_to_use.debug(f"CAUSAL DEBUG (New DI): Frame TS {current_di.timestamp:.2f} - Labels assigned:")
    for label_enum, count_val in label_counts.items():
        if count_val > 0: # Only print labels that have counts
            logger_to_use.debug(f"  Label {label_enum.name}: {count_val} points")
    logger_to_use.debug(f"  (Total points processed: {points_labeled_count})")
    # --- End logging ---
        
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