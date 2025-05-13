# src/core/m_detector/temporal.py
# This file is imported into MDetector class

from ..constants import OcclusionResult
import numpy as np

# def process_and_label_di_bidirectional(self, center_index):
#     """
#     Optimized bidirectional processing with smarter frame selection and iteration.
#     """
#     # Initial checks
#     if not self.is_ready_for_processing():
#         return {'points_labeled': 0, 'success': False, 'reason': 'Not ready for processing'}
    
#     # Check valid index
#     if center_index < 0 or center_index >= len(self.depth_image_library._images):
#         return {'points_labeled': 0, 'success': False, 'reason': 'Invalid center index'}
    
#     # Get center depth image
#     center_di = self.depth_image_library._images[center_index]
    
#     # --- Select DIs for Occlusion Check ---
#     historical_di_for_occ = None
#     if center_index > 0:
#         historical_di_for_occ = self.depth_image_library._images[center_index - 1]

#     future_di_for_occ = None
#     if center_index < len(self.depth_image_library._images) - 1:
#         future_di_for_occ = self.depth_image_library._images[center_index + 1]

#     # --- Prepare for Batch Processing ---
#     all_points_to_label_global = []
#     point_info_references = [] 
#     v_indices, h_indices = np.where(center_di.pixel_count > 0)
#     for v_idx, h_idx in zip(v_indices, h_indices):
#         pixel_content = center_di.get_pixel_info(v_idx, h_idx)
#         if pixel_content and pixel_content['points']:
#             for pt_info in pixel_content['points']:
#                 all_points_to_label_global.append(pt_info['global_pt'])
#                 # Store a way to get back to the original pt_info dict
#                 # (v_idx, h_idx, index_in_pixel_points_list)
#                 original_pt_list = center_di.pixel_points.get((v_idx, h_idx), [])
#                 try:
#                     # pt_idx_in_list = original_pt_list.index(pt_info) # This can be slow if lists are long
#                     # point_info_references.append({'pt_info': pt_info, 'id': (v_idx, h_idx, pt_idx_in_list)})
#                     point_info_references.append({'pt_info': pt_info})
#                 except ValueError:
#                     # Fallback if pt_info not found by identity, though it should be.
#                     # This part needs to be robust. Using a unique ID per point from creation is better.
#                     # For now, we assume pt_info is the dict we want to update.
#                     point_info_references.append({'pt_info': pt_info, 'id': None })


#     if not all_points_to_label_global:
#         # ... (return empty stats) ...
#         return {
#             'points_labeled': 0, 'label_counts': {label: 0 for label in OcclusionResult},
#             'success': True, 'timestamp': center_di.timestamp, 'frame_index': center_index,
#             'past_frames_used': 0, 'future_frames_used': 0
#         }
        
#     points_global_batch_np = np.array(all_points_to_label_global)

#     # --- Get Raw Occlusion Results (Batch) ---
#     past_occlusion_results_batch = np.full(len(points_global_batch_np), OcclusionResult.UNDETERMINED)
#     if historical_di_for_occ:
#         past_occlusion_results_batch = self.check_occlusion_batch(points_global_batch_np, historical_di_for_occ)

#     future_occlusion_results_batch = np.full(len(points_global_batch_np), OcclusionResult.UNDETERMINED)
#     if future_di_for_occ:
#         future_occlusion_results_batch = self.check_occlusion_batch(points_global_batch_np, future_di_for_occ)

#     # --- Determine Final Label for Each Point ---
#     total_points_labeled = 0
#     # Initialize label_counts for final labels
#     label_counts = {label: 0 for label in OcclusionResult} 
#     # Add string keys if your pt_info['label'] might store strings temporarily
#     label_counts["pending_classification"] = 0 
#     label_counts["non_event"] = 0


#     for i in range(len(points_global_batch_np)):
#         pt_info_dict_wrapper = point_info_references[i]
#         pt_info_to_update = pt_info_dict_wrapper['pt_info']
#         current_point_global = points_global_batch_np[i]

#         past_occ_raw = past_occlusion_results_batch[i]
#         future_occ_raw = future_occlusion_results_batch[i]

#         # --- Get "Before MC" Label ---
#         # Evidence for raw label (MC flags are present but will be ignored by the function if apply_map_consistency_effects=False)
#         evidence_for_raw_label = {
#             'past_raw_occ': past_occ_raw,
#             'future_raw_occ': future_occ_raw,
#             'is_map_consistent_past': False, # Placeholder, ignored by logic when apply_mc=False
#             'is_map_consistent_future': False # Placeholder, ignored by logic when apply_mc=False
#         }
#         label_before_mc = self._determine_final_label_bidirectional_simplified(
#             evidence_for_raw_label, 
#             apply_map_consistency_effects=False # IMPORTANT
#         )
#         pt_info_to_update['label_before_mc'] = label_before_mc

#         # --- Perform Actual Map Consistency Checks (if enabled) ---
#         actual_is_mc_past = False 
#         actual_is_mc_future = False

#         if self.map_consistency_enabled:
#             # Conditions for checking past MC
#             if past_occ_raw == OcclusionResult.OCCLUDING_IMAGE or \
#                past_occ_raw == OcclusionResult.EMPTY_IN_IMAGE or \
#                past_occ_raw == OcclusionResult.UNDETERMINED:
#                 actual_is_mc_past = self.is_map_consistent(
#                     current_point_global, 
#                     center_di.timestamp, 
#                     check_direction='past'
#                 )
#             # Conditions for checking future MC
#             if future_occ_raw == OcclusionResult.OCCLUDING_IMAGE or \
#                future_occ_raw == OcclusionResult.OCCLUDED_BY_IMAGE or \
#                future_occ_raw == OcclusionResult.EMPTY_IN_IMAGE or \
#                future_occ_raw == OcclusionResult.UNDETERMINED:
#                 actual_is_mc_future = self.is_map_consistent(
#                     current_point_global, 
#                     center_di.timestamp, 
#                     check_direction='future'
#                 )
        
#         # --- Get "After MC" (Final) Label ---
#         evidence_for_final_label = {
#             'past_raw_occ': past_occ_raw,
#             'is_map_consistent_past': actual_is_mc_past, # Use actual MC results
#             'future_raw_occ': future_occ_raw,
#             'is_map_consistent_future': actual_is_mc_future # Use actual MC results
#         }
        
#         final_label_after_mc = self._determine_final_label_bidirectional_simplified(
#             evidence_for_final_label, 
#             apply_map_consistency_effects=True # IMPORTANT
#         )
#         pt_info_to_update['label'] = final_label_after_mc
        
#         # Update overall label counts based on the final_label_after_mc
#         if isinstance(final_label_after_mc, OcclusionResult):
#             label_counts[final_label_after_mc] += 1
#         else: # Handle if it's a string (though it should be OcclusionResult)
#             label_counts[final_label_after_mc] = label_counts.get(final_label_after_mc, 0) + 1
            
#         total_points_labeled += 1
        
#     return {
#         'points_labeled': total_points_labeled, 'label_counts': label_counts,
#         'success': True, 'timestamp': center_di.timestamp, 'frame_index': center_index,
#         'past_frames_used': 1 if historical_di_for_occ else 0,
#         'future_frames_used': 1 if future_di_for_occ else 0
#     }

def process_and_label_di_bidirectional(self, center_index: int): # Added type hint for center_index
    """
    Optimized bidirectional processing with smarter frame selection and iteration.
    ADAPTED FOR NEW DepthImage STRUCTURE.
    """
    if not self.is_ready_for_processing(): # MDetector's readiness
        return {'points_labeled': 0, 'success': False, 'reason': 'MDetector not ready for processing'}
    
    if not (0 <= center_index < len(self.depth_image_library._images)):
        return {'points_labeled': 0, 'success': False, 'reason': 'Invalid center_index for DI library'}
    
    center_di = self.depth_image_library._images[center_index]

    # --- START DEBUG BLOCK for max_points_per_pixel ---
    if center_di.original_points_global_coords is not None:
        temp_pixel_storage = {} # Stores lists of original_idx per pixel
        
        # Project all points in center_di to get their pixel assignments
        # Note: project_points_batch returns local points, sph_coords, pixel_indices, valid_mask
        # We only need pixel_indices and valid_mask here.
        _, _, pixel_indices_all, valid_mask_all = \
            center_di.project_points_batch(center_di.original_points_global_coords)

        for original_idx in range(center_di.original_points_global_coords.shape[0]):
            if valid_mask_all[original_idx]:
                v_idx, h_idx = pixel_indices_all[original_idx]
                pixel_key = (v_idx, h_idx)
                if pixel_key not in temp_pixel_storage:
                    temp_pixel_storage[pixel_key] = []
                temp_pixel_storage[pixel_key].append(original_idx)

        max_ppp_config = self.config.get('depth_image', {}).get('max_points_per_pixel', 50)
        
        num_points_processed_by_new = center_di.original_points_global_coords.shape[0]
        num_points_would_be_processed_by_old = 0
        
        for pixel_key, point_indices_in_pixel in temp_pixel_storage.items():
            num_points_would_be_processed_by_old += min(len(point_indices_in_pixel), max_ppp_config)
            
        num_additional_points_processed_by_new = num_points_processed_by_new - num_points_would_be_processed_by_old
        
        if num_additional_points_processed_by_new > 0:
            self.logger.warning(f"Frame {center_index} (TS: {center_di.timestamp}):")
            self.logger.warning(f"  New system processes {num_points_processed_by_new} points.")
            self.logger.warning(f"  Old system would have processed approx. {num_points_would_be_processed_by_old} points (due to max_points_per_pixel={max_ppp_config}).")
            self.logger.warning(f"  New system processes approx. {num_additional_points_processed_by_new} MORE points.")
    # --- END DEBUG BLOCK ---

    if center_di.original_points_global_coords is None or center_di.original_points_global_coords.shape[0] == 0:
        # No points in the center_di to process
        return {
            'points_labeled': 0, 'label_counts': {label: 0 for label in OcclusionResult},
            'success': True, 'timestamp': center_di.timestamp, 'frame_index': center_index,
            'past_frames_used': 0, 'future_frames_used': 0, 'reason': 'Center DI has no points'
        }

    historical_di_for_occ = self.depth_image_library.get_image_by_index(center_index - 1) if center_index > 0 else None
    future_di_for_occ = self.depth_image_library.get_image_by_index(center_index + 1) # Handles out of bounds by returning None

    # --- Prepare for Batch Processing ---
    # We need to process ALL points that are part of the center_di.
    # The easiest way is to iterate through all original_indices of center_di.
    
    num_points_in_center_di = center_di.original_points_global_coords.shape[0]
    
    # These are the points we will get labels for.
    # We can directly use center_di.original_points_global_coords for batch operations.
    points_global_batch_np = center_di.original_points_global_coords

    # --- Get Raw Occlusion Results (Batch) ---
    # Initialize with UNDETERMINED. Ensure dtype matches OcclusionResult.value
    past_occlusion_results_batch = np.full(num_points_in_center_di, OcclusionResult.UNDETERMINED.value, dtype=np.int8)
    if historical_di_for_occ and historical_di_for_occ.original_points_global_coords is not None: # Ensure hist_di has points
        # check_occlusion_batch expects global points from current_di, and the historical_di itself
        past_occlusion_results_enums = self.check_occlusion_batch(points_global_batch_np, historical_di_for_occ)
        past_occlusion_results_batch = np.array([res.value for res in past_occlusion_results_enums], dtype=np.int8)


    future_occlusion_results_batch = np.full(num_points_in_center_di, OcclusionResult.UNDETERMINED.value, dtype=np.int8)
    if future_di_for_occ and future_di_for_occ.original_points_global_coords is not None: # Ensure future_di has points
        future_occlusion_results_enums = self.check_occlusion_batch(points_global_batch_np, future_di_for_occ)
        future_occlusion_results_batch = np.array([res.value for res in future_occlusion_results_enums], dtype=np.int8)

    # --- Determine Final Label for Each Point ---
    total_points_labeled = 0
    label_counts = {label: 0 for label in OcclusionResult}
    # Ensure string keys from config are also handled if _determine_final_label_bidirectional_simplified could return them
    # (though it's typed to return OcclusionResult)
    # label_counts["pending_classification"] = 0 
    # label_counts["non_event"] = 0

    # Iterate through each original point of the center_di
    for original_idx in range(num_points_in_center_di):
        current_point_global = points_global_batch_np[original_idx] # Same as center_di.original_points_global_coords[original_idx]

        # Get raw occlusion results for this specific point
        # Ensure that past_occlusion_results_batch and future_occlusion_results_batch are OcclusionResult ENUMs
        # if _determine_final_label_bidirectional_simplified expects enums.
        # The batch check returns enums, so we used .value to store. Now convert back or adapt downstream.
        # Let's assume _determine_final_label_bidirectional_simplified is adapted or already handles .value
        
        past_occ_raw_val = past_occlusion_results_batch[original_idx]
        future_occ_raw_val = future_occlusion_results_batch[original_idx]

        # --- Perform Actual Map Consistency Checks (if enabled) ---
        actual_is_mc_past = False 
        actual_is_mc_future = False

        if self.map_consistency_enabled:
            # Conditions for checking past MC (using OcclusionResult enum for comparison)
            # Need to convert raw_val back to Enum for these checks if is_map_consistent expects Enums
            # or if the logic here relies on Enum properties.
            # For now, assuming direct comparison with .value is okay for conditions.
            past_occ_raw_enum = OcclusionResult(past_occ_raw_val) # Convert for conditional logic
            if past_occ_raw_enum == OcclusionResult.OCCLUDING_IMAGE or \
               past_occ_raw_enum == OcclusionResult.EMPTY_IN_IMAGE or \
               past_occ_raw_enum == OcclusionResult.UNDETERMINED:
                # is_map_consistent takes global point, current_di's timestamp
                actual_is_mc_past = self.is_map_consistent(
                    current_point_global, 
                    center_di.timestamp, 
                    check_direction='past'
                    # return_debug_info=False # Assuming default
                )
            
            future_occ_raw_enum = OcclusionResult(future_occ_raw_val) # Convert for conditional logic
            # Conditions for checking future MC
            if future_occ_raw_enum == OcclusionResult.OCCLUDING_IMAGE or \
               future_occ_raw_enum == OcclusionResult.OCCLUDED_BY_IMAGE or \
               future_occ_raw_enum == OcclusionResult.EMPTY_IN_IMAGE or \
               future_occ_raw_enum == OcclusionResult.UNDETERMINED:
                actual_is_mc_future = self.is_map_consistent(
                    current_point_global, 
                    center_di.timestamp, 
                    check_direction='future'
                    # return_debug_info=False # Assuming default
                )
        
        # --- Get "After MC" (Final) Label ---
        # _determine_final_label_bidirectional_simplified expects OcclusionResult enums for past/future_raw_occ
        evidence_for_final_label = {
            'past_raw_occ': OcclusionResult(past_occ_raw_val), # Pass Enum
            'is_map_consistent_past': actual_is_mc_past,
            'future_raw_occ': OcclusionResult(future_occ_raw_val), # Pass Enum
            'is_map_consistent_future': actual_is_mc_future
        }
        
        final_label_after_mc_enum = self._determine_final_label_bidirectional_simplified(
            evidence_for_final_label, 
            apply_map_consistency_effects=True 
        )
        
        # Update the main label array in center_di
        center_di.mdet_labels_for_points[original_idx] = final_label_after_mc_enum.value
        # Optionally, if scores are generated by _determine_final_label_bidirectional_simplified:
        # center_di.mdet_scores_for_points[original_idx] = score_from_temporal_logic
        
        label_counts[final_label_after_mc_enum] += 1
        total_points_labeled += 1
        
    # --- START: TEMPORARY DEBUG PRINT ---
    if total_points_labeled > 0 and center_di.mdet_labels_for_points is not None:
        try:
            unique_labels, counts = np.unique(center_di.mdet_labels_for_points, return_counts=True)
            print(f"DEBUG (temporal.py): Timestamp {center_di.timestamp:.2f} - Labels assigned to center_di.mdet_labels_for_points:")
            for label_val, count in zip(unique_labels, counts):
                print(f"  Label {OcclusionResult(label_val).name}: {count} points")
            if not np.all(center_di.mdet_labels_for_points != OcclusionResult.UNDETERMINED.value) and \
               OcclusionResult.UNDETERMINED.value in unique_labels:
                print(f"  (Note: Some points might still be UNDETERMINED if processing logic didn't change them from initial state)")
            elif np.all(center_di.mdet_labels_for_points == OcclusionResult.UNDETERMINED.value):
                 print(f"  WARNING: All points in center_di.mdet_labels_for_points are still UNDETERMINED.")

        except Exception as e:
            print(f"DEBUG (temporal.py): Error during label stats print: {e}")
    elif center_di.original_points_global_coords is not None and center_di.original_points_global_coords.shape[0] > 0:
        print(f"DEBUG (temporal.py): Timestamp {center_di.timestamp:.2f} - No points were labeled, or mdet_labels_for_points is None. Total points in DI: {center_di.original_points_global_coords.shape[0]}")
    # --- END: TEMPORARY DEBUG PRINT ---
        
    return {
        'points_labeled': total_points_labeled, 'label_counts': label_counts,
        'success': True, 'timestamp': center_di.timestamp, 'frame_index': center_index,
        'past_frames_used': 1 if historical_di_for_occ and historical_di_for_occ.original_points_global_coords is not None else 0,
        'future_frames_used': 1 if future_di_for_occ and future_di_for_occ.original_points_global_coords is not None else 0,
        # Add the processed DI itself to the result, so NuScenesProcessor can access it
        'processed_di': center_di 
    }

def _determine_final_label_bidirectional_simplified(self, evidence: dict, 
                                                  apply_map_consistency_effects: bool = True) -> OcclusionResult:
    cfg = self.config.get('temporal_processing', {})

    past_raw_occ = evidence['past_raw_occ']
    future_raw_occ = evidence['future_raw_occ']
    
    # Use actual map consistency flags ONLY if apply_map_consistency_effects is True
    # Otherwise, treat them as False so they don't influence the logic.
    is_mc_past_effective = evidence.get('is_map_consistent_past', False) if apply_map_consistency_effects else False
    is_mc_future_effective = evidence.get('is_map_consistent_future', False) if apply_map_consistency_effects else False

    # Helper to convert config string to OcclusionResult
    def to_occ_res(label_str: str) -> OcclusionResult:
        if label_str == "DYNAMIC": return OcclusionResult.OCCLUDING_IMAGE
        if label_str == "STATIC": return OcclusionResult.OCCLUDED_BY_IMAGE
        if label_str == "EMPTY": return OcclusionResult.EMPTY_IN_IMAGE
        return OcclusionResult.UNDETERMINED

    # --- Special Override: Both Map Consistencies are True ---
    # This override should only apply if we are applying map consistency effects AND
    # the original map consistency flags (from evidence) were both true.
    both_mc_override_cfg = cfg.get('both_map_consistent_override', "STATIC").upper()
    if apply_map_consistency_effects and \
       both_mc_override_cfg == "STATIC" and \
       evidence.get('is_map_consistent_past', False) and \
       evidence.get('is_map_consistent_future', False): # Check original MC flags from evidence for the override
        return OcclusionResult.OCCLUDED_BY_IMAGE 

    # --- Determine Effective Past State (EPS) ---
    effective_past_state_str = ""
    if past_raw_occ == OcclusionResult.OCCLUDING_IMAGE:
        if is_mc_past_effective: # Uses the potentially modified mc_past
            effective_past_state_str = cfg.get('past_occluding_mc_true_outcome', "STATIC").upper()
        else:
            effective_past_state_str = "DYNAMIC"
    elif past_raw_occ == OcclusionResult.OCCLUDED_BY_IMAGE:
        effective_past_state_str = "STATIC"
    elif past_raw_occ == OcclusionResult.EMPTY_IN_IMAGE:
        effective_past_state_str = "AMBIGUOUS_PAST"
    else: # UNDETERMINED
        effective_past_state_str = "UNDETERMINED"

    # --- Determine Effective Future State (EFS) ---
    is_future_raw_dynamic_signal = (future_raw_occ == OcclusionResult.OCCLUDING_IMAGE or \
                                    future_raw_occ == OcclusionResult.OCCLUDED_BY_IMAGE or \
                                    future_raw_occ == OcclusionResult.EMPTY_IN_IMAGE)
    effective_future_state_str = ""
    if is_future_raw_dynamic_signal:
        if is_mc_future_effective: # Uses the potentially modified mc_future
            effective_future_state_str = cfg.get('future_dynamic_mc_true_outcome', "STATIC").upper()
        else:
            effective_future_state_str = "DYNAMIC"
    else: # UNDETERMINED
        effective_future_state_str = "UNDETERMINED"

    # --- Combine EPS and EFS using Configurable Rules (logic remains the same) ---
    final_label_str = ""
    if effective_past_state_str == "DYNAMIC":
        if effective_future_state_str == "DYNAMIC":
            final_label_str = cfg.get('both_dynamic_outcome', "DYNAMIC").upper()
        elif effective_future_state_str == "STATIC":
            final_label_str = cfg.get('past_dynamic_future_static_outcome', "DYNAMIC").upper()
        else: # Future is UNDETERMINED
            final_label_str = cfg.get('past_dynamic_future_undetermined_outcome', "DYNAMIC").upper()
    
    elif effective_past_state_str == "STATIC":
        if effective_future_state_str == "DYNAMIC":
            final_label_str = cfg.get('past_static_future_dynamic_outcome', "DYNAMIC").upper()
        elif effective_future_state_str == "STATIC":
            final_label_str = cfg.get('both_static_outcome', "STATIC").upper()
        else: # Future is UNDETERMINED
            final_label_str = cfg.get('past_static_future_undetermined_outcome', "STATIC").upper()

    elif effective_past_state_str == "AMBIGUOUS_PAST":
        if effective_future_state_str == "DYNAMIC":
            final_label_str = cfg.get('past_empty_future_dynamic_outcome', "DYNAMIC").upper()
        elif effective_future_state_str == "STATIC":
            final_label_str = cfg.get('past_empty_future_static_outcome', "STATIC").upper()
        else: 
            final_label_str = cfg.get('past_empty_future_undetermined_outcome', "UNDETERMINED").upper()
            if final_label_str == "EMPTY" and past_raw_occ == OcclusionResult.EMPTY_IN_IMAGE:
                 return OcclusionResult.EMPTY_IN_IMAGE

    elif effective_past_state_str == "UNDETERMINED":
        if effective_future_state_str == "DYNAMIC":
            final_label_str = "DYNAMIC" 
        elif effective_future_state_str == "STATIC":
            final_label_str = "STATIC"
        else: 
            final_label_str = cfg.get('default_ambiguous_outcome', "UNDETERMINED").upper()
            
    else: 
        final_label_str = cfg.get('default_ambiguous_outcome', "UNDETERMINED").upper()

    return to_occ_res(final_label_str)

def _determine_final_label(self, point_result_evidence: dict) -> OcclusionResult: # Renamed arg for clarity
    """
    Determines the final label based on accumulated past and future occlusion evidence.
    Map consistency is NOT YET APPLIED HERE in this focused debugging step.
    """
    class_votes = np.zeros(4, dtype=np.float32) # [OCCLUDING, OCCLUDED, EMPTY, UNDETERMINED]
    total_votes = 0.0

    # --- Process Past Results ---
    # Interpretation:
    # - OCCLUDING_IMAGE (center_pt occludes past_di): Dynamic signal for center_pt
    # - OCCLUDED_BY_IMAGE (center_pt occluded by past_di): Static signal for center_pt
    # - EMPTY_IN_IMAGE (center_pt projects to empty in past_di): Ambiguous, could be newly seen static or part of moving
    # - UNDETERMINED: No info
    for res_item in point_result_evidence.get('past', []):
        raw_past_result = res_item['result'] # This is an OcclusionResult enum
        weight = res_item['weight']
        
        # Map raw_past_result to a "vote" for dynamic or static
        if raw_past_result == OcclusionResult.OCCLUDING_IMAGE:
            class_votes[OcclusionResult.OCCLUDING_IMAGE.value] += weight # Vote for dynamic
        elif raw_past_result == OcclusionResult.OCCLUDED_BY_IMAGE:
            class_votes[OcclusionResult.OCCLUDED_BY_IMAGE.value] += weight # Vote for static
        elif raw_past_result == OcclusionResult.EMPTY_IN_IMAGE:
            # Less strong, could be neutral or slight dynamic
            class_votes[OcclusionResult.EMPTY_IN_IMAGE.value] += weight * 0.5 # Lesser weight or specific handling
        else: # UNDETERMINED
            class_votes[OcclusionResult.UNDETERMINED.value] += weight * 0.1 # Low weight

        total_votes += weight

    # --- Process Future Results ---
    # Interpretation (center_pt compared to future_di):
    # - Raw OCCLUDING_IMAGE (center_pt occludes future_di content): Dynamic (center_pt disappeared/moved)
    # - Raw OCCLUDED_BY_IMAGE (center_pt occluded by future_di content): Dynamic (center_pt disappeared/was covered)
    # - Raw EMPTY_IN_IMAGE (center_pt projects to empty in future_di): Dynamic (center_pt disappeared/moved)
    # - Raw UNDETERMINED: No info
    for res_item in point_result_evidence.get('future', []):
        raw_future_result = res_item['result'] # This is an OcclusionResult enum
        weight = res_item['weight']

        if raw_future_result == OcclusionResult.OCCLUDING_IMAGE:
            class_votes[OcclusionResult.OCCLUDING_IMAGE.value] += weight # Vote for dynamic
        elif raw_future_result == OcclusionResult.OCCLUDED_BY_IMAGE:
            class_votes[OcclusionResult.OCCLUDING_IMAGE.value] += weight # Also a vote for dynamic!
        elif raw_future_result == OcclusionResult.EMPTY_IN_IMAGE:
            class_votes[OcclusionResult.OCCLUDING_IMAGE.value] += weight # Also a vote for dynamic!
        else: # UNDETERMINED
            class_votes[OcclusionResult.UNDETERMINED.value] += weight * 0.1

        total_votes += weight
        
    if total_votes < 1e-6:
        return OcclusionResult.UNDETERMINED

    # --- Decision Logic (Simplified Voting) ---
    # Now, a high score for OCCLUDING_IMAGE means strong dynamic evidence from either past or future.
    # A high score for OCCLUDED_BY_IMAGE means strong static evidence (only from past).
    
    # Option 1: Prioritize Dynamic
    if class_votes[OcclusionResult.OCCLUDING_IMAGE.value] > class_votes[OcclusionResult.OCCLUDED_BY_IMAGE.value] and \
       class_votes[OcclusionResult.OCCLUDING_IMAGE.value] > class_votes[OcclusionResult.EMPTY_IN_IMAGE.value] and \
       class_votes[OcclusionResult.OCCLUDING_IMAGE.value] > class_votes[OcclusionResult.UNDETERMINED.value]:
        # Add confidence check
        if class_votes[OcclusionResult.OCCLUDING_IMAGE.value] / total_votes > self.config.get('temporal_processing', {}).get('dynamic_confidence_threshold', 0.5):
            return OcclusionResult.OCCLUDING_IMAGE

    # Option 2: Prioritize Static if strong past evidence
    if class_votes[OcclusionResult.OCCLUDED_BY_IMAGE.value] > class_votes[OcclusionResult.OCCLUDING_IMAGE.value] and \
       class_votes[OcclusionResult.OCCLUDED_BY_IMAGE.value] / total_votes > self.config.get('temporal_processing', {}).get('static_confidence_threshold', 0.5):
        return OcclusionResult.OCCLUDED_BY_IMAGE # This means static for causal, but for bidirectional it's more complex

    # Fallback
    return OcclusionResult.UNDETERMINED