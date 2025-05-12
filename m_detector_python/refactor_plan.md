Overall Guiding Principles:

    Iterative Approach: Implement changes in phases. Test and profile after each significant phase to measure impact and catch issues early.
    Data-Driven Optimization: Use the profiler as your guide. Focus on the functions consuming the most tottime.
    Clarity and Maintainability: While optimizing, strive to keep the code understandable. Add comments for complex optimizations.
    Correctness First: Ensure the logical output of the M-Detector remains correct after optimizations.

Detailed Refactoring and Optimization Plan

Phase 0: Setup & Baseline

    Version Control: Ensure your project is under version control (e.g., Git). Commit frequently. Create a new branch for this refactoring work.
    Consistent Profiling Setup:
        Use a fixed, representative dataset/scene for all profiling runs to ensure comparability.
        Use the same profiling command/script (cProfile with pstats or snakeviz).
        Action: Run the profiler on the current, unrefactored code with your chosen test scene. Save this baseline profile output. This is your reference.

Phase 1: Core DepthImage Refactoring (Foundation for NPZ Output & Performance)

    Goal: Modify DepthImage to store full, ordered arrays for original points, labels, and scores, and make pixel_points a lightweight structure linking to these arrays.

    Detailed Steps:

        Modify DepthImage.__init__:
            Remove initialization of self.pixel_points as collections.defaultdict(list) storing full pt_info dicts.
            Add new attributes, initialized to None or empty arrays of a defined size (e.g., 0-length initially, to be resized/replaced in add_points_batch):
                self.original_points_global_coords: Optional[np.ndarray] = None (N x 3, float32)
                self.mdet_labels_for_points: Optional[np.ndarray] = None (N, int8/int16 for OcclusionResult.value)
                self.mdet_scores_for_points: Optional[np.ndarray] = None (N, float32, if scores are generated)
                Performance Enhancement: self.local_sph_coords_for_points: Optional[np.ndarray] = None (N x 3, float32, for phi, theta, depth of each original point in this DI's own sensor frame).
            self.pixel_points: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
                This will now store lists of original_index integers for points projecting to that pixel.

        Modify DepthImage.add_points_batch:
            Store Original Data:
                self.original_points_global_coords = points_global_batch.copy() (or a view if mutations are carefully handled, but copy is safer).
                Initialize self.mdet_labels_for_points = np.full(batch_size, OcclusionResult.UNDETERMINED.value, dtype=np.int8).
                Initialize self.mdet_scores_for_points = np.zeros(batch_size, dtype=np.float32).
            Pre-calculate Local Spherical Coords (Performance):
                After points_local, sph_coords, pixel_indices, valid_mask = self.project_points_batch(points_global_batch):
                self.local_sph_coords_for_points = sph_coords.copy() (This stores phi, theta, depth in this DI's frame for all N points).
            Populate pixel_points with Indices:
                In the loop for i in valid_original_indices: (where i is the original index):
                    v_idx, h_idx = pixel_indices[i, 0], pixel_indices[i, 1]
                    pixel_key = (v_idx, h_idx)
                    self.pixel_points[pixel_key].append(i) (Append the original_index i).
            Optimize Pixel Stats Updates (Performance - P3.3 later, but design for it):
                The updates to self.pixel_min_depth[v_idx, h_idx], self.pixel_max_depth[v_idx, h_idx], and self.pixel_count[v_idx, h_idx] will be optimized later using np.ufunc.at. For now, ensure the logic is correct. depth for these updates comes from self.local_sph_coords_for_points[i, 2].

        Modify DepthImage.get_pixel_info (or create new accessors):
            This function is heavily used by is_map_consistent. It should now be very lightweight.
            Given v_idx, h_idx, it should return:
                min_depth_in_pixel = self.pixel_min_depth[v_idx, h_idx]
                max_depth_in_pixel = self.pixel_max_depth[v_idx, h_idx]
                count_in_pixel = self.pixel_count[v_idx, h_idx]
                original_indices_in_pixel: List[int] = self.pixel_points.get((v_idx, h_idx), [])
            The caller (e.g., is_map_consistent) will then use these original_indices_in_pixel to look up actual point data (like labels or local_sph_coords) from the main DepthImage arrays if needed.

        Add New Getter Methods for NPZ Assembly:
            def get_original_points_global(self) -> Optional[np.ndarray]: return self.original_points_global_coords
            def get_all_point_labels(self) -> Optional[np.ndarray]: return self.mdet_labels_for_points
            def get_all_point_scores(self) -> Optional[np.ndarray]: return self.mdet_scores_for_points
            def get_local_sph_coords(self) -> Optional[np.ndarray]: return self.local_sph_coords_for_points

        Test & Profile: After this phase, basic DI creation and point addition should work. Run your profiler. You might not see huge speedups yet, but get_pixel_info might improve. The main benefit comes when dependent functions are updated.

Phase 2: Adapting M-Detector Logic to New DepthImage Structure

    Goal: Update M-Detector functions to use the new DepthImage data access patterns and to update the new per-point label/score arrays.

    Detailed Steps:

        Modify MDetector.check_occlusion_pixel_level (if still used, check_occlusion_batch is primary):
            When accessing historical_depth_image pixel data, use the new get_pixel_info (or direct array access if more efficient and safe).
            It primarily needs min_depth_in_region and max_depth_in_region from historical_depth_image.

        Modify MDetector.check_occlusion_batch:
            Access historical_depth_image.pixel_min_depth, historical_depth_image.pixel_max_depth, historical_depth_image.pixel_count directly. This part is likely already efficient. No major changes expected here unless the internal loop is targeted by Numba later.

        Modify MDetector.is_map_consistent:
            When processing historical_di (referred to as di in its code):
                Call di.get_pixel_info(v_idx, h_idx) to get original_indices_in_pixel.
                Loop through these original_indices_in_pixel: for static_candidate_original_idx in original_indices_in_pixel:.
                Get the label of the static candidate: label_val = di.mdet_labels_for_points[static_candidate_original_idx], then convert OcclusionResult(label_val). Check if this label is in self.static_labels_for_map_check.
                Get its spherical coordinates in di's frame: static_sph_coords_in_di = di.local_sph_coords_for_points[static_candidate_original_idx].
                Perform comparisons using these values.
            Performance (P3.1 later): The inner loop over static candidates is a target for vectorization.

        Modify MDetector.actual_causal_processing_logic (was process_and_label_di):
            It iterates points from current_di. Instead of getting pt_info dicts to update, it should iterate from original_idx = 0 to len(current_di.original_points_global_coords) - 1.
            For each original_idx:
                Get point_global = current_di.original_points_global_coords[original_idx].
                Perform occlusion check against historical_di (e.g., by calling a modified check_occlusion_pixel_level or a single-point variant of check_occlusion_batch if that's more efficient, though batch is preferred).
                If needed, perform map consistency: self.is_map_consistent(point_global, current_di.timestamp).
                Determine final_label and final_score.
                Update:
                    current_di.mdet_labels_for_points[original_idx] = final_label.value
                    current_di.mdet_scores_for_points[original_idx] = final_score
            The label_counts can be aggregated during this loop.
            Alternative for Iteration: If iterating all N points is too slow, it could iterate through current_di.pixel_points to only process points that actually project into the DI, then use the original_index from pixel_points to update the main arrays. This matches the structure of process_and_label_di_bidirectional more closely.

        Modify MDetector.actual_bidirectional_processing_logic (was process_and_label_di_bidirectional):
            The existing structure that builds all_points_to_label_global and point_info_references (where pt_info now needs to contain original_index) is a good starting point.
            When final_label_after_mc and final_score_after_mc are determined for points_global_batch_np[i]:
                Retrieve original_idx = point_info_references[i]['pt_info']['original_index'].
                Update:
                    center_di.mdet_labels_for_points[original_idx] = final_label_after_mc.value
                    center_di.mdet_scores_for_points[original_idx] = final_score_after_mc (if scores are added).

        Test & Profile: Functionality should still be intact. Profile to see how tottime shifts. is_map_consistent and the processing functions are key.

Phase 3: Optimizing Hotspots (Iterative Performance Enhancements)

    Goal: Apply targeted optimizations to the functions identified as bottlenecks.

    Detailed Steps (Iterate: Implement one, Test, Profile):

        P3.1: MDetector.is_map_consistent Optimizations:
            Vectorize Inner Loop: When comparing sph_coords_target (from current_di) against static candidates in a pixel of historical_di:
                Fetch all static_candidate_original_idx for the pixel.
                Get all their local_sph_coords from historical_di.local_sph_coords_for_points[static_candidate_original_idx_array] (this gives an M x 3 array).
                Perform abs(phi_target - all_phi_static), abs(theta_target - all_theta_static), depth checks in a vectorized way.
                Use np.any(...) to see if any static candidate satisfies the conditions.
            Numba (Optional): If the outer loop (iterating relevant_dis) or the setup for vectorization is still slow, consider applying @njit to the whole is_map_consistent function or parts of it. Numba works well with NumPy.

        P3.2: MDetector.check_occlusion_batch Optimizations:
            Apply Numba (@njit): Decorate the check_occlusion_batch function. Numba should significantly speed up the Python loop for i, (idx, vs, ve, hs, he, d) in enumerate(...) and the NumPy array accesses within it. Ensure all inputs and outputs are Numba-compatible (mostly NumPy arrays and scalars).
            Review Indexing: Ensure min_depths[vs:ve, hs:he] slicing is efficient. Numba handles this well.

        P3.3: DepthImage.add_points_batch Optimizations:
            Use np.ufunc.at for Pixel Statistics:
                For self.pixel_count: After getting valid_v_indices and valid_h_indices (flat arrays of pixel coordinates for valid points):
                np.add.at(self.pixel_count, (valid_v_indices, valid_h_indices), 1)
                For self.pixel_min_depth:
                depths_of_valid_points = self.local_sph_coords_for_points[valid_original_indices, 2]
                np.minimum.at(self.pixel_min_depth, (valid_v_indices, valid_h_indices), depths_of_valid_points)
                For self.pixel_max_depth:
                np.maximum.at(self.pixel_max_depth, (valid_v_indices, valid_h_indices), depths_of_valid_points)
            Loop for self.pixel_points: The loop appending original_index to self.pixel_points[pixel_key] list might remain. If it appears as a bottleneck after other changes, Numba could be applied if self.pixel_points is changed to a Numba-compatible structure (e.g., list of lists of integers, if max_points_per_pixel is handled carefully). This is more advanced.

        P3.4: DepthImage.project_points_batch Review (Lower Priority if Numba is used elsewhere):
            This is called by add_points_batch and check_occlusion_batch. If those parent functions are Numba-fied, Numba might inline or optimize calls to it.
            Ensure all internal NumPy operations are efficient (e.g., avoid unnecessary copies, use appropriate dtypes). np.clip and arithmetic operations are generally fast.

        Test & Profile after each significant optimization.

Phase 4: NuScenesProcessor Update for NPZ Output

    Goal: Modify NuScenesProcessor.process_scene to use the new DepthImage getters and assemble the NPZ file in the desired format.

    Detailed Steps:

        Modify NuScenesProcessor.process_scene (Phase 3: Assemble NPZ data):
            Remove the call to extract_mdetector_points.
            After mdet_result = detector.decide_and_process_frame(...) is successful and you have di_that_was_processed:
                points_xyz_sweep = di_that_was_processed.get_original_points_global()
                labels_sweep = di_that_was_processed.get_all_point_labels()
                scores_sweep = di_that_was_processed.get_all_point_scores() (handle if None)
                If points_xyz_sweep is None or empty, skip or handle appropriately.
                Create a list collected_sweep_structured_arrays = [].
                For each processed sweep, create a temporary structured NumPy array:
                python

                # Inside the loop over processed sweeps
                num_points_in_sweep = len(points_xyz_sweep)
                sweep_structured_array = np.empty(num_points_in_sweep, dtype=[
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('mdet_label', 'i2'), # Adjust dtype as needed (i1 for OcclusionResult values)
                    ('mdet_score', 'f4')  # If scores are generated
                ])
                sweep_structured_array['x'] = points_xyz_sweep[:, 0]
                sweep_structured_array['y'] = points_xyz_sweep[:, 1]
                sweep_structured_array['z'] = points_xyz_sweep[:, 2]
                sweep_structured_array['mdet_label'] = labels_sweep
                if scores_sweep is not None:
                    sweep_structured_array['mdet_score'] = scores_sweep
                else: # If no scores, fill with a default (e.g. NaN or 0)
                    sweep_structured_array['mdet_score'] = np.nan 

                collected_sweep_structured_arrays.append(sweep_structured_array)

            After the scene processing loop, if collected_sweep_structured_arrays is not empty:
                all_points_predictions_scene = np.concatenate(collected_sweep_structured_arrays)
            Else:
                all_points_predictions_scene = np.empty(0, dtype=...) (same dtype as above)
            The points_predictions_indices array needs to be built correctly based on the number of points in each sweep_structured_array.
            Save all_points_predictions_scene and points_predictions_indices to the NPZ.
            Remove old NPZ fields like all_dynamic_points, all_occluded_points, etc.

        Test & Verify NPZ Output: Carefully check the structure and content of the generated NPZ files.

Phase 5: Validation & Final Profiling

    Update validation_utils.py: Ensure it correctly loads and interprets the new NPZ format (especially the mdet_label field and the definition of mdet_dynamic_label_value).
    Run Metrics Calculation: Verify that precision, recall, F1, ROC, etc., can be calculated correctly.
    Final Profiling: Run the profiler on the fully refactored and optimized code using the same baseline scene. Compare with the initial baseline to quantify the speedup.
    Code Review & Cleanup: Review changes for clarity, correctness, and maintainability. Add comments.
