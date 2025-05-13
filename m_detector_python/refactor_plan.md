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


baseline profiling:

Mon May 12 13:58:12 2025    mdetector_profile.prof

         432571239 function calls (422433002 primitive calls) in 720.494 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.000    0.000  720.494  720.494 {built-in method builtins.exec}
        1    0.353    0.353  720.494  720.494 <string>:1(<module>)
        1    0.002    0.002  720.141  720.141 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/scripts/run_mdetector_and_save.py:32(main)
        1    0.015    0.015  718.973  718.973 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/data_utils/nuscenes_helper.py:128(process_scene)
       82    0.001    0.000  685.573    8.361 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/base.py:232(decide_and_process_frame)
       79    0.256    0.003  685.572    8.678 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/base.py:213(_process_bidirectional_di_wrapper)
       79   25.785    0.326  685.316    8.675 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/temporal.py:7(process_and_label_di_bidirectional)
  3462188  150.443    0.000  524.352    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/map_consistency.py:7(is_map_consistent)
 10135567   94.905    0.000  288.547    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/depth_image.py:163(project_point_to_pixel_indices)
23683831/13546208   31.725    0.000  118.582    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
 10135884    8.105    0.000  107.111    0.000 <__array_function__ internals>:2(norm)
      157   51.620    0.329  100.830    0.642 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/occlusion_checks.py:71(check_occlusion_batch)
 10135884   48.619    0.000   89.212    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/linalg/linalg.py:2363(norm)
 11325037   71.755    0.000   71.755    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/depth_image.py:268(get_pixel_info)
 10135567   49.472    0.000   65.224    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/depth_image.py:63(_apply_transformation_to_point)
 20273953   33.063    0.000   33.063    0.000 {built-in method numpy.array}
       80    0.235    0.003   30.204    0.378 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/base.py:113(add_sweep_and_create_depth_image)
       80   16.834    0.210   28.871    0.361 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/depth_image.py:391(add_points_batch)
 10137123    6.885    0.000   27.454    0.000 <__array_function__ internals>:2(dot)
  3406074    2.084    0.000   22.892    0.000 <__array_function__ internals>:2(any)
 10149415   19.423    0.000   19.423    0.000 {method 'reduce' of 'numpy.ufunc' objects}
  3475684   12.413    0.000   16.868    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/temporal.py:158(_determine_final_label_bidirectional_simplified)
  3406074    2.898    0.000   16.248    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:2268(any)
 48205769   14.846    0.000   14.846    0.000 {method 'get' of 'dict' objects}
  3406074    4.612    0.000   13.349    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:69(_wrapreduction)
 21637000   12.895    0.000   12.895    0.000 {built-in method builtins.min}
  3462189    6.956    0.000    9.810    0.000 {method 'sort' of 'list' objects}
  3371512    1.676    0.000    9.245    0.000 {method 'min' of 'numpy.ndarray' objects}
 21637188    8.774    0.000    8.774    0.000 {built-in method builtins.max}
  3371512    1.036    0.000    7.569    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/_methods.py:42(_amin)
  3371512    1.223    0.000    7.534    0.000 {method 'max' of 'numpy.ndarray' objects}
  3371512    0.858    0.000    6.311    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/_methods.py:38(_amax)
 10135567    4.568    0.000    4.568    0.000 {method 'ravel' of 'numpy.ndarray' objects}
 10135727    3.308    0.000    4.450    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/linalg/linalg.py:112(isComplexType)
      157    1.465    0.009    3.333    0.021 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/occlusion_checks.py:139(<listcomp>)
 20271691    3.304    0.000    3.304    0.000 {built-in method builtins.issubclass}
 36703885    3.299    0.000    3.299    0.000 {method 'append' of 'list' objects}
       79    1.516    0.019    2.862    0.036 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/processing.py:10(extract_mdetector_points)
 27997680    2.520    0.000    2.520    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/map_consistency.py:50(<lambda>)
 15789296    2.460    0.000    2.460    0.000 {built-in method builtins.abs}
      237    1.604    0.007    2.339    0.010 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/depth_image.py:294(project_points_batch)
  3476237    1.611    0.000    2.224    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/enum.py:615(__hash__)
  3451019    0.969    0.000    1.869    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/enum.py:289(__call__)
 10138482    1.713    0.000    1.713    0.000 {built-in method numpy.asarray}
 10135884    1.548    0.000    1.548    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/linalg/linalg.py:2359(_norm_dispatcher)
 10137123    1.422    0.000    1.422    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/multiarray.py:736(dot)
  3017557    1.271    0.000    1.271    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/temporal.py:171(to_occ_res)
  6397175    1.198    0.000    1.198    0.000 {method 'upper' of 'str' objects}
  3406074    1.176    0.000    1.176    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:70(<dictcomp>)
  3451019    0.900    0.000    0.900    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/enum.py:531(__new__)
       80    0.000    0.000    0.820    0.010 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/depth_image_library.py:31(add_image)
       81    0.820    0.010    0.820    0.010 {method 'append' of 'collections.deque' objects}
        1    0.000    0.000    0.721    0.721 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/nuscenes/nuscenes.py:44(__init__)
       79    0.001    0.000    0.664    0.008 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/processing.py:42(<dictcomp>)
  3476237    0.613    0.000    0.613    0.000 {built-in method builtins.hash}
       13    0.000    0.000    0.556    0.043 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/nuscenes/nuscenes.py:134(__load_table__)
       13    0.002    0.000    0.556    0.043 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/json/__init__.py:274(load)
       13    0.000    0.000    0.511    0.039 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/json/__init__.py:299(loads)
       13    0.000    0.000    0.511    0.039 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/json/decoder.py:332(decode)
       13    0.510    0.039    0.510    0.039 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/json/decoder.py:343(raw_decode)
  3406074    0.507    0.000    0.507    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:2263(_any_dispatcher)
        1    0.000    0.000    0.408    0.408 <__array_function__ internals>:2(savez)
        1    0.000    0.000    0.408    0.408 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/lib/npyio.py:538(savez)
        1    0.000    0.000    0.408    0.408 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/lib/npyio.py:692(_savez)
  3406261    0.381    0.000    0.381    0.000 {method 'items' of 'dict' objects}
        2    0.000    0.000    0.339    0.170 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/zipfile.py:1811(close)
        1    0.000    0.000    0.339    0.339 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/zipfile.py:1935(_fpclose)
        1    0.339    0.339    0.339    0.339 {method 'close' of '_io.BufferedRandom' objects}
  1713177    0.334    0.000    0.334    0.000 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/core/m_detector/map_consistency.py:59(<lambda>)
  1744718    0.307    0.000    0.307    0.000 {built-in method builtins.isinstance}
      390    0.002    0.000    0.250    0.001 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/data_utils/nuscenes_helper.py:63(get_scene_sweep_data_sequence)
      389    0.010    0.000    0.246    0.001 /home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/src/data_utils/nuscenes_helper.py:20(get_lidar_sweep_data)
      389    0.002    0.000    0.195    0.001 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/nuscenes/utils/data_classes.py:246(from_file)
      389    0.191    0.000    0.192    0.000 {built-in method numpy.fromfile}
      320    0.001    0.000    0.188    0.001 <__array_function__ internals>:2(concatenate)
        1    0.082    0.082    0.164    0.164 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/nuscenes/nuscenes.py:150(__make_reverse_index__)
      317    0.001    0.000    0.164    0.001 <__array_function__ internals>:2(hstack)
  1769941    0.162    0.000    0.162    0.000 {built-in method builtins.len}
      317    0.003    0.000    0.161    0.001 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/shape_base.py:285(hstack)
      711    0.002    0.000    0.115    0.000 <__array_function__ internals>:2(clip)
      711    0.003    0.000    0.111    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:2046(clip)
      711    0.002    0.000    0.108    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/fromnumeric.py:51(_wrapfunc)
      711    0.003    0.000    0.105    0.000 {method 'clip' of 'numpy.ndarray' objects}
      711    0.006    0.000    0.103    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/_methods.py:125(_clip)
   112519    0.063    0.000    0.093    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/types.py:164(__get__)
   125211    0.051    0.000    0.084    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/nuscenes/nuscenes.py:207(get)
      790    0.002    0.000    0.062    0.000 <__array_function__ internals>:2(where)
       15    0.000    0.000    0.059    0.004 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/lib/format.py:627(write_array)
      711    0.058    0.000    0.058    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/_methods.py:106(_clip_dep_invoke_with_casting)
     1806    0.050    0.000    0.050    0.000 {built-in method numpy.zeros}
       31    0.000    0.000    0.048    0.002 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/zipfile.py:1122(write)
       17    0.024    0.001    0.043    0.003 {method 'read' of '_io.TextIOWrapper' objects}
     1422    0.017    0.000    0.035    0.000 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/numpy/core/_methods.py:91(_clip_dep_is_scalar_nan)
        1    0.000    0.000    0.034    0.034 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/yaml/__init__.py:117(safe_load)
        1    0.000    0.000    0.034    0.034 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/yaml/__init__.py:74(load)
        1    0.000    0.000    0.034    0.034 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/yaml/constructor.py:47(get_single_data)
        1    0.000    0.000    0.033    0.033 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/yaml/composer.py:29(get_single_node)
        1    0.000    0.000    0.032    0.032 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/yaml/composer.py:50(compose_document)
    184/1    0.001    0.000    0.032    0.032 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/yaml/composer.py:63(compose_node)
     13/1    0.000    0.000    0.032    0.032 /home/drugge/miniconda3/envs/mdet_env/lib/python3.7/site-packages/yaml/composer.py:117(compose_mapping_node)
