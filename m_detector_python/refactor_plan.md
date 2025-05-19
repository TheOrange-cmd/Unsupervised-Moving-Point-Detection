Phase 0: Preparation & Refactoring (Minor but Important)

Before diving into new features, a few preparatory steps will make the additions cleaner:

    Consolidate Occlusion Check Configuration:
        Current: epsilon_depth_occlusion, neighbor_search_pixels_h/v are used for the pixel-based check. angular_threshold_deg_h/v are in config but not used by check_occlusion_batch.
        Action: Decide if the "finer occlusion check" for Tests 2 & 3 will use these angular thresholds. If so, ensure they are clearly named and accessible. If not, they can remain for now.
        File: src/core/m_detector/base.py (in _load_occlusion_check_config).

    Clarify DepthImage.mdet_labels_for_points Usage:
        Current: This array is populated by process_and_label_di with the outcome of (currently only partial) Test 1 and its MCC interaction.
        Future: This array will hold the final label after all three tests (and their respective MCCs) are considered. If clustering/region growth is applied, it will further refine these labels on the DepthImage object itself.
        Action: No immediate code change, but keep this evolution in mind. The NuScenesProcessor will save whatever is in this array at the end.

    Review MDetector.config Access:
        Ensure all necessary parameters for the new tests and processing steps (e.g., voxel size for clustering, M2, M3 thresholds) are added to your m_detector_config.yaml and loaded into self.config in MDetector.
        File: config/m_detector_config.yaml, src/core/m_detector/base.py (_load_..._config methods).

    (Optional but Recommended) Create a Dedicated Occlusion Check for Tests 2 & 3:
        The paper states: "This finer occlusion check will be used in the test two and test three...". This implies a different mechanism than the pixel-summary based one in check_occlusion_batch.
        Action: Plan to create a new method in src/core/m_detector/occlusion_checks.py, e.g., check_occlusion_point_level_detailed(self, point_to_check_global: np.ndarray, point_in_historical_di_global: np.ndarray, historical_di: DepthImage) -> bool:. This function would take two specific points (one current, one historical candidate from the neighborhood) and check if they meet the angular (εh, εv) and depth (εd) criteria for occlusion. This will be called repeatedly within Tests 2 and 3.
        Alternatively, if the number of points in a pixel neighborhood is small, you could adapt check_occlusion_batch or check_occlusion_pixel_level to iterate through individual points in the historical pixel neighborhood if a specific flag is passed. However, a dedicated function might be cleaner.

No major file restructuring seems immediately necessary for these additions. The existing MDetector class and its helper modules (occlusion_checks.py, map_consistency.py) provide a good foundation. We might add a postprocessing.py for clustering/region growth later.

Phase 1: Implementing Test 2 & Test 3

    Define Test Parameters in Config:
        Ensure event_tests.test2_M2_depth_images and event_tests.test3_M3_depth_images are in your m_detector_config.yaml and loaded by MDetector.

    Modify MDetector.process_and_label_di:
        This function will become the orchestrator for all three tests.
        Overall Flow per point pt_idx:
        a. Initialize final_label_for_P = OcclusionResult.UNDETERMINED.
        b. Execute Test 1 Logic (largely as is, but isolate its outcome):
        i. Get raw_occlusion_P_vs_immediate_hist.
        ii. If OCCLUDING_IMAGE and map_consistency_enabled, call is_map_consistent.
        iii.Determine outcome_test1 (e.g., OCCLUDING_IMAGE if dynamic, OCCLUDED_BY_IMAGE if static after MCC, or EMPTY_IN_IMAGE, UNDETERMINED).
        c. Execute Test 2 Logic (New):
        i. Loop for k_hist_idx from 0 to M2 - 1 to get the M2 historical DIs.
        ii. For each historical_di_k:
        1. Project point_global_P into historical_di_k. Get sph_coords_curr_in_hist_k.
        2. Find neighboring points N in historical_di_k around the projection of point_global_P (using pixel neighborhood and then the "finer" angular check if you implement the dedicated point-level occlusion).
        3. Recursive Occlusion Check: For p_curr to be an event by Test 2, it must be OCCLUDED_BY_IMAGE by points in all M2 DIs, AND those historical points must satisfy their own chain of being occluded by points in their subsequent DIs (within the M2 window).
        * This requires careful iteration. For p_curr vs historical_di_k (let's call the occluding point found in historical_di_k as p_hist_k):
        * Check if p_curr is occluded by p_hist_k (using check_occlusion_point_level_detailed or similar). Apply MCC to this check. If fails or MCC rejects, Test 2 fails for p_curr.
        * Then, for this p_hist_k, check if it's occluded by points in historical_di_k+1 (up to M2). Apply MCC. If fails, Test 2 fails.
        * Continue this chain.
        iv. If all checks pass, outcome_test2 = OcclusionResult.OCCLUDING_IMAGE (or a specific "event_parallel_away" if you want more granular internal labels before final mapping). Otherwise outcome_test2 = OcclusionResult.UNDETERMINED.
        d. Execute Test 3 Logic (New, similar to Test 2 but reversed occlusion):
        i. Loop for k_hist_idx from 0 to M3 - 1.
        ii. For each historical_di_k:
        1. Project point_global_P.
        2. Find neighbors N.
        3. Recursive Occlusion Check: p_curr must occlude points in all M3 DIs, AND those historical points must occlude points in their subsequent DIs. Apply MCC at each step.
        iv. If all pass, outcome_test3 = OcclusionResult.OCCLUDING_IMAGE. Otherwise UNDETERMINED.
        e. Final Label Aggregation:
        * Paper: "If any of the tests are positive, the current point will be labeled as an event."
        * So, if outcome_test1 == OCCLUDING_IMAGE OR outcome_test2 == OCCLUDING_IMAGE OR outcome_test3 == OCCLUDING_IMAGE, then final_label_for_P = OcclusionResult.OCCLUDING_IMAGE.
        * Else if any test resulted in OCCLUDED_BY_IMAGE (and no dynamic signal from others), then OCCLUDED_BY_IMAGE.
        * Else if any test resulted in EMPTY_IN_IMAGE (and no dynamic/static signal), then EMPTY_IN_IMAGE.
        * Else UNDETERMINED. (This logic needs careful thought to match the paper's intent if multiple tests give non-UNDETERMINED but non-OCCLUDING results). A simpler approach might be: if any test returns OCCLUDING_IMAGE, it's dynamic. Otherwise, if any test returns OCCLUDED_BY_IMAGE, it's static. Otherwise EMPTY_IN_IMAGE if applicable, else UNDETERMINED.
        f. Store final_label_for_P.value in current_di.mdet_labels_for_points[pt_idx].

    Helper for Finer Occlusion Check (in occlusion_checks.py):
        Implement check_occlusion_point_level_detailed (or your chosen alternative). This function will take current_point_global (the point whose label is being determined, or a historical point in the recursive check), historical_candidate_point_global (a point from the neighborhood in a historical DI), and the historical_di itself.
        It needs to:
            Project current_point_global into historical_di to get its spherical coords (phi_curr, theta_curr, d_curr) relative to historical_di.
            The historical_candidate_point_global already has its spherical coords stored in historical_di.local_sph_coords_for_points (let them be phi_hist_cand, theta_hist_cand, d_hist_cand).
            Check abs(phi_curr - phi_hist_cand) <= angular_threshold_h_rad.
            Check abs(theta_curr - theta_hist_cand) <= angular_threshold_v_rad.
            Check depth: d_curr < d_hist_cand - epsilon_depth_occlusion (for current occluding historical) or d_curr > d_hist_cand + epsilon_depth_occlusion (for current occluded by historical).
        Return True if occlusion condition met, False otherwise.

    Map Consistency Integration:
        The paper states: "map consistency check is applied... after each of the above occlusion test [within Tests 2 and 3]."
        This means if an occlusion check (e.g., p_curr occludes p_hist_k) within Test 2/3 is positive, you then call is_map_consistent for p_curr (or p_hist_k if that's the point being evaluated for "eventness" in a recursive step). If is_map_consistent returns True (it's near static map points), then that specific occlusion event is rejected/ignored for Test 2/3.

Phase 2: Implementing Clustering and Region Growth

This phase modifies the labels produced by process_and_label_di. It operates on a DepthImage after its points have initial event labels.

    Create New Module/Functions:
        Consider a new file, e.g., src/core/m_detector/postprocessing.py or src/core/m_detector/refinement.py.
        Functions needed:
            refine_labels_with_clustering_growth(self, depth_image: DepthImage) -> None: This would be a method of MDetector or a standalone function called by it. It modifies depth_image.mdet_labels_for_points in place.

    Clustering Logic (within refine_labels_with_clustering_growth):
    a. Identify Event Points: Get all points in depth_image currently labeled as OCCLUDING_IMAGE.
    b. Voxelize Event Points:
    i. Define voxel size Lv (from config).
    ii. For each event point, determine its voxel index.
    iii.Create a set of unique "event voxels."
    c. Cluster Event Voxels:
    i. Use sklearn.cluster.DBSCAN on the center coordinates of these event voxels. Parameters for DBSCAN (eps, min_samples) will need tuning and configuration.
    ii. Identify core voxels and border voxels belonging to significant clusters. Reject isolated event voxels (noise).
    d. Update Labels based on Voxel Clusters:
    i. All original raw points (event or non-event from previous steps) that fall into the accepted clustered event voxels are now re-labeled as OCCLUDING_IMAGE. This can bring back some FNs.

    Region Growth Logic (within refine_labels_with_clustering_growth, after clustering):
    a. For each cluster of event voxels from the previous step:
    i. Get the AABB of the voxels in the cluster.
    ii. Expand AABB (e.g., "twice its size").
    iii.Fit a ground plane within this expanded AABB using RANSAC on the raw points within the expansion space. Remove ground points/voxels.
    iv. Recursively grow the event voxel set: Start with the clustered event voxels. Check their non-ground neighbors. If a neighbor contains raw points, merge it into the event voxel set. Continue until the boundary of the expansion space is reached.
    b. Update Labels based on Grown Regions: All original raw points falling into any of the final grown event voxels are labeled OCCLUDING_IMAGE.

    Integration into NuScenesProcessor (or similar workflow):
        After detector.decide_and_process_frame() returns a processed_di_object_ref, and before assembling data for HDF5:
        python

        if processed_di_object: # This is the DepthImage
            detector.refine_labels_with_clustering_growth(processed_di_object)
            # Now processed_di_object.mdet_labels_for_points contains the refined labels
            # Proceed to save to HDF5

High-Level Overview for Clustering & Region Growth:

    Input: A DepthImage object whose mdet_labels_for_points have been populated by the three event detection tests.
    Voxelize Event Points: Identify points labeled as events and assign them to voxels.
    Cluster Event Voxels: Use DBSCAN on the event voxels to find dense regions of activity and discard isolated event voxels (noise).
    Initial Label Update: Re-label all raw points within the accepted event voxel clusters as events.
    Region Grow from Clusters: For each voxel cluster:
    a. Define an expansion space around it.
    b. Remove ground from this space.
    c. Iteratively add neighboring non-ground voxels (that contain points) to the event region.
    Final Label Update: Re-label all raw points within these grown event regions as events.
    Output: The DepthImage object with its mdet_labels_for_points array modified in place.
