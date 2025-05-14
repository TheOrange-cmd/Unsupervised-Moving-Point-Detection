Detailed Refactoring Plan (Incorporating Logging)

The overarching goals remain: improve clarity, maintainability, adherence to the Single Responsibility Principle, and implement robust logging.

Step 1: Integrate spdlog Logging Library

    Action: Add spdlog as a project dependency.
        Detail: Use CMake's FetchContent module or add spdlog as a git submodule. Ensure it's correctly linked in CMakeLists.txt.
    Action: Replace the debug_en boolean parameter.
        Detail: Remove debug_en from DynObjFilterParams. Add a std::string log_level parameter (defaulting to "info"). Update config_loader.cpp to load log_level.
    Action: Create and configure named loggers.
        Detail: In a central place (e.g., DynObjFilter constructor or a dedicated setup function), create shared spdlog::logger instances (e.g., spdlog::stdout_color_mt("Filter"), spdlog::stdout_color_mt("Consistency"), spdlog::stdout_color_mt("Utils")). Configure sinks as needed (e.g., console output via stdout_color_mt, potentially add a file sink like basic_logger_mt or rotating_logger_mt later).
    Action: Implement logger level control based on configuration.
        Detail: In the setup function, use spdlog::level::from_str(params_.log_level) to get the base level. Set this level globally (spdlog::set_level()) or on the primary logger(s). Optionally, add a log_levels map to the config/params to allow overriding levels per logger (Filter: "debug", Consistency: "trace", etc.) and implement logic to apply these overrides using spdlog::get("LoggerName")->set_level().
    Action: Replace all std::cout and std::cerr calls.
        Detail: Go through all source files (.cpp). Replace debugging std::cout with logger->debug(...), informational messages with logger->info(...), warnings with logger->warn(...), and errors (std::cerr) with logger->error(...) or logger->critical(...). Use spdlog's Python-style formatting (e.g., logger->debug("Processing point {} at time {}", index, timestamp);). Remove the if (params_.debug_en) checks previously guarding std::cout calls, as the logger level now controls output.

Step 2: Refactor DynObjFilter (src/filtering/dyn_obj_filter.cpp)

    Action: Extract point processing logic from processBufferedFrames.
        Detail: Create a new private helper function, e.g., ProcessSinglePointResult processSinglePoint(const pcl::PointXYZI& raw_point, size_t original_index, const ScanFrame& current_frame). This function will perform steps 1-6 currently inside the for loop over raw_point: isPointInvalid check, point_soph creation/population, isSelfPoint check, checkAppearingPoint call (and future CASE2/3 calls), and population of ProcessedPointInfo. It should return a struct/pair containing the point_soph::Ptr (if valid for map insertion, nullptr otherwise) and the ProcessedPointInfo. The loop in processBufferedFrames will then call this helper, collect the valid point_soph::Ptrs into processed_points, and collect all ProcessedPointInfo into current_frame_info_vec.
    Action: Separate map management from point insertion in updateDepthMaps.
        Detail: updateDepthMaps will retain the logic for checking time thresholds, creating the first map, creating new maps, or rotating the map list (depth_map_list_). After ensuring the latest_map exists (either newly created or the existing back of the deque), it will call a new helper function: addPointsToMap(DepthMap& map, const std::vector<std::shared_ptr<point_soph>>& points).
        Detail: The new addPointsToMap function will contain the loop over processed_points, the grid index check (pos >= 0 && pos < MAX_2D_N), adding the point_soph::Ptr to map.depth_map[pos], and updating the map statistics (min/max_depth_all, min/max_depth_static, etc.) for that specific map.
    Action: Add detailed comments and potentially refactor checkAppearingPoint.
        Detail: Add comments clearly explaining the logic flow for each condition: invalid projection, self-point check, neighbors.empty(), neighbors.size() < 3 (sparsity), neighbors.size() >= 3 with InterpolationStatus::SUCCESS (and the depth_diff < -threshold check), and neighbors.size() >= 3 with interpolation failure.
        Detail: Identify the CASE1 threshold calculation logic within checkAppearingPoint and plan its extraction into a shared utility helper function (see Step 4).
    Action: Implement setupPointForMapCheck helper function.
        Detail: Create a (potentially static or private member) function setupPointForMapCheck(const point_soph& p_world, const DepthMap& map, const DynObjFilterParams& params, point_soph& p_map_frame). This function takes the original point p_world (with global coords), the target map, and populates the output p_map_frame with the results of SphericalProjection, local coordinates relative to the map, and copies relevant fields like time, is_distort. checkAppearingPoint will use this helper.

Step 3: Refactor ConsistencyChecks (src/filtering/consistency_checks.cpp)

    Action: Refactor checkDepthConsistency using forEachNeighborCell.
        Detail: Define a small helper struct NeighborDepthStats containing members like count_close, count_farther, count_closer, sum_abs_diff_close, static_neighbors_evaluated. Inside checkDepthConsistency, initialize this struct. Call PointCloudUtils::forEachNeighborCell with the appropriate range (depth_cons_hor_num*, depth_cons_ver_num*). The lambda passed to forEachNeighborCell will contain the logic currently inside the nested loops: getting points from map_info.depth_map[pos], iterating these points, applying time/angle/status filters, and updating the NeighborDepthStats struct based on depth comparisons. After the forEachNeighborCell call, checkDepthConsistency applies the final decision rules (Rule 1, Rule 2) based on the values in the populated NeighborDepthStats struct.
    Action: Refactor findOcclusionRelationshipInMap using forEachNeighborCell.
        Detail: Replace the outer nested for loops (iterating ind_hor, ind_ver) with a call to PointCloudUtils::forEachNeighborCell, passing the appropriate range (occ_hor_num*, occ_ver_num*). The lambda passed will contain the logic for the Case 3 optimization check (if applicable), getting points from map_info.depth_map[neighbor_pos], and the inner loop that iterates through points_in_pixel. The inner loop logic (calling checkOcclusionRelationship and depth_checker, updating point_to_update on success) remains within the lambda. Add comments explaining the Case 3 min-depth optimization.
    Action: Add clarifying comments.
        Detail: Ensure comments clearly explain the meaning of returning true (consistent) or false (inconsistent) for checkMapConsistency, checkDepthConsistency, and checkAccelerationLimit specifically in the context of CASE1, CASE2, and CASE3 rules.

Step 4: Create/Consolidate Utility Helpers

    Action: Implement/Move Threshold Calculation Helpers.
        Detail: Create calculateInterpolationThreshold(const point_soph& p, ConsistencyCheckType check_type, int map_index_diff, const DynObjFilterParams& params) potentially in consistency_checks.cpp or point_cloud_utils.cpp. This function encapsulates the threshold logic involving interp_thr*, scaling by map_index_diff, and the specific adjustments for CASE1 based on depth (interp_start_depth1, interp_kp1, interp_kd1) and distortion (enlarge_distort). checkMapConsistency and checkAppearingPoint can call this.
        Detail: Finalize calculateOcclusionDepthThreshold (move from test helpers if desired, place likely in consistency_checks.cpp) to centralize the dynamic depth threshold calculation used in checkOcclusionRelationship.
    Action: Add setupPointForMapCheck (as described in Step 2).

Step 5: General Code Cleanup & Quality

    Action: Replace Logging Placeholders (Covered by Step 1).
    Action: Review Constants.
        Detail: Ensure constants defined in dyn_obj_datatypes.h (MAX_1D, HASH_PRIM, etc.) are appropriate and clearly documented. Check for any unnamed magic numbers in the code and replace them with named constants where applicable. Ensure low-level constants like CACHE_VALID_THRESHOLD, BARY_DEGENERACY_EPSILON are clearly named.
    Action: Standardize Error Handling.
        Detail: Review all functions. Use exceptions (std::runtime_error, std::invalid_argument) for critical, unrecoverable errors (e.g., failure to load essential config parameters, invalid function arguments that indicate a programming error). Use return codes or status enums (like InterpolationStatus) for expected, recoverable "failure" conditions during algorithmic processing (e.g., interpolation not succeeding due to lack of neighbors).
    Action: Update Comments.
        Detail: Review comments in all modified files (.h and .cpp). Ensure they are accurate, up-to-date, and explain the purpose and reasoning ("why") behind code sections, not just restating the code ("what"). Pay particular attention to comments explaining the different cases (1/2/3) and the meaning of consistency checks.
    Action: Refine Config Loading (src/config/config_loader.cpp).
        Detail: Add comments within load_config to clearly group related parameters (e.g., # --- Case 1 Parameters ---). Implement a validation step after loading all base parameters from YAML but before calculating derived parameters. This step should check for invalid values (e.g., negative thresholds where positive are expected, non-positive resolutions) and throw an std::invalid_argument if critical errors are found.
