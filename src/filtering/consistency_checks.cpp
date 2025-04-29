#include "filtering/consistency_checks.h"
#include "point_cloud_utils/point_cloud_utils.h" // For interpolateDepth, isSelfPoint
#include <cmath>     // For std::fabs
#include <algorithm> // For std::max
#include <iomanip>

namespace ConsistencyChecks {

    // Helper to convert InterpolationStatus to string for printing
    inline const char* interpolationStatusToString(PointCloudUtils::InterpolationStatus status) {
        switch (status) {
            case PointCloudUtils::InterpolationStatus::SUCCESS: return "SUCCESS";
            case PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS: return "NOT_ENOUGH_NEIGHBORS";
            case PointCloudUtils::InterpolationStatus::NO_VALID_TRIANGLE: return "NO_VALID_TRIANGLE";
            default: return "UNKNOWN_STATUS";
        }
    }

    bool checkMapConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        int map_index_diff)
    {
        // --- 1. Determine Interpolation Type, Thresholds, and Case String ---
        PointCloudUtils::InterpolationNeighborType interp_neighbor_type;
        float base_interp_threshold = 0.0f;
        float threshold_scaling = 1.0f;
        const char* case_str = ""; // For printing

        switch (check_type) {
            case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                case_str = "CASE1";
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::STATIC_ONLY;
                base_interp_threshold = params.interp_thr1;
                if (p.vec(2) > params.interp_start_depth1) {
                    base_interp_threshold += ((p.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
                }
                if (params.dataset == 0 && p.is_distort) {
                    // Assuming enlarge_distort > 1.0 only when distortion enlargement is active
                    if (params.enlarge_distort > 1.0f) {
                       base_interp_threshold *= params.enlarge_distort;
                    }
                }
                break;

            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                case_str = "CASE2";
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                base_interp_threshold = params.interp_thr2;
                threshold_scaling = static_cast<float>(std::max(1, map_index_diff));
                break;

            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
            default:
                case_str = "CASE3"; // Default to Case 3 settings
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                base_interp_threshold = params.interp_thr3;
                threshold_scaling = static_cast<float>(std::max(1, map_index_diff));
                break;
        }

        float current_interp_threshold = base_interp_threshold * threshold_scaling;

        // --- DEBUG: Print Input and Parameters ---
        std::cout << std::fixed << std::setprecision(5); // Ensure consistent float output
        std::cout << "[MapCheck " << case_str << "] Point p: H=" << p.hor_ind << " V=" << p.ver_ind
                  << " Az=" << p.vec(0) << " El=" << p.vec(1) << " D=" << p.vec(2) << " T=" << p.time
                  << " Distort=" << p.is_distort << std::endl;
        std::cout << "[MapCheck " << case_str << "] Params: BaseThr=" << base_interp_threshold
                  << " Scale=" << threshold_scaling << " FinalThr=" << current_interp_threshold
                  << " MapDiff=" << map_index_diff << " InterpType=" << static_cast<int>(interp_neighbor_type) << std::endl;


        // --- 2. Check if Point is in Self-Region ---
        bool point_is_inside_self_box =
            (p.local.x() >= params.self_x_b && p.local.x() <= params.self_x_f &&
             p.local.y() >= params.self_y_r && p.local.y() <= params.self_y_l);

        if (point_is_inside_self_box) {
             std::cout << "[MapCheck " << case_str << "] Point Local Coords: (" << p.local.x() << ", " << p.local.y() << ", " << p.local.z() << ")" << std::endl;
             std::cout << "[MapCheck " << case_str << "] Self Box X: [" << params.self_x_b << ", " << params.self_x_f << "], Y: [" << params.self_y_r << ", " << params.self_y_l << "]" << std::endl;
             std::cout << "[MapCheck " << case_str << "] -> Returning FALSE (Point inside self-box)" << std::endl;
            return false;
        }

        // --- 3. Perform Interpolation ---
        // Assuming PointCloudUtils::interpolateDepth also includes its own debug prints now
        PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
            p, map_info, params, interp_neighbor_type);

        // --- 4. Evaluate Result ---
        if (result.status == PointCloudUtils::InterpolationStatus::SUCCESS) {
            float actual_diff = std::fabs(result.depth - p.vec(2));
            bool is_consistent = actual_diff < current_interp_threshold;

            // --- DEBUG: Print Final Comparison ---
            std::cout << "[MapCheck " << case_str << "] Interpolation SUCCESS: CenterDepth=" << p.vec(2)
                      << ", InterpDepth=" << result.depth
                      << ", AbsDiff=" << actual_diff
                      << ", Threshold=" << current_interp_threshold
                      << ". Consistent=" << (is_consistent ? "TRUE" : "FALSE") << std::endl;
            std::cout << "[MapCheck " << case_str << "] -> Returning " << (is_consistent ? "TRUE" : "FALSE") << std::endl;

            return is_consistent;
        } else {
            // Interpolation failed
            // --- DEBUG: Print Interpolation Failure ---
            std::cout << "[MapCheck " << case_str << "] Interpolation FAILED: Status="
                      << interpolationStatusToString(result.status) << std::endl;
            std::cout << "[MapCheck " << case_str << "] -> Returning FALSE" << std::endl;

            return false; // Inconsistent - couldn't verify via interpolation
        }
    }

    bool checkDepthConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type)
    {
        // --- Select parameters based on check_type ---
        int hor_num, ver_num;
        float hor_thr, ver_thr, max_thr, depth_thr, k_depth;
        const char* case_str = ""; // For printing
    
        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                hor_num = params.depth_cons_hor_num2;
                ver_num = params.depth_cons_ver_num2;
                hor_thr = params.depth_cons_hor_thr2;
                ver_thr = params.depth_cons_ver_thr2;
                max_thr = params.depth_cons_depth_max_thr2;
                depth_thr = params.depth_cons_depth_thr2;
                k_depth = params.k_depth2;
                case_str = "CASE2";
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                hor_num = params.depth_cons_hor_num3;
                ver_num = params.depth_cons_ver_num3;
                hor_thr = params.depth_cons_hor_thr3;
                ver_thr = params.depth_cons_ver_thr3;
                max_thr = params.depth_cons_depth_max_thr3;
                depth_thr = params.depth_cons_depth_thr3;
                k_depth = params.k_depth3;
                case_str = "CASE3";
                break;
            case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                throw std::invalid_argument("checkDepthConsistency is not applicable for CASE1_FALSE_REJECTION.");
        }
    
        // --- DEBUG: Print Input and Parameters ---
        std::cout << std::fixed << std::setprecision(5); // For consistent float output
        std::cout << "[DepthCheck " << case_str << "] Point p: H=" << p.hor_ind << " V=" << p.ver_ind
                  << " Az=" << p.vec(0) << " El=" << p.vec(1) << " D=" << p.vec(2) << " T=" << p.time << std::endl;
        std::cout << "[DepthCheck " << case_str << "] Params: hor_num=" << hor_num << " ver_num=" << ver_num
                  << " hor_thr=" << hor_thr << " ver_thr=" << ver_thr << " max_thr=" << max_thr
                  << " depth_thr=" << depth_thr << " k_depth=" << k_depth << " frame_dur=" << params.frame_dur << std::endl;
    
        // --- Initialize counters ---
        float sum_abs_diff_close = 0;
        int count_close = 0;
        int count_farther = 0;
        int count_closer = 0;
        int count_total_considered = 0;
        int neighbors_in_map_count = 0; // Count how many neighbors we iterate over
    
        // --- Iterate through neighborhood defined by grid indices ---
        for (int ind_hor = -hor_num; ind_hor <= hor_num; ++ind_hor) {
            for (int ind_ver = -ver_num; ind_ver <= ver_num; ++ind_ver) {
                int neighbor_hor_ind = (p.hor_ind + ind_hor + MAX_1D) % MAX_1D;
                int neighbor_ver_ind = p.ver_ind + ind_ver;
    
                if (neighbor_ver_ind < 0 || neighbor_ver_ind >= MAX_1D_HALF) continue;
    
                int pos_new = neighbor_hor_ind * MAX_1D_HALF + neighbor_ver_ind;
                const auto& points_in_pixel = map_info.depth_map[pos_new];
    
                // --- Process points within the neighbor pixel ---
                for (const auto& neighbor_ptr : points_in_pixel) {
                    if (!neighbor_ptr) continue;
                    const point_soph& neighbor = *neighbor_ptr;
                    neighbors_in_map_count++; // Count this potential neighbor
    
                    // --- DEBUG: Print Neighbor Info ---
                    std::cout << "[DepthCheck " << case_str << "] Checking Neighbor: H=" << neighbor.hor_ind << " V=" << neighbor.ver_ind
                              << " Az=" << neighbor.vec(0) << " El=" << neighbor.vec(1) << " D=" << neighbor.vec(2)
                              << " T=" << neighbor.time << " Dyn=" << neighbor.dyn << std::endl;
    
                    // Filter neighbors based on time difference and angular proximity
                    float time_diff = std::fabs(neighbor.time - p.time);
                    float az_diff = std::fabs(neighbor.vec(0) - p.vec(0));
                    float el_diff = std::fabs(neighbor.vec(1) - p.vec(1));
                    bool time_ok = time_diff < params.frame_dur;
                    bool az_ok = az_diff < hor_thr;
                    bool el_ok = el_diff < ver_thr;
                    bool status_ok = neighbor.dyn == STATIC;
    
                    // --- DEBUG: Print Filter Results ---
                    std::cout << "[DepthCheck " << case_str << "]   Filters: time_diff=" << time_diff << (time_ok ? " (OK)" : " (FAIL)")
                              << ", az_diff=" << az_diff << (az_ok ? " (OK)" : " (FAIL)")
                              << ", el_diff=" << el_diff << (el_ok ? " (OK)" : " (FAIL)")
                              << ", status=" << neighbor.dyn << (status_ok ? " (OK)" : " (FAIL)") << std::endl;
    
                    if (time_ok && az_ok && el_ok)
                    {
                        count_total_considered++;
    
                        if (status_ok) {
                            float depth_diff = p.vec(2) - neighbor.vec(2);
                            float abs_depth_diff = std::fabs(depth_diff);
    
                            // --- DEBUG: Print Depth Comparison ---
                            std::cout << "[DepthCheck " << case_str << "]   Static Neighbor Considered: depth_diff=" << depth_diff
                                      << ", abs_depth_diff=" << abs_depth_diff << ", max_thr=" << max_thr << std::endl;
    
                            if (abs_depth_diff < max_thr) {
                                count_close++;
                                sum_abs_diff_close += abs_depth_diff;
                                std::cout << "[DepthCheck " << case_str << "]     -> Increment count_close (now " << count_close << ")" << std::endl;
                            } else if (depth_diff > 0) {
                                count_farther++;
                                std::cout << "[DepthCheck " << case_str << "]     -> Increment count_farther (now " << count_farther << ")" << std::endl;
                            } else {
                                count_closer++;
                                std::cout << "[DepthCheck " << case_str << "]     -> Increment count_closer (now " << count_closer << ")" << std::endl;
                            }
                        } else {
                             std::cout << "[DepthCheck " << case_str << "]   Neighbor passed time/angle but not STATIC." << std::endl;
                        }
                    } else {
                         std::cout << "[DepthCheck " << case_str << "]   Neighbor failed time/angle filter." << std::endl;
                    }
                }
            }
        }
    
        // --- DEBUG: Print Final Counts ---
        std::cout << std::fixed << std::setprecision(5); // Ensure precision is set
        std::cout << "[DepthCheck " << case_str << "] Final Counts: total_considered=" << count_total_considered
                << ", close=" << count_close << ", farther=" << count_farther << ", closer=" << count_closer
                << ", sum_abs_diff_close=" << sum_abs_diff_close << ", neighbors_in_map=" << neighbors_in_map_count << std::endl;

        // Calculate how many static neighbors were actually categorized
        int static_neighbors_evaluated = count_close + count_farther + count_closer;
        std::cout << "[DepthCheck " << case_str << "] Static Neighbors Evaluated: " << static_neighbors_evaluated << std::endl;


        // --- Final consistency decision logic (REVISED) ---

        // If no neighbors met the initial time/angle criteria OR
        // if neighbors met time/angle but NONE were STATIC and categorized, consider inconsistent.
        if (count_total_considered == 0 || static_neighbors_evaluated == 0) {
            std::cout << "[DepthCheck " << case_str << "] -> Returning FALSE (No suitable STATIC neighbors found/evaluated)" << std::endl;
            return false; // MODIFIED: Return false if no static points were evaluated
        }

        // Check average depth difference for 'close' static neighbors (Rule 1)
        if (count_close > 1) {
            float avg_abs_diff_close = sum_abs_diff_close / static_cast<float>(count_close - 1);
            float current_depth_threshold = std::max(depth_thr, k_depth * p.vec(2));
            std::cout << "[DepthCheck " << case_str << "] Rule 1 Check: avg_abs_diff_close=" << avg_abs_diff_close
                    << ", current_depth_threshold=" << current_depth_threshold << std::endl;
            if (avg_abs_diff_close > current_depth_threshold) {
                std::cout << "[DepthCheck " << case_str << "] -> Returning FALSE (Rule 1 Failed: Avg diff too high)" << std::endl;
                return false;
            }
            std::cout << "[DepthCheck " << case_str << "] Rule 1 Passed." << std::endl;
        } else {
            std::cout << "[DepthCheck " << case_str << "] Rule 1 Skipped (count_close <= 1)." << std::endl;
        }

        // Check if point p is consistently closer OR farther than *all* significantly different static neighbors (Rule 2)
        // This rule only applies if Rule 1 didn't already cause a 'false' return.
        std::cout << "[DepthCheck " << case_str << "] Rule 2 Check: count_closer=" << count_closer << ", count_farther=" << count_farther << std::endl;
        if (count_closer == 0 || count_farther == 0) {
            // No mix of significantly closer/farther points among static neighbors. -> Consistent
            std::cout << "[DepthCheck " << case_str << "] -> Returning TRUE (Rule 2 Passed: No mix of closer/farther)" << std::endl;
            return true;
        } else {
            // Mix of significantly closer AND farther static neighbors found. -> Inconsistent
            std::cout << "[DepthCheck " << case_str << "] -> Returning FALSE (Rule 2 Failed: Mix of closer/farther)" << std::endl;
            return false;
        }
    }

    bool checkAccelerationLimit(
        float velocity1,
        float velocity2,
        double time_delta_between_velocity_centers,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type)
    {
        // --- Select parameters and case string based on check_type ---
        float acceleration_threshold;
        const char* case_str = ""; // For printing

        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                acceleration_threshold = params.acc_thr2;
                case_str = "CASE2";
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                acceleration_threshold = params.acc_thr3;
                case_str = "CASE3";
                break;
            case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                // Throw exception *before* attempting to print using case_str
                throw std::invalid_argument("checkAccelerationLimit is not applicable for CASE1_FALSE_REJECTION.");
            default:
                 // Handle unexpected check_type if necessary, maybe throw or default
                 throw std::invalid_argument("checkAccelerationLimit called with unexpected check_type.");
        }

        // --- DEBUG: Print Inputs and Parameters ---
        std::cout << std::fixed << std::setprecision(5); // Set consistent float formatting
        std::cout << "[AccelCheck " << case_str << "] Inputs: V1=" << velocity1 << ", V2=" << velocity2
                  << ", DeltaT=" << time_delta_between_velocity_centers << std::endl;
        std::cout << "[AccelCheck " << case_str << "] Param: AccelThr=" << acceleration_threshold << std::endl;


        // Handle negligible time difference
        constexpr double epsilon_time = 1e-6; // 1 microsecond
        if (time_delta_between_velocity_centers <= epsilon_time) {
            constexpr float epsilon_vel = 1e-4; // Velocity tolerance for near-zero time delta
            float delta_v_abs = std::fabs(velocity1 - velocity2);
            bool is_consistent = delta_v_abs < epsilon_vel;

            // --- DEBUG: Print Near-Zero DeltaT Path ---
            std::cout << "[AccelCheck " << case_str << "] Near-Zero DeltaT detected (<= " << epsilon_time << ")." << std::endl;
            std::cout << "[AccelCheck " << case_str << "] Comparing |V1-V2|=" << delta_v_abs << " vs VelEpsilon=" << epsilon_vel
                      << ". Consistent=" << (is_consistent ? "TRUE" : "FALSE") << std::endl;
            std::cout << "[AccelCheck " << case_str << "] -> Returning " << (is_consistent ? "TRUE" : "FALSE") << std::endl;

            return is_consistent;
        }

        // Main Check: |v1 - v2| < delta_t * accel_threshold
        float delta_v_abs = std::fabs(velocity1 - velocity2);
        // Use double for the limit calculation as time_delta is double
        double velocity_change_limit = time_delta_between_velocity_centers * static_cast<double>(acceleration_threshold);
        bool is_consistent = delta_v_abs < velocity_change_limit;

        // --- DEBUG: Print Main Comparison Path ---
        std::cout << "[AccelCheck " << case_str << "] Main Check: Comparing |V1-V2|=" << delta_v_abs
                  << " vs Limit (DeltaT*AccelThr)=" << velocity_change_limit
                  << ". Consistent=" << (is_consistent ? "TRUE" : "FALSE") << std::endl;
        std::cout << "[AccelCheck " << case_str << "] -> Returning " << (is_consistent ? "TRUE" : "FALSE") << std::endl;

        return is_consistent;
    }

    bool checkOcclusionRelationship(
        const point_soph& potential_occluder,
        const point_soph& potential_occluded,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type)
    {
        // --- Select parameters based on check_type ---
        float occ_hor_thr, occ_ver_thr;
        float k_depth_max_thr, d_depth_max_thr;
        float base_depth_offset; // Combined occ_depth_thr2 or map_cons_depth_thr3
        float v_min_thr;
        bool is_case2 = false; // Flag for specific logic if needed

        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                occ_hor_thr = params.occ_hor_thr2;
                occ_ver_thr = params.occ_ver_thr2;
                k_depth_max_thr = params.k_depth_max_thr2;
                d_depth_max_thr = params.d_depth_max_thr2;
                base_depth_offset = params.occ_depth_thr2;
                v_min_thr = params.v_min_thr2;
                is_case2 = true;
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                occ_hor_thr = params.occ_hor_thr3;
                occ_ver_thr = params.occ_ver_thr3;
                k_depth_max_thr = params.k_depth_max_thr3;
                d_depth_max_thr = params.d_depth_max_thr3;
                base_depth_offset = params.map_cons_depth_thr3; // Note different param name
                v_min_thr = params.v_min_thr3;
                break;
            default:
                throw std::invalid_argument("checkOcclusionRelationship received an invalid check_type.");
        }

        // --- Initial Checks (Common to Case 2 & 3 IsOccluded) ---

        // Check for invalid status of the potentially occluded point
        if (potential_occluded.dyn == INVALID) {
            return false;
        }

        // Dataset 0 distortion check
        if (params.dataset == 0 && (potential_occluded.is_distort || potential_occluder.is_distort)) {
            return false;
        }

        // Self-occlusion check (using local coordinates)
        const auto& p_local = potential_occluder.local;
        const auto& p_occ_local = potential_occluded.local;
        if ((p_local(0) > params.self_x_b && p_local(0) < params.self_x_f && p_local(1) < params.self_y_l && p_local(1) > params.self_y_r) ||
            (p_occ_local(0) > params.self_x_b && p_occ_local(0) < params.self_x_f && p_occ_local(1) < params.self_y_l && p_occ_local(1) > params.self_y_r))
        {
            return false;
        }

        // --- Time Check ---
        // Occluder must be later in time than the occluded point for this logic
        double delta_t = potential_occluder.time - potential_occluded.time;
        if (delta_t <= 0) // Original used > 0 check, so <= 0 means false
        {
            return false;
        }

        // --- Core Occlusion Condition ---

        // Calculate dynamic depth threshold
        float depth_thr_adaptive = std::max(static_cast<float>(params.cutoff_value), k_depth_max_thr * (potential_occluder.vec(2) - d_depth_max_thr));
        float depth_thr_velocity = v_min_thr * static_cast<float>(delta_t);
        float depth_threshold = std::min(depth_thr_adaptive + base_depth_offset, depth_thr_velocity);

        // Apply distortion enlargement factor for dataset 0
        if (params.dataset == 0 && potential_occluder.is_distort) { // Original checked p.is_distort here
            depth_threshold *= params.enlarge_distort;
        }
        // Original Case 3 also checked p.is_distort, but Case 2 did not explicitly.
        // Let's assume it applies to the occluder (p) in both cases if dataset==0.
        // If this assumption is wrong, adjust the condition.

        // Check depth relationship: potential_occluded must be farther than potential_occluder by the threshold
        bool depth_check_passed = potential_occluded.vec(2) > potential_occluder.vec(2) + depth_threshold;

        if (!depth_check_passed) {
            return false;
        }

        // Check angular proximity (using spherical coordinates vec(0)=azimuth, vec(1)=elevation)
        bool angular_check_passed =
            std::fabs(potential_occluder.vec(0) - potential_occluded.vec(0)) < occ_hor_thr &&
            std::fabs(potential_occluder.vec(1) - potential_occluded.vec(1)) < occ_ver_thr;

        return angular_check_passed; // If depth and angular checks pass
    }

    bool findOcclusionRelationshipInMap(
        point_soph& point_to_update, // Non-const ref to allow updating indices
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type) // The type received from the caller
    {
        // --- Select parameters based on check_type ---
        int occ_hor_num, occ_ver_num;
        bool update_occu_index; // True to update occu_index, false to update is_occu_index
    
        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                occ_hor_num = params.occ_hor_num2;
                occ_ver_num = params.occ_ver_num2;
                update_occu_index = true;
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                occ_hor_num = params.occ_hor_num3;
                occ_ver_num = params.occ_ver_num3;
                update_occu_index = false;
                break;
            default:
                throw std::invalid_argument("findOcclusionRelationshipInMap received an invalid check_type.");
        }
    
        // --- Search Neighborhood ---
        // Ensure point_to_update has valid initial indices
        if (point_to_update.position < 0 || point_to_update.position >= MAX_2D_N) {
            // Or handle this error appropriately
            return false;
        }

        // Iterate through the search window centered on point_to_update's indices
        for (int ind_hor = -occ_hor_num; ind_hor <= occ_hor_num; ++ind_hor)
        {
            for (int ind_ver = -occ_ver_num; ind_ver <= occ_ver_num; ++ind_ver)
            {
                // Calculate neighbor cell index with wrap-around
                // Ensure intermediate calculations don't underflow with modulo
                int neighbor_hor_ind = (point_to_update.hor_ind + ind_hor + MAX_1D) % MAX_1D;
                int neighbor_ver_ind = (point_to_update.ver_ind + ind_ver + MAX_1D_HALF) % MAX_1D_HALF; // Assuming positive MAX_1D_HALF
                // Clamp vertical index just in case modulo resulted in negative (shouldn't with positive MAX_1D_HALF)
                // or if ver_ind was somehow out of range initially.
                neighbor_ver_ind = std::max(0, std::min(neighbor_ver_ind, MAX_1D_HALF - 1));


                int neighbor_pos = neighbor_hor_ind * MAX_1D_HALF + neighbor_ver_ind;

                // Bounds check for the calculated position
                if (neighbor_pos < 0 || neighbor_pos >= MAX_2D_N) {
                    continue; // Should not happen with correct modulo/clamping, but safety first
                }

                // Optimization: Check min depth in the neighbor cell
                // If the closest point in the neighbor cell is farther than the point we are checking,
                // then no point in that cell can satisfy the p_occ.vec(2) > p.vec(2) + threshold condition.
                if (map_info.min_depth_all[neighbor_pos] > point_to_update.vec(2)) // Using precomputed min_depth_all
                {
                    continue;
                }

                // Get points in the neighbor cell
                const std::vector<point_soph::Ptr>& points_in_pixel = map_info.depth_map[neighbor_pos];

                // Check each point in the neighbor cell
                for (int j = 0; j < points_in_pixel.size(); ++j)
                {
                    const point_soph::Ptr& p_neighbor_ptr = points_in_pixel[j];
                    if (!p_neighbor_ptr) continue; // Safety check for null pointers
                    const point_soph& p_neighbor = *p_neighbor_ptr;

                    // Check 1: Occlusion Relationship
                    // The call order matches the original: check relationship between point_to_update and p_neighbor
                    bool occlusion_holds = checkOcclusionRelationship(point_to_update, p_neighbor, params, check_type);

                    if (occlusion_holds) {
                        // Check 2: Depth Consistency of the neighbor point
                        // Assumes checkDepthConsistency exists and uses the corresponding type
                        bool depth_consistent = checkDepthConsistency(p_neighbor, map_info, params, check_type);

                        if (depth_consistent) {
                            // --- Match Found ---
                            // Update the correct index on point_to_update
                            if (update_occu_index) {
                                point_to_update.occu_index[0] = map_info.map_index;
                                point_to_update.occu_index[1] = neighbor_pos;
                                point_to_update.occu_index[2] = j;
                            } else {
                                point_to_update.is_occu_index[0] = map_info.map_index;
                                point_to_update.is_occu_index[1] = neighbor_pos;
                                point_to_update.is_occu_index[2] = j;
                            }
                            // Update occ_vec as in original code
                            point_to_update.occ_vec = point_to_update.vec;

                            return true; // Found the first valid neighbor
                        }
                    }
                } // End loop through points in cell
            } // End loop vertical offset
        } // End loop horizontal offset

        return false; // No suitable neighbor found
    }
    

} // namespace ConsistencyChecks