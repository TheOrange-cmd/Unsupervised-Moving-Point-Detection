#include "filtering/consistency_checks.h"
#include "point_cloud_utils/point_cloud_utils.h" // For interpolateDepth, isSelfPoint
#include <cmath>     // For std::fabs
#include <algorithm> // For std::max
#include <iomanip>

namespace ConsistencyChecks {

    // Helper to convert InterpolationStatus to string for printing
    // inline const char* interpolationStatusToString(PointCloudUtils::InterpolationStatus status) {
    //     switch (status) {
    //         case PointCloudUtils::InterpolationStatus::SUCCESS: return "SUCCESS";
    //         case PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS: return "NOT_ENOUGH_NEIGHBORS";
    //         case PointCloudUtils::InterpolationStatus::NO_VALID_TRIANGLE: return "NO_VALID_TRIANGLE";
    //         default: return "UNKNOWN_STATUS";
    //     }
    // }

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
                // --- Use interp_thr1 for CASE1 threshold base ---
                base_interp_threshold = params.interp_thr1;
                if (p.vec(2) > params.interp_start_depth1) {
                    base_interp_threshold += ((p.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
                }
                if (params.dataset == 0 && p.is_distort && params.enlarge_distort > 1.0f) {
                   base_interp_threshold *= params.enlarge_distort;
                }
                break;
    
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                case_str = "CASE2";
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                // --- Use interp_thr2 for CASE2 threshold base ---
                base_interp_threshold = params.interp_thr2;
                threshold_scaling = static_cast<float>(std::max(1, map_index_diff));
                break;
    
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
            default:
                case_str = "CASE3"; // Default to Case 3 settings
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                 // --- Use interp_thr3 for CASE3 threshold base ---
                base_interp_threshold = params.interp_thr3;
                threshold_scaling = static_cast<float>(std::max(1, map_index_diff));
                break;
        }
        float current_interp_threshold = base_interp_threshold * threshold_scaling;
    
        // --- DEBUG: Print Input and Parameters ---
        // (Keep these prints as they are helpful)
        std::cout << std::fixed << std::setprecision(5);
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
        PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
            p, map_info, params, interp_neighbor_type);
    
        // --- 4. Evaluate Result ---
        if (result.status == PointCloudUtils::InterpolationStatus::SUCCESS) {
            float depth_diff = p.vec(2) - result.depth; // current_depth - interpolated_depth
            bool is_consistent = false;
    
            // --- REVERTED/CORRECTED CONSISTENCY LOGIC PER CASE ---
            switch (check_type) {
                case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                    // Consistent if current point is NOT significantly IN FRONT of the interpolated static surface.
                    is_consistent = (depth_diff >= -current_interp_threshold);
                    break;
                case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                     // Consistent if current point IS significantly IN FRONT of the interpolated surface.
                     is_consistent = (depth_diff < -current_interp_threshold);
                     break;
                case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                     // Consistent if current point IS significantly BEHIND the interpolated surface.
                     is_consistent = (depth_diff > current_interp_threshold);
                     break;
                default:
                     // Fallback: check if close (absolute difference)
                     is_consistent = (std::fabs(depth_diff) <= current_interp_threshold);
                     break;
            }
            // --- END REVERTED/CORRECTED LOGIC ---
    
            // --- DEBUG: Print Final Comparison ---
            std::cout << "[MapCheck " << case_str << "] Interpolation SUCCESS: CenterDepth=" << p.vec(2)
                      << ", InterpDepth=" << result.depth
                      << ", Diff=" << depth_diff // Signed diff is more informative here
                      << ", Threshold=" << current_interp_threshold
                      << ". Consistent=" << (is_consistent ? "TRUE" : "FALSE") << std::endl;
            std::cout << "[MapCheck " << case_str << "] -> Returning " << (is_consistent ? "TRUE" : "FALSE") << std::endl;
    
            return is_consistent;
    
        } else {
            // Interpolation failed (NOT_ENOUGH_NEIGHBORS or NO_VALID_TRIANGLE or other)
            // --- REVERTED: ANY failure means inconsistent ---
            std::cout << "[MapCheck " << case_str << "] Interpolation FAILED: Status="
                      << PointCloudUtils::interpolationStatusToString(result.status) << std::endl;
            std::cout << "[MapCheck " << case_str << "] -> Returning FALSE" << std::endl;
            return false;
            // --- END REVERTED LOGIC ---
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
                    bool status_ok = neighbor.dyn == DynObjLabel::STATIC;
    
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

    // Helper to get case string (can be shared)
    inline const char* getCaseString(ConsistencyCheckType check_type) {
        switch (check_type) {
            case ConsistencyCheckType::CASE1_FALSE_REJECTION: return "CASE1"; // Although not used here
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH: return "CASE2";
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH: return "CASE3";
            default: return "INVALID_CASE";
        }
    }

    bool checkOcclusionRelationship(
        const point_soph& potential_occluder, // p
        const point_soph& potential_occluded, // p_occ
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type)
    {
        // --- Select parameters and case string ---
        float occ_hor_thr, occ_ver_thr;
        float k_depth_max_thr, d_depth_max_thr;
        float base_depth_offset;
        float v_min_thr;
        const char* case_str = getCaseString(check_type);

        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                occ_hor_thr = params.occ_hor_thr2;
                occ_ver_thr = params.occ_ver_thr2;
                k_depth_max_thr = params.k_depth_max_thr2;
                d_depth_max_thr = params.d_depth_max_thr2;
                base_depth_offset = params.occ_depth_thr2;
                v_min_thr = params.v_min_thr2;
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
                // Log before throwing if possible, or just throw
                std::cerr << "[OccRelCheck " << case_str << "] ERROR: Invalid check_type received." << std::endl;
                throw std::invalid_argument("checkOcclusionRelationship received an invalid check_type.");
        }

        // --- DEBUG: Print Inputs and Selected Parameters ---
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "[OccRelCheck " << case_str << "] Occluder (P):  H=" << potential_occluder.hor_ind << " V=" << potential_occluder.ver_ind
                  << " D=" << potential_occluder.vec(2) << " T=" << potential_occluder.time << " Dist=" << potential_occluder.is_distort << " Dyn=" << potential_occluder.dyn << std::endl;
        std::cout << "[OccRelCheck " << case_str << "] Occluded (PO): H=" << potential_occluded.hor_ind << " V=" << potential_occluded.ver_ind
                  << " D=" << potential_occluded.vec(2) << " T=" << potential_occluded.time << " Dist=" << potential_occluded.is_distort << " Dyn=" << potential_occluded.dyn << std::endl;
        std::cout << "[OccRelCheck " << case_str << "] Params: HorThr=" << occ_hor_thr << " VerThr=" << occ_ver_thr
                  << " KDepth=" << k_depth_max_thr << " DDepth=" << d_depth_max_thr << " BaseOffset=" << base_depth_offset
                  << " VMinThr=" << v_min_thr << std::endl;


        // --- Initial Checks ---
        if (potential_occluded.dyn == DynObjLabel::INVALID) {
            std::cout << "[OccRelCheck " << case_str << "] -> Returning FALSE (Occluded point dyn status is INVALID)" << std::endl;
            return false;
        }

        // *** Replace manual self-check with calls to isSelfPoint ***
        bool p_in_self = PointCloudUtils::isSelfPoint(potential_occluder.local, params);
        bool pocc_in_self = PointCloudUtils::isSelfPoint(potential_occluded.local, params);
        // **********************************************************

        if (p_in_self || pocc_in_self) {
             std::cout << "[OccRelCheck " << case_str << "] -> Returning FALSE (Self-occlusion check failed: P_in=" << p_in_self << ", PO_in=" << pocc_in_self << ")" << std::endl;
            return false;
        }

        double delta_t = potential_occluder.time - potential_occluded.time;
        if (delta_t <= 0) {
             std::cout << "[OccRelCheck " << case_str << "] -> Returning FALSE (Time delta check failed: DeltaT=" << delta_t << " <= 0)" << std::endl;
            return false;
        }

        // --- Core Occlusion Condition ---
        float depth_thr_adaptive = std::max(static_cast<float>(params.cutoff_value), k_depth_max_thr * (potential_occluder.vec(2) - d_depth_max_thr));
        float depth_thr_velocity = v_min_thr * static_cast<float>(delta_t);
        float depth_threshold_base = std::min(depth_thr_adaptive + base_depth_offset, depth_thr_velocity);
        float depth_threshold = depth_threshold_base; // Start with base

        bool distortion_enlargement_applied = false;
        if (params.dataset == 0 && potential_occluder.is_distort && params.enlarge_distort > 1.0f) {
            depth_threshold *= params.enlarge_distort;
            distortion_enlargement_applied = true;
        }

        // --- DEBUG: Print Depth Threshold Calculation ---
        std::cout << "[OccRelCheck " << case_str << "] Depth Thr Calc: Adaptive=" << depth_thr_adaptive << " VelocityBased=" << depth_thr_velocity
                  << " BaseThr=" << depth_threshold_base << " DistEnlarge=" << (distortion_enlargement_applied ? "YES" : "NO")
                  << " FinalDepthThr=" << depth_threshold << std::endl;


        // Check depth relationship
        float required_occluded_depth = potential_occluder.vec(2) + depth_threshold;
        bool depth_check_passed = potential_occluded.vec(2) > required_occluded_depth;

        // --- DEBUG: Print Depth Check ---
        std::cout << "[OccRelCheck " << case_str << "] Depth Check: PO.D=" << potential_occluded.vec(2) << " vs (P.D + Thr)=" << required_occluded_depth
                  << ". Passed=" << (depth_check_passed ? "TRUE" : "FALSE") << std::endl;

        if (!depth_check_passed) {
            std::cout << "[OccRelCheck " << case_str << "] -> Returning FALSE (Depth check failed)" << std::endl;
            return false;
        }

        // Check angular proximity
        float az_diff = std::fabs(potential_occluder.vec(0) - potential_occluded.vec(0));
        float el_diff = std::fabs(potential_occluder.vec(1) - potential_occluded.vec(1));
        bool angular_check_passed = (az_diff < occ_hor_thr) && (el_diff < occ_ver_thr);

        // --- DEBUG: Print Angular Check ---
        std::cout << "[OccRelCheck " << case_str << "] Angular Check: |AzDiff|=" << az_diff << " vs ThrH=" << occ_hor_thr
                  << ", |ElDiff|=" << el_diff << " vs ThrV=" << occ_ver_thr
                  << ". Passed=" << (angular_check_passed ? "TRUE" : "FALSE") << std::endl;

        // Final result
        std::cout << "[OccRelCheck " << case_str << "] -> Returning " << (angular_check_passed ? "TRUE" : "FALSE") << " (Final angular check result)" << std::endl;
        return angular_check_passed;
    }

    bool findOcclusionRelationshipInMap(
        point_soph& point_to_update,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        DepthConsistencyChecker depth_checker)
    {
        // --- Select parameters and case string ---
        int occ_hor_num, occ_ver_num;
        bool update_occu_index;
        const char* case_str = getCaseString(check_type);

        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH: // point_to_update is potential occluder, searching for occluded
                occ_hor_num = params.occ_hor_num2;
                occ_ver_num = params.occ_ver_num2;
                update_occu_index = true; // Update p.occu_index with neighbor info
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH: // point_to_update is potential occluded, searching for occluder
                occ_hor_num = params.occ_hor_num3;
                occ_ver_num = params.occ_ver_num3;
                update_occu_index = false; // Update p.is_occu_index with neighbor info
                break;
            default:
                 std::cerr << "[FindOccRel " << case_str << "] ERROR: Invalid check_type received." << std::endl;
                throw std::invalid_argument("findOcclusionRelationshipInMap received an invalid check_type.");
        }

        // --- DEBUG: Print Entry Info ---
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "[FindOccRel " << case_str << "] Checking point (P): H=" << point_to_update.hor_ind << " V=" << point_to_update.ver_ind
                  << " Pos=" << point_to_update.position << " D=" << point_to_update.vec(2) << " T=" << point_to_update.time << std::endl;
        std::cout << "[FindOccRel " << case_str << "] Params: HorNum=" << occ_hor_num << " VerNum=" << occ_ver_num
                  << " UpdateOccuIdx=" << (update_occu_index ? "TRUE" : "FALSE") << " MapIdx=" << map_info.map_index << std::endl;


        // --- Input Validation ---
        if (point_to_update.position < 0 || point_to_update.position >= MAX_2D_N) {
            std::cout << "[FindOccRel " << case_str << "] -> Returning FALSE (Invalid initial point position: " << point_to_update.position << ")" << std::endl;
            return false;
        }
        if (point_to_update.hor_ind < 0 || point_to_update.hor_ind >= MAX_1D || point_to_update.ver_ind < 0 || point_to_update.ver_ind >= MAX_1D_HALF) {
             std::cout << "[FindOccRel " << case_str << "] -> Returning FALSE (Invalid initial point indices: H=" << point_to_update.hor_ind << ", V=" << point_to_update.ver_ind << ")" << std::endl;
             return false;
        }


        // --- Search Neighborhood ---
        for (int ind_hor = -occ_hor_num; ind_hor <= occ_hor_num; ++ind_hor) {
            for (int ind_ver = -occ_ver_num; ind_ver <= occ_ver_num; ++ind_ver) {
                // Skip the center point itself (should be added if not already present)
                // if (ind_hor == 0 && ind_ver == 0) continue;

                int neighbor_hor_ind = (point_to_update.hor_ind + ind_hor + MAX_1D) % MAX_1D;
                int neighbor_ver_ind = point_to_update.ver_ind + ind_ver;

                if (neighbor_ver_ind < 0 || neighbor_ver_ind >= MAX_1D_HALF) {
                    continue;
                }

                int neighbor_pos = neighbor_hor_ind * MAX_1D_HALF + neighbor_ver_ind;

                // Print constants once for verification
                static bool constants_printed = false;
                if (!constants_printed) {
                    std::cout << "[FindOccRel DEBUG] Constants: MAX_1D=" << MAX_1D << ", MAX_1D_HALF=" << MAX_1D_HALF << ", MAX_2D_N=" << MAX_2D_N << std::endl;
                    constants_printed = true;
                }
                // Print calculation details, especially for wrap-around
                if (point_to_update.hor_ind == 0 && ind_hor < 0) { // Focus on the specific test case
                     std::cout << "[FindOccRel DEBUG " << case_str << "] Wrap Check: ind_hor=" << ind_hor
                               << ", ind_ver=" << ind_ver << ", neigh_h=" << neighbor_hor_ind
                               << ", neigh_v=" << neighbor_ver_ind << ", calculated neighbor_pos=" << neighbor_pos
                               << std::endl;
                }


                // Bounds check (redundant if vertical check is done, but safe)
                if (neighbor_pos < 0 || neighbor_pos >= MAX_2D_N) {
                     std::cout << "[FindOccRel " << case_str << "] Skipping neighbor position out of bounds: " << neighbor_pos << " (H=" << neighbor_hor_ind << ", V=" << neighbor_ver_ind << ")" << std::endl;
                    continue;
                }

                // --- DEBUG: Print Current Neighbor Cell ---
                // This can be verbose, enable if needed:
                // std::cout << "[FindOccRel " << case_str << "] Checking Neighbor Cell: Pos=" << neighbor_pos << " (dH=" << ind_hor << ", dV=" << ind_ver << ")" << std::endl;

                // --- Min Depth Optimization (Only applicable and valid for Case 3) ---
                if (check_type == ConsistencyCheckType::CASE3_OCCLUDED_SEARCH) {
                    float neighbor_min_depth = map_info.min_depth_all[neighbor_pos];
                    // If the CLOSEST point in the neighbor cell (PN, potential occluder)
                    // is already FARTHER than the point being checked (P, potential occluded),
                    // then no point PN in that cell can satisfy P.D > PN.D + threshold.
                    if (neighbor_min_depth > point_to_update.vec(2)) {
                        std::cout << "[FindOccRel " << case_str << "] Skipping Cell Pos=" << neighbor_pos
                                << ": NeighborMinD=" << neighbor_min_depth << " > P.D=" << point_to_update.vec(2)
                                << " (Opt for Case3)" << std::endl;
                        continue; // Skip this cell
                    }
                }

                // Get points in the neighbor cell
                const std::vector<point_soph::Ptr>& points_in_pixel = map_info.depth_map[neighbor_pos];
                if (points_in_pixel.empty()) {
                    // Optional log for skipping empty cell
                    // std::cout << "[FindOccRel " << case_str << "] Skipping Cell Pos=" << neighbor_pos << ": Empty cell" << std::endl;
                    continue;
                }

                // --- DEBUG: Log cell size if not skipped ---
                // Log min depth here only if useful, maybe not necessary now optimization is clear
                std::cout << "[FindOccRel " << case_str << "] Cell Pos=" << neighbor_pos << " has " << points_in_pixel.size() << " points." << std::endl;

                // Check each point in the neighbor cell
                for (int j = 0; j < points_in_pixel.size(); ++j) {
                    const point_soph::Ptr& p_neighbor_ptr = points_in_pixel[j];
                    if (!p_neighbor_ptr) continue;
                    const point_soph& p_neighbor = *p_neighbor_ptr;

                    // --- DEBUG: Print Current Neighbor Point ---
                    // Can be verbose, enable if needed:
                    // std::cout << "[FindOccRel " << case_str << "]  Checking Pn[" << j << "]: H=" << p_neighbor.hor_ind << " V=" << p_neighbor.ver_ind << " D=" << p_neighbor.vec(2) << " T=" << p_neighbor.time << std::endl;


                    // Check 1: Occlusion Relationship
                    // Note the order of arguments depends on the case!
                    bool occlusion_holds = false;
                    if (check_type == ConsistencyCheckType::CASE2_OCCLUDER_SEARCH) {
                        // point_to_update (P) is potential occluder, p_neighbor (PN) is potential occluded
                         std::cout << "[FindOccRel " << case_str << "]   Calling checkOcclusionRelationship(P, PN[" << j << "])..." << std::endl;
                        occlusion_holds = checkOcclusionRelationship(point_to_update, p_neighbor, params, check_type);
                         std::cout << "[FindOccRel " << case_str << "]   ...Result: " << (occlusion_holds ? "TRUE" : "FALSE") << std::endl;
                    } else { // CASE3_OCCLUDED_SEARCH
                        // p_neighbor (PN) is potential occluder, point_to_update (P) is potential occluded
                         std::cout << "[FindOccRel " << case_str << "]   Calling checkOcclusionRelationship(PN[" << j << "], P)..." << std::endl;
                        occlusion_holds = checkOcclusionRelationship(p_neighbor, point_to_update, params, check_type);
                         std::cout << "[FindOccRel " << case_str << "]   ...Result: " << (occlusion_holds ? "TRUE" : "FALSE") << std::endl;
                    }


                    if (occlusion_holds) {
                        // Check 2: Depth Consistency of the *neighbor* point (p_neighbor)
                        // We need to ensure the neighbor point itself is consistent with its surroundings in the map.
                        // The check_type passed to checkDepthConsistency should likely match the overall check_type.
                        std::cout << "[FindOccRel " << case_str << "]   Occlusion holds. Calling checkDepthConsistency(PN[" << j << "])..." << std::endl;
                        bool depth_consistent = depth_checker(p_neighbor, map_info, params, check_type);
                        std::cout << "[FindOccRel " << case_str << "]   ...Result: " << (depth_consistent ? "TRUE" : "FALSE") << std::endl;


                        if (depth_consistent) {
                            // --- Match Found ---
                            std::cout << "[FindOccRel " << case_str << "] *** Match Found! ***" << std::endl;
                            std::cout << "[FindOccRel " << case_str << "]   Neighbor: CellPos=" << neighbor_pos << ", PointIdx=" << j << std::endl;
                            if (update_occu_index) {
                                point_to_update.occu_index[0] = map_info.map_index;
                                point_to_update.occu_index[1] = neighbor_pos;
                                point_to_update.occu_index[2] = j;
                                std::cout << "[FindOccRel " << case_str << "]   Updating P.occu_index = [" << point_to_update.occu_index[0] << "," << point_to_update.occu_index[1] << "," << point_to_update.occu_index[2] << "]" << std::endl;
                            } else {
                                point_to_update.is_occu_index[0] = map_info.map_index;
                                point_to_update.is_occu_index[1] = neighbor_pos;
                                point_to_update.is_occu_index[2] = j;
                                std::cout << "[FindOccRel " << case_str << "]   Updating P.is_occu_index = [" << point_to_update.is_occu_index[0] << "," << point_to_update.is_occu_index[1] << "," << point_to_update.is_occu_index[2] << "]" << std::endl;
                            }
                            // Update occ_vec (Copying current point's vec seems odd, maybe should copy neighbor's? Check original logic)
                            // Assuming original logic is correct:
                            point_to_update.occ_vec = point_to_update.vec;
                             std::cout << "[FindOccRel " << case_str << "]   Updating P.occ_vec." << std::endl;
                             std::cout << "[FindOccRel " << case_str << "] -> Returning TRUE (Match found)" << std::endl;

                            return true; // Found the first valid neighbor
                        }
                    }
                } // End loop through points in cell
            } // End loop vertical offset
        } // End loop horizontal offset

        std::cout << "[FindOccRel " << case_str << "] No suitable neighbor found after checking all cells." << std::endl;
        std::cout << "[FindOccRel " << case_str << "] -> Returning FALSE" << std::endl;
        return false; // No suitable neighbor found
    }


} // namespace ConsistencyChecks