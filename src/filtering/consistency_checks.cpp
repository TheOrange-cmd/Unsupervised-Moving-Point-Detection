#include "filtering/consistency_checks.h"
#include "point_cloud_utils/point_cloud_utils.h" // For interpolateDepth, isSelfPoint
#include <cmath>     // For std::fabs
#include <algorithm> // For std::max
#include <iomanip>

namespace ConsistencyChecks {

    bool checkMapConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        int map_index_diff)
    {
        // --- 1. Determine Interpolation Type and Thresholds based on Case ---
        PointCloudUtils::InterpolationNeighborType interp_neighbor_type;
        float base_interp_threshold = 0.0f;
        float threshold_scaling = 1.0f; // Scaling factor based on map age (for Case 2/3)

        switch (check_type) {
            case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::STATIC_ONLY;
                base_interp_threshold = params.interp_thr1;
                // Add Case 1 specific scaling based on depth
                if (p.vec(2) > params.interp_start_depth1) {
                    base_interp_threshold += ((p.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
                }
                // Note: Original Case 1 also had angular/depth thresholds for neighbor search,
                // and specific enlargements (distort, low elevation). We are omitting the
                // neighbor search part here and focusing on the interpolation check.
                // Apply distortion enlargement to the interpolation threshold for Case 1 if applicable.
                if (params.dataset == 0 && p.is_distort) {
                    // Assuming enlarge_distort is a parameter scaling factor (e.g., 1.5)
                    // Need to add enlarge_distort to DynObjFilterParams if not already there.
                    // Using a placeholder value if not available:
                    const float enlarge_factor = params.enlarge_distort; // Get from params
                    base_interp_threshold *= enlarge_factor;
                }
                break;

            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                base_interp_threshold = params.interp_thr2;
                // Apply scaling based on map age (difference in indices)
                threshold_scaling = static_cast<float>(std::max(1, map_index_diff)); // Ensure scaling is at least 1
                break;

            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
            default: // Default to Case 3 settings if type is unexpected
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                base_interp_threshold = params.interp_thr3;
                // Apply scaling based on map age (difference in indices)
                threshold_scaling = static_cast<float>(std::max(1, map_index_diff)); // Ensure scaling is at least 1
                // Note: Original Case 3 had cur_v_min threshold enlarged for distortion,
                // but not the interpolation threshold itself.
                break;
        }

        float current_interp_threshold = base_interp_threshold * threshold_scaling;
        // Apply a minimum cutoff to the threshold, similar to original logic?
        // e.g., current_interp_threshold = std::max(params.cutoff_value, current_interp_threshold);
        // Let's omit this for now unless params like interp_thr* are expected to be very small.


        // --- 2. Check if Point is in Self-Region (Skip Interpolation If So) ---
        // Original code performed interpolation *only if* the point was OUTSIDE the self-box.
        // Define the self-box using parameters from the loaded config.
        // Assuming p.local are the coordinates to check against the self-box.
        bool point_is_inside_self_box =
            (p.local.x() >= params.self_x_b && p.local.x() <= params.self_x_f &&
            p.local.y() >= params.self_y_r && p.local.y() <= params.self_y_l);

        if (point_is_inside_self_box) {
            // Points within the self-region are not checked via interpolation.
            return false;
        }

        // --- 3. Perform Interpolation ---
        PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
            p, map_info, params, interp_neighbor_type);

        // --- 4. Evaluate Result ---
        if (result.status == PointCloudUtils::InterpolationStatus::SUCCESS) {
            // Check if the interpolated depth is close enough to the actual depth
            if (std::fabs(result.depth - p.vec(2)) < current_interp_threshold) {
                return true; // Consistent with the map based on interpolation
            } else {
                // std::cout << "Inconsistent: Depth diff " << std::fabs(result.depth - p.vec(2)) << " >= " << current_interp_threshold << std::endl; // Debug
                return false; // Inconsistent - depth differs too much
            }
        } else {
            // Interpolation failed (NOT_ENOUGH_NEIGHBORS or NO_VALID_TRIANGLE)
            // Original code returned false in these scenarios when interpolation was attempted.
            // std::cout << "Inconsistent: Interpolation failed (" << static_cast<int>(result.status) << ")" << std::endl; // Debug
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

// Implementations for other consistency checks go here...

} // namespace ConsistencyChecks