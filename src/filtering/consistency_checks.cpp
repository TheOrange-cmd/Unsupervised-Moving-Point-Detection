#include "filtering/consistency_checks.h"
#include "point_cloud_utils/point_cloud_utils.h" // For interpolateDepth, isSelfPoint
#include <cmath>     // For std::fabs
#include <algorithm> // For std::max

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

// Implementations for other consistency checks go here...

} // namespace ConsistencyChecks