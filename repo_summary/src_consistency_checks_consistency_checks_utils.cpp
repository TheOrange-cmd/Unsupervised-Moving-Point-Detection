#include "consistency_checks/consistency_checks_utils.h"
#include "consistency_checks/consistency_checks.h"

#include <cmath>     // For std::fabs, std::max
#include <algorithm> // For std::max, std::min
#include <stdexcept> // For std::invalid_argument
#include <iomanip>   // For logging

// Logging
#include <spdlog/spdlog.h>

namespace ConsistencyChecks {

    // --- Definition of getCaseStringUtil ---
    const char* getCaseStringUtil(ConsistencyCheckType check_type) {
        // Use fully qualified enum members here because the enum is defined in this namespace
        switch (check_type) {
            case ConsistencyCheckType::CASE1_FALSE_REJECTION: return "CASE1";
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH: return "CASE2";
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH: return "CASE3";
            default: return "INVALID_CASE";
        }
    }

    float calculateInterpolationThreshold(
        const point_soph& p,
        ConsistencyCheckType check_type,
        int map_index_diff,
        const DynObjFilterParams& params)
    {
        auto logger = spdlog::get("Consistency"); // Or use "Utils" logger if preferred
        const char* case_str = getCaseStringUtil(check_type);

        float base_threshold = 0.0f;
        float scale_factor = 1.0f;
        bool apply_case1_adjustments = false;

        switch (check_type) {
            case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                base_threshold = params.interp_thr1;
                // Scaling for Case 1 is typically 1.0, not based on map_index_diff
                scale_factor = 1.0f;
                apply_case1_adjustments = true;
                break;
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                base_threshold = params.interp_thr2;
                scale_factor = static_cast<float>(std::max(1, map_index_diff)); // Scale by map difference
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                base_threshold = params.interp_thr3;
                scale_factor = static_cast<float>(std::max(1, map_index_diff)); // Scale by map difference
                break;
            default:
                if (logger) logger->error("[CalcInterpThr {}] ERROR: Invalid check_type received.", case_str);
                throw std::invalid_argument("calculateInterpolationThreshold received an invalid check_type.");
        }

        float current_threshold = base_threshold;
        bool depth_adj_applied = false;
        float depth_adj = 0.0f;
        bool distort_adj_applied = false;
        float distort_adj_factor = 1.0f;

        // Apply CASE1 specific adjustments *after* selecting base threshold
        if (apply_case1_adjustments) {
            // Depth-dependent term
            if (p.vec(2) > params.interp_start_depth1) {
                depth_adj = (p.vec(2) - params.interp_start_depth1) * params.interp_kp1; // Removed +interp_kd1 based on previous code structure review, double check original intent if needed
                current_threshold += depth_adj;
                depth_adj_applied = true;
            }
            // Distortion term (applied multiplicatively to the threshold *including* depth adjustment)
            // Note: Original code applied multiplicatively, ensure this matches intent.
            if (params.dataset == 0 && p.is_distort && params.enlarge_distort > 1.0f) {
                distort_adj_factor = params.enlarge_distort;
                current_threshold *= distort_adj_factor; // Apply multiplicatively
                distort_adj_applied = true;
            }
        }

        // Apply map difference scaling *last* (except for CASE1 where it's usually 1.0)
        current_threshold *= scale_factor;

        if (logger && logger->should_log(spdlog::level::trace)) { // Check level before formatting complex message
             logger->trace("[CalcInterpThr {}] Point D={:.3f} Distort={}. BaseThr={:.3f} DepthAdj={} ({:.3f}) DistortAdj={} ({:.3f}) Scale={:.1f} -> FinalThr={:.3f}",
                          case_str, p.vec(2), p.is_distort,
                          base_threshold, depth_adj_applied, depth_adj,
                          distort_adj_applied, distort_adj_factor,
                          scale_factor, current_threshold);
        }

        return current_threshold;
    }


    float calculateOcclusionDepthThreshold(
        const point_soph& occluder_point, // Point assumed to be closer
        ConsistencyCheckType check_type,
        const DynObjFilterParams& params)
    {
        auto logger = spdlog::get("Consistency"); // Or use "Utils" logger
        const char* case_str = getCaseStringUtil(check_type);

        float k_depth = 0.0f;
        float depth_thr = 0.0f; // Base threshold

        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                k_depth = params.k_depth2;
                depth_thr = params.depth_cons_depth_thr2;
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                k_depth = params.k_depth3;
                depth_thr = params.depth_cons_depth_thr3;
                break;
            case ConsistencyCheckType::CASE1_FALSE_REJECTION: // Invalid for this function
            default:
                if (logger) logger->error("[CalcOccDepthThr {}] ERROR: Invalid check_type received.", case_str);
                throw std::invalid_argument("calculateOcclusionDepthThreshold received an invalid check_type.");
        }

        // Calculate threshold: max(base_threshold, k_factor * occluder_depth)
        float occluder_depth = occluder_point.vec(2);
        float calculated_threshold = std::max(depth_thr, k_depth * occluder_depth);

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[CalcOccDepthThr {}] Occluder D={:.3f}. BaseThr={:.3f} KDepth={:.3f} -> FinalThr={:.3f}",
                         case_str, occluder_depth, depth_thr, k_depth, calculated_threshold);
        }

        return calculated_threshold;
    }

} // namespace ConsistencyChecks