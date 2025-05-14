/**
 * @file consistency_checks_map.cpp
 * @brief Implements map-based consistency checks (interpolation, depth neighborhood).
*/

#include "consistency_checks/consistency_checks.h" // Includes function declarations, enums, DepthMap, point_soph, DynObjFilterParams
#include "consistency_checks/consistency_checks_utils.h" // Include the new utility functions header

// Include necessary utility headers
#include "point_cloud_utils/interpolation_utils.h" // For interpolateDepth, statusToString
#include "point_cloud_utils/grid_search_utils.h"   // For forEachNeighborCell
#include "point_cloud_utils/point_validity.h"      // For isSelfPoint (if needed directly)

#include <cmath>     // For std::fabs, std::max
#include <algorithm> // For std::min, std::max
#include <iomanip>   // For logging output formatting

// Logging
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h> // Required to log custom types like DynObjLabel via streams

namespace ConsistencyChecks {

    // --- Map Consistency Check (Interpolation Based - Refactored) ---
    bool checkMapConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        int map_index_diff)
    {
        auto logger = spdlog::get("Consistency");

        // --- 1. Determine Interpolation Type and Case String ---
        PointCloudUtils::InterpolationNeighborType interp_neighbor_type;
        const char* case_str = getCaseStringUtil(check_type); // Use util function

        switch (check_type) {
            case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::STATIC_ONLY;
                break;
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
            default:
                interp_neighbor_type = PointCloudUtils::InterpolationNeighborType::ALL_VALID;
                // Keep warn level as is - no should_log needed typically
                if (check_type != ConsistencyCheckType::CASE3_OCCLUDED_SEARCH && logger) {
                     logger->warn("[MapCheck {}] Unexpected check_type ({}), defaulting to CASE3 logic.", case_str, static_cast<int>(check_type));
                }
                break;
        }

        // --- Calculate Interpolation Threshold using Utility ---
        float current_interp_threshold = calculateInterpolationThreshold(
            p, check_type, map_index_diff, params);

        // Apply should_log for trace
        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[MapCheck {}] Point p: H={} V={} D={:.3f} T={:.3f} Distort={}",
                        case_str, p.hor_ind, p.ver_ind, p.vec(2), p.time, p.is_distort);
            logger->trace("[MapCheck {}] Params: FinalThr={:.3f} MapDiff={} InterpType={}",
                        case_str, current_interp_threshold,
                        map_index_diff, static_cast<int>(interp_neighbor_type));
        }

        // --- 2. Check if Point is in Self-Region ---
        bool point_is_inside_self_box =
            (p.local.x() >= params.self_x_b && p.local.x() <= params.self_x_f &&
            p.local.y() >= params.self_y_r && p.local.y() <= params.self_y_l);

        if (point_is_inside_self_box) {
            // Apply should_log for trace and debug
            if (logger && logger->should_log(spdlog::level::trace)) {
                logger->trace("[MapCheck {}] Point Local Coords: ({:.3f}, {:.3f}, {:.3f})", case_str, p.local.x(), p.local.y(), p.local.z());
                logger->trace("[MapCheck {}] Self Box X: [{:.3f}, {:.3f}], Y: [{:.3f}, {:.3f}]", case_str, params.self_x_b, params.self_x_f, params.self_y_r, params.self_y_l);
            }
            if (logger && logger->should_log(spdlog::level::debug)) {
                logger->debug("[MapCheck {}] -> Returning FALSE (Point inside self-box)", case_str);
            }
            return false;
        }

        // --- 3. Perform Interpolation ---
        PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
            p, map_info, params, interp_neighbor_type);

        // --- 4. Evaluate Result ---
        if (result.status == PointCloudUtils::InterpolationStatus::SUCCESS) {
            float depth_diff = p.vec(2) - result.depth;
            bool is_consistent = false;

            switch (check_type) {
                case ConsistencyCheckType::CASE1_FALSE_REJECTION:
                    is_consistent = (depth_diff >= -current_interp_threshold);
                    break;
                case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                    is_consistent = (depth_diff < -current_interp_threshold);
                    break;
                case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                    is_consistent = (depth_diff > current_interp_threshold);
                    break;
                default:
                    is_consistent = (std::fabs(depth_diff) <= current_interp_threshold);
                    break;
            }

            // Apply should_log for trace and debug
            if (logger && logger->should_log(spdlog::level::trace)) {
                logger->trace("[MapCheck {}] Interpolation SUCCESS: CenterD={:.3f}, InterpD={:.3f}, Diff={:.3f}, Thr={:.3f}. Consistent={}",
                            case_str, p.vec(2), result.depth, depth_diff, current_interp_threshold, is_consistent);
            }
            if (logger && logger->should_log(spdlog::level::debug)) {
                logger->debug("[MapCheck {}] -> Returning {}", case_str, is_consistent);
            }
            return is_consistent;

        } else { // Interpolation failed
            // Apply should_log for trace and debug
            if (logger && logger->should_log(spdlog::level::trace)) {
                logger->trace("[MapCheck {}] Interpolation FAILED: Status={}",
                            case_str, PointCloudUtils::interpolationStatusToString(result.status));
            }
            if (logger && logger->should_log(spdlog::level::debug)) {
                logger->debug("[MapCheck {}] -> Returning FALSE (Interpolation Failed)", case_str);
            }
            return false;
        }
    }


    // --- Helper Struct for checkDepthConsistency ---
    struct NeighborDepthStats {
        float sum_abs_diff_close = 0.0f;
        int count_close = 0;
        int count_farther = 0;
        int count_closer = 0;
        int count_total_considered = 0;
        int static_neighbors_evaluated = 0;
    };


    // --- Depth Consistency Check (Neighborhood Based - Refactored) ---
    bool checkDepthConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type)
    {
        auto logger = spdlog::get("Consistency");

        // --- Select parameters based on check_type ---
        int hor_num, ver_num;
        float hor_thr, ver_thr, max_thr;
        const char* case_str = getCaseStringUtil(check_type);

        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                hor_num = params.depth_cons_hor_num2;
                ver_num = params.depth_cons_ver_num2;
                hor_thr = params.depth_cons_hor_thr2;
                ver_thr = params.depth_cons_ver_thr2;
                max_thr = params.depth_cons_depth_max_thr2;
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                hor_num = params.depth_cons_hor_num3;
                ver_num = params.depth_cons_ver_num3;
                hor_thr = params.depth_cons_hor_thr3;
                ver_thr = params.depth_cons_ver_thr3;
                max_thr = params.depth_cons_depth_max_thr3;
                break;
            case ConsistencyCheckType::CASE1_FALSE_REJECTION:
            default:
                // Keep error level as is
                if (logger) logger->error("[DepthCheck {}] Invalid check_type ({}) received.", case_str, static_cast<int>(check_type));
                throw std::invalid_argument("checkDepthConsistency received an invalid check_type.");
        }

        // Apply should_log for trace
        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[DepthCheck {}] Point p: H={} V={} D={:.3f} T={:.3f}",
                        case_str, p.hor_ind, p.ver_ind, p.vec(2), p.time);
            logger->trace("[DepthCheck {}] Params: hor_num={} ver_num={} hor_thr={:.3f} ver_thr={:.3f} max_thr={:.3f} frame_dur={:.3f}",
                        case_str, hor_num, ver_num, hor_thr, ver_thr, max_thr, params.frame_dur);
        }

        // --- Initialize stats ---
        NeighborDepthStats stats;

        // --- Iterate using forEachNeighborCell ---
        PointCloudUtils::forEachNeighborCell(
            p,
            hor_num,
            ver_num,
            true, // include_center
            true, // wrap_horizontal
            [&](int neighbor_pos) {
                if (neighbor_pos < 0 || neighbor_pos >= MAX_2D_N) return;
                const auto& points_in_pixel = map_info.depth_map[neighbor_pos];

                for (const auto& neighbor_ptr : points_in_pixel) {
                    if (!neighbor_ptr) continue;
                    const point_soph& neighbor = *neighbor_ptr;

                    float time_diff = std::fabs(neighbor.time - p.time);
                    float az_diff = std::fabs(neighbor.vec(0) - p.vec(0));
                    float el_diff = std::fabs(neighbor.vec(1) - p.vec(1));
                    bool time_ok = time_diff < params.frame_dur;
                    bool az_ok = az_diff < hor_thr;
                    bool el_ok = el_diff < ver_thr;
                    bool status_ok = neighbor.dyn == DynObjLabel::STATIC;

                    // Apply should_log for trace (inside loop)
                    if (logger && logger->should_log(spdlog::level::trace)) {
                         logger->trace("[DepthCheck {}]  Neighbor@({},{}): D={:.3f} T={:.3f} Dyn={} | Filters: dT={:.3f}({}) dAz={:.3f}({}) dEl={:.3f}({}) Stat({})",
                                     case_str, neighbor.hor_ind, neighbor.ver_ind, neighbor.vec(2), neighbor.time, neighbor.dyn,
                                     time_diff, time_ok, az_diff, az_ok, el_diff, el_ok, status_ok);
                    }

                    if (time_ok && az_ok && el_ok) {
                        stats.count_total_considered++;
                        if (status_ok) {
                            stats.static_neighbors_evaluated++;
                            float depth_diff = p.vec(2) - neighbor.vec(2);
                            float abs_depth_diff = std::fabs(depth_diff);

                            // Apply should_log for trace (inside loop)
                            if (logger && logger->should_log(spdlog::level::trace)) {
                                logger->trace("[DepthCheck {}]   Static Neighbor Considered: depth_diff={:.3f}, abs_depth_diff={:.3f}, max_thr={:.3f}", case_str, depth_diff, abs_depth_diff, max_thr);
                            }

                            if (abs_depth_diff < max_thr) {
                                stats.count_close++;
                                stats.sum_abs_diff_close += abs_depth_diff;
                                // Apply should_log for trace (inside loop)
                                if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[DepthCheck {}]     -> count_close = {}", case_str, stats.count_close);
                            } else if (depth_diff > 0) {
                                stats.count_farther++;
                                // Apply should_log for trace (inside loop)
                                if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[DepthCheck {}]     -> count_farther = {}", case_str, stats.count_farther);
                            } else {
                                stats.count_closer++;
                                // Apply should_log for trace (inside loop)
                                if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[DepthCheck {}]     -> count_closer = {}", case_str, stats.count_closer);
                            }
                        }
                    }
                } // End loop through points in pixel
            } // End lambda
        ); // End forEachNeighborCell

        // Apply should_log for trace
        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[DepthCheck {}] Final Stats: total_considered={} static_evaluated={} close={} farther={} closer={} sum_abs_diff_close={:.3f}",
                    case_str, stats.count_total_considered, stats.static_neighbors_evaluated,
                    stats.count_close, stats.count_farther, stats.count_closer, stats.sum_abs_diff_close);
        }

        // --- Final consistency decision logic ---
        if (stats.static_neighbors_evaluated == 0) {
            // Apply should_log for debug
            if (logger && logger->should_log(spdlog::level::debug)) {
                logger->debug("[DepthCheck {}] -> Returning FALSE (No suitable STATIC neighbors evaluated)", case_str);
            }
            return false;
        }

        // Rule 1: Check average depth difference for 'close' static neighbors
        if (stats.count_close > 0) {
            double avg_abs_diff_close = static_cast<double>(stats.sum_abs_diff_close) / static_cast<double>(stats.count_close);
            double current_depth_threshold = static_cast<double>(
                calculateOcclusionDepthThreshold(p, check_type, params)
            );

            // Apply should_log for trace
            if (logger && logger->should_log(spdlog::level::trace)) {
                 logger->trace("[DepthCheck {}] Rule 1 Check: avg_abs_diff_close={:.4f} vs current_depth_threshold={:.4f} (calculated for p.depth={:.3f})",
                              case_str, avg_abs_diff_close, current_depth_threshold, p.vec(2));
            }

            if (avg_abs_diff_close > current_depth_threshold) {
                // Apply should_log for debug
                if (logger && logger->should_log(spdlog::level::debug)) {
                    logger->debug("[DepthCheck {}] -> Returning FALSE (Rule 1 Failed: Avg diff > threshold)", case_str);
                }
                return false;
            }
            // Apply should_log for trace
            if (logger && logger->should_log(spdlog::level::trace)) {
                logger->trace("[DepthCheck {}] Rule 1 Passed.", case_str);
            }
        } else {
            // Apply should_log for trace
            if (logger && logger->should_log(spdlog::level::trace)) {
                logger->trace("[DepthCheck {}] Rule 1 Skipped (count_close == 0).", case_str);
            }
        }

        // Rule 2: Check mix of closer/farther neighbors
        // Apply should_log for trace
        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[DepthCheck {}] Rule 2 Check: count_closer={} count_farther={}", case_str, stats.count_closer, stats.count_farther);
        }
        if (stats.count_closer == 0 || stats.count_farther == 0) {
            // Apply should_log for debug
            if (logger && logger->should_log(spdlog::level::debug)) {
                logger->debug("[DepthCheck {}] -> Returning TRUE (Rule 2 Passed: No mix of closer/farther)", case_str);
            }
            return true;
        } else {
            // Apply should_log for debug
            if (logger && logger->should_log(spdlog::level::debug)) {
                logger->debug("[DepthCheck {}] -> Returning FALSE (Rule 2 Failed: Mix of closer/farther)", case_str);
            }
            return false;
        }
    }

} // namespace ConsistencyChecks