/**
 * @file consistency_checks_occlusion.cpp
 * @brief Implements occlusion-related consistency checks.
*/

#include "consistency_checks/consistency_checks.h" // Includes function declarations, enums, DepthMap, point_soph, DynObjFilterParams
#include "consistency_checks/consistency_checks_utils.h" 

// Include necessary utility headers
#include "point_cloud_utils/point_validity.h"      // For isSelfPoint
#include "point_cloud_utils/grid_search_utils.h"   // For forEachNeighborCell

#include <cmath>     // For std::fabs, std::max, std::min
#include <algorithm> // For std::max, std::min
#include <iomanip>   // For logging output formatting
#include <vector>    // For std::vector

// Logging
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h> // Required to log custom types like DynObjLabel via streams


namespace ConsistencyChecks {

    // Remove the local getCaseString, use the one from utils
    // inline const char* getCaseString(ConsistencyCheckType check_type) { ... } // <<--- REMOVE THIS

    // --- Occlusion Relationship Check (Pairwise - Refactored) ---
    bool checkOcclusionRelationship(
        const point_soph& potential_occluder, // p
        const point_soph& potential_occluded, // p_occ
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type)
    {
        auto logger = spdlog::get("Consistency");

        // --- Select parameters and case string ---
        float occ_hor_thr, occ_ver_thr;
        // k_depth_max_thr, d_depth_max_thr removed - handled by utility
        float base_depth_offset; // Still need this offset
        float v_min_thr;
        const char* case_str = getCaseStringUtil(check_type); // <<--- USE UTILITY

        switch (check_type) {
            case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                occ_hor_thr = params.occ_hor_thr2;
                occ_ver_thr = params.occ_ver_thr2;
                // k_depth_max_thr = params.k_depth_max_thr2; // Removed
                // d_depth_max_thr = params.d_depth_max_thr2; // Removed
                base_depth_offset = params.occ_depth_thr2;
                v_min_thr = params.v_min_thr2;
                break;
            case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                occ_hor_thr = params.occ_hor_thr3;
                occ_ver_thr = params.occ_ver_thr3;
                // k_depth_max_thr = params.k_depth_max_thr3; // Removed
                // d_depth_max_thr = params.d_depth_max_thr3; // Removed
                base_depth_offset = params.map_cons_depth_thr3; // Note different param name
                v_min_thr = params.v_min_thr3;
                break;
            default:
                if (logger) logger->error("[OccRelCheck {}] ERROR: Invalid check_type received.", case_str);
                throw std::invalid_argument("checkOcclusionRelationship received an invalid check_type.");
        }

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[OccRelCheck {}] Occluder (P):  H={} V={} D={:.3f} T={:.3f} Dist={} Dyn={}",
                        case_str, potential_occluder.hor_ind, potential_occluder.ver_ind, potential_occluder.vec(2), potential_occluder.time, potential_occluder.is_distort, potential_occluder.dyn);
            logger->trace("[OccRelCheck {}] Occluded (PO): H={} V={} D={:.3f} T={:.3f} Dist={} Dyn={}",
                        case_str, potential_occluded.hor_ind, potential_occluded.ver_ind, potential_occluded.vec(2), potential_occluded.time, potential_occluded.is_distort, potential_occluded.dyn);
            // Updated log to remove KDepth, DDepth
            logger->trace("[OccRelCheck {}] Params: HorThr={:.3f} VerThr={:.3f} BaseOffset={:.3f} VMinThr={:.3f}",
                        case_str, occ_hor_thr, occ_ver_thr, base_depth_offset, v_min_thr);
        }

        // --- Initial Checks ---
        if (potential_occluded.dyn == DynObjLabel::INVALID) {
            if (logger) logger->trace("[OccRelCheck {}] -> Returning FALSE (Occluded point dyn status is INVALID)", case_str);
            return false;
        }

        bool p_in_self = PointCloudUtils::isSelfPoint(potential_occluder.local, params);
        bool pocc_in_self = PointCloudUtils::isSelfPoint(potential_occluded.local, params);

        if (p_in_self || pocc_in_self) {
            if (logger) logger->trace("[OccRelCheck {}] -> Returning FALSE (Self-occlusion check failed: P_in={}, PO_in={})", case_str, p_in_self, pocc_in_self);
            return false;
        }

        double delta_t = potential_occluder.time - potential_occluded.time;
        if (delta_t <= 0) {
            if (logger) logger->trace("[OccRelCheck {}] -> Returning FALSE (Time delta check failed: DeltaT={:.4f} <= 0)", case_str, delta_t);
            return false;
        }

        // --- Core Occlusion Condition ---

        // --- Calculate dynamic depth threshold using Utility ---
        // 1. Get the base adaptive threshold from the utility function (depends on occluder)
        float adaptive_threshold = calculateOcclusionDepthThreshold(
            potential_occluder, check_type, params
        );

        // 2. Calculate the velocity-based limit
        float depth_thr_velocity = v_min_thr * static_cast<float>(delta_t);

        // 3. Combine: Use the adaptive threshold + offset, but cap it by the velocity limit
        float depth_threshold_base = std::min(adaptive_threshold + base_depth_offset, depth_thr_velocity);

        // 4. Apply distortion enlargement (if applicable)
        float depth_threshold = depth_threshold_base; // Start with combined base
        bool distortion_enlargement_applied = false;
        if (params.dataset == 0 && potential_occluder.is_distort && params.enlarge_distort > 1.0f) {
            depth_threshold *= params.enlarge_distort;
            distortion_enlargement_applied = true;
        }

        if (logger && logger->should_log(spdlog::level::trace)) {
            // Updated log message
            logger->trace("[OccRelCheck {}] Depth Thr Calc: Adaptive(Util)={:.3f} BaseOffset={:.3f} VelocityBased={:.3f} CombinedBase={:.3f} DistEnlarge={} FinalDepthThr={:.3f}",
                        case_str, adaptive_threshold, base_depth_offset, depth_thr_velocity, depth_threshold_base, distortion_enlargement_applied, depth_threshold);
        }

        // --- Check depth relationship ---
        float required_occluded_depth = potential_occluder.vec(2) + depth_threshold;
        bool depth_check_passed = potential_occluded.vec(2) > required_occluded_depth;

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[OccRelCheck {}] Depth Check: PO.D={:.3f} vs (P.D + Thr)={:.3f}. Passed={}",
                        case_str, potential_occluded.vec(2), required_occluded_depth, depth_check_passed);
        }

        // --- Check angular proximity ---
        float az_diff = std::fabs(potential_occluder.vec(0) - potential_occluded.vec(0)); // Assuming vec(0) is azimuth
        float el_diff = std::fabs(potential_occluder.vec(1) - potential_occluded.vec(1)); // Assuming vec(1) is elevation
        bool angular_check_passed = (az_diff < occ_hor_thr) && (el_diff < occ_ver_thr);

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[OccRelCheck {}] Angular Check: |AzDiff|={:.4f} vs ThrH={:.3f}, |ElDiff|={:.4f} vs ThrV={:.3f}. Passed={}",
                        case_str, az_diff, occ_hor_thr, el_diff, occ_ver_thr, angular_check_passed);
        }

        // --- Final result ---
        bool overall_result = depth_check_passed && angular_check_passed;
        if (logger) logger->debug("[OccRelCheck {}] -> Returning {}", case_str, overall_result);
        return overall_result;
    }


    // --- Find Occlusion Relationship in Map (Neighborhood Search - Refactored) ---
    // No changes needed here related to threshold utils, but use getCaseStringUtil
    bool findOcclusionRelationshipInMap(
        point_soph& point_to_update, // P
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        DepthConsistencyChecker depth_checker)
    {
        auto logger = spdlog::get("Consistency");

        // --- Select parameters and case string ---
        int occ_hor_num, occ_ver_num;
        bool update_occu_index;
        const char* case_str = getCaseStringUtil(check_type); // <<--- USE UTILITY

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
                if (logger) logger->error("[FindOccRel {}] ERROR: Invalid check_type received.", case_str);
                throw std::invalid_argument("findOcclusionRelationshipInMap received an invalid check_type.");
        }

        if (logger && logger->should_log(spdlog::level::trace)) { // Added level check
            logger->trace("[FindOccRel {}] Checking point (P): H={} V={} Pos={} D={:.3f} T={:.3f}",
                        case_str, point_to_update.hor_ind, point_to_update.ver_ind, point_to_update.position, point_to_update.vec(2), point_to_update.time);
            logger->trace("[FindOccRel {}] Params: HorNum={} VerNum={} UpdateOccuIdx={} MapIdx={}",
                        case_str, occ_hor_num, occ_ver_num, update_occu_index, map_info.map_index);
        }

        // --- Input Validation ---
        if (point_to_update.position < 0 || point_to_update.position >= MAX_2D_N ||
            point_to_update.hor_ind < 0 || point_to_update.hor_ind >= MAX_1D ||
            point_to_update.ver_ind < 0 || point_to_update.ver_ind >= MAX_1D_HALF)
        {
            if (logger) logger->warn("[FindOccRel {}] -> Returning FALSE (Invalid initial point indices/position: H={}, V={}, Pos={})",
                                    case_str, point_to_update.hor_ind, point_to_update.ver_ind, point_to_update.position);
            return false;
        }

        // --- Search Neighborhood using forEachNeighborCell ---
        bool match_found = false;

        PointCloudUtils::forEachNeighborCell(
            point_to_update,
            occ_hor_num,
            occ_ver_num,
            true, // include_center
            true, // wrap_horizontal
            // Lambda function
            [&](int neighbor_pos) {
                if (match_found) return;
                if (neighbor_pos < 0 || neighbor_pos >= MAX_2D_N) return;

                // --- Case 3 Min Depth Optimization ---
                if (check_type == ConsistencyCheckType::CASE3_OCCLUDED_SEARCH) {
                    if (map_info.min_depth_all[neighbor_pos] >= 0.0f &&
                        map_info.min_depth_all[neighbor_pos] > point_to_update.vec(2))
                    {
                         if (logger && logger->should_log(spdlog::level::trace)) { // Added level check
                             logger->trace("[FindOccRel {}] Skipping Cell Pos={}: NeighborMinD={:.3f} > P.D={:.3f} (Opt for Case3)",
                                         case_str, neighbor_pos, map_info.min_depth_all[neighbor_pos], point_to_update.vec(2));
                         }
                        return;
                    }
                }

                const auto& points_in_pixel = map_info.depth_map[neighbor_pos];
                if (points_in_pixel.empty()) return;

                if (logger && logger->should_log(spdlog::level::trace)) { // Added level check
                    logger->trace("[FindOccRel {}] Cell Pos={} has {} points.", case_str, neighbor_pos, points_in_pixel.size());
                }

                for (int j = 0; j < points_in_pixel.size(); ++j) {
                    const point_soph::Ptr& p_neighbor_ptr = points_in_pixel[j];
                    if (!p_neighbor_ptr) continue;
                    const point_soph& p_neighbor = *p_neighbor_ptr;

                    // Check 1: Occlusion Relationship
                    bool occlusion_holds = false;
                    // Logging inside the loop can be verbose, ensure trace level is appropriate
                    if (check_type == ConsistencyCheckType::CASE2_OCCLUDER_SEARCH) {
                        if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[FindOccRel {}]   Calling checkOcclusionRelationship(P, PN[{}@{}])", case_str, j, neighbor_pos);
                        occlusion_holds = checkOcclusionRelationship(point_to_update, p_neighbor, params, check_type);
                    } else { // CASE3_OCCLUDED_SEARCH
                        if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[FindOccRel {}]   Calling checkOcclusionRelationship(PN[{}@{}], P)", case_str, j, neighbor_pos);
                        occlusion_holds = checkOcclusionRelationship(p_neighbor, point_to_update, params, check_type);
                    }
                     if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[FindOccRel {}]   ...Result: {}", case_str, occlusion_holds);

                    if (occlusion_holds) {
                        // Check 2: Depth Consistency of the *neighbor* point (PN)
                        if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[FindOccRel {}]   Occlusion holds. Calling depth_checker(PN[{}@{}])", case_str, j, neighbor_pos);
                        bool depth_consistent = depth_checker(p_neighbor, map_info, params, check_type);
                        if (logger && logger->should_log(spdlog::level::trace)) logger->trace("[FindOccRel {}]   ...Result: {}", case_str, depth_consistent);

                        if (depth_consistent) {
                            // --- Match Found ---
                            if (logger) {
                                logger->debug("[FindOccRel {}] *** Match Found! ***", case_str);
                                logger->trace("[FindOccRel {}]   Neighbor: CellPos={}, PointIdx={}", case_str, neighbor_pos, j);
                            }

                            if (update_occu_index) { // Case 2
                                point_to_update.occu_index[0] = map_info.map_index;
                                point_to_update.occu_index[1] = neighbor_pos;
                                point_to_update.occu_index[2] = j;
                                if (logger) logger->trace("[FindOccRel {}]   Updating P.occu_index = [{},{},{}]", case_str, point_to_update.occu_index[0], point_to_update.occu_index[1], point_to_update.occu_index[2]);
                            } else { // Case 3
                                point_to_update.is_occu_index[0] = map_info.map_index;
                                point_to_update.is_occu_index[1] = neighbor_pos;
                                point_to_update.is_occu_index[2] = j;
                                if (logger) logger->trace("[FindOccRel {}]   Updating P.is_occu_index = [{},{},{}]", case_str, point_to_update.is_occu_index[0], point_to_update.is_occu_index[1], point_to_update.is_occu_index[2]);
                            }
                            point_to_update.occ_vec = point_to_update.vec;
                            if (logger) logger->trace("[FindOccRel {}]   Updating P.occ_vec.", case_str);

                            match_found = true;
                            return; // Exit lambda for this cell and inner loop
                        }
                    }
                } // End loop through points in cell
            } // End lambda
        ); // End forEachNeighborCell

        // Return the final status
        if (logger) {
            if (match_found) {
                logger->debug("[FindOccRel {}] -> Returning TRUE (Match found)", case_str);
            } else {
                logger->debug("[FindOccRel {}] -> Returning FALSE (No match found after checking all cells)", case_str);
            }
        }
        return match_found;
    }

} // namespace ConsistencyChecks