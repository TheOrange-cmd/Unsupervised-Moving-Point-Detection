// In src/filtering/dyn_obj_filter_consistency.cpp

#include "filtering/dyn_obj_filter.h" // Need full class definition
#include <vector>
#include <cmath> // For std::fabs
#include <string> // For std::string in logging
#include <algorithm> // For std::min

// Include necessary utility headers
#include "point_cloud_utils/point_validity.h"
#include "point_cloud_utils/interpolation_utils.h"
#include "point_cloud_utils/projection_utils.h"
#include "point_cloud_utils/logging_utils.h"
#include "point_cloud_utils/logging_context.h" // For g_current_logging_seq_id

// Logging
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>


// --- Setup Point For Map Check ---
// (Keep the existing setupPointForMapCheck function as it is)
void DynObjFilter::setupPointForMapCheck(
    const point_soph& p_world,
    const DepthMap& map,
    point_soph& p_map_frame)
{
    auto logger = spdlog::get("Filter_Consistency");
    point_soph p_spherical_in_map;
    PointCloudUtils::SphericalProjection(const_cast<point_soph&>(p_world),
                                         map.map_index, map.project_R, map.project_T,
                                         params_, p_spherical_in_map);
    if (PointCloudUtils::should_log_point_details(logger)) {
        logger->trace("setupPointForMapCheck: Point {}: Projection for MapIdx {} done.", p_world.original_index, map.map_index);
    }
    p_map_frame.local = map.project_R * p_world.glob + map.project_T;
    p_map_frame.vec = p_spherical_in_map.vec;
    p_map_frame.hor_ind = p_spherical_in_map.hor_ind;
    p_map_frame.ver_ind = p_spherical_in_map.ver_ind;
    p_map_frame.position = p_spherical_in_map.position;
    p_map_frame.time = p_world.time;
    p_map_frame.is_distort = p_world.is_distort;
    p_map_frame.glob = p_world.glob;
    p_map_frame.intensity = p_world.intensity;
    p_map_frame.dyn = p_world.dyn;
    p_map_frame.original_index = p_world.original_index;
    if (PointCloudUtils::should_log_point_details(logger)) {
         uint64_t current_seq_id = PointCloudUtils::g_current_logging_seq_id;
        logger->trace("setupPointForMapCheck: Point {} (Frame {}): In Map Frame: H={}, V={}, Pos={}, D={:.3f}",
            p_world.original_index, current_seq_id,
            p_map_frame.hor_ind, p_map_frame.ver_ind, p_map_frame.position, p_map_frame.vec(2));
        logger->trace("setupPointForMapCheck: Point {} (Frame {}): Local Coords (Rel Map): X={:.3f}, Y={:.3f}, Z={:.3f}",
            p_world.original_index, current_seq_id,
            p_map_frame.local.x(), p_map_frame.local.y(), p_map_frame.local.z());
        logger->trace("setupPointForMapCheck: Point {} (Frame {}): Map Pose T: ({:.3f}, {:.3f}, {:.3f})",
            p_world.original_index, current_seq_id, 
            map.project_T.x(), map.project_T.y(), map.project_T.z());
        // --- ADD THESE LINES ---
        logger->trace("setupPointForMapCheck: Point {} (Frame {}): Input p_world.glob: ({:.3f}, {:.3f}, {:.3f})",
                      p_world.original_index, current_seq_id,
                      p_world.glob.x(), p_world.glob.y(), p_world.glob.z());
        logger->trace("setupPointForMapCheck: Point {} (Frame {}): Input map.project_T: ({:.3f}, {:.3f}, {:.3f})",
                      p_world.original_index, current_seq_id,
                      map.project_T.x(), map.project_T.y(), map.project_T.z());
        // Optional: Log Rotation Matrix (can be verbose)
        // logger->trace("setupPointForMapCheck: Point {} (Frame {}): Input map.project_R:\n{}",
        //               p_world.original_index, current_seq_id, map.project_R);
        // --- END ADDED LINES ---
     }
}


// --- Check Appearing Point (Case 1) ---
bool DynObjFilter::checkAppearingPoint(point_soph& p) {
    // Use the "Filter_Consistency" logger for normal trace logs
    auto logger = spdlog::get("Filter_Consistency");
    if (!logger) { /* Handle missing logger if necessary */ }

    // --- Get Current Sequence ID from TLS ---
    uint64_t current_seq_id_for_log = PointCloudUtils::g_current_logging_seq_id;

    // --- Define a flag for targeted debugging ---
    // Check point index is 5 or 6 AND the sequence ID from TLS is 6
    bool is_target_debug_point = ( (p.original_index == 5 || p.original_index == 6) &&
                                   current_seq_id_for_log == 6 );

    // --- Conditional Entry Log (Normal Trace) ---
    if (PointCloudUtils::should_log_point_details(logger)) {
        logger->trace("[checkAppearingPoint] Entered for point idx: {} (Frame Seq ID: {})", p.original_index, current_seq_id_for_log);
    }

    // --- Original Logic Start ---
    int depth_map_num = depth_map_list_.size();
    int available_maps_to_check = depth_map_num;

    int min_maps_needed = std::min(params_.occluded_map_thr1, params_.occluded_map_thr2);
    if (available_maps_to_check < min_maps_needed) {
        if (PointCloudUtils::should_log_point_details(logger)) {
            logger->trace("[checkAppearingPoint] Point {}: Skipping: Not enough maps ({}) < min threshold required ({}).",
                          p.original_index, available_maps_to_check, min_maps_needed);
        }
        return false; // Not enough history to make a decision
    }

    int geometric_inconsistent_count = 0;
    int empty_neighbor_inconsistent_count = 0;

    // Iterate through historical maps (newest relevant to oldest relevant)
    // Note: depth_map_list_ contains maps from oldest [0] to newest [-1]
    // The *current* frame's map is not yet in the list when this check runs.
    // We check against maps up to depth_map_num - 1.
    for (int i = depth_map_num - 1; i >= 0; --i) {
        if (!depth_map_list_[i]) {
             if (PointCloudUtils::should_log_point_details(logger)) {
                 logger->warn("[checkAppearingPoint] Point {}: Encountered null depth map pointer at list index {}. Skipping map.", p.original_index, i);
             }
             continue; // Skip this map if pointer is null
        }
        const DepthMap& map = *depth_map_list_[i];
        point_soph point_in_map_frame; // To store point coords relative to the historical map's frame

        if (PointCloudUtils::should_log_point_details(logger)) {
            logger->trace("[checkAppearingPoint] Point {}: Checking MapIdx={} (List Idx={})",
                          p.original_index, map.map_index, i);
        }
        // if (is_target_debug_point) { // Use existing flag for targeted logging
        //     spdlog::warn("[TLS DEBUG @ Consistency] Before setupPointForMapCheck: SeqID={}, PointIdx={}",
        //                  current_seq_id_for_log, p.original_index);
        // }
        // Project the current point (p) into the coordinate system of the historical map
        setupPointForMapCheck(p, map, point_in_map_frame);

        bool is_geometric_inconsistent = false;      // Flag for this specific map check
        bool is_empty_neighbor_inconsistent = false; // Flag for this specific map check
        bool is_consistent = true;                   // Default assumption for this map check

        // Condition 1 & 2: Invalid projection or self-point in map frame
        if (point_in_map_frame.position < 0 || point_in_map_frame.position >= MAX_2D_N || point_in_map_frame.vec(2) <= 0.0f) {
            // Point projects outside the map grid or has non-positive depth
            is_geometric_inconsistent = true; // Treat as geometrically inconsistent for this map
            is_consistent = false;
            if (PointCloudUtils::should_log_point_details(logger)) {
                logger->trace("[checkAppearingPoint] Point {}: MapIdx={} -> Geo Inconsistent (Invalid Projection: pos={}, z={:.3f})",
                              p.original_index, map.map_index, point_in_map_frame.position, point_in_map_frame.vec(2));
            }
        }
        else if (PointCloudUtils::isSelfPoint(point_in_map_frame.local, params_)) {
            // Point falls within the self-filter box relative to the historical map's sensor pose
            is_consistent = true; // Treat as consistent (ignore self-points)
            if (PointCloudUtils::should_log_point_details(logger)) {
                logger->trace("[checkAppearingPoint] Point {}: MapIdx={} -> Consistent (Inside Self-Box)",
                              p.original_index, map.map_index);
            }
        }
        // Condition 3: Check against historical map content (interpolation)
        else {
            // Find neighboring static points in the historical map grid
            std::vector<V3F> neighbors = PointCloudUtils::findInterpolationNeighbors(
                point_in_map_frame, map, params_, PointCloudUtils::InterpolationNeighborType::STATIC_ONLY);

            if (neighbors.empty()) {
                // No static neighbors found in the historical map around the projected point
                is_empty_neighbor_inconsistent = true; // Treat as empty-neighbor inconsistent for this map
                is_consistent = false;
                if (PointCloudUtils::should_log_point_details(logger)) {
                    logger->trace("[checkAppearingPoint] Point {}: MapIdx={} -> EmptyN Inconsistent (No Static Neighbors)",
                                  p.original_index, map.map_index);
                }
            }
            else if (neighbors.size() < 3) {
                // Too few neighbors to reliably interpolate
                is_consistent = true; // Treat as consistent (insufficient evidence)
                if (PointCloudUtils::should_log_point_details(logger)) {
                    logger->trace("[checkAppearingPoint] Point {}: MapIdx={} -> Consistent (Sparse Static Neighbors: {})",
                                  p.original_index, map.map_index, neighbors.size());
                }
            }
            else { // Sufficient Neighbors Found (>= 3)
                // Attempt barycentric interpolation to find historical depth at the projected location
                V2F target_projection = point_in_map_frame.vec.head<2>(); // Azimuth, Elevation

                PointCloudUtils::InterpolationResult interp_result = PointCloudUtils::computeBarycentricDepth(
                    target_projection, neighbors, params_);

                // --->>> HARDCODED DEBUG LOG 1 <<<---
                if (is_target_debug_point) {
                    spdlog::warn("[DEBUG SYNTHETIC] Point {}: MapIdx={}: Checking Interp Status: {}", p.original_index, map.map_index, PointCloudUtils::interpolationStatusToString(interp_result.status));
                }
                // --->>> END DEBUG LOG 1 <<<---

                if (interp_result.status == PointCloudUtils::InterpolationStatus::SUCCESS) {
                    // Interpolation succeeded, compare depths
                    float current_depth_in_map_frame = point_in_map_frame.vec(2); // Current point's depth relative to historical map pose
                    float historical_static_depth = interp_result.depth;          // Interpolated historical depth
                    float depth_diff = current_depth_in_map_frame - historical_static_depth;

                    // Calculate the dynamic threshold
                    float threshold = params_.interp_thr1; // Base threshold
                    int map_index_diff = 0;
                    if (!depth_map_list_.empty() && depth_map_list_.back()) {
                        // depth_map_list_.back() is the *newest* map in the history list (before the current frame)
                        map_index_diff = depth_map_list_.back()->map_index - map.map_index;
                    }
                    threshold *= (1.0f + 0.1f * map_index_diff); // Increase threshold slightly for older maps
                    // Add depth-dependent term
                    if (current_depth_in_map_frame > params_.interp_start_depth1) {
                         threshold += params_.interp_kp1 * (current_depth_in_map_frame - params_.interp_start_depth1);
                    }
                    // Skip dataset/distort check for synthetic tests

                    // --->>> HARDCODED DEBUG LOG 2 (MODIFIED) <<<---
                    // Log details regardless of the outcome of the check
                    if (is_target_debug_point) {
                        spdlog::warn("[DEBUG SYNTHETIC] Point {}: MapIdx={}: Interp SUCCESS. CurZ={:.3f}, HistZ={:.3f}, Diff={:.3f}, Thr={:.3f}, MapIdxDiff={}, BaseThr={:.3f}. Check: Diff < -Thr?",
                                         p.original_index, map.map_index, current_depth_in_map_frame, historical_static_depth, depth_diff, threshold, map_index_diff, params_.interp_thr1);
                    }
                    // --->>> END DEBUG LOG 2 <<<---

                    // Check if the current point is significantly *in front* of the historical surface
                    if (depth_diff < -threshold) {
                        is_geometric_inconsistent = true; // Mark as geometrically inconsistent for this map
                        is_consistent = false;
                        // --->>> HARDCODED DEBUG LOG 3 <<<---
                        if (is_target_debug_point) {
                             spdlog::warn("[DEBUG SYNTHETIC] Point {}: MapIdx={}: Geometric Inconsistency DETECTED.", p.original_index, map.map_index);
                        }
                        // --->>> END DEBUG LOG 3 <<<---
                         if (PointCloudUtils::should_log_point_details(logger)) { // Keep original trace log
                             logger->trace("[checkAppearingPoint] Point {}: MapIdx={} -> Geo Inconsistent (Interp SUCCESS, In Front: CurZ={:.3f}, HistZ={:.3f}, Diff={:.3f} < Thr={:.3f})",
                                                  p.original_index, map.map_index, current_depth_in_map_frame, historical_static_depth, depth_diff, -threshold);
                         }
                    } else { // Consistent based on depth check
                        is_consistent = true;
                        // --->>> HARDCODED DEBUG LOG 4 <<<---
                         if (is_target_debug_point) {
                             spdlog::warn("[DEBUG SYNTHETIC] Point {}: MapIdx={}: Consistent (Geometric check failed: Diff >= -Thr).", p.original_index, map.map_index);
                         }
                        // --->>> END DEBUG LOG 4 <<<---
                         if (PointCloudUtils::should_log_point_details(logger)) { // Keep original trace log
                             logger->trace("[checkAppearingPoint] Point {}: MapIdx={} -> Consistent (Interp SUCCESS, Not In Front: CurZ={:.3f}, HistZ={:.3f}, Diff={:.3f} >= Thr={:.3f})",
                                                  p.original_index, map.map_index, current_depth_in_map_frame, historical_static_depth, depth_diff, -threshold);
                         }
                    }
                } else { // Interpolation FAILED (e.g., NO_VALID_TRIANGLE)
                    is_consistent = true; // Treat as consistent if interpolation fails (cannot prove inconsistency)
                    // --->>> HARDCODED DEBUG LOG 5 <<<---
                    if (is_target_debug_point) {
                        spdlog::warn("[DEBUG SYNTHETIC] Point {}: MapIdx={}: Interp FAILED, status {}, treated as Consistent.", p.original_index, map.map_index, PointCloudUtils::interpolationStatusToString(interp_result.status));
                    }
                    // --->>> END DEBUG LOG 5 <<<---
                    if (PointCloudUtils::should_log_point_details(logger)) { // Keep original trace log
                        logger->trace("[checkAppearingPoint] Point {}: MapIdx={} -> Consistent (Interp FAILED: {} with >=3 neighbors)",
                                             p.original_index, map.map_index, PointCloudUtils::interpolationStatusToString(interp_result.status));
                    }
                }
            } // End neighbor count checks (>= 3)
        } // End Condition 3 block (interpolation checks)

        // Update counts based on the outcome for this map
        if (is_geometric_inconsistent) {
            geometric_inconsistent_count++;
            if (PointCloudUtils::should_log_point_details(logger)) {
                logger->trace("[checkAppearingPoint] Point {}:   -> Geo Inconsistent Count = {}",
                              p.original_index, geometric_inconsistent_count);
            }
        } else if (is_empty_neighbor_inconsistent) {
            empty_neighbor_inconsistent_count++;
             if (PointCloudUtils::should_log_point_details(logger)) {
                 logger->trace("[checkAppearingPoint] Point {}:   -> EmptyN Inconsistent Count = {}",
                               p.original_index, empty_neighbor_inconsistent_count);
             }
        } else { // is_consistent
             if (PointCloudUtils::should_log_point_details(logger)) {
                 logger->trace("[checkAppearingPoint] Point {}:   -> Consistent with map {}.",
                               p.original_index, map.map_index);
             }
        }

        // --->>> HARDCODED DEBUG LOG 6 <<<---
        if (is_target_debug_point) {
            spdlog::warn("[DEBUG SYNTHETIC] Point {}: After MapIdx={}: GeoCnt={}, EmptyCnt={}", p.original_index, map.map_index, geometric_inconsistent_count, empty_neighbor_inconsistent_count);
        }
        // --->>> END DEBUG LOG 6 <<<---

        // --- Check if *either* threshold is met ---
        bool geometric_threshold_met = (geometric_inconsistent_count >= params_.occluded_map_thr1);
        bool empty_neighbor_threshold_met = (empty_neighbor_inconsistent_count >= params_.occluded_map_thr2);

        if (geometric_threshold_met || empty_neighbor_threshold_met) {
             // --->>> HARDCODED DEBUG LOG 7 <<<---
             if (is_target_debug_point) {
                std::string reason = geometric_threshold_met ? "Geo" : "EmptyN";
                 spdlog::warn("[DEBUG SYNTHETIC] Point {}: Threshold MET ({}), returning TRUE.", p.original_index, reason);
             }
             // --->>> END DEBUG LOG 7 <<<---
            if (PointCloudUtils::should_log_point_details(logger)) { // Keep original trace log
                 std::string reason = geometric_threshold_met ?
                    fmt::format("Geometric threshold reached: {} >= {}", geometric_inconsistent_count, params_.occluded_map_thr1) :
                    fmt::format("Empty Neighbor threshold reached: {} >= {}", empty_neighbor_inconsistent_count, params_.occluded_map_thr2);
                logger->trace("[checkAppearingPoint] Point {}: -> Returning TRUE ({}). Point labeled APPEARING.",
                                     p.original_index, reason);
            }
            return true; // Point is labeled APPEARING
        }

        // --- Optimization: Early Exit Check ---
        // Check if it's still possible to reach either threshold with the remaining maps
        int remaining_maps_to_check = i; // Maps from index i-1 down to 0
        bool can_reach_geometric_thr = (geometric_inconsistent_count + remaining_maps_to_check >= params_.occluded_map_thr1);
        bool can_reach_empty_thr = (empty_neighbor_inconsistent_count + remaining_maps_to_check >= params_.occluded_map_thr2);

        if (!can_reach_geometric_thr && !can_reach_empty_thr) {
             // --->>> HARDCODED DEBUG LOG 8 <<<---
             if (is_target_debug_point) {
                 spdlog::warn("[DEBUG SYNTHETIC] Point {}: Early Exit, returning FALSE.", p.original_index);
             }
             // --->>> END DEBUG LOG 8 <<<---
            if (PointCloudUtils::should_log_point_details(logger)) { // Keep original trace log
                logger->trace("[checkAppearingPoint] Point {}: -> Returning FALSE (Early exit: Cannot reach either threshold. GeoCnt={}, EmptyCnt={}, Remaining={}, GeoThr={}, EmptyThr={})",
                                     p.original_index, geometric_inconsistent_count, empty_neighbor_inconsistent_count, remaining_maps_to_check, params_.occluded_map_thr1, params_.occluded_map_thr2);
            }
            return false; // Cannot possibly reach threshold, label remains STATIC (or previous)
        }

    } // End loop through historical maps

    // If loop finishes without returning true, the thresholds were not met
    // --->>> HARDCODED DEBUG LOG 9 <<<---
    if (is_target_debug_point) {
        spdlog::warn("[DEBUG SYNTHETIC] Point {}: Loop Finished, returning FALSE.", p.original_index);
    }
    // --->>> END DEBUG LOG 9 <<<---
    if (PointCloudUtils::should_log_point_details(logger)) { // Keep original trace log
        logger->trace("[checkAppearingPoint] Point {}: -> Returning FALSE (Loop finished, thresholds not reached. GeoCnt={}, EmptyCnt={}, GeoThr={}, EmptyThr={})",
                      p.original_index, geometric_inconsistent_count, empty_neighbor_inconsistent_count, params_.occluded_map_thr1, params_.occluded_map_thr2);
    }
    return false; // Label remains STATIC (or previous)
}
