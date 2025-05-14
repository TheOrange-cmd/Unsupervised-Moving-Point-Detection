/**
 * @file interpolation_utils.cpp
 * @brief Implements functions for depth interpolation using barycentric coordinates.
*/

#include "point_cloud_utils/interpolation_utils.h"
#include "point_cloud_utils/grid_search_utils.h" // Need forEachNeighborCell
#include "point_cloud_utils/logging_utils.h"     // <-- ADDED THIS INCLUDE
#include "point_cloud_utils/logging_context.h"   // <-- ADDED THIS INCLUDE (to read context for log messages)
#include "common/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "common/types.h"
#include <cmath>        // For std::fabs, M_PI
#include <vector>
#include <string>
#include <spdlog/spdlog.h> // For logging
#include <spdlog/fmt/ostr.h> // Required to log custom types like DynObjLabel via streams

// Define constants used in calculations
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
constexpr float TWO_PI = 2.0f * M_PI;
// Epsilon for checking if barycentric denominator is too close to zero (degenerate triangle)
constexpr float BARY_DEGENERACY_EPSILON = 1e-12f;
// Epsilon for checking if point is inside triangle (allows points slightly outside due to float precision)
constexpr float BARY_INSIDE_CHECK_EPSILON = 1e-6f;
 
 namespace PointCloudUtils {
 
     // --- Enum to String Helpers ---
     std::string interpolationStatusToString(InterpolationStatus status) {
         switch (status) {
             case InterpolationStatus::SUCCESS: return "SUCCESS";
             case InterpolationStatus::NOT_ENOUGH_NEIGHBORS: return "NOT_ENOUGH_NEIGHBORS";
             case InterpolationStatus::NO_VALID_TRIANGLE: return "NO_VALID_TRIANGLE";
             case InterpolationStatus::DEGENERACY: return "DEGENERACY";
             default: return "UNKNOWN_STATUS";
         }
     }
 
     std::string interpolationNeighborTypeToString(InterpolationNeighborType type) {
         switch (type) {
             case InterpolationNeighborType::ALL_VALID: return "ALL_VALID";
             case InterpolationNeighborType::STATIC_ONLY: return "STATIC_ONLY";
             default: return "UNKNOWN";
         }
     }
 
 
     // --- Core Logic Functions ---
     std::vector<V3F> findInterpolationNeighbors(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type)
    {
        auto logger = spdlog::get("Utils_Interpolation"); // Using new logger name
        std::vector<V3F> neighbors;
        uint64_t current_seq_id_local = g_current_logging_seq_id;

        // General entry log at trace level
        if (logger->should_log(spdlog::level::trace)) {
            logger->trace("[findInterpNeighbors] ENTER for Point OriginalIdx={}, H={}, V={}, FrameID={}, MapIdx={}, Type: {}. Params: HorNum={}, VerNum={}, HorThr={:.4f}, VerThr={:.4f}",
                p.original_index, p.hor_ind, p.ver_ind, current_seq_id_local, map_info.map_index, 
                interpolationNeighborTypeToString(type), // Use to_string helper
                params.interp_hor_num, params.interp_ver_num, params.interp_hor_thr, params.interp_ver_thr);
        }
    
        if (map_info.depth_map.size() != MAX_2D_N) {
             if (logger) logger->warn("[findInterpNeighbors] map_info.depth_map size ({}) != MAX_2D_N ({}). Cannot find neighbors for point OriginalIdx={} in FrameID={}.", map_info.depth_map.size(), MAX_2D_N, p.original_index, current_seq_id_local);
             return neighbors;
        }
    
        forEachNeighborCell(p, params.interp_hor_num, params.interp_ver_num, true, true,
            [&](int pos) { // Capture necessary variables
                 if (pos < 0 || pos >= MAX_2D_N) return;
    
                const auto& cell_points = map_info.depth_map[pos];
    
                if (logger->should_log(spdlog::level::trace) && !cell_points.empty()) {
                     logger->trace("  [findInterpNeighbors] Point OriginalIdx={}, FrameID={}: Checking cell_pos={}. Contains {} points.", p.original_index, current_seq_id_local, pos, cell_points.size());
                }
    
                if (cell_points.empty()) return;
    
                for (const auto& neighbor_ptr : cell_points) {
                    if (!neighbor_ptr) continue;
    
                    // Time check
                    float time_diff = std::fabs(neighbor_ptr->time - p.time);
                    bool time_ok = (time_diff >= params.frame_dur);
                    if (!time_ok) {
                        if (logger->should_log(spdlog::level::trace)) {
                            logger->trace("    [findInterpNeighbors] Point OriginalIdx={}, FrameID={}: Reject Neighbor OriginalIdx={}: TimeDiff ({:.4f} < {:.4f})",
                                p.original_index, current_seq_id_local, neighbor_ptr->original_index, time_diff, params.frame_dur);
                        }
                        continue;
                    }
    
                    // Angular check
                    float hor_diff_raw = neighbor_ptr->vec.x() - p.vec.x();
                    float ver_diff = neighbor_ptr->vec.y() - p.vec.y();
                    float hor_diff = hor_diff_raw;
                    if (hor_diff > M_PI) hor_diff -= TWO_PI;
                    else if (hor_diff <= -M_PI) hor_diff += TWO_PI;
                    bool angular_ok = (std::fabs(hor_diff) < params.interp_hor_thr && std::fabs(ver_diff) < params.interp_ver_thr);
                    if (!angular_ok) {
                        if (logger->should_log(spdlog::level::trace)) {
                            logger->trace("    [findInterpNeighbors] Point OriginalIdx={}, FrameID={}: Reject Neighbor OriginalIdx={}: AngularDist (Hor diff {:.4f} vs Thr {:.4f} OR Ver diff {:.4f} vs Thr {:.4f})",
                                p.original_index, current_seq_id_local, neighbor_ptr->original_index, std::fabs(hor_diff), params.interp_hor_thr, std::fabs(ver_diff), params.interp_ver_thr);
                        }
                        continue;
                    }
    
                    // Label check
                    bool label_ok = false;
                    if (type == InterpolationNeighborType::ALL_VALID) {
                        label_ok = true;
                    } else { // STATIC_ONLY
                        label_ok = (neighbor_ptr->dyn == DynObjLabel::STATIC);
                        if (!label_ok) {
                            if (logger->should_log(spdlog::level::trace)) {
                                logger->trace("    [findInterpNeighbors] Point OriginalIdx={}, FrameID={}: Reject Neighbor OriginalIdx={}: LabelNotStatic (Type STATIC_ONLY, but label is {})",
                                    p.original_index, current_seq_id_local, neighbor_ptr->original_index, neighbor_ptr->dyn); // spdlog handles enum via ostream
                            }
                        }
                    }
                    if (!label_ok) continue;
    
                    // If all checks pass, add neighbor
                    neighbors.push_back(neighbor_ptr->vec);
    
                    if (logger->should_log(spdlog::level::trace)) {
                        logger->trace("    [findInterpNeighbors] Point OriginalIdx={}, FrameID={}: ACCEPT Neighbor OriginalIdx={}: Label={}, Depth={:.3f}, Coords=({:.4f}, {:.4f}, {:.3f})",
                            p.original_index, current_seq_id_local, neighbor_ptr->original_index, neighbor_ptr->dyn, neighbor_ptr->vec.z(),
                            neighbor_ptr->vec.x(), neighbor_ptr->vec.y(), neighbor_ptr->vec.z());
                    }
                }
            }
        );
    
        // Final log of how many neighbors were found
        if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details to general trace
            logger->trace("[findInterpNeighbors] EXIT for Point OriginalIdx={}, FrameID={}: Found {} valid neighbors (Type: {}).",
                         p.original_index, current_seq_id_local, neighbors.size(), interpolationNeighborTypeToString(type));
        }
    
        return neighbors;
    }

    InterpolationResult computeBarycentricDepth(
        const V2F& target_point_projection, // This is (azimuth, elevation) of the target point
        const std::vector<V3F>& neighbors,  // These are (azimuth, elevation, depth) of neighbors
        const DynObjFilterParams& params)
    {
        auto logger = spdlog::get("Utils_Interpolation"); // Using new logger name
        InterpolationResult result;
        // Assuming g_current_logging_point_ptr and g_current_logging_seq_id are set by the caller's context
        const point_soph* p_context = g_current_logging_point_ptr;
        uint64_t current_seq_id_local = g_current_logging_seq_id;
        size_t target_original_idx = (p_context ? p_context->original_index : -1); // Get original_index if context is set

        if (logger->should_log(spdlog::level::trace)) {
            logger->trace("[computeBarycentricDepth] ENTER for TargetPoint OriginalIdx={}, FrameID={}. TargetProj=({:.4f}, {:.4f}). Received {} neighbors.",
                          target_original_idx, current_seq_id_local, target_point_projection.x(), target_point_projection.y(), neighbors.size());
            if (!neighbors.empty()) { // Log all neighbor coordinates if trace is on and list is not empty
                for(size_t idx = 0; idx < neighbors.size(); ++idx) {
                     logger->trace("  Neighbor {}: ProjAz={:.4f}, ProjEl={:.4f}, Depth={:.3f}", idx, neighbors[idx].x(), neighbors[idx].y(), neighbors[idx].z());
                }
            }
        }
    
        if (neighbors.size() < 3) {
           if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
               logger->trace("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Failed: {} neighbors found (need >= 3).", target_original_idx, current_seq_id_local, neighbors.size());
           }
           result.status = InterpolationStatus::NOT_ENOUGH_NEIGHBORS;
           return result;
        }
    
        result.status = InterpolationStatus::NO_VALID_TRIANGLE;
        bool degeneracy_encountered = false;
        bool found_valid_triangle = false;
        float min_area = std::numeric_limits<float>::max();
        float best_depth = 0.0f;
        V3F best_bary_coords; // To store u,v,w of best triangle
        std::array<size_t, 3> best_triangle_indices = {0,0,0};


        int degenerate_count = 0;
        int outside_count = 0;
        int attempted_count = 0;
    
        for (size_t i = 0; i < neighbors.size(); ++i) {
            for (size_t j = i + 1; j < neighbors.size(); ++j) {
                for (size_t k = j + 1; k < neighbors.size(); ++k) {
                    attempted_count++;
    
                    V2F v0_orig = neighbors[i].head<2>(); // Az, El of neighbor i
                    V2F v1_orig = neighbors[j].head<2>(); // Az, El of neighbor j
                    V2F v2_orig = neighbors[k].head<2>(); // Az, El of neighbor k
    
                    auto unwrap_azimuth = [&](float neighbor_az, float target_az) -> float {
                        float diff = neighbor_az - target_az;
                        if (diff > M_PI)       neighbor_az -= TWO_PI;
                        else if (diff <= -M_PI) neighbor_az += TWO_PI;
                        return neighbor_az;
                    };

                    V2F v0 = v0_orig; v0.x() = unwrap_azimuth(v0.x(), target_point_projection.x());
                    V2F v1 = v1_orig; v1.x() = unwrap_azimuth(v1.x(), target_point_projection.x());
                    V2F v2 = v2_orig; v2.x() = unwrap_azimuth(v2.x(), target_point_projection.x());
                    V2F target_unwrapped = target_point_projection; 
                    // If target_point_projection itself needs unwrapping relative to a common median, do it here.
                    // For now, assuming neighbors are unwrapped relative to the target's original azimuth.

                    if (logger->should_log(spdlog::level::trace)) { // Log every triangle attempt at trace
                        logger->trace("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Checking triangle NeighIdxs({}, {}, {}): "
                                      "V0_orig({:.4f},{:.4f}), V1_orig({:.4f},{:.4f}), V2_orig({:.4f},{:.4f}). "
                                      "V0_unwrapped({:.4f},{:.4f}), V1_unwrapped({:.4f},{:.4f}), V2_unwrapped({:.4f},{:.4f}). "
                                      "Target_unwrapped({:.4f},{:.4f})",
                                      target_original_idx, current_seq_id_local, i, j, k,
                                      v0_orig.x(), v0_orig.y(), v1_orig.x(), v1_orig.y(), v2_orig.x(), v2_orig.y(),
                                      v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y(),
                                      target_unwrapped.x(), target_unwrapped.y());
                    }
    
                    V2F vec_v0v1 = v1 - v0;
                    V2F vec_v0v2 = v2 - v0;
                    V2F vec_v0p = target_unwrapped - v0;
    
                    float d00 = vec_v0v1.dot(vec_v0v1);
                    float d01 = vec_v0v1.dot(vec_v0v2);
                    float d11 = vec_v0v2.dot(vec_v0v2);
                    float d20 = vec_v0p.dot(vec_v0v1);
                    float d21 = vec_v0p.dot(vec_v0v2);
                    float denom = d00 * d11 - d01 * d01;
    
                    if (std::fabs(denom) < BARY_DEGENERACY_EPSILON) {
                        degeneracy_encountered = true;
                        degenerate_count++;
                        if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
                            logger->trace("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Skipping degenerate triangle NeighIdxs({}, {}, {}): denom={:.4g}. V0_unwrapped({:.4f},{:.4f}), V1_unwrapped({:.4f},{:.4f}), V2_unwrapped({:.4f},{:.4f})",
                                          target_original_idx, current_seq_id_local, i, j, k, denom, v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y());
                        }
                        continue;
                     }
    
                    float bary_v = (d11 * d20 - d01 * d21) / denom; // Barycentric v
                    float bary_w = (d00 * d21 - d01 * d20) / denom; // Barycentric w
                    float bary_u = 1.0f - bary_v - bary_w;         // Barycentric u
    
                    bool is_outside = (bary_u < -BARY_INSIDE_CHECK_EPSILON ||
                                       bary_v < -BARY_INSIDE_CHECK_EPSILON ||
                                       bary_w < -BARY_INSIDE_CHECK_EPSILON);
                    bool is_inside = !is_outside;
    
                    if (is_inside) {
                        float area = 0.5f * std::fabs(vec_v0v1.x() * vec_v0v2.y() - vec_v0v1.y() * vec_v0v2.x());
                        float current_depth = bary_u * neighbors[i].z() + bary_v * neighbors[j].z() + bary_w * neighbors[k].z();

                        if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
                             logger->trace("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Triangle NeighIdxs({}, {}, {}) is VALID. BaryU={:.3f}, BaryV={:.3f}, BaryW={:.3f}. Area={:.4g}, CalcDepth={:.3f}.",
                                           target_original_idx, current_seq_id_local, i, j, k, bary_u, bary_v, bary_w, area, current_depth);
                        }

                        if (area < min_area) {
                            min_area = area;
                            best_depth = current_depth;
                            found_valid_triangle = true;
                            result.status = InterpolationStatus::SUCCESS; // Tentative success
                            best_bary_coords = V3F(bary_u, bary_v, bary_w);
                            best_triangle_indices = {i,j,k};
                            if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
                                 logger->trace("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Updating BEST triangle. NewMinArea={:.4g}, NewBestDepth={:.3f}.", target_original_idx, current_seq_id_local, min_area, best_depth);
                            }
                        }
                    } else { // is_outside
                        outside_count++;
                        if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
                            logger->trace("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Target OUTSIDE triangle NeighIdxs({}, {}, {}): BaryU={:.3f}, BaryV={:.3f}, BaryW={:.3f}. V0_unwrapped({:.4f},{:.4f}), V1_unwrapped({:.4f},{:.4f}), V2_unwrapped({:.4f},{:.4f}), Target_unwrapped({:.4f},{:.4f})",
                                          target_original_idx, current_seq_id_local, i, j, k, bary_u, bary_v, bary_w, v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y(), target_unwrapped.x(), target_unwrapped.y());
                        }
                    }
                }
            }
        }
    
        if (found_valid_triangle) {
            result.depth = best_depth;
            // result.status is already SUCCESS
            if (logger->should_log(spdlog::level::debug)) { // Keep this summary at debug
                logger->debug("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Final Result: SUCCESS. BestDepth={:.3f} (from NeighIdxs({},{},{}), Area={:.4g}, BaryUWV=({:.3f},{:.3f},{:.3f})). Attempts={}, Degenerate={}, Outside={}",
                             target_original_idx, current_seq_id_local, result.depth, best_triangle_indices[0], best_triangle_indices[1], best_triangle_indices[2], min_area, best_bary_coords.x(), best_bary_coords.y(), best_bary_coords.z(), attempted_count, degenerate_count, outside_count);
            }
        } else {
            // result.status is NO_VALID_TRIANGLE (or NOT_ENOUGH_NEIGHBORS if caught earlier)
            if (degeneracy_encountered && result.status == InterpolationStatus::NO_VALID_TRIANGLE) { // If only degenerates were found
                 result.status = InterpolationStatus::DEGENERACY;
            }
            if (logger->should_log(spdlog::level::debug)) { // Keep this summary at debug
                logger->debug("[computeBarycentricDepth] TargetPoint OriginalIdx={}, FrameID={}: Final Result: {}. Attempts={}, Degenerate={}, Outside={}. DegeneracyEncountered={}",
                             target_original_idx, current_seq_id_local, interpolationStatusToString(result.status), attempted_count, degenerate_count, outside_count, degeneracy_encountered);
            }
        }
        return result;
    }
 
    // InterpolationResult computeBarycentricDepth(
    //     const V2F& target_point_projection,
    //     const std::vector<V3F>& neighbors,
    //     const DynObjFilterParams& params)
    // {
    //     auto logger = spdlog::get("Utils_Interpolation");
    //     InterpolationResult result;
    
    //     // --- ADDED LOG: Log input neighbor count ---
    //     if (PointCloudUtils::should_log_point_details(logger)) {
    //         logger->trace("[computeBarycentricDepth] ENTER. Received {} neighbors.", neighbors.size());
    //         // Optional: Log neighbor coordinates if needed, but can be verbose
    //         // for(size_t idx = 0; idx < neighbors.size(); ++idx) {
    //         //      logger->trace("  Neighbor {}: ({:.4f}, {:.4f}, {:.3f})", idx, neighbors[idx].x(), neighbors[idx].y(), neighbors[idx].z());
    //         // }
    //     }
    //     // --- END ADDED LOG ---
    
    
    //     if (neighbors.size() < 3) {
    //        if (PointCloudUtils::should_log_point_details(logger)) {
    //            logger->trace("[computeBarycentricDepth] Failed: {} neighbors found (need >= 3).", neighbors.size());
    //        }
    //        result.status = InterpolationStatus::NOT_ENOUGH_NEIGHBORS;
    //        return result;
    //     }
    
    //     result.status = InterpolationStatus::NO_VALID_TRIANGLE;
    //     bool degeneracy_encountered = false;
    //     bool found_valid_triangle = false;
    //     float min_area = std::numeric_limits<float>::max();
    //     float best_depth = 0.0f;
    //     // --- ADDED: Counters for failure reasons ---
    //     int degenerate_count = 0;
    //     int outside_count = 0;
    //     int attempted_count = 0;
    //     // --- END ADDED ---
    
    
    //     for (size_t i = 0; i < neighbors.size(); ++i) {
    //         for (size_t j = i + 1; j < neighbors.size(); ++j) {
    //             for (size_t k = j + 1; k < neighbors.size(); ++k) {
    //                 attempted_count++; // Increment attempt counter
    
    //                 V2F v0_orig = neighbors[i].head<2>();
    //                 V2F v1_orig = neighbors[j].head<2>();
    //                 V2F v2_orig = neighbors[k].head<2>();
    
    //                 auto unwrap_azimuth = [&](float neighbor_az, float target_az) -> float { // Explicitly specify -> float
    //                     float diff = neighbor_az - target_az;
    //                     if (diff > M_PI)       neighbor_az -= TWO_PI;
    //                     else if (diff <= -M_PI) neighbor_az += TWO_PI;
    //                     return neighbor_az; // Now the compiler knows for sure it returns float
    //                 };
    //                 V2F v0 = v0_orig; v0.x() = unwrap_azimuth(v0.x(), target_point_projection.x());
    //                 V2F v1 = v1_orig; v1.x() = unwrap_azimuth(v1.x(), target_point_projection.x());
    //                 V2F v2 = v2_orig; v2.x() = unwrap_azimuth(v2.x(), target_point_projection.x());
    //                 V2F target_unwrapped = target_point_projection; // Assuming target is already relative or unwrapping is handled
    
    //                 // --- Optional ADDED LOG: Log triangle vertices being checked ---
    //                 // if (PointCloudUtils::should_log_point_details(logger)) {
    //                 //     logger->trace("[computeBarycentricDepth] Checking triangle ({}, {}, {}): V0({:.4f},{:.4f}), V1({:.4f},{:.4f}), V2({:.4f},{:.4f}), Target({:.4f},{:.4f})",
    //                 //                   i, j, k, v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y(), target_unwrapped.x(), target_unwrapped.y());
    //                 // }
    //                 // --- END ADDED LOG ---
    
    //                 V2F vec_v0v1 = v1 - v0;
    //                 V2F vec_v0v2 = v2 - v0;
    //                 V2F vec_v0p = target_unwrapped - v0;
    
    //                 float d00 = vec_v0v1.dot(vec_v0v1);
    //                 float d01 = vec_v0v1.dot(vec_v0v2);
    //                 float d11 = vec_v0v2.dot(vec_v0v2);
    //                 float d20 = vec_v0p.dot(vec_v0v1);
    //                 float d21 = vec_v0p.dot(vec_v0v2);
    
    //                 float denom = d00 * d11 - d01 * d01;
    
    //                 if (std::fabs(denom) < BARY_DEGENERACY_EPSILON) {
    //                     degeneracy_encountered = true;
    //                     degenerate_count++; // Increment degeneracy counter
    //                     // --- MODIFIED LOG: Add more details ---
    //                     if (PointCloudUtils::should_log_point_details(logger)) {
    //                         logger->trace("[computeBarycentricDepth] Skipping degenerate triangle ({}, {}, {}): denom={:.4g}. V0({:.4f},{:.4f}), V1({:.4f},{:.4f}), V2({:.4f},{:.4f})",
    //                                       i, j, k, denom, v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y());
    //                     }
    //                     // --- END MODIFIED LOG ---
    //                     continue;
    //                  }
    
    //                 float v = (d11 * d20 - d01 * d21) / denom;
    //                 float w = (d00 * d21 - d01 * d20) / denom;
    //                 float u = 1.0f - v - w;
    
    //                 bool is_outside = (u < -BARY_INSIDE_CHECK_EPSILON ||
    //                     v < -BARY_INSIDE_CHECK_EPSILON ||
    //                     w < -BARY_INSIDE_CHECK_EPSILON);
    //                 bool is_inside = !is_outside;
    
    //                 if (is_inside) {
    //                     float area = 0.5f * std::fabs(vec_v0v1.x() * vec_v0v2.y() - vec_v0v1.y() * vec_v0v2.x());
    
    //                     if (area < min_area) {
    //                         // ... (update best_depth, min_area, found_valid_triangle as before) ...
    //                         if (PointCloudUtils::should_log_point_details(logger)) {
    //                              logger->trace("[computeBarycentricDepth] Found valid triangle ({}, {}, {}). Area={:.4g}, Depth={:.3f}. Updating best.", i, j, k, area, best_depth);
    //                         }
    //                     } else {
    //                          if (PointCloudUtils::should_log_point_details(logger)) {
    //                             logger->trace("[computeBarycentricDepth] Found valid triangle ({}, {}, {}). Area={:.4g}. Not better than min_area={:.4g}.", i, j, k, area, min_area);
    //                         }
    //                     }
    //                 } else {
    //                     outside_count++; // Increment outside counter
    //                     // --- MODIFIED LOG: Add more details ---
    //                     if (PointCloudUtils::should_log_point_details(logger)) {
    //                         logger->trace("[computeBarycentricDepth] Target outside triangle ({}, {}, {}): u={:.3f}, v={:.3f}, w={:.3f}. V0({:.4f},{:.4f}), V1({:.4f},{:.4f}), V2({:.4f},{:.4f}), Target({:.4f},{:.4f})",
    //                                       i, j, k, u, v, w, v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y(), target_unwrapped.x(), target_unwrapped.y());
    //                     }
    //                     // --- END MODIFIED LOG ---
    //                 }
    //             }
    //         }
    //     }
    
    //     if (found_valid_triangle) {
    //         result.depth = best_depth;
    //         result.status = InterpolationStatus::SUCCESS;
    //         // --- MODIFIED LOG: Add attempt counts ---
    //         if (PointCloudUtils::should_log_point_details(logger)) {
    //             logger->debug("[computeBarycentricDepth] Final Result: SUCCESS. Best Depth = {:.3f} (from triangle area {:.4g}). Attempts={}, Degenerate={}, Outside={}",
    //                          result.depth, min_area, attempted_count, degenerate_count, outside_count);
    //         }
    //         // --- END MODIFIED LOG ---
    //     } else {
    //         // --- MODIFIED LOG: Add attempt counts ---
    //          if (PointCloudUtils::should_log_point_details(logger)) {
    //             logger->debug("[computeBarycentricDepth] Final Result: {}. Attempts={}, Degenerate={}, Outside={}. Degeneracy encountered: {}",
    //                          interpolationStatusToString(result.status), attempted_count, degenerate_count, outside_count, degeneracy_encountered);
    //         }
    //         // --- END MODIFIED LOG ---
    //     }
    
    //     return result;
    // }
 
 
    InterpolationResult interpolateDepth(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type)
    {
        auto logger = spdlog::get("Utils_Interpolation"); // Using new logger name
        uint64_t current_seq_id_local = g_current_logging_seq_id;

        if (logger->should_log(spdlog::level::trace)) {
            logger->trace("[interpolateDepth] ENTER for Point OriginalIdx={}, H={}, V={}, FrameID={}, MapIdx={}, NeighborType: {}",
                         p.original_index, p.hor_ind, p.ver_ind, current_seq_id_local, map_info.map_index, interpolationNeighborTypeToString(type));
        }

        // Set logging context for functions called from here
        LoggingContextSetter context(current_seq_id_local, p);

        std::vector<V3F> neighbors = findInterpolationNeighbors(p, map_info, params, type);
        V2F target_projection = p.vec.head<2>();
        InterpolationResult result = computeBarycentricDepth(target_projection, neighbors, params);

        if (logger->should_log(spdlog::level::debug)) { // Keep summary at debug
            logger->debug("[interpolateDepth] EXIT for Point OriginalIdx={}, FrameID={}: ResultStatus={}, CalcDepth={:.3f}",
                         p.original_index, current_seq_id_local, interpolationStatusToString(result.status), result.depth);
        }
        return result;
    }
 
 } // namespace PointCloudUtils