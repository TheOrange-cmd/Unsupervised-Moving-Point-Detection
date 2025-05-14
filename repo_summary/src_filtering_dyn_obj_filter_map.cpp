/**
 * @file dyn_obj_filter_map.cpp
 * @brief Implements the depth map management and point insertion logic for DynObjFilter.
 */
#include "filtering/dyn_obj_filter.h" // Need full class definition

#include <deque>
#include <vector>
#include <memory> // For std::shared_ptr
#include <algorithm> // For std::min, std::max
#include "common/types.h"
// Include necessary utility headers
#include "point_cloud_utils/projection_utils.h" // For SphericalProjection used in addPointsToMap

// Logging
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

// extern const int DEBUG_POINT_IDX;
// extern const uint64_t DEBUG_FRAME_SEQ_ID;
// extern uint64_t g_current_logging_seq_id;

// --- Update Depth Maps ---
void DynObjFilter::updateDepthMaps(
    const std::vector<std::shared_ptr<point_soph>>& points_to_add,
    const ScanFrame& current_frame_info)
{
    // Use the "Filter_Map" logger
    auto logger = spdlog::get("Filter_Map");
    DepthMap::Ptr target_map = nullptr;

    // --- Map List Management ---
    if (depth_map_list_.empty()) {
        int current_map_idx = ++map_index_;
        if (logger) logger->debug("Creating first DepthMap (Index: {}) at time {:.6f}", current_map_idx, current_frame_info.timestamp);
        target_map = std::make_shared<DepthMap>(
            current_frame_info.sensor_pose.rotation(),
            current_frame_info.sensor_pose.translation(),
            current_frame_info.timestamp,
            current_map_idx
        );
        depth_map_list_.push_back(target_map);
    } else {
        double time_since_last_map = current_frame_info.timestamp - depth_map_list_.back()->time;
        if (time_since_last_map >= params_.depth_map_dur) {
            int next_map_idx = ++map_index_;
            if (logger) logger->debug("Time threshold reached ({:.3f}s >= {:.3f}s). ", time_since_last_map, params_.depth_map_dur);

            if (depth_map_list_.size() >= params_.max_depth_map_num) {
                if (logger) logger->debug("Rotating DepthMap (Reusing Index: {}, New Index: {}) at time {:.6f}",
                                         depth_map_list_.front()->map_index, next_map_idx, current_frame_info.timestamp);
                target_map = depth_map_list_.front();
                depth_map_list_.pop_front();
                target_map->Reset(
                    current_frame_info.sensor_pose.rotation(),
                    current_frame_info.sensor_pose.translation(),
                    current_frame_info.timestamp,
                    next_map_idx
                );
                depth_map_list_.push_back(target_map);
            } else {
                 if (logger) logger->debug("Creating new DepthMap (Index: {}) at time {:.6f}", next_map_idx, current_frame_info.timestamp);
                 target_map = std::make_shared<DepthMap>(
                    current_frame_info.sensor_pose.rotation(),
                    current_frame_info.sensor_pose.translation(),
                    current_frame_info.timestamp,
                    next_map_idx
                );
                depth_map_list_.push_back(target_map);
            }
        } else {
            target_map = depth_map_list_.back();
             if (logger) logger->trace("Using existing DepthMap (Index: {})", target_map->map_index);
        }
    }

    // --- Add Points to the Target Map ---
    if (!target_map) {
         if (logger) logger->error("[updateDepthMaps] No target depth map available! Cannot add points.");
         else std::cerr << "Error: No target depth map available in updateDepthMaps!" << std::endl;
         return;
    }

    addPointsToMap(*target_map, points_to_add);
}

// --- Add Points To Map ---
// void DynObjFilter::addPointsToMap(
//     DepthMap& map,
//     const std::vector<std::shared_ptr<point_soph>>& points_to_add)
// {
//     // Use the "Filter_Map" logger
//     auto logger = spdlog::get("Filter_Map");
//     int points_actually_added = 0;

//     for (const auto& p_soph : points_to_add) {
//         if (!p_soph) continue;

//         // Project point into the target map's frame to get correct grid index
//         point_soph p_spherical_in_map;
//         // Note: SphericalProjection uses "Utils" logger internally if enabled
//         PointCloudUtils::SphericalProjection(*p_soph, map.map_index, map.project_R, map.project_T, params_, p_spherical_in_map);
//         int pos = p_spherical_in_map.position;

//         if (pos >= 0 && pos < MAX_2D_N) {
//             if (map.depth_map[pos].size() < params_.max_pixel_points) {
//                 map.depth_map[pos].push_back(p_soph);
//                 points_actually_added++;
//                 size_t current_cell_size = map.depth_map[pos].size();
//                 size_t current_point_index_in_cell = current_cell_size - 1;

//                 // --- Statistics Update ---
//                 float current_depth = p_spherical_in_map.vec(2);
//                 bool is_first_in_cell = (current_cell_size == 1);

//                 if (is_first_in_cell || current_depth > map.max_depth_all[pos]) {
//                     map.max_depth_all[pos] = current_depth;
//                     map.max_depth_index_all[pos] = current_point_index_in_cell;
//                 }
//                 if (is_first_in_cell || current_depth < map.min_depth_all[pos]) {
//                      map.min_depth_all[pos] = current_depth;
//                      map.min_depth_index_all[pos] = current_point_index_in_cell;
//                 }
//                 if (p_soph->dyn == DynObjLabel::STATIC) {
//                      bool first_static_in_cell = true;
//                      if (!is_first_in_cell) {
//                          for(size_t k = 0; k < current_point_index_in_cell; ++k) {
//                              if(map.depth_map[pos][k] && map.depth_map[pos][k]->dyn == DynObjLabel::STATIC) {
//                                  first_static_in_cell = false;
//                                  break;
//                              }
//                          }
//                      }
//                      if (first_static_in_cell) {
//                          map.min_depth_static[pos] = current_depth;
//                          map.max_depth_static[pos] = current_depth;
//                      } else {
//                          map.min_depth_static[pos] = std::min(map.min_depth_static[pos], current_depth);
//                          map.max_depth_static[pos] = std::max(map.max_depth_static[pos], current_depth);
//                      }
//                       if (logger) logger->trace("Updated static depth for cell {}: min={:.3f}, max={:.3f}", pos, map.min_depth_static[pos], map.max_depth_static[pos]);
//                 }
//                 // --- End Statistics Update ---

//             } else {
//                  if (logger) logger->trace("Pixel {} full ({} points), cannot add point.", pos, params_.max_pixel_points);
//             }
//         } else {
//              if (logger) logger->warn("Invalid projected position index {} calculated for point. Cannot add to map.", pos);
//         }
//     }
//     if (logger) {
//         // Log at debug level - info might be too noisy
//         logger->debug("Added {} points to DepthMap index {}", points_actually_added, map.map_index);
//     }
// }
void DynObjFilter::addPointsToMap(
    DepthMap& map,
    const std::vector<std::shared_ptr<point_soph>>& points_to_add)
{
    auto logger = spdlog::get("Filter_Map");
    int points_actually_added = 0;
    // Use the externally declared global variable
    uint64_t current_seq_id_local = PointCloudUtils::g_current_logging_seq_id;


    for (const auto& p_soph : points_to_add) {
        if (!p_soph) continue;

        point_soph p_spherical_in_map;
        PointCloudUtils::SphericalProjection(*p_soph, map.map_index, map.project_R, map.project_T, params_, p_spherical_in_map);
        int pos = p_spherical_in_map.position;
         // Use the externally declared debug constants
        bool is_debug_point = (p_soph->original_index == DEBUG_POINT_IDX && current_seq_id_local == DEBUG_FRAME_SEQ_ID);


        if (pos >= 0 && pos < MAX_2D_N) {
            if (map.depth_map[pos].size() < params_.max_pixel_points) {
                map.depth_map[pos].push_back(p_soph);
                points_actually_added++;
                size_t current_cell_size = map.depth_map[pos].size();
                size_t current_point_index_in_cell = current_cell_size - 1;

                float current_depth = p_spherical_in_map.vec(2);
                bool is_first_in_cell = (current_cell_size == 1);

                // --- Log min/max_depth_all updates ---
                if (is_debug_point && logger) {
                    if (is_first_in_cell || current_depth > map.max_depth_all[pos]) {
                        logger->trace("  Point {} (Frame {}), Cell {}: Updating max_depth_all from {:.3f} to {:.3f}", p_soph->original_index, current_seq_id_local, pos, is_first_in_cell ? -std::numeric_limits<float>::infinity() : map.max_depth_all[pos], current_depth);
                    }
                     if (is_first_in_cell || current_depth < map.min_depth_all[pos]) {
                        logger->trace("  Point {} (Frame {}), Cell {}: Updating min_depth_all from {:.3f} to {:.3f}", p_soph->original_index, current_seq_id_local, pos, is_first_in_cell ? std::numeric_limits<float>::infinity() : map.min_depth_all[pos], current_depth);
                    }
                }
                // --- End Log ---

                if (is_first_in_cell || current_depth > map.max_depth_all[pos]) {
                    map.max_depth_all[pos] = current_depth;
                    map.max_depth_index_all[pos] = current_point_index_in_cell;
                }
                if (is_first_in_cell || current_depth < map.min_depth_all[pos]) {
                     map.min_depth_all[pos] = current_depth;
                     map.min_depth_index_all[pos] = current_point_index_in_cell;
                }

                // Check if the point being added is STATIC
                if (p_soph->dyn == DynObjLabel::STATIC) {
                     bool first_static_in_cell = true;
                     // Check previous points in the *same cell* if this isn't the first point overall
                     if (!is_first_in_cell) {
                         for(size_t k = 0; k < current_point_index_in_cell; ++k) {
                             // Check if pointer is valid AND label is STATIC
                             if(map.depth_map[pos][k] && map.depth_map[pos][k]->dyn == DynObjLabel::STATIC) {
                                 first_static_in_cell = false;
                                 break;
                             }
                         }
                     }

                     float old_min_static = map.min_depth_static[pos];
                     float old_max_static = map.max_depth_static[pos];

                     if (first_static_in_cell) {
                         map.min_depth_static[pos] = current_depth;
                         map.max_depth_static[pos] = current_depth;
                     } else {
                         // Only update if not the first static point, otherwise keep initial values
                         map.min_depth_static[pos] = std::min(map.min_depth_static[pos], current_depth);
                         map.max_depth_static[pos] = std::max(map.max_depth_static[pos], current_depth);
                     }

                     // Log static depth updates
                     if (logger && (is_debug_point || map.min_depth_static[pos] != old_min_static || map.max_depth_static[pos] != old_max_static)) {
                          logger->trace("  Point {} (Frame {}), Cell {}: STATIC point added. Updated static depth: min={:.3f} (was {:.3f}), max={:.3f} (was {:.3f}). FirstStatic={}",
                                       p_soph->original_index, current_seq_id_local, pos,
                                       map.min_depth_static[pos], old_min_static,
                                       map.max_depth_static[pos], old_max_static,
                                       first_static_in_cell);
                     }

                } // End if STATIC
            } else {
                 if (logger && is_debug_point) {
                      logger->trace("  Point {} (Frame {}), Cell {}: Pixel full ({} points), cannot add.", p_soph->original_index, current_seq_id_local, pos, params_.max_pixel_points);
                 }
            }
        } else {
             if (logger && is_debug_point) {
                 logger->warn("  Point {} (Frame {}): Invalid projected position index {} calculated. Cannot add to map.", p_soph->original_index, current_seq_id_local, pos);
             }
        }
    } // End loop through points_to_add

    if (logger) {
        logger->debug("Added {} points to DepthMap index {}", points_actually_added, map.map_index);
    }
}