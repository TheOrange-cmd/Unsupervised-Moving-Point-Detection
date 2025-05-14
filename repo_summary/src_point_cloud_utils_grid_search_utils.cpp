/**
 * @file grid_search_utils.cpp
 * @brief Implements utility functions for searching and iterating over the spherical projection grid.
*/

#include "point_cloud_utils/grid_search_utils.h"
#include "common/dyn_obj_datatypes.h" // For point_soph, DepthMap, MAX_ constants
#include "config/config_loader.h"     // For DynObjFilterParams
#include <vector>
#include <algorithm> // For std::min, std::max
#include <limits>    // For std::numeric_limits
#include <spdlog/spdlog.h> // For logging

// Note: forEachNeighborCell is a template function and defined entirely in the header.
 
namespace PointCloudUtils {
 
    bool checkVerticalFov(const point_soph& p, const DepthMap& map_info, const DynObjFilterParams& params) {
        auto logger = spdlog::get("Utils_Grid");
        uint64_t frame_id = g_current_logging_seq_id; // Get current frame/sequence ID

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[checkVerticalFov] Point OriginalIdx={}, H={}, V={}, FrameID={}. Params: Down={}, Up={}",
                          p.original_index, p.hor_ind, p.ver_ind, frame_id, params.pixel_fov_down, params.pixel_fov_up);
        }

        bool found_support_down = false;
        bool found_support_up = false;

        if (p.hor_ind < 0 || p.hor_ind >= MAX_1D || p.ver_ind < 0 || p.ver_ind >= MAX_1D_HALF) {
            if (logger) logger->warn("[checkVerticalFov] Point OriginalIdx={}, FrameID={}: Called with invalid indices: H={}, V={}. Returning true (isolated).", p.original_index, frame_id, p.hor_ind, p.ver_ind);
            return true;
        }
        if (map_info.depth_map.size() != MAX_2D_N) {
            if (logger) logger->error("[checkVerticalFov] Point OriginalIdx={}, FrameID={}: map_info.depth_map size ({}) != MAX_2D_N ({}). Aborting check, returning true.", p.original_index, frame_id, map_info.depth_map.size(), MAX_2D_N);
            return true;
        }

        const int search_limit_down_param = params.pixel_fov_down; // Original param
        const int search_limit_up_param = params.pixel_fov_up;     // Original param
        const int actual_search_start_down = p.ver_ind;
        const int actual_search_end_down = std::max(0, p.ver_ind - search_limit_down_param); // Corrected logic: search from p.ver_ind down by search_limit_down_param pixels
        const int actual_search_start_up = p.ver_ind; // Redundant if loop starts at p.ver_ind
        const int actual_search_end_up = std::min(MAX_1D_HALF - 1, p.ver_ind + search_limit_up_param); // Corrected logic: search from p.ver_ind up by search_limit_up_param pixels


        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[checkVerticalFov] Point OriginalIdx={}, FrameID={}: Downwards search: V_start={}, V_end_clamped={}. Upwards search: V_start={}, V_end_clamped={}",
                          p.original_index, frame_id, actual_search_start_down, actual_search_end_down, p.ver_ind, actual_search_end_up);
        }
        
        // --- Check Downwards ---
        // Search from current p.ver_ind down to p.ver_ind - search_limit_down, clamped at 0
        for (int v_idx = p.ver_ind; v_idx >= actual_search_end_down; --v_idx) {
            int cur_pos = p.hor_ind * MAX_1D_HALF + v_idx;
            if (cur_pos < 0 || cur_pos >= MAX_2D_N) continue;

            if (!map_info.depth_map[cur_pos].empty()) {
                found_support_down = true;
                if (logger && logger->should_log(spdlog::level::trace)) {
                    logger->trace("[checkVerticalFov] Point OriginalIdx={}, FrameID={}: Found support DOWN at V_idx={}, Pos={}", p.original_index, frame_id, v_idx, cur_pos);
                }
                break;
            }
        }

        // --- Check Upwards ---
        // Search from current p.ver_ind up to p.ver_ind + search_limit_up, clamped at MAX_1D_HALF - 1
        for (int v_idx = p.ver_ind; v_idx <= actual_search_end_up; ++v_idx) {
            int cur_pos = p.hor_ind * MAX_1D_HALF + v_idx;
            if (cur_pos < 0 || cur_pos >= MAX_2D_N) continue;

            if (!map_info.depth_map[cur_pos].empty()) {
                found_support_up = true;
                 if (logger && logger->should_log(spdlog::level::trace)) {
                    logger->trace("[checkVerticalFov] Point OriginalIdx={}, FrameID={}: Found support UP at V_idx={}, Pos={}", p.original_index, frame_id, v_idx, cur_pos);
                }
                break;
            }
        }

        bool is_isolated = !(found_support_up && found_support_down);
        if (logger && logger->should_log(spdlog::level::trace)) { // Log result at trace
            logger->trace("[checkVerticalFov] Point OriginalIdx={}, FrameID={}: Result: Isolated={}, UpSupport={}, DownSupport={}.",
                         p.original_index, frame_id, is_isolated, found_support_up, found_support_down);
        }
        return is_isolated;
    }
 
    std::vector<int> findNeighborCells(
        const point_soph& center_point,
        int hor_range,
        int ver_range,
        bool include_center,
        bool wrap_horizontal)
    {
        auto logger = spdlog::get("Utils_Grid"); 
        uint64_t frame_id = g_current_logging_seq_id;

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[findNeighborCells] CenterPoint OriginalIdx={}, H={}, V={}, FrameID={}. Ranges: Hor={}, Ver={}, IncludeCenter={}, WrapH={}",
                          center_point.original_index, center_point.hor_ind, center_point.ver_ind, frame_id,
                          hor_range, ver_range, include_center, wrap_horizontal);
        }

        std::vector<int> neighbor_indices;
 
        // Estimate size for reservation
        int effective_hor_range = std::max(0, hor_range);
        int effective_ver_range = std::max(0, ver_range);
        size_t max_neighbors_estimate = static_cast<size_t>(2 * effective_hor_range + 1) *
                                        static_cast<size_t>(2 * effective_ver_range + 1);
        if (!include_center && max_neighbors_estimate > 0) {
            max_neighbors_estimate--;
        }
        if (max_neighbors_estimate > 0) {
            neighbor_indices.reserve(max_neighbors_estimate);
        }

        // Use the template function from the header to populate the vector
        forEachNeighborCell(center_point, hor_range, ver_range, include_center, wrap_horizontal,
            [&neighbor_indices, logger, frame_id, &center_point](int pos) { // Added logger, frame_id, center_point for context
                neighbor_indices.push_back(pos);
            }
        );

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[findNeighborCells] CenterPoint OriginalIdx={}, FrameID={}: Found {} neighbor cells.", center_point.original_index, frame_id, neighbor_indices.size());
        }
        return neighbor_indices;
     }
 
    void findNeighborStaticDepthRange(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        float& min_depth,
        float& max_depth)
    {
        auto logger = spdlog::get("Utils_Grid"); 
        uint64_t frame_id = g_current_logging_seq_id;
        bool first_valid_neighbor_found = false;
        const int search_range = std::max(0, params.checkneighbor_range);

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[findNeighborStaticDepthRange] Point OriginalIdx={}, H={}, V={}, FrameID={}. SearchRangeParam={}. Initial min_depth={:.3f}, max_depth={:.3f}",
                          p.original_index, p.hor_ind, p.ver_ind, frame_id, params.checkneighbor_range, min_depth, max_depth);
        }
 
        // Handle map_info validity checks upfront
        if (map_info.depth_map.size() != MAX_2D_N ||
            map_info.min_depth_static.size() != MAX_2D_N ||
            map_info.max_depth_static.size() != MAX_2D_N) {
            if (logger) logger->warn("[findNeighborStaticDepthRange] map_info vector sizes (depth={}, min={}, max={}) do not match MAX_2D_N ({}). Results may be unreliable.",
                                    map_info.depth_map.size(), map_info.min_depth_static.size(), map_info.max_depth_static.size(), MAX_2D_N);
            // Decide whether to return or proceed cautiously. Proceeding for now.
        }

        // If range is 0, only check the center cell
        bool include_center_cell = (search_range == 0);
        bool wrap_around = true; // Wrap horizontally by default for this function

        forEachNeighborCell(p, search_range, search_range, (search_range == 0), true,
            [&](int pos) {
                if (pos < 0 || pos >= MAX_2D_N) return;

                if (!map_info.depth_map[pos].empty() && map_info.min_depth_static[pos] > 0.0f)
                {
                    float cur_min_depth = map_info.min_depth_static[pos];
                    float cur_max_depth = map_info.max_depth_static[pos];

                    if (logger && logger->should_log(spdlog::level::trace)) {
                        logger->trace("[findNeighborStaticDepthRange] (lambda) Point OriginalIdx={}, FrameID={}: CellPos={}, ValidStaticNeighbor. CellMinD={:.3f}, CellMaxD={:.3f}",
                                      p.original_index, frame_id, pos, cur_min_depth, cur_max_depth);
                    }

                    if (!first_valid_neighbor_found) {
                        min_depth = cur_min_depth;
                        max_depth = cur_max_depth;
                        first_valid_neighbor_found = true;
                    } else {
                        min_depth = std::min(min_depth, cur_min_depth);
                        max_depth = std::max(max_depth, cur_max_depth);
                    }
                     if (logger && logger->should_log(spdlog::level::trace)) {
                        logger->trace("[findNeighborStaticDepthRange] (lambda) Point OriginalIdx={}, FrameID={}: Updated overall MinD={:.3f}, MaxD={:.3f}",
                                      p.original_index, frame_id, min_depth, max_depth);
                    }
                } else if (logger && logger->should_log(spdlog::level::trace)) {
                     logger->trace("[findNeighborStaticDepthRange] (lambda) Point OriginalIdx={}, FrameID={}: CellPos={}, No valid static points (empty_map_cell={}, min_static_depth={:.3f})",
                                      p.original_index, frame_id, pos, map_info.depth_map[pos].empty(), map_info.min_depth_static[pos]);
                }
            }
        );

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[findNeighborStaticDepthRange] Point OriginalIdx={}, FrameID={}: Final result: MinDepth={:.3f}, MaxDepth={:.3f}. FirstValidFound={}",
                         p.original_index, frame_id, min_depth, max_depth, first_valid_neighbor_found);
        }
    }
 
 
    std::vector<std::shared_ptr<point_soph>> findPointsInCells(
        const std::vector<int>& cell_indices,
        const DepthMap& map_info)
    {
        auto logger = spdlog::get("Utils_Grid"); // Or "Utils_Grid"
        uint64_t frame_id = g_current_logging_seq_id; // Assuming this is relevant for context, even if not point-specific

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[findPointsInCells] FrameID={}: Received {} cell indices to check. MapIndex={}",
                          frame_id, cell_indices.size(), map_info.map_index);
            // Optionally log all cell_indices if the list isn't too long
            // std::string indices_str; for(int idx : cell_indices) indices_str += std::to_string(idx) + " ";
            // logger->trace("  Cell Indices: {}", indices_str);
        }
         std::vector<std::shared_ptr<point_soph>> found_points;
         const size_t map_total_size = map_info.depth_map.size();
 
         if (map_total_size != MAX_2D_N) {
              if (logger) logger->warn("[findPointsInCells] map_info.depth_map size ({}) != MAX_2D_N ({}). Indices may be invalid.", map_total_size, MAX_2D_N);
         }
 
         // Optional: Pre-calculate reservation size
         size_t estimated_total_points = 0;
         for (int cell_index : cell_indices) {
             if (cell_index >= 0 && static_cast<size_t>(cell_index) < map_total_size) {
                 estimated_total_points += map_info.depth_map[cell_index].size();
             }
         }
         if (estimated_total_points > 0) {
             found_points.reserve(estimated_total_points);
         }
 
         // Collect points
         for (int cell_index : cell_indices) {
            // Check index validity
            if (cell_index >= 0 && static_cast<size_t>(cell_index) < map_info.depth_map.size()) {
                const auto& points_ptrs_in_cell = map_info.depth_map[cell_index];
                if (logger && logger->should_log(spdlog::level::trace) && !points_ptrs_in_cell.empty()) {
                    logger->trace("[findPointsInCells] FrameID={}: CellIndex {} contains {} point_ptrs.", frame_id, cell_index, points_ptrs_in_cell.size());
                }
                for (const auto& point_ptr : points_ptrs_in_cell) {
                    if (point_ptr) {
                        found_points.push_back(point_ptr);
                        // if (logger && logger->should_log(spdlog::level::trace)) {
                        //    logger->trace("[findPointsInCells] FrameID={}: Added point OriginalIdx {} from cell {}.", frame_id, point_ptr->original_index, cell_index);
                        // }
                    }
                }
            } else {
                if (logger) logger->warn("[findPointsInCells] Skipping invalid cell index {} (Map size: {}).", cell_index, map_total_size);
            }
         }
 
        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[findPointsInCells] FrameID={}: Found total {} points from given cells.", frame_id, found_points.size());
        }
        return found_points;
     }
 
 } // namespace PointCloudUtils