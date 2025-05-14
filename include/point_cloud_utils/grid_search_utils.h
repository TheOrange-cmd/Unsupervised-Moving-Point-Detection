/**
 * @file grid_search_utils.h
 * @brief Defines utility functions for searching and iterating over the spherical projection grid.
*/

#ifndef GRID_SEARCH_UTILS_H
#define GRID_SEARCH_UTILS_H

#include "common/dyn_obj_datatypes.h" // Includes point_soph, DepthMap, Eigen types etc.
#include "config/config_loader.h"     // Include to get DynObjFilterParams definition
#include <vector>
#include <functional> // For std::function (though template is preferred)
#include <algorithm>  // For std::max, std::min
#include <limits>     // For std::numeric_limits
#include <spdlog/spdlog.h> // For logging
 
namespace PointCloudUtils {

    /**
     * @brief Checks if a point has neighboring points vertically above and below it
     *        within specified limits in the same horizontal column of the depth map.
     * Used to identify points potentially isolated vertically, especially near FOV edges.
     * @param p The point to check (must contain valid `hor_ind` and `ver_ind`).
     * @param map_info The depth map containing other points.
     * @param params Parameters struct containing vertical FoV limits (`pixel_fov_down`, `pixel_fov_up`).
     * @return bool Returns `true` if the point LACKS support either above OR below (potentially isolated),
     *              `false` if it HAS support both above AND below.
     */
    bool checkVerticalFov(const point_soph& p, const DepthMap& map_info, const DynObjFilterParams& params);

    /**
     * @brief Internal helper template to iterate over neighbor cells and execute a callback.
     *
     * Iterates through a grid neighborhood defined by ranges around a center point's
     * spherical indices (`hor_ind`, `ver_ind`). Handles grid boundaries and optional
     * horizontal wrap-around. For each valid neighbor cell within the bounds, it
     * executes the provided callback function with the cell's 1D linearized index.
     *
     * @tparam Func Type of the callback function (e.g., lambda `[](int pos){...}`).
     * @param center_point The point defining the center of the search (must have valid `hor_ind`, `ver_ind`).
     * @param hor_range Half-size of the horizontal search window (non-negative).
     * @param ver_range Half-size of the vertical search window (non-negative).
     * @param include_center If true, the center cell itself is included in the iteration.
     * @param wrap_horizontal If true, handles horizontal index wrap-around using modulo arithmetic;
     *                        otherwise, checks bounds strictly within [0, MAX_1D-1].
     * @param callback A function-like object (lambda, functor) that accepts a single `int` (the 1D cell index).
     *                 The callback is responsible for handling the index (e.g., accessing `map_info.depth_map[pos]`).
     */
    template <typename Func>
    void forEachNeighborCell(
        const point_soph& center_point,
        int hor_range,
        int ver_range,
        bool include_center,
        bool wrap_horizontal,
        Func callback) // Accept any callable 'Func' directly
    {
        // Check if center point indices are valid first
        if (center_point.hor_ind < 0 || center_point.hor_ind >= MAX_1D ||
            center_point.ver_ind < 0 || center_point.ver_ind >= MAX_1D_HALF) {
            // Get logger and log warning if indices are bad
            auto logger = spdlog::get("Utils");
            if (logger) {
                logger->warn("[forEachNeighborCell] Center point indices out of bounds (H={}, V={}). Aborting search.",
                            center_point.hor_ind, center_point.ver_ind);
            }
            return;
        }

        // Use effective non-negative ranges for loop bounds
        int effective_hor_range = std::max(0, hor_range);
        int effective_ver_range = std::max(0, ver_range);

        // Determine loop bounds using effective ranges
        int h_start = center_point.hor_ind - effective_hor_range;
        int h_end = center_point.hor_ind + effective_hor_range;
        // Clamp vertical bounds to the valid grid range [0, MAX_1D_HALF - 1]
        int v_start = std::max(0, center_point.ver_ind - effective_ver_range);
        int v_end = std::min(MAX_1D_HALF - 1, center_point.ver_ind + effective_ver_range);

        for (int h = h_start; h <= h_end; ++h) {
            int current_h = h;

            // Handle horizontal indexing (wrap or clamp)
            if (wrap_horizontal) {
                // Modulo arithmetic for wrap-around
                current_h = (h % MAX_1D + MAX_1D) % MAX_1D;
            } else {
                // Skip if outside bounds when not wrapping
                if (current_h < 0 || current_h >= MAX_1D) {
                    continue;
                }
            }

            // Iterate vertically within clamped bounds
            for (int v = v_start; v <= v_end; ++v) {
                // Skip the center cell if requested
                if (!include_center && current_h == center_point.hor_ind && v == center_point.ver_ind) {
                    continue;
                }

                // Calculate the 1D linearized index
                int pos = current_h * MAX_1D_HALF + v;

                // Bounds check for the final 1D index (safety)
                if (pos < 0 || pos >= MAX_2D_N) {
                    // This should ideally not happen if MAX_ constants are correct, but good safeguard.
                    auto logger = spdlog::get("Utils");
                    if (logger) {
                        logger->error("[forEachNeighborCell] Calculated invalid 1D index {} (H={}, V={}). Skipping.",
                                    pos, current_h, v);
                    }
                    continue;
                }

                // Execute the provided callback with the valid 1D index
                callback(pos);
            }
        }
    }


    /**
     * @brief Finds the 1D linearized indices of neighboring cells in the spherical grid.
     *
     * Uses `forEachNeighborCell` to collect the indices.
     *
     * @param center_point The point defining the center of the search (must have valid `hor_ind`, `ver_ind`).
     * @param hor_range The half-size of the horizontal search window (e.g., 1 for 3 wide).
     * @param ver_range The half-size of the vertical search window (e.g., 1 for 3 high).
     * @param include_center If true, the index of the center_point's own cell is included in the result. Defaults to false.
     * @param wrap_horizontal If true, handles horizontal index wrap-around correctly. Defaults to true.
     * @return std::vector<int> A vector containing the 1D linearized indices (`position`) of the neighboring cells.
     */
    std::vector<int> findNeighborCells(
        const point_soph& center_point,
        int hor_range,
        int ver_range,
        bool include_center = false,
        bool wrap_horizontal = true
    );

    /**
     * @brief Finds the minimum and maximum static depth among valid neighbors of a point.
     *
     * Searches a square neighborhood (defined by `params.checkneighbor_range`) around
     * the point 'p' in the spherical index grid, handling horizontal wrap-around.
     * For each neighbor cell within the grid bounds that contains points, it considers
     * the pre-computed static min/max depth associated with that cell (`map_info.min_depth_static`, `map_info.max_depth_static`).
     *
     * @param p The center point (must contain valid `hor_ind` and `ver_ind`).
     * @param map_info The depth map containing neighbor information and pre-computed static depths.
     * @param params Parameters struct containing `checkneighbor_range`.
     * @param[out] min_depth Reference to a float that will be updated with the minimum static depth found among valid neighbors.
     *                       The caller should initialize this (e.g., to `std::numeric_limits<float>::max()`).
     *                       If no valid neighbors with static points are found, its value remains the initial value.
     * @param[out] max_depth Reference to a float that will be updated with the maximum static depth found among valid neighbors.
     *                       The caller should initialize this (e.g., to `0.0f`).
     *                       If no valid neighbors with static points are found, its value remains the initial value.
     */
    void findNeighborStaticDepthRange(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        float& min_depth,
        float& max_depth
    );

    /**
     * @brief Retrieves shared pointers to all `point_soph` objects stored within specified grid cells.
     *
     * Given a list of 1D cell indices and a DepthMap, this function collects
     * `std::shared_ptr<point_soph>` pointing to the points residing in those cells.
     * Invalid indices (out of bounds) are ignored. Null pointers within cells are skipped.
     *
     * @param cell_indices A vector of 1D integer indices representing the grid cells to query.
     * @param map_info The DepthMap containing the grid data (`map_info.depth_map`).
     * @return std::vector<std::shared_ptr<point_soph>> A vector containing shared pointers to all valid points found.
     */
    std::vector<std::shared_ptr<point_soph>> findPointsInCells(
        const std::vector<int>& cell_indices,
        const DepthMap& map_info
    );

} // namespace PointCloudUtils

#endif // GRID_SEARCH_UTILS_H