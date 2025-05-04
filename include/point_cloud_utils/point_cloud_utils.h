// file: point_cloud_utils/point_cloud_utils.h

// @file point_cloud_utils.h
//
// point_cloud_utils.h
//

#ifndef POINTCLOUD_UTILS_H
#define POINTCLOUD_UTILS_H

#include "filtering/dyn_obj_datatypes.h" // Includes point_soph, DepthMap, Eigen types etc.
#include "config/config_loader.h"     // Include to get DynObjFilterParams definition

// Define a namespace for these utility functions
namespace PointCloudUtils {

    struct AABB { // Axis-Aligned Bounding Box
        V3D min_corner;
        V3D max_corner;
    };

    // --- Interpolation Related Types ---
    enum class InterpolationNeighborType { STATIC_ONLY, ALL_VALID };
    enum class InterpolationStatus { SUCCESS, NOT_ENOUGH_NEIGHBORS, NO_VALID_TRIANGLE };
    struct InterpolationResult {
        InterpolationStatus status = InterpolationStatus::NOT_ENOUGH_NEIGHBORS;
        float depth = 0.0f;
    };

    // Constant used for barycentric calculations
    constexpr float BARYCENTRIC_EPSILON = 1e-5f;

    /**
     * @brief Projects a point into spherical coordinates, potentially using a cache.
     * @param p Input point (contains global coords, cache is updated).
     * @param depth_index Index for accessing the cache within p.
     * @param rot Rotation matrix for projection.
     * @param transl Translation vector for projection.
     * @param params The configuration parameters struct (DynObjFilterParams).
     * @param p_spherical Output point containing spherical coordinates and indices.
     */
    void SphericalProjection(point_soph &p, int depth_index, const M3D &rot, const V3D &transl, const DynObjFilterParams& params, point_soph &p_spherical);

    /**
     * @brief Checks if a point is invalid (e.g., too close).
     * @param body Point coordinates (Eigen V3D).
     * @param intensity Point intensity (currently unused in implementation).
     * @param params The configuration parameters struct (DynObjFilterParams).
     * @return True if the point is invalid, false otherwise.
     */
    bool isPointInvalid(const V3D& point, float blind_distance, int dataset_id);

    /**
     * @brief Checks if a point falls within predefined "self" regions for a specific dataset.
     * @param point The 3D point coordinates.
     * @param dataset_id Identifier for the dataset (filtering rules depend on this).
     * @return True if the point is considered part of the "self" for the given dataset, false otherwise.
     */
    bool isSelfPoint(const V3D& point, int dataset_id);

    // Helper function for readability, not part of the original code
    /**
     * @brief Checks if a point is inside an Axis-Aligned Bounding Box (AABB).
     * @param point The 3D point coordinates.
     * @param box The Axis-Aligned Bounding Box defined by min and max corners.
     * @return True if the point is inside the AABB, false otherwise.
     */
    bool isInsideAABB(const V3D& point, const AABB& box);

    /**
     * @brief Checks if a point has neighboring points vertically above and below it
     *        within specified limits in the same horizontal column of the depth map.
     * @param p The point to check (must contain valid hor_ind and ver_ind).
     * @param map_info The depth map containing other points.
     * @param params Parameters struct containing vertical FoV limits (pixel_fov_down, pixel_fov_up).
     * @return bool Returns true if the point LACKS support either above or below (potentially isolated),
     *              false if it HAS support both above and below.
     */
    bool checkVerticalFov(const point_soph& p, const DepthMap& map_info, const DynObjFilterParams& params);

    /**
     * @brief Internal helper to iterate over neighbor cells and execute a callback.
     *
     * Iterates through a grid neighborhood defined by ranges around a center point.
     * Handles grid boundaries and optional horizontal wrap-around. For each valid
     * neighbor cell within the bounds, it executes the provided callback function
     * with the cell's 1D linearized index.
     *
     * @tparam Func Type of the callback function (usually a lambda).
     * @param center_point The point defining the center of the search.
     * @param hor_range Half-size of the horizontal search window.
     * @param ver_range Half-size of the vertical search window.
     * @param include_center If true, the center cell itself is included in the iteration.
     * @param wrap_horizontal If true, handles horizontal index wrap-around; otherwise, checks bounds [0, MAX_1D-1].
     * @param callback A function-like object (lambda, functor) that accepts a single int (the 1D cell index).
     */
    template <typename Func> // Use template for the callable
    void forEachNeighborCell(
        const point_soph& center_point,
        int hor_range,
        int ver_range,
        bool include_center,
        bool wrap_horizontal,
        Func func) // Accept any callable 'Func' directly
    {
        // Check if center point indices are valid first
        if (center_point.hor_ind < 0 || center_point.hor_ind >= MAX_1D ||
            center_point.ver_ind < 0 || center_point.ver_ind >= MAX_1D_HALF) {
            // Optional: Add warning if desired
            // std::cerr << "[forEachNeighborCell] Warning: Center point indices out of bounds." << std::endl;
            return;
        }

        // Use effective non-negative ranges for loop bounds
        int effective_hor_range = std::max(0, hor_range);
        int effective_ver_range = std::max(0, ver_range);

        // Determine loop bounds using effective ranges
        int h_start = center_point.hor_ind - effective_hor_range;
        int h_end = center_point.hor_ind + effective_hor_range;
        int v_start = std::max(0, center_point.ver_ind - effective_ver_range);
        int v_end = std::min(MAX_1D_HALF - 1, center_point.ver_ind + effective_ver_range);

        // --- (Optional: Remove Debug Prints) ---
        // std::cout << "[forEachNeighborCell] Debug Info:" << std::endl;
        // ... debug prints ...
        // ---

        for (int h = h_start; h <= h_end; ++h) {
            int current_h = h;

            if (wrap_horizontal) {
                if (current_h < 0) {
                    current_h += MAX_1D;
                } else if (current_h >= MAX_1D) {
                    current_h -= MAX_1D;
                }
            } else {
                if (current_h < 0 || current_h >= MAX_1D) {
                    continue;
                }
            }

            for (int v = v_start; v <= v_end; ++v) {
                if (!include_center && h == center_point.hor_ind && v == center_point.ver_ind) {
                    continue;
                }

                int pos = current_h * MAX_1D_HALF + v;
                // --- FIX: Call the templated 'func' directly ---
                func(pos);
            }
        }
    }

    /**
     * @brief Finds the 1D linearized indices of neighboring cells in the spherical grid.
     *
     * Searches a neighborhood defined by horizontal and vertical ranges around
     * the center_point's spherical projection indices (hor_ind, ver_ind).
     *
     * @param center_point The point defining the center of the search (must have valid hor_ind, ver_ind).
     * @param hor_range The half-size of the horizontal search window (e.g., 1 for 3 wide).
     * @param ver_range The half-size of the vertical search window (e.g., 1 for 3 high).
     * @param include_center If true, the index of the center_point's own cell is included in the result.
     * @param wrap_horizontal If true, handles horizontal index wrap-around correctly. If false, checks bounds [0, MAX_1D-1].
     * @return std::vector<int> A vector containing the 1D linearized indices (positions) of the neighboring cells.
     */
    std::vector<int> findNeighborCells(
        const point_soph& center_point,
        int hor_range,
        int ver_range,
        bool include_center = false,
        bool wrap_horizontal = true // Default to enabling wrap-around
    );

    /**
     * @brief Finds the minimum and maximum static depth among valid neighbors of a point.
     *
     * Searches a square neighborhood (defined by params.checkneighbor_range) around
     * the point 'p' in the spherical index grid, handling horizontal wrap-around.
     * For each neighbor cell within the grid bounds that contains points, it considers
     * the pre-computed static min/max depth associated with that cell.
     *
     * @param p The center point (must contain valid hor_ind and ver_ind).
     * @param map_info The depth map containing neighbor information and pre-computed static depths.
     * @param params Parameters struct containing checkneighbor_range.
     * @param[out] min_depth Reference to a float that will be updated with the minimum static depth found among neighbors.
     *                       The caller should initialize this (e.g., to std::numeric_limits<float>::max() or 0.0f).
     *                       If no valid neighbors are found, its value might remain unchanged or be the initial value.
     * @param[out] max_depth Reference to a float that will be updated with the maximum static depth found among neighbors.
     *                       The caller should initialize this (e.g., to 0.0f or std::numeric_limits<float>::lowest()).
     *                       If no valid neighbors are found, its value might remain unchanged or be the initial value.
     *
     * @note Relies on the caller to initialize min_depth and max_depth appropriately before the call.
     *       Common initialization: min_depth = std::numeric_limits<float>::max(), max_depth = 0.0f.
     */
    void findNeighborStaticDepthRange(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        float& min_depth,
        float& max_depth
    );

    /**
     * @brief Finds suitable neighboring points (as V3F vectors) for depth interpolation.
     *
     * Searches a neighborhood defined by interp_hor_num and interp_ver_num, handling wrap-around.
     * Filters points within neighbor cells based on time difference, angular distance, and type.
     *
     * @param p The point for which to find neighbors (center of search).
     * @param map_info The depth map data containing points and grid structure.
     * @param params Filter parameters (interp range, thresholds, frame_dur, etc.).
     * @param type Whether to collect only static or all valid neighbors (InterpolationNeighborType).
     * @return std::vector<V3F> Vector of 3D spherical coordinates (azimuth, elevation, depth) of valid neighbors.
     */
    std::vector<V3F> findInterpolationNeighbors(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type
    );

    // --- Core Logic Functions ---

    /**
     * @brief Finds suitable neighboring points for depth interpolation.
     * @param p The point for which to find neighbors.
     * @param map_info The depth map data.
     * @param params Filter parameters (interp range, thresholds, frame_dur).
     * @param type Whether to collect only static or all valid neighbors.
     * @return Vector of 3D coordinates (V3F) of valid neighbors.
     */
    std::vector<V3F> findInterpolationNeighbors(
        const point_soph& p,
        const DepthMap& map_info, // Assuming DepthMap holds the grid
        const DynObjFilterParams& params,
        InterpolationNeighborType type);

    /**
     * @brief Attempts to find a triangle of neighbors enclosing p's projection and interpolates depth.
     * @param target_point_projection 2D projection of the point to interpolate for (e.g., p.vec.head<2>()).
     * @param neighbors Vector of 3D coordinates (V3F) of potential neighbors.
     * @param params Filter parameters (needed for thresholds like interp_hor_thr if used internally).
     * @return InterpolationResult indicating success/failure and the depth.
     */
    InterpolationResult computeBarycentricDepth(
        const V2F& target_point_projection, // Assuming V2F type for 2D
        const std::vector<V3F>& neighbors,
        const DynObjFilterParams& params); // May only need specific thresholds


    // --- Main Public Function ---

    /**
     * @brief Interpolates the depth for a point using barycentric interpolation on neighbors.
     * @param p The point to interpolate depth for.
     * @param map_info The depth map data.
     * @param params Filter parameters.
     * @param type Whether to use only static or all valid neighbors.
     * @return InterpolationResult indicating success/failure and the depth.
     */
    InterpolationResult interpolateDepth(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type);

    // END interpolation related functions

    /**
     * @brief Retrieves all point_soph objects stored within specified grid cells.
     *
     * Given a list of 1D cell indices and a DepthMap, this function collects
     * copies of all point_soph objects residing in those cells. Invalid indices
     * (out of bounds) are ignored.
     *
     * @param cell_indices A vector of 1D integer indices representing the grid cells to query.
     * @param map_info The DepthMap containing the grid data (map_info.depth_map).
     * @return std::vector<point_soph> A vector containing copies of all points found in the specified cells.
     *         The order of points generally follows the order of cell_indices and the order within each cell's vector.
     */
    std::vector<std::shared_ptr<point_soph>> findPointsInCells( // <-- Return shared_ptr vector
        const std::vector<int>& cell_indices,
        const DepthMap& map_info
    );

} // namespace PointCloudUtils

#endif // POINTCLOUD_UTILS_H