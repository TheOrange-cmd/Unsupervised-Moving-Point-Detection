// file: include/point_cloud_utils/point_cloud_utils.h

@file point_cloud_utils.h
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
     * @brief Finds the minimum and maximum static depth among valid neighbors of a point.
     *
     * Searches a square neighborhood (defined by params.checkneighbor_range) around
     * the point 'p' in the spherical index grid. For each neighbor cell within
     * the grid bounds that contains points, it considers the pre-computed static
     * min/max depth associated with that cell.
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
     * @note This function does NOT handle horizontal index wrap-around. Neighbors are checked
     *       only within the non-wrapped index range.
     * @note The function relies on the caller to initialize min_depth and max_depth appropriately before the call.
     *       A common initialization is min_depth = std::numeric_limits<float>::max(), max_depth = 0.0f.
     *       Alternatively, initialize both to 0.0f, and check after the call if they are still 0.0f to see if neighbors were found.
     */
    void findNeighborStaticDepthRange(const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        float& min_depth,
        float& max_depth);

    // START interpolation related functions

    // --- Helper Types ---
    enum class InterpolationNeighborType { STATIC_ONLY, ALL_VALID };
    enum class InterpolationStatus { SUCCESS, NOT_ENOUGH_NEIGHBORS, NO_VALID_TRIANGLE };
    struct InterpolationResult {
        InterpolationStatus status = InterpolationStatus::NOT_ENOUGH_NEIGHBORS;
        float depth = 0.0f;
    };

    // --- Helper Function (Internal or in .cpp) ---
    // Calculates wrapped index safely
    inline int getWrappedIndex(int base_idx, int offset, int max_dim) {
        int idx = base_idx + offset;
        return (idx % max_dim + max_dim) % max_dim; // Handles negative results correctly
    }

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

} // namespace PointCloudUtils

#endif // POINTCLOUD_UTILS_H