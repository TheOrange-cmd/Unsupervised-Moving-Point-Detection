#ifndef POINTCLOUD_UTILS_H
#define POINTCLOUD_UTILS_H

#include "dyn_obj_datatypes.h" // Includes point_soph, DepthMap, Eigen types etc.
#include "config_loader.h"     // Include to get DynObjFilterParams definition

// Define a namespace for these utility functions
namespace PointCloudUtils {

    struct AABB { // Axis-Aligned Bounding Box
        V3D min_corner;
        V3D max_corner;
    };

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

    // /**
    //  * @brief Checks if a point is near the vertical Field of View limits based on neighbors.
    //  * @param p The point to check (uses hor_ind, ver_ind).
    //  * @param map_info The depth map containing neighbor information.
    //  * @param params The configuration parameters struct (DynObjFilterParams).
    //  * @return True if the point is near the FoV limit, false otherwise.
    //  */
    // bool CheckVerFoV(const point_soph & p, const DepthMap &map_info, const DynObjFilterParams& params);

    // /**
    //  * @brief Finds the min/max static depth values in the neighborhood of a point.
    //  * @param p The point defining the neighborhood center.
    //  * @param map_info The depth map containing neighbor depth information.
    //  * @param params The configuration parameters struct (DynObjFilterParams).
    //  * @param max_depth Output: Maximum static depth found in the neighborhood.
    //  * @param min_depth Output: Minimum static depth found in the neighborhood.
    //  */
    // void CheckNeighbor(const point_soph & p, const DepthMap &map_info, const DynObjFilterParams& params, float &max_depth, float &min_depth);

    // /**
    //  * @brief Interpolates depth using only STATIC neighbors from a historical depth map.
    //  * @param p Input point (spherical coords used, cache is updated).
    //  * @param map_index Index of the historical map (used for cache offset calculation).
    //  * @param base_map_index Index of the earliest map in the current processing window (for cache offset).
    //  * @param depth_map The historical 2D depth map containing potential neighbors.
    //  * @param params The configuration parameters struct (DynObjFilterParams).
    //  * @return Interpolated depth, -1 if no neighbors, -2 if triangulation failed.
    //  */
    // float DepthInterpolationStatic(point_soph & p, int map_index, int base_map_index, const DepthMap2D &depth_map, const DynObjFilterParams& params);

    // /**
    //  * @brief Interpolates depth using ALL neighbors from a historical depth map.
    //  * @param p Input point (spherical coords used).
    //  * @param depth_map The historical 2D depth map containing potential neighbors.
    //  * @param params The configuration parameters struct (DynObjFilterParams).
    //  * @return Interpolated depth, -1 if not enough neighbors, -2 if triangulation failed.
    //  */
    // float DepthInterpolationAll(const point_soph & p, const DepthMap2D &depth_map, const DynObjFilterParams& params);

} // namespace PointCloudUtils

#endif // POINTCLOUD_UTILS_H