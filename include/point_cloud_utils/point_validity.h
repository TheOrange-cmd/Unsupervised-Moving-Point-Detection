/**
 * @file point_validity.h
 * @brief Defines utility functions for checking point validity based on proximity, bounding boxes, etc.
*/

#ifndef POINT_VALIDITY_H
#define POINT_VALIDITY_H

#include "common/dyn_obj_datatypes.h" // Includes V3D, AABB
#include "config/config_loader.h"     // Include to get DynObjFilterParams definition
#include <vector>                     // For self_boxes definition

namespace PointCloudUtils {

/**
 * @brief Checks if a point is invalid based on configured criteria.
 * Invalid points include those too close to the sensor (within blind_dis)
 * and optionally those within a specific configurable bounding box near the origin.
 * @param point The 3D point coordinates in the local sensor frame.
 * @param params The configuration parameters containing blind_dis and invalid box settings.
 * @return True if the point is considered invalid, false otherwise.
 */
bool isPointInvalid(const V3D& point, const DynObjFilterParams& params);

/**
 * @brief Checks if a point falls within the configured ego-vehicle bounding box(es).
 * This function checks against a simple rectangular region defined by params
 * OR potentially against a list of predefined AABBs based on dataset type (implementation detail).
 * @param local_point The 3D point coordinates in the local sensor/ego frame.
 * @param params The configuration parameters containing self-filtering settings (e.g., self_x_f, self_x_b, dataset type).
 * @return True if the point is considered part of the "self" region, false otherwise.
 */
bool isSelfPoint(const V3D& local_point, const DynObjFilterParams& params);

struct AABB { // Axis-Aligned Bounding Box
    V3D min_corner;
    V3D max_corner;
};

/**
 * @brief Checks if a point is inside an Axis-Aligned Bounding Box (AABB).
 * @param point The 3D point coordinates.
 * @param box The Axis-Aligned Bounding Box defined by min and max corners.
 * @return True if the point is strictly inside the AABB (exclusive boundaries), false otherwise.
 */
bool isInsideAABB(const V3D& point, const AABB& box);

// Example definition of dataset-specific boxes (could be moved or loaded differently)
extern const std::vector<AABB> self_boxes_dataset0; // Example for KITTI

} // namespace PointCloudUtils

#endif // POINT_VALIDITY_H