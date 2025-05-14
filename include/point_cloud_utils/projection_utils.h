/**
 * @file projection_utils.h
 * @brief Defines utility functions related to point projection, particularly spherical projection with caching.
*/

#ifndef PROJECTION_UTILS_H
#define PROJECTION_UTILS_H

#include "common/dyn_obj_datatypes.h" // Includes point_soph, Eigen types etc.
#include "config/config_loader.h"     // Include to get DynObjFilterParams definition

namespace PointCloudUtils {

// Constant used for cache validation threshold
constexpr double CACHE_VALID_THRESHOLD = 1e-5; // Renamed from 10e-5 for clarity

/**
 * @brief Projects a point into spherical coordinates, potentially using a cache.
 *
 * Transforms the point's global coordinates (`p.glob`) to the local frame defined
 * by `rot` and `transl`. Calculates the spherical projection (azimuth, elevation, range)
 * and corresponding grid indices (`hor_ind`, `ver_ind`, `position`).
 * Uses a simple cache (`p.last_vecs`, `p.last_positions`) based on `depth_index`
 * to avoid redundant calculations if the transformation hasn't changed significantly
 * (determined by `CACHE_VALID_THRESHOLD` on the range component).
 *
 * @param p Input point (contains global coords `glob`, cache is read and updated).
 * @param depth_index Index used for accessing the cache slot within `p`.
 * @param rot Rotation matrix transforming global to local frame.
 * @param transl Translation vector transforming global to local frame.
 * @param params The configuration parameters struct (needed for sensor resolutions).
 * @param[out] p_spherical Output point structure populated with spherical coordinates (`vec`),
 *                         grid indices (`hor_ind`, `ver_ind`), and 1D position (`position`).
 *                         Note: `p_spherical.glob` and `p_spherical.local` are NOT set by this function.
 */
void SphericalProjection(point_soph &p, int depth_index, const M3D &rot, const V3D &transl, const DynObjFilterParams& params, point_soph &p_spherical);

} // namespace PointCloudUtils

#endif // PROJECTION_UTILS_H