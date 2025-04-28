#ifndef CONSISTENCY_CHECKS_H
#define CONSISTENCY_CHECKS_H

#include "filtering/dyn_obj_datatypes.h" // Includes point_soph, DepthMap
#include "config/config_loader.h"     // Includes DynObjFilterParams
#include "point_cloud_utils/point_cloud_utils.h" // For Interpolation types

// Define a namespace for consistency check functions
namespace ConsistencyChecks {

    /**
     * @brief Specifies the context or case for the map consistency check,
     *        influencing which parameters and neighbor types are used.
     */
    enum class ConsistencyCheckType {
        CASE1_FALSE_REJECTION, // Corresponds to original Case1FalseRejection/Case1MapConsistencyCheck
        CASE2_OCCLUDER_SEARCH, // Corresponds to original Case2MapConsistencyCheck (checking point P against older map)
        CASE3_OCCLUDED_SEARCH  // Corresponds to original Case3MapConsistencyCheck (checking point P against older map)
    };

    /**
     * @brief Checks if a point 'p' is consistent with a given DepthMap based on interpolation.
     *
     * This function refactors the core logic found in the original Case1, Case2, and Case3
     * map consistency checks, primarily relying on depth interpolation using neighbors
     * from the provided map_info. The function checks if a point p (from a relatively recent time t_p) is 
     * geometrically consistent with the scene as it was recorded in an
     *  older map (map_info from time t_map, where t_map < t_p).
     * 
     * An inconsistent point (checkMapConsistency returns false) is a strong indicator that the
     * point likely belongs to an object exhibiting independent motion relative to the ego-vehicle and
     * the static background, or represents a significant change like appearing/disappearing 
     * or occluding/disoccluding. A consistent point generally belongs to the static background 
     * whose structure hasn't changed significantly relative to the sensor between the two timeframes.
     *
     * The behavior (neighbor type selection, thresholds) is determined by the check_type parameter.
     *
     * @param p The point_soph object to check for consistency. Its spherical coordinates (p.vec)
     *          and local coordinates (p.local) are used.
     * @param map_info The DepthMap representing the environment against which 'p' is checked.
     * @param params The configuration parameters containing thresholds (interp_thr*, etc.)
     *               and settings (dataset_id, self-region bounds).
     * @param check_type Enum indicating which case's rules/thresholds to apply (CASE1, CASE2, CASE3).
     * @param map_index_diff The difference between the current frame index and the map_info frame index.
     *                       Used for scaling thresholds in Case 2 and Case 3 as per original logic. Defaults to 1.
     * @return bool True if the point 'p' is considered consistent with the map 'map_info'
     *              based on the interpolation check and thresholds for the specified case.
     *              False otherwise (including if interpolation fails or the point is in the self-region).
     */
    bool checkMapConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        int map_index_diff = 1 // Default difference to 1 if not provided
    );

    // Forward declarations for other checks if they go in this file
    // bool checkDepthConsistency(...);
    // bool checkVelocityConsistency(...);

} // namespace ConsistencyChecks

#endif // CONSISTENCY_CHECKS_H