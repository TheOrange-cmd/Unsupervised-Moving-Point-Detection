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

    /**
     * @brief Checks if a point 'p' has a depth consistent with nearby STATIC points in an older map.
     *
     * This function refactors the logic from the original Case2DepthConsistencyCheck and
     * Case3DepthConsistencyCheck. It examines static points within a defined angular and
     * temporal neighborhood of 'p' in the provided 'map_info'.
     *
     * Consistency is determined based on two criteria:
     * 1. The average absolute depth difference between 'p' and 'close' static neighbors
     *    (those within 'max_thr') must not exceed a dynamic threshold ('current_depth_threshold').
     * 2. Among static neighbors with a significant depth difference (greater than 'max_thr'),
     *    'p' must be consistently either closer than all of them OR farther than all of them.
     *    A mix of significantly closer and significantly farther neighbors indicates inconsistency.
     *
     * @param p The point_soph object to check for depth consistency.
     * @param map_info The DepthMap (representing an older timeframe) containing potential neighbors.
     * @param params The configuration parameters holding thresholds (depth_cons_*, k_depth*, etc.).
     * @param check_type Enum indicating which case's parameters to use (CASE2 or CASE3).
     *                   CASE1 is not supported by this check.
     * @return bool True if 'p' is considered depth-consistent with the static points in 'map_info'
     *              according to the rules for the specified case. False otherwise (including if
     *              no suitable static neighbors are found).
     * @throws std::invalid_argument if check_type is CASE1_FALSE_REJECTION.
     */
    bool checkDepthConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

    /**
     * @brief Checks if the change between two velocity measurements is consistent with a maximum acceleration limit.
     *
     * This function embodies the logic previously in Case2VelCheck and Case3VelCheck.
     * It calculates the implied acceleration between two velocity measurements and compares it
     * to a given threshold.
     *
     * @param velocity1 The first velocity measurement (e.g., from time t-2 to t-1).
     * @param velocity2 The second velocity measurement (e.g., from time t-1 to t).
     * @param time_delta_between_velocity_centers The time difference between the midpoints of the intervals
     *                                            over which velocity1 and velocity2 were calculated.
     *                                            (e.g., ((t + t-1)/2) - ((t-1 + t-2)/2) = (t - t-2)/2 ).
     *                                            Alternatively, if velocities are considered instantaneous,
     *                                            this is simply the time difference between the velocity measurements.
     * @param acceleration_threshold The maximum plausible acceleration (magnitude).
     * @return True if the absolute difference in velocities is less than the allowed change
     *         (acceleration_threshold * time_delta_between_velocity_centers), false otherwise.
     */
    bool checkAccelerationLimit(
        float velocity1,
        float velocity2,
        double time_delta_between_velocity_centers,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

    /**
     * @brief Checks if two points satisfy the geometric and temporal conditions for an occlusion relationship.
     *
     * Replaces `Case2IsOccluded` and `Case3IsOccluding`.
     * Checks if `potential_occluder` (closer, later point) could be occluding `potential_occluded` (farther, earlier point)
     * based on depth difference, angular proximity, time difference, and specific thresholds defined by `check_type`.
     *
     * @param potential_occluder The point that might be closer and occluding (usually the point from the current frame).
     * @param potential_occluded The point that might be farther and occluded (usually the point from a previous frame).
     * @param params Configuration parameters containing thresholds.
     * @param check_type Determines which set of thresholds (Case 2 or Case 3) to use.
     * @return True if the occlusion relationship holds, false otherwise.
     */
    bool checkOcclusionRelationship(
        const point_soph& potential_occluder,
        const point_soph& potential_occluded,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

    /**
     * @brief Searches a neighborhood in a depth map for a point that has an occlusion relationship
     *        with the given point_to_update.
     *
     * Replaces `Case2SearchPointOccludingP` and `Case3SearchPointOccludedbyP`.
     * Iterates through neighbor cells in map_info around point_to_update. For each neighbor point found,
     * it calls checkOcclusionRelationship and checkDepthConsistency. If both pass, it updates
     * point_to_update's occu_index (for Case 2) or is_occu_index (for Case 3) and returns true.
     *
     * @param point_to_update The point to check against the map (passed by non-const reference to allow updates).
     * @param map_info The depth map containing potential neighbors.
     * @param params Configuration parameters containing thresholds and search window sizes.
     * @param check_type Determines search window size, which relationship to check (Case 2 or 3),
     *                   which depth consistency check to use, and which index on point_to_update to set.
     * @return True if a valid occluding/occluded neighbor is found, false otherwise.
     */
    bool findOcclusionRelationshipInMap(
        point_soph& point_to_update, // Non-const ref to allow updating indices
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

} // namespace ConsistencyChecks

#endif // CONSISTENCY_CHECKS_H