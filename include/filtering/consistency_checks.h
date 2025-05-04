// file: filtering/consistency_checks.h

#ifndef CONSISTENCY_CHECKS_H
#define CONSISTENCY_CHECKS_H

#include "filtering/dyn_obj_datatypes.h" // Includes point_soph, DepthMap
#include "config/config_loader.h"     // Includes DynObjFilterParams
#include "point_cloud_utils/point_cloud_utils.h" // For Interpolation types
#include <functional> // Required for std::function

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

    // Type alias for the depth consistency checking function signature
    using DepthConsistencyChecker = std::function<bool(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type
    )>;

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
     * @brief Checks if two points satisfy the geometric, temporal, and configuration-specific conditions for an occlusion relationship.
     * @details This function serves as the core logic for determining if a potential occlusion scenario exists between
     *          exactly two points, one assumed to be closer and later (`potential_occluder`) and the other farther
     *          and earlier (`potential_occluded`). It evaluates several criteria:
     *          1.  **Validity Checks:** Ensures the `potential_occluded` point is not marked as `INVALID` and checks for
     *              specific distortion conditions based on `params.dataset`.
     *          2.  **Self-Occlusion:** Checks if either point's local coordinates fall within the ego-vehicle's
     *              defined bounding box (`params.self_*`), preventing self-detection from causing occlusion flags.
     *          3.  **Temporal Order:** Verifies that `potential_occluder.time` is strictly greater than `potential_occluded.time`.
     *          4.  **Angular Proximity:** Checks if the absolute difference in azimuth and elevation between the two points
     *              is within the thresholds specified by `params.occ_hor_thr*` and `params.occ_ver_thr*` corresponding
     *              to the given `check_type`.
     *          5.  **Depth Relationship:** Verifies that the `potential_occluded` point is farther than the `potential_occluder`
     *              point by at least a dynamically calculated threshold. This threshold depends on the `check_type` and considers
     *              base offsets (`params.occ_depth_thr2`, `params.map_cons_depth_thr3`), velocity estimates (`params.v_min_thr*`),
     *              adaptive components based on occluder depth (`params.k_depth_max_thr*`, `params.d_depth_max_thr*`, `params.cutoff_value`),
     *              and potentially distortion (`params.enlarge_distort`).
     *          This function replaces the logic previously found in `Case2IsOccluded` and `Case3IsOccluding`.
     *
     * @param potential_occluder The point assumed to be physically closer to the sensor and occurring later in time.
     *                           - For `CASE2_OCCLUDER_SEARCH`, this is typically the point being updated (`P`).
     *                           - For `CASE3_OCCLUDED_SEARCH`, this is typically a neighbor point from the map (`PN`).
     * @param potential_occluded The point assumed to be physically farther from the sensor and occurring earlier in time.
     *                           - For `CASE2_OCCLUDER_SEARCH`, this is typically a neighbor point from the map (`PN`).
     *                           - For `CASE3_OCCLUDED_SEARCH`, this is typically the point being updated (`P`).
     * @param params Configuration parameters object containing various thresholds:
     *               - Occlusion angular thresholds (`occ_hor_thr2/3`, `occ_ver_thr2/3`).
     *               - Occlusion depth thresholds (base offsets `occ_depth_thr2`, `map_cons_depth_thr3`).
     *               - Dynamic depth threshold parameters (`k_depth_max_thr2/3`, `d_depth_max_thr2/3`, `cutoff_value`).
     *               - Minimum velocity thresholds (`v_min_thr2/3`).
     *               - Self-occlusion box dimensions (`self_x_b`, `self_x_f`, `self_y_l`, `self_y_r`).
     *               - Distortion parameters (`dataset`, `enlarge_distort`).
     * @param check_type Crucially determines:
     *                   1. Which set of parameters (Case 2 or Case 3 specific versions) to use for angular and depth checks.
     *                   2. How to interpret the roles of `potential_occluder` and `potential_occluded` (as described above).
     *                   Must be either `ConsistencyCheckType::CASE2_OCCLUDER_SEARCH` or `ConsistencyCheckType::CASE3_OCCLUDED_SEARCH`.
     * @return `true` if all applicable validity, self-occlusion, temporal, angular, and depth conditions are met
     *         according to the specified `check_type`. Returns `false` if any condition fails.
     * @throws std::invalid_argument if `check_type` is not `CASE2_OCCLUDER_SEARCH` or `CASE3_OCCLUDED_SEARCH`.
     */
    bool checkOcclusionRelationship(
        const point_soph& potential_occluder,
        const point_soph& potential_occluded,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

    /**
     * @brief Searches a neighborhood within a depth map for the *first* point that forms a valid occlusion relationship
     *        with a given point, also passing a depth consistency check.
     * @details This function orchestrates the search for occlusion evidence related to `point_to_update` (`P`).
     *          It iterates through neighbor cells in `map_info` relative to `P`'s grid position. The size of the
     *          neighborhood searched is determined by `params.occ_hor_num*` and `params.occ_ver_num*` based on `check_type`.
     *          For each neighbor point (`PN`) found within the search window:
     *          1.  **(Case 3 Optimization):** If `check_type` is `CASE3_OCCLUDED_SEARCH`, a quick check compares `P`'s depth
     *              to the minimum depth recorded for `PN`'s cell (`map_info.min_depth_all`). If `P` is already farther
     *              than the closest possible point in `PN`'s cell, that cell can be skipped.
     *          2.  **Occlusion Check:** Calls `checkOcclusionRelationship`, passing `P` and `PN` in the correct order
     *              (`P` as occluder for Case 2, `PN` as occluder for Case 3) according to `check_type`.
     *          3.  **Depth Consistency Check:** If `checkOcclusionRelationship` returns `true`, it then calls the
     *              provided `depth_checker` function (which defaults to the real `ConsistencyChecks::checkDepthConsistency`)
     *              on the neighbor point `PN` to ensure `PN` itself is considered depth-consistent with its *own* local neighborhood.
     *          If *both* checks pass for a neighbor `PN`, the function immediately updates the appropriate index array
     *          on `point_to_update` (`occu_index` for Case 2, `is_occu_index` for Case 3) with the map index, the
     *          position (`PN.position`), and the index within the cell vector (`j`) of the found neighbor `PN`.
     *          It then returns `true`.
     *          If the entire neighborhood is searched without finding a suitable neighbor that passes both checks,
     *          the function returns `false`, and `point_to_update` remains unchanged.
     *          This function replaces the logic previously found in `Case2SearchPointOccludingP` and `Case3SearchPointOccludedbyP`.
     *
     * @param point_to_update The point (`P`) whose occlusion relationship with the map is being investigated.
     *                        Passed by non-const reference because its `occu_index` or `is_occu_index` member
     *                        will be updated if a suitable neighbor is found.
     * @param map_info The depth map data structure containing potential neighbor points (`PN`) organized in a grid.
     *                 It should contain `depth_map` (the grid of point vectors) and potentially `min_depth_all` for optimization.
     * @param params Configuration parameters object containing:
     *               - Occlusion search window sizes (`occ_hor_num2/3`, `occ_ver_num2/3`).
     *               - Thresholds used by `checkOcclusionRelationship`.
     *               - Parameters used by the `depth_checker` function.
     * @param check_type Determines:
     *                   1. The search window size (`occ_*_num*`).
     *                   2. Which relationship `checkOcclusionRelationship` should evaluate (P occluding PN or PN occluding P).
     *                   3. Which parameters `checkOcclusionRelationship` and `depth_checker` should use (Case 2 vs Case 3).
     *                   4. Which index on `point_to_update` to set (`occu_index` vs `is_occu_index`).
     *                   Must be either `ConsistencyCheckType::CASE2_OCCLUDER_SEARCH` or `ConsistencyCheckType::CASE3_OCCLUDED_SEARCH`.
     * @param depth_checker A function object (e.g., `std::function`, lambda, function pointer) that performs the
     *                      depth consistency check on a potential neighbor point (`PN`). It defaults to the actual
     *                      `ConsistencyChecks::checkDepthConsistency` implementation. This allows injecting mock
     *                      checkers for unit testing.
     * @return `true` if a neighbor `PN` satisfying both `checkOcclusionRelationship` and `depth_checker` was found
     *         within the search window, and `point_to_update`'s corresponding index was updated.
     *         `false` if no such neighbor was found after searching the entire relevant neighborhood.
     * @throws std::invalid_argument if `check_type` is not `CASE2_OCCLUDER_SEARCH` or `CASE3_OCCLUDED_SEARCH`.
     */
    bool findOcclusionRelationshipInMap(
        point_soph& point_to_update,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        DepthConsistencyChecker depth_checker = ConsistencyChecks::checkDepthConsistency // Default!
    );

} // namespace ConsistencyChecks

#endif // CONSISTENCY_CHECKS_H