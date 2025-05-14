// include/consistency_checks/consistency_checks.h

#ifndef CONSISTENCY_CHECKS_H
#define CONSISTENCY_CHECKS_H

#include "common/dyn_obj_datatypes.h" // Includes point_soph, DepthMap
#include "config/config_loader.h"     // Includes DynObjFilterParams
#include <functional> // Required for std::function
#include <stdexcept>  // For std::invalid_argument

// Forward declarations for utility functions used internally (optional, but can reduce header dependencies)
namespace PointCloudUtils {
    struct InterpolationResult;
    enum class InterpolationNeighborType;
    enum class InterpolationStatus;
    // No need to forward declare grid_search utils if consistency_checks.h doesn't directly call them.
}

// Define a namespace for consistency check functions
namespace ConsistencyChecks {

    /**
     * @brief Specifies the context or case for consistency checks,
     *        influencing which parameters, thresholds, and logic are applied.
     */
    enum class ConsistencyCheckType {
        CASE1_FALSE_REJECTION, // Check if a point *inconsistent* with static map (potential APPEARING)
        CASE2_OCCLUDER_SEARCH, // Check if a point is *consistent* with being an OCCLUDER relative to map
        CASE3_OCCLUDED_SEARCH  // Check if a point is *consistent* with being OCCLUDED relative to map
    };

    /**
     * @brief Checks if a point 'p' is consistent with a given historical DepthMap based on interpolation.
     *
     * This function evaluates if a point `p` (from a newer scan) aligns geometrically with the surface
     * represented in an older `map_info`. It uses interpolation of neighboring points in `map_info`.
     * The interpretation of "consistent" depends heavily on `check_type`:
     *
     * - **CASE1_FALSE_REJECTION:**
     *   - **Goal:** Identify points that *contradict* the static map (potential dynamic points).
     *   - **Logic:** Interpolates using only STATIC neighbors in `map_info`.
     *   - **Returns `true` (Consistent):** If `p` is *behind* or *close to* the interpolated static surface.
     *     This suggests `p` might be part of the static background or a dynamic object behind the old surface.
     *   - **Returns `false` (Inconsistent):** If `p` is significantly *in front* of the interpolated static surface,
     *     or if interpolation fails (no neighbors, ambiguity). This suggests `p` might be an APPEARING dynamic point.
     *
     * - **CASE2_OCCLUDER_SEARCH:**
     *   - **Goal:** Check if `p` could plausibly be an OCCLUDER relative to the surface in `map_info`.
     *   - **Logic:** Interpolates using ALL valid neighbors in `map_info`.
     *   - **Returns `true` (Consistent):** If `p` is significantly *in front* of the interpolated surface in `map_info`.
     *     This supports the hypothesis that `p` is part of an object that was occluding the background seen in `map_info`.
     *   - **Returns `false` (Inconsistent):** If `p` is *behind* or *close to* the interpolated surface, or if interpolation fails.
     *
     * - **CASE3_OCCLUDED_SEARCH:**
     *   - **Goal:** Check if `p` could plausibly be OCCLUDED relative to the surface in `map_info`.
     *   - **Logic:** Interpolates using ALL valid neighbors in `map_info`.
     *   - **Returns `true` (Consistent):** If `p` is significantly *behind* the interpolated surface in `map_info`.
     *     This supports the hypothesis that `p` was occluded by something closer, whose surface is represented in `map_info`.
     *   - **Returns `false` (Inconsistent):** If `p` is *in front* or *close to* the interpolated surface, or if interpolation fails.
     *
     * @param p The point_soph object to check. Assumes its projection relative to `map_info` is cached or calculated.
     * @param map_info The historical DepthMap.
     * @param params Configuration parameters.
     * @param check_type Enum indicating the context (CASE1, CASE2, CASE3).
     * @param map_index_diff Difference between current map index and `map_info.map_index` (for threshold scaling).
     * @return bool True if consistent according to the rules of `check_type`, False otherwise.
     */
    bool checkMapConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        int map_index_diff = 1
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
     * Evaluates `p` against STATIC neighbors in `map_info` within an angular/temporal window.
     * Used primarily in CASE2 and CASE3 to validate potential occluders/occluded points found during
     * the `findOcclusionRelationshipInMap` search.
     *
     * - **Goal:** Ensure a point identified in an occlusion relationship isn't itself part of a noisy/inconsistent area.
     * - **Logic:** Applies Rule 1 (average difference of close neighbors) and Rule 2 (no mix of closer/farther distant neighbors).
     * - **Returns `true` (Consistent):** If `p` passes both Rule 1 and Rule 2 relative to static neighbors in `map_info`.
     * - **Returns `false` (Inconsistent):** If `p` fails either Rule 1 or Rule 2, or if no suitable static neighbors are found.
     *
     * @param p The point_soph object to check.
     * @param map_info The historical DepthMap.
     * @param params Configuration parameters.
     * @param check_type Enum indicating context (CASE2 or CASE3). CASE1 is invalid.
     * @return bool True if depth-consistent, False otherwise.
     * @throws std::invalid_argument if check_type is CASE1_FALSE_REJECTION.
     */
    bool checkDepthConsistency(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

    /**
     * @brief Checks if the change between two velocity measurements implies a plausible acceleration.
     *
     * Used in CASE2 and CASE3 to check the kinematic plausibility of velocity changes derived from
     * point correspondences over time.
     *
     * - **Goal:** Filter out velocity changes that imply unrealistically high accelerations.
     * - **Logic:** Compares `abs(velocity1 - velocity2)` against `time_delta * acceleration_threshold`.
     * - **Returns `true` (Consistent):** If the implied acceleration is below the threshold.
     * - **Returns `false` (Inconsistent):** If the implied acceleration exceeds the threshold.
     *
     * @param velocity1 First velocity measurement.
     * @param velocity2 Second velocity measurement.
     * @param time_delta_between_velocity_centers Time difference between measurements.
     * @param params Configuration parameters (contains acc_thr2/3).
     * @param check_type Enum indicating context (CASE2 or CASE3). CASE1 is invalid.
     * @return bool True if acceleration is plausible, False otherwise.
     * @throws std::invalid_argument if check_type is CASE1_FALSE_REJECTION.
     */
    bool checkAccelerationLimit(
        float velocity1,
        float velocity2,
        double time_delta_between_velocity_centers,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

    /**
     * @brief Checks if two points satisfy geometric/temporal conditions for an occlusion relationship.
     *
     * Core check used by `findOcclusionRelationshipInMap`. Evaluates if `potential_occluder` (closer, later)
     * and `potential_occluded` (farther, earlier) meet angular proximity and dynamic depth separation criteria
     * based on the specified `check_type` (CASE2 or CASE3) and parameters. Also handles self-occlusion checks.
     *
     * - **Goal:** Determine if a specific pair of points represents a geometrically plausible occlusion.
     * - **Logic:** Checks time order, self-occlusion, angular difference, and depth difference against dynamic thresholds.
     * - **Returns `true` (Consistent Relationship):** If all conditions for occlusion are met for the given `check_type`.
     * - **Returns `false` (Inconsistent Relationship):** If any condition fails.
     *
     * @param potential_occluder The point assumed closer and later.
     * @param potential_occluded The point assumed farther and earlier.
     * @param params Configuration parameters.
     * @param check_type Enum indicating context (CASE2 or CASE3). CASE1 is invalid.
     * @return bool True if the pair forms a valid occlusion relationship, False otherwise.
     * @throws std::invalid_argument if `check_type` is CASE1_FALSE_REJECTION.
     */
    bool checkOcclusionRelationship(
        const point_soph& potential_occluder,
        const point_soph& potential_occluded,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type);

    /**
     * @brief Searches a map neighborhood for the first point forming a valid & consistent occlusion relationship.
     *
     * Orchestrates the search within `map_info` for a neighbor (`PN`) of `point_to_update` (`P`) that satisfies
     * both `checkOcclusionRelationship(P, PN or PN, P)` and `depth_checker(PN)`. Updates `P`'s `occu_index`
     * or `is_occu_index` upon finding the first match. Uses `forEachNeighborCell` internally.
     *
     * - **Goal:** Find evidence in the map supporting `P` being an occluder (Case 2) or occluded (Case 3).
     * - **Logic:** Iterates neighbors using `forEachNeighborCell`, applies Case 3 optimization, calls
     *   `checkOcclusionRelationship` and `depth_checker`. Stops on first full match.
     * - **Returns `true` (Match Found):** If a suitable neighbor `PN` was found and `P` was updated.
     * - **Returns `false` (No Match):** If the entire neighborhood was searched without finding a match.
     *
     * @param point_to_update The point `P` being investigated (updated on success).
     * @param map_info The historical DepthMap to search within.
     * @param params Configuration parameters.
     * @param check_type Enum indicating context (CASE2 or CASE3). CASE1 is invalid.
     * @param depth_checker Function to check depth consistency of the neighbor `PN` (defaults to `checkDepthConsistency`).
     * @return bool True if a match was found, False otherwise.
     * @throws std::invalid_argument if `check_type` is CASE1_FALSE_REJECTION.
     */
    bool findOcclusionRelationshipInMap(
        point_soph& point_to_update,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        ConsistencyCheckType check_type,
        DepthConsistencyChecker depth_checker = ConsistencyChecks::checkDepthConsistency
    );

} // namespace ConsistencyChecks

#endif // CONSISTENCY_CHECKS_H