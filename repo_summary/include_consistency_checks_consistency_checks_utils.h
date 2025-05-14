#ifndef CONSISTENCY_CHECKS_UTILS_H
#define CONSISTENCY_CHECKS_UTILS_H

#include "common/dyn_obj_datatypes.h" // For point_soph, ConsistencyCheckType
#include "config/config_loader.h"     // For DynObjFilterParams
#include <string> // For std::string

// Forward declare logger if needed for utils-specific logging, or include spdlog header
// #include <spdlog/spdlog.h>
#include "consistency_checks/consistency_checks.h"

namespace ConsistencyChecks {

    /**
     * @brief Calculates the dynamic depth threshold for interpolation-based consistency checks.
     *
     * This threshold determines how much difference between a point's measured depth and an
     * interpolated depth from a historical map is considered acceptable or significant,
     * depending on the consistency check type (CASE1, CASE2, CASE3).
     *
     * The calculation considers:
     * - A base threshold specific to the check type (interp_thr1/2/3).
     * - Scaling based on the temporal difference between maps (map_index_diff).
     * - For CASE1: Adjustments based on the point's depth and distortion status.
     *
     * @param p The point_soph object being checked (its depth and distortion status are used for CASE1).
     * @param check_type The context (CASE1, CASE2, CASE3) influencing parameters and logic.
     * @param map_index_diff The difference in indices between the current map and the historical map being checked against.
     * @param params Configuration parameters containing thresholds (interp_thr*, interp_start_depth1, etc.).
     * @return float The calculated interpolation threshold.
     * @throws std::invalid_argument if check_type is invalid.
     */
    float calculateInterpolationThreshold(
        const point_soph& p,
        ConsistencyCheckType check_type,
        int map_index_diff,
        const DynObjFilterParams& params
    );

    /**
     * @brief Calculates the dynamic depth threshold used for occlusion checks (CASE2/CASE3).
     *
     * This threshold determines the minimum depth separation required between a potential occluder
     * and a potential occluded point. It typically depends on the depth of the *occluder* point.
     *
     * The calculation considers:
     * - A base threshold specific to the check type (depth_cons_depth_thr2/3).
     * - A depth-dependent scaling factor (k_depth2/3) applied to the occluder's depth.
     * - The maximum of the base threshold and the scaled depth-dependent term.
     *
     * @param occluder_point The point_soph object assumed to be the occluder (closer point).
     * @param check_type The context (CASE2 or CASE3) influencing parameters.
     * @param params Configuration parameters containing thresholds (depth_cons_depth_thr*, k_depth*).
     * @return float The calculated occlusion depth threshold.
     * @throws std::invalid_argument if check_type is invalid or CASE1.
     */
    float calculateOcclusionDepthThreshold(
        const point_soph& occluder_point,
        ConsistencyCheckType check_type,
        const DynObjFilterParams& params
    );

    // Helper to get case string
    const char* getCaseStringUtil(ConsistencyCheckType check_type);


} // namespace ConsistencyChecks

#endif // CONSISTENCY_CHECKS_UTILS_H