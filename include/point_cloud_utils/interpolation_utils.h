/**
 * @file interpolation_utils.h
 * @brief Defines types and functions for depth interpolation using barycentric coordinates.
*/

#ifndef INTERPOLATION_UTILS_H
#define INTERPOLATION_UTILS_H

#include "common/dyn_obj_datatypes.h" // Includes point_soph, DepthMap, Eigen types etc.
#include "config/config_loader.h"     // Include to get DynObjFilterParams definition
#include <vector>
#include <string>

 
namespace PointCloudUtils {
 
    // --- Interpolation Related Types ---

    /** @brief Specifies which types of neighboring points to consider for interpolation. */
    enum class InterpolationNeighborType {
        STATIC_ONLY, /**< Consider only neighbors marked as STATIC. */
        ALL_VALID    /**< Consider all neighbors that pass time/angular filters, regardless of label. */
    };

    /** @brief Indicates the result status of an interpolation attempt. */
    enum class InterpolationStatus {
        SUCCESS,              /**< Interpolation succeeded. */
        NOT_ENOUGH_NEIGHBORS, /**< Fewer than 3 valid neighbors found. */
        NO_VALID_TRIANGLE,    /**< Enough neighbors found, but none formed a triangle containing the target projection. */
        DEGENERACY            /**< A potential triangle was degenerate (collinear points). */ // Added possibility
    };

    /** @brief Holds the result of a depth interpolation attempt. */
    struct InterpolationResult {
        InterpolationStatus status = InterpolationStatus::NOT_ENOUGH_NEIGHBORS; /**< Status code indicating outcome. */
        float depth = 0.0f; /**< Interpolated depth value (only valid if status is SUCCESS). */
    };

    /** @brief Converts an InterpolationStatus enum to a human-readable string. */
    std::string interpolationStatusToString(InterpolationStatus status);

    /** @brief Converts an InterpolationNeighborType enum to a human-readable string. */
    std::string interpolationNeighborTypeToString(InterpolationNeighborType type);

    // Constant used for barycentric calculations
    constexpr float BARYCENTRIC_EPSILON = 1e-5f;


    // --- Core Logic Functions ---

    /**
     * @brief Finds suitable neighboring points (as V3F vectors) for depth interpolation.
     *
     * Searches a neighborhood defined by `params.interp_hor_num` and `params.interp_ver_num`
     * around point `p` in the `map_info` grid, handling horizontal wrap-around.
     * Filters points within neighbor cells based on time difference (`params.frame_dur`),
     * angular distance (`params.interp_hor_thr`, `params.interp_ver_thr`), and the specified `type`
     * (either `STATIC_ONLY` or `ALL_VALID`).
     *
     * @param p The point for which to find neighbors (center of search, provides time, projection `vec`).
     * @param map_info The depth map data containing points and grid structure.
     * @param params Filter parameters (interp range, thresholds, frame_dur, etc.).
     * @param type Whether to collect only static or all valid neighbors (`InterpolationNeighborType`).
     * @return std::vector<V3F> Vector of 3D spherical coordinates (azimuth, elevation, depth) of valid neighbors.
     *                          Returns an empty vector if `map_info` is invalid.
     */
    std::vector<V3F> findInterpolationNeighbors(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type
    );

    /**
     * @brief Attempts to find a triangle of neighbors enclosing the target's projection and interpolates depth using barycentric coordinates.
     *
     * Iterates through all combinations of 3 neighbors from the input vector. For each triplet,
     * it checks if the 2D projection of the `target_point_projection` lies within the triangle formed
     * by the neighbors' 2D projections. Handles azimuth wrap-around during the check.
     * If a valid enclosing triangle is found, it calculates the interpolated depth based on the
     * barycentric coordinates and the neighbors' depths.
     *
     * @param target_point_projection 2D projection (azimuth, elevation) of the point to interpolate for.
     * @param neighbors Vector of 3D spherical coordinates (azimuth, elevation, depth) of potential neighbors found previously.
     * @param params Filter parameters (currently unused in this specific function but kept for consistency).
     * @return InterpolationResult indicating `SUCCESS` and the interpolated depth if found,
     *         or `NOT_ENOUGH_NEIGHBORS` / `NO_VALID_TRIANGLE` / `DEGENERACY` on failure.
     */
    InterpolationResult computeBarycentricDepth(
        const V2F& target_point_projection,
        const std::vector<V3F>& neighbors,
        const DynObjFilterParams& params // Keep for potential future use
    );


    // --- Main Public Function ---

    /**
     * @brief Interpolates the depth for a point using barycentric interpolation on neighbors.
     *
     * This function orchestrates the interpolation process:
     * 1. Calls `findInterpolationNeighbors` to find suitable neighbors based on the specified `type`.
     * 2. Calls `computeBarycentricDepth` to perform the interpolation using the found neighbors.
     *
     * @param p The point to interpolate depth for (must have valid `time` and `vec`).
     * @param map_info The depth map data containing potential neighbors.
     * @param params Filter parameters used for neighbor finding and potentially interpolation.
     * @param type Whether to use only static or all valid neighbors (`InterpolationNeighborType`).
     * @return InterpolationResult indicating success/failure and the interpolated depth.
     */
    InterpolationResult interpolateDepth(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type
    );
 
} // namespace PointCloudUtils

// --- BEGIN {fmt} Formatter Specialization for InterpolationNeighborType ---
// Needs to be after the enum definition and after spdlog/fmt headers are included (likely via dyn_obj_datatypes.h)
template <>
struct fmt::formatter<PointCloudUtils::InterpolationNeighborType> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin();
        if (it != ctx.end() && *it != '}')
            throw format_error("invalid format");
        return it;
    }

    template <typename FormatContext>
    auto format(PointCloudUtils::InterpolationNeighborType type, FormatContext& ctx) const -> decltype(ctx.out()) {
        std::string_view name = "UNKNOWN_TYPE";
        switch (type) {
            case PointCloudUtils::InterpolationNeighborType::ALL_VALID:   name = "ALL_VALID"; break;
            case PointCloudUtils::InterpolationNeighborType::STATIC_ONLY: name = "STATIC_ONLY"; break;
        }
        return fmt::format_to(ctx.out(), "{}", name);
    }
};
// --- END {fmt} Formatter Specialization ---

#endif // INTERPOLATION_UTILS_H