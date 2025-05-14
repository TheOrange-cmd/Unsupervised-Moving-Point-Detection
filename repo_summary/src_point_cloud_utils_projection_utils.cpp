/**
 * @file projection_utils.cpp
 * @brief Implements utility functions related to point projection, particularly spherical projection with caching.
 */
#include "point_cloud_utils/projection_utils.h"
#include "point_cloud_utils/logging_utils.h"  
#include "point_cloud_utils/logging_context.h"
#include "common/dyn_obj_datatypes.h" // For point_soph, M3D, V3D, HASH_PRIM
#include "config/config_loader.h"     // For DynObjFilterParams
#include <cmath>                      // For std::fabs
#include <spdlog/spdlog.h>            // For logging

namespace PointCloudUtils {

// Original Signature
void SphericalProjection(point_soph &p, int depth_index, const M3D &rot,
    const V3D &transl, const DynObjFilterParams& params, point_soph &p_spherical)
{
    auto logger = spdlog::get("Utils_Projection"); // Using new logger name
    uint64_t current_seq_id = g_current_logging_seq_id; // Get from context

    if (logger->should_log(spdlog::level::trace)) {
        logger->trace("[SphericalProjection] ENTER for Point OriginalIdx={}, DepthIndex={}, FrameID={}. Input p.glob=({:.3f},{:.3f},{:.3f}). Rot, Transl provided.",
                      p.original_index, depth_index, current_seq_id, p.glob.x(), p.glob.y(), p.glob.z());
    }

    const int cache_idx = depth_index % HASH_PRIM;

    if(std::fabs(p.last_vecs.at(cache_idx)[2]) > CACHE_VALID_THRESHOLD)
    {
        p_spherical.vec = p.last_vecs.at(cache_idx);
        p_spherical.hor_ind = p.last_positions.at(cache_idx)[0];
        p_spherical.ver_ind = p.last_positions.at(cache_idx)[1];
        p_spherical.position = p.last_positions.at(cache_idx)[2];
        
        if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
            logger->trace("[SphericalProjection] Point OriginalIdx={}, FrameID={}: Cache HIT. SphericalVec=({:.3f},{:.3f},{:.3f}), H_ind={}, V_ind={}, Pos_1D={}",
                          p.original_index, current_seq_id, p_spherical.vec.x(), p_spherical.vec.y(), p_spherical.vec.z(),
                          p_spherical.hor_ind, p_spherical.ver_ind, p_spherical.position);
        }
    }
    else
    {
        if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
            logger->trace("[SphericalProjection] Point OriginalIdx={}, FrameID={}: Cache MISS.", p.original_index, current_seq_id);
        }

        V3D p_proj(rot * (p.glob - transl));
        if (logger->should_log(spdlog::level::trace)) {
            logger->trace("  [SphericalProjection] Point OriginalIdx={}, FrameID={}: Transformed p_proj=({:.3f},{:.3f},{:.3f})",
                          p.original_index, current_seq_id, p_proj.x(), p_proj.y(), p_proj.z());
        }

        p_spherical.GetVec(p_proj, params.hor_resolution_max, params.ver_resolution_max);
        if (logger->should_log(spdlog::level::trace)) {
            logger->trace("  [SphericalProjection] Point OriginalIdx={}, FrameID={}: After GetVec: SphericalVec=({:.3f},{:.3f},{:.3f}), H_ind={}, V_ind={}, Pos_1D={}",
                          p.original_index, current_seq_id, p_spherical.vec.x(), p_spherical.vec.y(), p_spherical.vec.z(),
                          p_spherical.hor_ind, p_spherical.ver_ind, p_spherical.position);
        }

        p.last_vecs.at(cache_idx) = p_spherical.vec;
        p.last_positions.at(cache_idx)[0] = p_spherical.hor_ind;
        p.last_positions.at(cache_idx)[1] = p_spherical.ver_ind;
        p.last_positions.at(cache_idx)[2] = p_spherical.position;
    }
}

} // namespace PointCloudUtils