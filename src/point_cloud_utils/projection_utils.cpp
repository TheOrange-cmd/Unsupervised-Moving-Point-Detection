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
// void SphericalProjection(point_soph &p, int depth_index, const M3D &rot,
//     const V3D &transl, const DynObjFilterParams& params, point_soph &p_spherical)
// {
//     auto logger = spdlog::get("Utils_Projection"); // Using new logger name
//     uint64_t current_seq_id = g_current_logging_seq_id; // Get from context

//     if (logger->should_log(spdlog::level::trace)) {
//         logger->trace("[SphericalProjection] ENTER for Point OriginalIdx={}, DepthIndex={}, FrameID={}. Input p.glob=({:.3f},{:.3f},{:.3f}). Rot, Transl provided.",
//                       p.original_index, depth_index, current_seq_id, p.glob.x(), p.glob.y(), p.glob.z());
//     }

//     const int cache_idx = depth_index % HASH_PRIM;

//     if(std::fabs(p.last_vecs.at(cache_idx)[2]) > CACHE_VALID_THRESHOLD)
//     {
//         p_spherical.vec = p.last_vecs.at(cache_idx);
//         p_spherical.hor_ind = p.last_positions.at(cache_idx)[0];
//         p_spherical.ver_ind = p.last_positions.at(cache_idx)[1];
//         p_spherical.position = p.last_positions.at(cache_idx)[2];
        
//         if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
//             logger->trace("[SphericalProjection] Point OriginalIdx={}, FrameID={}: Cache HIT. SphericalVec=({:.3f},{:.3f},{:.3f}), H_ind={}, V_ind={}, Pos_1D={}",
//                           p.original_index, current_seq_id, p_spherical.vec.x(), p_spherical.vec.y(), p_spherical.vec.z(),
//                           p_spherical.hor_ind, p_spherical.ver_ind, p_spherical.position);
//         }
//     }
//     else
//     {
//         if (logger->should_log(spdlog::level::trace)) { // Changed from should_log_point_details
//             logger->trace("[SphericalProjection] Point OriginalIdx={}, FrameID={}: Cache MISS.", p.original_index, current_seq_id);
//         }

//         V3D p_proj(rot * (p.glob - transl));
//         if (logger->should_log(spdlog::level::trace)) {
//             logger->trace("  [SphericalProjection] Point OriginalIdx={}, FrameID={}: Transformed p_proj=({:.3f},{:.3f},{:.3f})",
//                           p.original_index, current_seq_id, p_proj.x(), p_proj.y(), p_proj.z());
//         }

//         p_spherical.GetVec(p_proj, params.hor_resolution_max, params.ver_resolution_max);
//         if (logger->should_log(spdlog::level::trace)) {
//             logger->trace("  [SphericalProjection] Point OriginalIdx={}, FrameID={}: After GetVec: SphericalVec=({:.3f},{:.3f},{:.3f}), H_ind={}, V_ind={}, Pos_1D={}",
//                           p.original_index, current_seq_id, p_spherical.vec.x(), p_spherical.vec.y(), p_spherical.vec.z(),
//                           p_spherical.hor_ind, p_spherical.ver_ind, p_spherical.position);
//         }

//         p.last_vecs.at(cache_idx) = p_spherical.vec;
//         p.last_positions.at(cache_idx)[0] = p_spherical.hor_ind;
//         p.last_positions.at(cache_idx)[1] = p_spherical.ver_ind;
//         p.last_positions.at(cache_idx)[2] = p_spherical.position;
//     }
// }

void SphericalProjection(point_soph &p_in, // Input point (contains p_in.glob)
                         int depth_index,  // Used for caching, often p_in.original_index or map_index
                         const M3D &rot_w2s, // Rotation World-to-Sensor (R_w2s)
                         const V3D &transl_w2s_prime, // Translation World-to-Sensor (T_w2s_prime = -R_w2s * T_sensor_origin_w)
                         const DynObjFilterParams& params,
                         point_soph &p_spherical_out) // Output structure
{
    auto logger = spdlog::get("Utils_Projection");
    uint64_t current_seq_id = PointCloudUtils::g_current_logging_seq_id; // Get from context for logging

    // --- Log Inputs ---
    // Use should_log_point_details if you want to target specific points/frames for these detailed logs
    // Otherwise, use logger->should_log(spdlog::level::trace) for general trace verbosity
    bool log_details_for_this_call = false;
    if (logger) { // Ensure logger exists
        // Example: Log if trace is enabled OR if it's a specifically targeted debug point
        log_details_for_this_call = logger->should_log(spdlog::level::trace) ||
                                    (PointCloudUtils::g_current_logging_point_ptr == &p_in && // Ensure context matches this point
                                     PointCloudUtils::should_log_point_details(logger)); // Your existing targeted logic
    }
    // If this SphericalProjection is called from a utility binding (like mdet.spherical_projection)
    // g_current_logging_point_ptr might not be &p_in.
    // For the mdet.spherical_projection call, we rely on the binding itself setting the logger to trace.
    // For calls within the filter (like from addPointsToMap), LoggingContextSetter should handle it.
    // For simplicity in this direct utility debug, let's assume trace level on "Utils_Projection" is enough.
    log_details_for_this_call = logger && logger->should_log(spdlog::level::trace);


    if (log_details_for_this_call) {
        logger->trace("[SphericalProjection] FRAME_ID={}, Point OriginalIdx={}, DepthIndex(cache_key)={}",
                      current_seq_id, p_in.original_index, depth_index);
        logger->trace("  Input p_in.glob (World Coords): ({:.4f}, {:.4f}, {:.4f})",
                      p_in.glob.x(), p_in.glob.y(), p_in.glob.z());
        logger->trace("  Input rot_w2s (R_w2s) (row0): ({:.4f}, {:.4f}, {:.4f})",
                      rot_w2s(0,0), rot_w2s(0,1), rot_w2s(0,2));
        logger->trace("  Input transl_w2s_prime (T_w2s_prime): ({:.4f}, {:.4f}, {:.4f})",
                      transl_w2s_prime.x(), transl_w2s_prime.y(), transl_w2s_prime.z());
    }

    const int cache_idx = depth_index % HASH_PRIM;
    bool cache_is_valid = false;
    if (p_in.last_vecs.size() > static_cast<size_t>(cache_idx)) {
        // Assuming p_in.last_vecs[cache_idx].z() stores depth and is non-zero if valid.
        // CACHE_VALID_THRESHOLD could be a small positive number like 1e-3f.
        cache_is_valid = (std::fabs(p_in.last_vecs.at(cache_idx).z()) > CACHE_VALID_THRESHOLD);
    }

    if(cache_is_valid)
    {
        p_spherical_out.vec = p_in.last_vecs.at(cache_idx);
        p_spherical_out.hor_ind = p_in.last_positions.at(cache_idx)[0];
        p_spherical_out.ver_ind = p_in.last_positions.at(cache_idx)[1];
        p_spherical_out.position = p_in.last_positions.at(cache_idx)[2];
        
        if (log_details_for_this_call) {
            logger->trace("  Cache HIT. Output SphericalVec=({:.3f},{:.3f},{:.3f}), H_ind={}, V_ind={}, Pos_1D={}",
                          p_spherical_out.vec.x(), p_spherical_out.vec.y(), p_spherical_out.vec.z(),
                          p_spherical_out.hor_ind, p_spherical_out.ver_ind, p_spherical_out.position);
        }
    }
    else
    {
        if (log_details_for_this_call) {
            logger->trace("  Cache MISS.");
        }

        // --- Perform Transformation to Sensor Frame ---
        // This is the CORRECTED transformation based on our understanding of inputs
        V3D point_in_sensor_frame = rot_w2s * p_in.glob + transl_w2s_prime;

        if (log_details_for_this_call) {
            // Log the original calculation for comparison if you want to see its output too
            V3D p_glob_minus_transl_orig = p_in.glob - transl_w2s_prime; // Original C++'s intermediate step
            V3D p_in_sensor_frame_orig_cpp_math = rot_w2s * p_glob_minus_transl_orig; // Original C++'s result
            logger->trace("    Original C++ math intermediate (p.glob - transl): ({:.4f}, {:.4f}, {:.4f})",
                          p_glob_minus_transl_orig.x(), p_glob_minus_transl_orig.y(), p_glob_minus_transl_orig.z());
            logger->trace("    Original C++ math p_proj (sensor frame by that math): ({:.4f}, {:.4f}, {:.4f})",
                          p_in_sensor_frame_orig_cpp_math.x(), p_in_sensor_frame_orig_cpp_math.y(), p_in_sensor_frame_orig_cpp_math.z());
            
            // Log the corrected calculation's result
            logger->trace("    Corrected point_in_sensor_frame (rot*glob + transl): ({:.4f}, {:.4f}, {:.4f})",
                          point_in_sensor_frame.x(), point_in_sensor_frame.y(), point_in_sensor_frame.z());
        }

        // --- Calculate Spherical Coordinates using point_soph::GetVec ---
        // GetVec takes the point in sensor frame and calculates spherical vec, hor_ind, ver_ind, position,
        // and stores them in p_spherical_out.
        // It uses params.hor_resolution_max and params.ver_resolution_max internally.
        p_spherical_out.GetVec(point_in_sensor_frame, params.hor_resolution_max, params.ver_resolution_max);
        // If your point_soph::GetVec signature was changed to take `const DynObjFilterParams& params`, use that:
        // p_spherical_out.GetVec(point_in_sensor_frame, params);


        if (log_details_for_this_call) {
            logger->trace("  After p_spherical_out.GetVec(): Output SphericalVec=({:.3f},{:.3f},{:.3f}), H_ind={}, V_ind={}, Pos_1D={}",
                          p_spherical_out.vec.x(), p_spherical_out.vec.y(), p_spherical_out.vec.z(),
                          p_spherical_out.hor_ind, p_spherical_out.ver_ind, p_spherical_out.position);
        }

        // Update cache (using p_in's cache members, as p_spherical_out might be a temporary or different object)
        if (p_in.last_vecs.size() > static_cast<size_t>(cache_idx)) { // Bounds check
            p_in.last_vecs.at(cache_idx) = p_spherical_out.vec; // Cache the calculated spherical vector
            p_in.last_positions.at(cache_idx)[0] = p_spherical_out.hor_ind;
            p_in.last_positions.at(cache_idx)[1] = p_spherical_out.ver_ind;
            p_in.last_positions.at(cache_idx)[2] = p_spherical_out.position;
        }
    }
    // p_spherical_out now contains the results.
    // The original `p_in` also has its cache updated if it was a cache miss.
}


} // namespace PointCloudUtils