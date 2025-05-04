// file: test/test_consistency_helpers.hpp

#ifndef TEST_CONSISTENCY_HELPERS_HPP
#define TEST_CONSISTENCY_HELPERS_HPP

#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <string>
#include <iostream>
#include <Eigen/Eigen> 
#include <algorithm> // For std::max, std::min

// Headers for the code being tested and its dependencies
#include "filtering/consistency_checks.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "point_cloud_utils/point_cloud_utils.h" 
#include "common/types.h"


// --- Standalone Helper Functions (Marked Inline) ---

// Helper function to create a point with basic info and calculated indices
inline point_soph createTestPointWithIndices(
    float x, float y, float z, // Local Cartesian coordinates for projection
    const DynObjFilterParams& params,
    double time = 0.0,
    dyn_obj_flg status = STATIC,
    bool is_distort = false)
{
    point_soph p;
    V3D local_coords(x, y, z);
    p.local = local_coords;
    p.glob = local_coords; // Simplify for tests
    p.time = time;
    p.dyn = status;
    p.is_distort = is_distort;

    float hor_res = params.hor_resolution_max;
    float ver_res = params.ver_resolution_max;
    p.GetVec(local_coords, hor_res, ver_res);

    if (p.position < 0 || p.position >= MAX_2D_N) {
        std::cerr << "Warning: createTestPointWithIndices generated invalid position " << p.position
                  << " for coords (" << x << "," << y << "," << z << "). Clamping to 0." << std::endl;
        p.position = std::max(0, std::min(p.position, MAX_2D_N - 1)); // Clamp instead of just 0
        // Recalculate indices based on clamped position? Or rely on GetVec's clamping?
        // Assuming GetVec handles clamping correctly based on its implementation.
    }
    return p;
}

// Helper function to convert Spherical (az, el, depth) to Cartesian (x, y, z)
inline V3D sphericalToCartesian(float az, float el, float depth) {
    depth = std::max(0.0f, depth);
    float x = depth * std::cos(el) * std::cos(az);
    float y = depth * std::cos(el) * std::sin(az);
    float z = depth * std::sin(el);
    return V3D(x, y, z);
}

// Helper function to convert grid indices back to approximate spherical angles
inline void indicesToSpherical(int hor_ind, int ver_ind, float hor_res, float ver_res, float& az, float& el) {
    az = (static_cast<float>(hor_ind) + 0.5f) * hor_res - M_PI;
    el = (static_cast<float>(ver_ind) + 0.5f) * ver_res - (0.5f * M_PI);
    el = std::max(-0.5f * (float)M_PI, std::min(0.5f * (float)M_PI, el));
}

// Standalone helper to calculate the dynamic depth threshold
inline float calculateOcclusionDepthThreshold(
    const point_soph& occluder,
    const point_soph& occluded,
    const DynObjFilterParams& p, // Pass params explicitly
    ConsistencyChecks::ConsistencyCheckType type)
{
    float k_depth_max, d_depth_max, base_offset, v_min;
    bool is_case2 = false; // Keep for potential future use, though not strictly needed now

     switch (type) {
        case ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
            k_depth_max = p.k_depth_max_thr2; d_depth_max = p.d_depth_max_thr2;
            base_offset = p.occ_depth_thr2; v_min = p.v_min_thr2;
            is_case2 = true;
            break;
        case ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
             k_depth_max = p.k_depth_max_thr3; d_depth_max = p.d_depth_max_thr3;
             base_offset = p.map_cons_depth_thr3; v_min = p.v_min_thr3;
             break;
        default:
            // Consider throwing or returning NaN instead of -1.0f for clearer error
            throw std::invalid_argument("calculateOcclusionDepthThreshold received invalid check_type");
            // return std::numeric_limits<float>::quiet_NaN();
     }
     double delta_t = occluder.time - occluded.time;
     // Check delta_t validity based on how checkOcclusionRelationship uses it
     if (delta_t <= 0) {
         // The checkOcclusionRelationship returns false for delta_t <= 0, so threshold is irrelevant.
         // Returning NaN or a very large value might be appropriate, or throw.
         // Let's return NaN to indicate the condition isn't met for threshold calculation.
         return std::numeric_limits<float>::quiet_NaN();
     }

     // Calculate adaptive threshold component
     float depth_thr_adaptive = p.cutoff_value; // Start with cutoff
     if (occluder.vec(2) > d_depth_max) { // Apply linear part only if depth > d_depth_max
         depth_thr_adaptive = std::max(depth_thr_adaptive, k_depth_max * (occluder.vec(2) - d_depth_max));
     }

     // Calculate velocity-based threshold component
     float depth_thr_velocity = v_min * static_cast<float>(delta_t);

     // Combine: Min of velocity-based and (adaptive + base offset)
     float threshold = std::min(depth_thr_adaptive + base_offset, depth_thr_velocity);

     // Apply distortion enlargement factor if applicable
     if (p.dataset == 0 && occluder.is_distort && p.enlarge_distort > 1.0f) {
          threshold *= p.enlarge_distort;
     }

     // Ensure threshold is not negative (shouldn't happen with max(cutoff, ...) but good practice)
     return std::max(0.0f, threshold);
}


// --- Test Fixture Declaration ---

class ConsistencyChecksTest : public ::testing::Test {
    protected:
        DynObjFilterParams params;
        DepthMap test_map;
        point_soph center_point; // Can be initialized in SetUp
    
        const std::string config_path = "../test_cpp/config/test_full_config.yaml"; // Adjust if needed
    
        // Declare SetUp - Definition can be here or in a .cpp file
        void SetUp() override;
    
        // Declare helper methods belonging to the fixture
        inline std::shared_ptr<point_soph> addPointToMap(const point_soph& p) {
            // ... (definition as before) ...
            if (p.position >= 0 && p.position < MAX_2D_N) {
                auto new_point_ptr = std::make_shared<point_soph>(p);
                test_map.depth_map[p.position].push_back(new_point_ptr);
                // Update min_depth_all if necessary for optimization testing
                if (test_map.min_depth_all.size() == MAX_2D_N) { // Basic check if initialized
                     test_map.min_depth_all[p.position] = std::min(test_map.min_depth_all[p.position], p.vec(2));
                }
                return new_point_ptr;
            } else {
                std::cerr << "Warning: Attempted to add point with invalid position "
                          << p.position << " to map." << std::endl;
                return nullptr;
            }
        }
    
    
        // Declare other fixture helpers...
        point_soph addNeighborInRelativeCell(
            const point_soph& target,
            DepthMap& map, // Note: map is now a member, maybe remove from signature? No, keep for clarity.
            const DynObjFilterParams& params, // Same, params is member. Keep for clarity.
            int delta_hor_ind,
            int delta_ver_ind,
            float depth,
            double time_offset_sec,
            dyn_obj_flg status = STATIC); // Definition in .cpp
    
        void addInterpolationTriangle(
            const point_soph& target,
            float depth1, float depth2, float depth3,
            dyn_obj_flg status = STATIC, double time_offset = -1.0); // Definition in .cpp
    };

#endif // TEST_CONSISTENCY_HELPERS_HPP