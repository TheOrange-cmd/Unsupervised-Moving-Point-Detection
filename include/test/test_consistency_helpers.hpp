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

// Headers for the code being tested and its dependencies
#include "filtering/consistency_checks.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "point_cloud_utils/point_cloud_utils.h" // For types used internally
#include "common/types.h" // Include your types.h


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


// --- Test Fixture Declaration ---

class ConsistencyChecksTest : public ::testing::Test {
protected:
    DynObjFilterParams params;
    DepthMap test_map;
    point_soph center_point; // Can be initialized in SetUp

    const std::string config_path = "../test/config/test_full_config.yaml"; // Adjust if needed

    // Declare SetUp - Definition can be here or in a .cpp file
    void SetUp() override;

    // Declare helper methods belonging to the fixture
    // Define them inline here if simple, or define in .cpp file if complex
    inline std::shared_ptr<point_soph> addPointToMap(const point_soph& p) {
        if (p.position >= 0 && p.position < MAX_2D_N) {
            // Create the shared pointer
            auto new_point_ptr = std::make_shared<point_soph>(p);
            // Add it to the map
            test_map.depth_map[p.position].push_back(new_point_ptr);
            // Return the pointer
            return new_point_ptr;
        } else {
            std::cerr << "Warning: Attempted to add point with invalid position "
                      << p.position << " to map." << std::endl;
            // Return nullptr to indicate failure/no addition
            return nullptr;
        }
    }

    // Declare other fixture helpers...
    point_soph addNeighborInRelativeCell(
        const point_soph& target,
        DepthMap& map,
        const DynObjFilterParams& params,
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