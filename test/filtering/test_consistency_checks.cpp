#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <string>
#include <iostream> // For potential debug output

// Headers for the code being tested and its dependencies
#include "filtering/consistency_checks.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "point_cloud_utils/point_cloud_utils.h" // For types used internally by consistency_checks

// --- Constants (Ensure these match your project's definitions) ---
// Assuming these are defined elsewhere, e.g., dyn_obj_datatypes.h or constants.h
// #define MAX_1D 1800 // Example value
// #define MAX_1D_HALF 16 // Example value
// #define MAX_2D_N (MAX_1D * MAX_1D_HALF) // Example value

// Helper function to create a point with basic info and calculated indices
// Assumes params are loaded for resolution info.
point_soph createTestPointWithIndices(
    float x, float y, float z, // Local Cartesian coordinates for projection
    const DynObjFilterParams& params,
    double time = 0.0,
    dyn_obj_flg status = STATIC,
    bool is_distort = false)
{
    point_soph p;
    V3D local_coords(x, y, z);
    p.local = local_coords; // Store local coordinates
    p.glob = local_coords; // Assume global = local for simplicity in tests unless specified
    p.time = time;
    p.dyn = status;
    p.is_distort = is_distort;

    // Calculate spherical coordinates and indices using GetVec
    // Need non-const resolutions for the GetVec signature in the provided point_soph code
    float hor_res = params.hor_resolution_max;
    float ver_res = params.ver_resolution_max;
    p.GetVec(local_coords, hor_res, ver_res); // Calculates vec, hor_ind, ver_ind, position

    // Ensure indices are valid (GetVec should handle clamping/wrapping)
    if (p.position < 0 || p.position >= MAX_2D_N) {
        // Handle error or throw? For tests, maybe force into valid range if GetVec fails?
        std::cerr << "Warning: createTestPointWithIndices generated invalid position " << p.position
                  << " for coords (" << x << "," << y << "," << z << "). Clamping to 0." << std::endl;
        p.position = 0;
        p.hor_ind = 0;
        p.ver_ind = 0;
    }

    return p;
}

// Helper function to convert Spherical (az, el, depth) to Cartesian (x, y, z)
V3D sphericalToCartesian(float az, float el, float depth) {
    // Ensure depth is positive
    depth = std::max(0.0f, depth);
    float x = depth * std::cos(el) * std::cos(az);
    float y = depth * std::cos(el) * std::sin(az);
    float z = depth * std::sin(el);
    return V3D(x, y, z);
}

// Helper function to convert grid indices back to approximate spherical angles
// This is the inverse of the logic in GetVec (simplified)
void indicesToSpherical(int hor_ind, int ver_ind, float hor_res, float ver_res, float& az, float& el) {
    // Azimuth: index = floor((az + PI) / hor_res) => az ~ (index * hor_res) - PI
    // Add 0.5 * hor_res to aim for the center of the cell
    az = (static_cast<float>(hor_ind) + 0.5f) * hor_res - M_PI;

    // Elevation: index = floor((el + PI/2) / ver_res) => el ~ (index * ver_res) - PI/2
    // Add 0.5 * ver_res to aim for the center of the cell
    el = (static_cast<float>(ver_ind) + 0.5f) * ver_res - (0.5f * M_PI);

    // Clamp elevation to valid range [-PI/2, PI/2] just in case
    el = std::max(-0.5f * (float)M_PI, std::min(0.5f * (float)M_PI, el));
    // Azimuth wrapping handled implicitly by trig functions later
}


class ConsistencyChecksTest : public ::testing::Test {
protected:
    DynObjFilterParams params;
    DepthMap test_map;
    point_soph center_point; // The point being checked for consistency

    // Path to the config file (relative to build directory where test runs)
    // Adjust this path if your build/test setup differs.
    const std::string config_path = "../test/config/test_full_config.yaml";

    void SetUp() override {
        // Load parameters from the full config file
        try {
            bool loaded = load_config(config_path, params);
            ASSERT_TRUE(loaded) << "Failed to load config file: " << config_path;
            ASSERT_GT(params.hor_resolution_max, 0) << "Horizontal resolution invalid.";
            ASSERT_GT(params.ver_resolution_max, 0) << "Vertical resolution invalid.";
        } catch (const std::exception& e) {
            FAIL() << "Exception during config loading: " << e.what();
        }

        // Initialize/Clear the depth map
        test_map.depth_map.assign(MAX_2D_N, std::vector<std::shared_ptr<point_soph>>());
        // Initialize static min/max if they are used directly or indirectly (optional)
        // test_map.min_depth_static.assign(MAX_2D_N, std::numeric_limits<float>::max());
        // test_map.max_depth_static.assign(MAX_2D_N, 0.0f);
        // ... same for min/max_depth_all ...

        // Initialize a default center point (can be overridden in tests)
        // Place it somewhere away from edges and the self-box by default
        center_point = createTestPointWithIndices(20.0, 5.0, 0.0, params, 0.1); // time=0.1
        ASSERT_GE(center_point.position, 0) << "Default center_point has invalid position.";
        ASSERT_LT(center_point.position, MAX_2D_N) << "Default center_point has invalid position.";
    }

    // Helper to add a point (as shared_ptr) to the test map
    void addPointToMap(const point_soph& p) {
        if (p.position >= 0 && p.position < MAX_2D_N) {
            test_map.depth_map[p.position].push_back(std::make_shared<point_soph>(p));
        } else {
            std::cerr << "Warning: Attempted to add point with invalid position "
                      << p.position << " to map." << std::endl;
        }
    }

    // Helper to create neighbors for interpolation tests
    // Creates 3 neighbors guaranteed to be in different cells adjacent to the target
    void addInterpolationTriangle(const point_soph& target, float depth1, float depth2, float depth3,
        dyn_obj_flg status = STATIC, double time_offset = -1.0) {

    int target_hor_ind = target.hor_ind;
    int target_ver_ind = target.ver_ind;

    // Define neighbor indices in adjacent cells (handle wrapping/clamping)
    // Neighbor 1: Lower-left cell
    int n1_hor_ind = (target_hor_ind - 1 + MAX_1D) % MAX_1D; // Wrap horizontally
    int n1_ver_ind = std::max(0, target_ver_ind - 1);       // Clamp vertically

    // Neighbor 2: Lower-right cell
    int n2_hor_ind = (target_hor_ind + 1) % MAX_1D;         // Wrap horizontally
    int n2_ver_ind = std::max(0, target_ver_ind - 1);       // Clamp vertically (same row as n1)

    // Neighbor 3: Cell directly above
    int n3_hor_ind = target_hor_ind;                        // Same column as target
    int n3_ver_ind = std::min(MAX_1D_HALF - 1, target_ver_ind + 1); // Clamp vertically

    // --- Convert target indices back to approximate spherical angles ---
    float az1, el1, az2, el2, az3, el3;
    indicesToSpherical(n1_hor_ind, n1_ver_ind, params.hor_resolution_max, params.ver_resolution_max, az1, el1);
    indicesToSpherical(n2_hor_ind, n2_ver_ind, params.hor_resolution_max, params.ver_resolution_max, az2, el2);
    indicesToSpherical(n3_hor_ind, n3_ver_ind, params.hor_resolution_max, params.ver_resolution_max, az3, el3);

    // --- Convert spherical back to Cartesian ---
    V3D n1_local = sphericalToCartesian(az1, el1, depth1);
    V3D n2_local = sphericalToCartesian(az2, el2, depth2);
    V3D n3_local = sphericalToCartesian(az3, el3, depth3);

    // --- Create point_soph objects using the calculated local Cartesian coords ---
    // These points *should* now map back to the intended nX_hor_ind, nX_ver_ind
    point_soph n1 = createTestPointWithIndices(n1_local.x(), n1_local.y(), n1_local.z(), params, target.time + time_offset, status);
    point_soph n2 = createTestPointWithIndices(n2_local.x(), n2_local.y(), n2_local.z(), params, target.time + time_offset, status);
    point_soph n3 = createTestPointWithIndices(n3_local.x(), n3_local.y(), n3_local.z(), params, target.time + time_offset, status);

    // --- Add to map ---
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    // --- Debug Print (Crucial for this attempt) ---
    std::cout << "Target Pos=" << target.position << " (H:" << target.hor_ind << ", V:" << target.ver_ind << ")" << std::endl;
    std::cout << "Added Neighbors (Cell-based):" << std::endl;
    std::cout << " N1 Target (H:" << n1_hor_ind << ", V:" << n1_ver_ind << ") -> Actual Pos=" << n1.position << " (H:" << n1.hor_ind << ", V:" << n1.ver_ind << ") Proj=(" << n1.vec(0) << "," << n1.vec(1) << ")" << std::endl;
    std::cout << " N2 Target (H:" << n2_hor_ind << ", V:" << n2_ver_ind << ") -> Actual Pos=" << n2.position << " (H:" << n2.hor_ind << ", V:" << n2.ver_ind << ") Proj=(" << n2.vec(0) << "," << n2.vec(1) << ")" << std::endl;
    std::cout << " N3 Target (H:" << n3_hor_ind << ", V:" << n3_ver_ind << ") -> Actual Pos=" << n3.position << " (H:" << n3.hor_ind << ", V:" << n3.ver_ind << ") Proj=(" << n3.vec(0) << "," << n3.vec(1) << ")" << std::endl;
    }


};

// --- Test Cases ---

TEST_F(ConsistencyChecksTest, PointInsideSelfBox) {
    // Modify center_point's local coordinates to be inside the self-box defined by params
    center_point.local.x() = (params.self_x_b + params.self_x_f) / 2.0;
    center_point.local.y() = (params.self_y_r + params.self_y_l) / 2.0;
    center_point.local.z() = 0.0; // Z doesn't matter for the check

    // Add some arbitrary points to the map (shouldn't affect the outcome)
    addInterpolationTriangle(center_point, center_point.vec(2), center_point.vec(2), center_point.vec(2));

    // Check consistency for all cases - should return false due to self-box
    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION)) << "Case 1 failed self-box check";
    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, 2)) << "Case 2 failed self-box check";
    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH, 2)) << "Case 3 failed self-box check";
}

// --- Case 1 Tests (STATIC_ONLY neighbors) ---

TEST_F(ConsistencyChecksTest, Case1_InterpolationSucceedsAndPasses) {
    center_point.vec(2) = 15.0f; // Set depth for center point
    float neighbor_depth = 15.1f; // Close depth
    float expected_interp_depth = neighbor_depth; // Approx

    // Add a valid triangle of STATIC neighbors with depths close to center_point
    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, STATIC, -params.frame_dur * 2); // Old static points

    // Calculate expected threshold for Case 1
    float threshold = params.interp_thr1;
     if (center_point.vec(2) > params.interp_start_depth1) {
         threshold += ((center_point.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
     }
    // Assume center_point is not distorted for this test

    ASSERT_LT(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be within threshold";

    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be consistent when interpolation passes threshold.";
}

TEST_F(ConsistencyChecksTest, Case1_InterpolationSucceedsButFailsThreshold) {
    center_point.vec(2) = 15.0f;
    float neighbor_depth = 17.0f; // Far depth
    float expected_interp_depth = neighbor_depth; // Approx

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, STATIC, -params.frame_dur * 2);

    // Calculate expected threshold for Case 1
    float threshold = params.interp_thr1;
     if (center_point.vec(2) > params.interp_start_depth1) {
         threshold += ((center_point.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
     }

    ASSERT_GE(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be outside threshold";

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be inconsistent when interpolation result is outside threshold.";
}

TEST_F(ConsistencyChecksTest, Case1_InterpolationFailsNotEnoughNeighbors) {
    center_point.vec(2) = 15.0f;

    // Add only one STATIC neighbor
    point_soph n1 = createTestPointWithIndices(center_point.local.x()+1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*2, STATIC);
    n1.vec(2) = 15.1f; // Close depth, but shouldn't matter
    addPointToMap(n1);

    // Add one close static point *conceptually* for the old logic check
    // The new logic should ignore this if interpolation fails.
    // point_soph close_neighbor = createTestPointWithIndices(center_point.local.x()+0.1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*2, STATIC);
    // close_neighbor.vec(2) = 15.05f; // Very close
    // addPointToMap(close_neighbor); // Add if you want to be extra sure it doesn't interfere

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be inconsistent when interpolation fails (not enough neighbors), even if a close static point exists.";
}

TEST_F(ConsistencyChecksTest, Case1_InterpolationFailsBadTriangle) {
    center_point.vec(2) = 15.0f;

    // Add 3 STATIC neighbors, but make them collinear (same elevation, different azimuth)
    float depth = 15.1f;
    float az_offset = params.hor_resolution_max * 1.5f;
    V3D p_local = center_point.local;

    V3D p1_local = p_local; p1_local.x() -= p_local.norm() * az_offset; p1_local = p1_local.normalized() * depth;
    V3D p2_local = p_local;                                              p2_local = p2_local.normalized() * depth;
    V3D p3_local = p_local; p3_local.x() += p_local.norm() * az_offset; p3_local = p3_local.normalized() * depth;

    point_soph n1 = createTestPointWithIndices(p1_local.x(), p1_local.y(), p1_local.z(), params, center_point.time - params.frame_dur*2, STATIC);
    point_soph n2 = createTestPointWithIndices(p2_local.x(), p2_local.y(), p2_local.z(), params, center_point.time - params.frame_dur*2, STATIC);
    point_soph n3 = createTestPointWithIndices(p3_local.x(), p3_local.y(), p3_local.z(), params, center_point.time - params.frame_dur*2, STATIC);

    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be inconsistent when interpolation fails (bad triangle).";
}

TEST_F(ConsistencyChecksTest, Case1_DistortionEnlargement) {
    // Requires dataset 0 and is_distort = true
    if (params.dataset != 0) {
        GTEST_SKIP() << "Skipping distortion test because dataset is not 0";
    }
    if (params.enlarge_distort <= 1.0f) {
         GTEST_SKIP() << "Skipping distortion test because enlarge_distort is not > 1.0";
    }

    center_point = createTestPointWithIndices(20.0, 5.0, 0.0, params, 0.1, STATIC, true); // is_distort = true
    center_point.vec(2) = 15.0f;
    float neighbor_depth = 15.5f; // Choose a depth difference
    float expected_interp_depth = neighbor_depth; // Approx

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, STATIC, -params.frame_dur * 2);

    // Calculate threshold WITHOUT distortion
    float threshold_no_distort = params.interp_thr1;
     if (center_point.vec(2) > params.interp_start_depth1) {
         threshold_no_distort += ((center_point.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
     }
    // Calculate threshold WITH distortion
    float threshold_with_distort = threshold_no_distort * params.enlarge_distort;

    float actual_diff = std::fabs(expected_interp_depth - center_point.vec(2));

    // Ensure the chosen depth difference falls between the two thresholds
    ASSERT_GE(actual_diff, threshold_no_distort) << "Test setup error: Depth diff should be >= non-distorted threshold.";
    ASSERT_LT(actual_diff, threshold_with_distort) << "Test setup error: Depth diff should be < distorted threshold.";

    // Expect TRUE because the enlarged threshold should allow it
    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be consistent due to distortion enlargement.";
}


// --- Case 2 Tests (ALL_VALID neighbors, older map) ---

TEST_F(ConsistencyChecksTest, Case2_InterpolationSucceedsAndPasses) {
    center_point.vec(2) = 25.0f;
    float neighbor_depth = 25.2f; // Close depth
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 3; // Simulate map being 3 steps older

    // Add a valid triangle - can include DYNAMIC points if they are old enough
    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, UNCERTAIN, -params.frame_dur * (map_diff + 1)); // Old enough dynamic points
    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, STATIC, -params.frame_dur * (map_diff + 1)); // Old static points

    float threshold = params.interp_thr2 * static_cast<float>(map_diff);

    ASSERT_LT(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be within threshold";

    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, map_diff))
        << "Case 2: Point should be consistent when interpolation passes threshold.";
}

TEST_F(ConsistencyChecksTest, Case2_InterpolationFailsThreshold) {
    center_point.vec(2) = 25.0f;
    float neighbor_depth = 28.0f; // Far depth
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 3;

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, STATIC, -params.frame_dur * (map_diff + 1));

    float threshold = params.interp_thr2 * static_cast<float>(map_diff);

    ASSERT_GE(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be outside threshold";

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, map_diff))
        << "Case 2: Point should be inconsistent when interpolation fails threshold.";
}

TEST_F(ConsistencyChecksTest, Case2_InterpolationFailsNotEnoughValidNeighbors) {
    center_point.vec(2) = 25.0f;
    int map_diff = 3;

    // Add only one OLD STATIC neighbor
    point_soph n1 = createTestPointWithIndices(center_point.local.x()+1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*(map_diff+1), STATIC);
    n1.vec(2) = 25.1f;
    addPointToMap(n1);
    // Add one RECENT DYNAMIC neighbor (should be ignored by findInterpolationNeighbors for ALL_VALID if time diff < frame_dur)
    point_soph n2 = createTestPointWithIndices(center_point.local.x()-1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*0.5, UNCERTAIN);
    n2.vec(2) = 25.1f;
    addPointToMap(n2);

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, map_diff))
        << "Case 2: Point should be inconsistent when interpolation fails (not enough valid neighbors).";
}


// --- Case 3 Tests (ALL_VALID neighbors, older map) ---

TEST_F(ConsistencyChecksTest, Case3_InterpolationSucceedsAndPasses) {
    center_point.vec(2) = 30.0f;
    // Adjust depth slightly to be clearly within threshold
    float neighbor_depth = 30.19f; // WAS 30.2f
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 4;

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, STATIC, -params.frame_dur * (map_diff + 1));

    float threshold = params.interp_thr3 * static_cast<float>(map_diff);

    // This assertion should now pass
    ASSERT_LT(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be within threshold";

    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH, map_diff))
        << "Case 3: Point should be consistent when interpolation passes threshold.";
}

TEST_F(ConsistencyChecksTest, Case3_InterpolationFailsThreshold) {
    center_point.vec(2) = 30.0f;
    float neighbor_depth = 33.0f; // Far depth
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 4;

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, STATIC, -params.frame_dur * (map_diff + 1));

    float threshold = params.interp_thr3 * static_cast<float>(map_diff);

    ASSERT_GE(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be outside threshold";

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH, map_diff))
        << "Case 3: Point should be inconsistent when interpolation fails threshold.";
}