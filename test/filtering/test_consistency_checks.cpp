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
        test_map.time = 0.0; // Reset timestamp if needed
        // Initialize static min/max if they are used directly or indirectly (optional)
        // test_map.min_depth_static.assign(MAX_2D_N, std::numeric_limits<float>::max());
        // test_map.max_depth_static.assign(MAX_2D_N, 0.0f);
        // ... same for min/max_depth_all ...

        // Initialize a default center point (can be overridden in tests)
        // Place it somewhere away from edges and the self-box by default
        center_point = createTestPointWithIndices(20.0, 5.0, 0.0, params, 0.1); // time=0.1
        center_point.vec(2) = 20.0;
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

    point_soph addNeighborInRelativeCell(
        const point_soph& target,
        DepthMap& map, // Pass map by reference
        const DynObjFilterParams& params, // Pass params
        int delta_hor_ind, // e.g., -1, 0, 1
        int delta_ver_ind, // e.g., -1, 0, 1
        float depth,
        double time_offset_sec, // Time offset in SECONDS
        dyn_obj_flg status = STATIC)
    {
        int neighbor_hor_ind = (target.hor_ind + delta_hor_ind + MAX_1D) % MAX_1D;
        int neighbor_ver_ind = target.ver_ind + delta_ver_ind;
    
        // Clamp vertical index
        neighbor_ver_ind = std::max(0, std::min(MAX_1D_HALF - 1, neighbor_ver_ind));
    
        float neighbor_az, neighbor_el;
        indicesToSpherical(neighbor_hor_ind, neighbor_ver_ind, params.hor_resolution_max, params.ver_resolution_max, neighbor_az, neighbor_el);
    
        V3D neighbor_local = sphericalToCartesian(neighbor_az, neighbor_el, depth);
    
        // Use target time + offset
        double neighbor_time = target.time + time_offset_sec;
    
        point_soph neighbor_point = createTestPointWithIndices(
            neighbor_local.x(), neighbor_local.y(), neighbor_local.z(),
            params, neighbor_time, status);
    
        // Debug print in helper
        std::cout << "[addNeighborInRelativeCell] Target H=" << target.hor_ind << ", V=" << target.ver_ind
                  << ". Adding neighbor with delta H=" << delta_hor_ind << ", V=" << delta_ver_ind
                  << " (Target Cell " << neighbor_hor_ind << "," << neighbor_ver_ind << ")"
                  << " at Depth=" << depth << " Time=" << neighbor_time
                  << ". Resulting Point H=" << neighbor_point.hor_ind << ", V=" << neighbor_point.ver_ind
                  << ", Az=" << neighbor_point.vec(0) << ", El=" << neighbor_point.vec(1) << std::endl;
    
        // Add the point to the map passed by reference
        addPointToMap(neighbor_point); // Use a direct add function if 'addPointToMap' isn't suitable
                                                    // Or make addPointToMap take the map as arg.
                                                    // Let's assume addPointToMap uses the fixture's test_map implicitly for now.
                                                    // If not, adjust this call.
    
        return neighbor_point;
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

// --- Test Cases for Map Consistency ---

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

// --- Test Cases for Depth Consistency ---

// Test Case: No suitable neighbors found (empty map)
TEST_F(ConsistencyChecksTest, DepthConsistency_NoNeighbors) {
    // center_point is setup in the fixture
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should return false when no neighbors exist in the map.";
}

// Test Case: Neighbors exist but are filtered out (time, angle, status)
TEST_F(ConsistencyChecksTest, DepthConsistency_NoSuitableNeighbors) {
    // Add neighbors outside the time window
    addPointToMap(createTestPointWithIndices(center_point.local.x() + 0.1, center_point.local.y(), center_point.local.z(), params, center_point.time + params.frame_dur * 2.0, STATIC));
    // Add neighbors outside the angular threshold (assuming default thresholds are not huge)
    addPointToMap(createTestPointWithIndices(center_point.local.x() + 5.0, center_point.local.y() + 5.0, center_point.local.z(), params, center_point.time, STATIC));
    // Add non-static neighbor
    addPointToMap(createTestPointWithIndices(center_point.local.x(), center_point.local.y(), center_point.local.z(), params, center_point.time, UNCERTAIN)); // Assuming UNCERTAIN is not STATIC

    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should return false when neighbors exist but none meet the criteria (time, angle, static).";
}

// Test Case: Pass - Only one close static neighbor (Rule 1 skipped, Rule 2 passes trivially)
TEST_F(ConsistencyChecksTest, DepthConsistency_PassOneCloseNeighbor) {
    center_point.vec(2) = 20.0f;
    float neighbor_depth = 20.1f;
    float depth_diff = center_point.vec(2) - neighbor_depth;
    ASSERT_LT(std::fabs(depth_diff), params.depth_cons_depth_max_thr2);

    // Add neighbor in the *same cell* (delta 0,0) - within search window [-1,1], [-2,2]
    addNeighborInRelativeCell(center_point, test_map, params, 0, 0, neighbor_depth, -params.frame_dur * 0.5, STATIC);

    // *** ADD THIS DEBUG CHECK ***
    int target_cell_index = center_point.hor_ind * MAX_1D_HALF + center_point.ver_ind;
    std::cout << "[TEST DEBUG] Size of map cell [" << target_cell_index << "] after add: "
            << test_map.depth_map[target_cell_index].size() << std::endl;
    // Also check if the pointers are the same if size > 1
    if (test_map.depth_map[target_cell_index].size() > 1) {
        std::cout << "[TEST DEBUG] Pointer 1: " << test_map.depth_map[target_cell_index][0].get() << std::endl;
        std::cout << "[TEST DEBUG] Pointer 2: " << test_map.depth_map[target_cell_index][1].get() << std::endl;
    }
    // ***************************

    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass with only one close static neighbor.";
}

// Test Case: Pass - Multiple close neighbors, average diff below threshold, no significant diffs
TEST_F(ConsistencyChecksTest, DepthConsistency_PassAvgDiffBelowThreshold) {
    center_point.vec(2) = 20.0f;
    float current_depth_threshold = std::max(params.depth_cons_depth_thr2, params.k_depth2 * center_point.vec(2));
    float depth1 = 20.0f + current_depth_threshold * 0.4f;
    float depth2 = 20.0f - current_depth_threshold * 0.3f;
    float depth3 = 20.0f + current_depth_threshold * 0.2f;
    // Average abs diff = (| -0.4 | + | 0.3 | + | 0.2 |)*thr / (3-1) = (0.4+0.3+0.2)*thr/2 = 0.9*thr/2 = 0.45*thr
    ASSERT_LT(0.45f * current_depth_threshold, current_depth_threshold); // Verify test setup avg diff < threshold
    ASSERT_LT(std::fabs(center_point.vec(2) - depth1), params.depth_cons_depth_max_thr2); // Verify within max_thr
    ASSERT_LT(std::fabs(center_point.vec(2) - depth2), params.depth_cons_depth_max_thr2);
    ASSERT_LT(std::fabs(center_point.vec(2) - depth3), params.depth_cons_depth_max_thr2);

    // Add neighbors in adjacent cells within search window [-1,1], [-2,2]
    addNeighborInRelativeCell(center_point, test_map, params, 0, 0, depth1, -params.frame_dur * 0.5, STATIC); // Same cell
    addNeighborInRelativeCell(center_point, test_map, params, 1, 0, depth2, -params.frame_dur * 0.5, STATIC); // Cell right
    addNeighborInRelativeCell(center_point, test_map, params, 0, 1, depth3, -params.frame_dur * 0.5, STATIC); // Cell above

    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass when average depth diff of close neighbors is below threshold.";
}

// Test Case: Fail - Multiple close neighbors, average diff ABOVE threshold
TEST_F(ConsistencyChecksTest, DepthConsistency_FailAvgDiffAboveThreshold) {
    center_point.vec(2) = 20.0f;
    float current_depth_threshold = std::max(params.depth_cons_depth_thr2, params.k_depth2 * center_point.vec(2));

    // Add neighbors with larger (but still < max_thr) depth differences, ensuring average is ABOVE threshold
    float depth1 = 20.0f + current_depth_threshold * 1.1f; // diff = -1.1*thr
    float depth2 = 20.0f - current_depth_threshold * 1.2f; // diff = +1.2*thr
    // Average abs diff = (| -1.1 | + | 1.2 |)*thr / (2-1) = (1.1+1.2)*thr/1 = 2.3*thr
    ASSERT_GT(2.3f * current_depth_threshold, current_depth_threshold); // Verify test setup avg diff > threshold
    ASSERT_LT(std::fabs(center_point.vec(2) - depth1), params.depth_cons_depth_max_thr2); // Ensure still < max_thr
    ASSERT_LT(std::fabs(center_point.vec(2) - depth2), params.depth_cons_depth_max_thr2);

    addPointToMap(createTestPointWithIndices(center_point.local.x() + 0.01, center_point.local.y(), depth1, params, center_point.time - params.frame_dur * 0.5, STATIC));
    addPointToMap(createTestPointWithIndices(center_point.local.x() - 0.01, center_point.local.y(), depth2, params, center_point.time - params.frame_dur * 0.5, STATIC));

    // Expect false because avg diff is too high (Rule 1 fails).
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should fail when average depth diff of close neighbors is above threshold.";
}

// Test Case: Pass - Rule 1 passes (or skipped), Rule 2 passes (only significantly FARTHER neighbors)
TEST_F(ConsistencyChecksTest, DepthConsistency_PassOnlyFartherNeighbors) {
    center_point.vec(2) = 20.0f;
    float max_thr = params.depth_cons_depth_max_thr2;

    // Add one close neighbor (Rule 1 will be skipped or pass easily)
    addNeighborInRelativeCell(center_point, test_map, params, 0, 0, 20.0f, -params.frame_dur * 0.5, STATIC);

    // Add neighbors significantly FARTHER than p (p.depth > neighbor.depth)
    float farther_depth1 = center_point.vec(2) - max_thr * 1.5f;
    float farther_depth2 = center_point.vec(2) - max_thr * 2.0f;
    // Add in adjacent cells within search window [-1,1], [-2,2]
    addNeighborInRelativeCell(center_point, test_map, params, -1, 0, farther_depth1, -params.frame_dur * 0.5, STATIC); // Cell left
    addNeighborInRelativeCell(center_point, test_map, params, 0, -1, farther_depth2, -params.frame_dur * 0.5, STATIC); // Cell below

    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass when significantly different neighbors are all farther.";
}

// Test Case: Pass - Rule 1 passes (or skipped), Rule 2 passes (only significantly CLOSER neighbors)
TEST_F(ConsistencyChecksTest, DepthConsistency_PassOnlyCloserNeighbors) {
    center_point.vec(2) = 20.0f;
    float max_thr = params.depth_cons_depth_max_thr2;

    // Add one close neighbor
    addNeighborInRelativeCell(center_point, test_map, params, 0, 0, 20.0f, -params.frame_dur * 0.5, STATIC);

    // Add neighbors significantly CLOSER than p (p.depth < neighbor.depth)
    float closer_depth1 = center_point.vec(2) + max_thr * 1.5f;
    float closer_depth2 = center_point.vec(2) + max_thr * 2.0f;
    // Add in adjacent cells within search window [-1,1], [-2,2]
    addNeighborInRelativeCell(center_point, test_map, params, 1, 1, closer_depth1, -params.frame_dur * 0.5, STATIC); // Cell up-right
    addNeighborInRelativeCell(center_point, test_map, params, 0, 2, closer_depth2, -params.frame_dur * 0.5, STATIC); // Cell 2 above

    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass when significantly different neighbors are all closer.";
}

// Test Case: Fail - Rule 1 passes (or skipped), Rule 2 fails (MIXED closer/farther neighbors)
TEST_F(ConsistencyChecksTest, DepthConsistency_FailMixedCloserFartherNeighbors) {
    center_point.vec(2) = 20.0f;
    float max_thr = params.depth_cons_depth_max_thr2;

    // Add one close neighbor
    addPointToMap(createTestPointWithIndices(center_point.local.x() + 0.01, center_point.local.y(), 20.0f, params, center_point.time - params.frame_dur * 0.5, STATIC));

    // Add one significantly FARTHER neighbor
    float farther_depth = center_point.vec(2) - max_thr * 1.5f;
    addPointToMap(createTestPointWithIndices(center_point.local.x() - 0.01, center_point.local.y(), farther_depth, params, center_point.time - params.frame_dur * 0.5, STATIC));

    // Add one significantly CLOSER neighbor
    float closer_depth = center_point.vec(2) + max_thr * 1.5f;
    addPointToMap(createTestPointWithIndices(center_point.local.x(), center_point.local.y() + 0.01, closer_depth, params, center_point.time - params.frame_dur * 0.5, STATIC));

    // Expect false because count_closer > 0 AND count_farther > 0 (Rule 2 fails).
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should fail when there is a mix of significantly closer and farther neighbors.";
}

// Test Case: Ensure CASE3 uses its specific parameters (simple check)
// This test assumes CASE3 parameters are different from CASE2 in the config
// We'll set up a scenario that passes with CASE2 params but fails with CASE3 (or vice versa)
TEST_F(ConsistencyChecksTest, DepthConsistency_UsesCase3Params) {
    // Example: Make CASE3 max_thr much smaller than CASE2 max_thr
    // Ensure your test config has different values for depth_cons_depth_max_thr2 and depth_cons_depth_max_thr3
    if (params.depth_cons_depth_max_thr2 <= params.depth_cons_depth_max_thr3) {
        GTEST_SKIP() << "Skipping test: depth_cons_depth_max_thr2 is not greater than depth_cons_depth_max_thr3 in config.";
    }

    center_point.vec(2) = 20.0f;
    // Choose a depth difference that is < max_thr2 but >= max_thr3
    float depth_diff = (params.depth_cons_depth_max_thr2 + params.depth_cons_depth_max_thr3) / 2.0f;
    float neighbor_depth = center_point.vec(2) + depth_diff; // This makes diff negative

    ASSERT_LT(std::fabs(depth_diff), params.depth_cons_depth_max_thr2) << "Test setup error: Diff should be < max_thr2";
    ASSERT_GE(std::fabs(depth_diff), params.depth_cons_depth_max_thr3) << "Test setup error: Diff should be >= max_thr3";

    // Add one neighbor. With CASE2, it's "close". With CASE3, it's "significantly closer".
    addPointToMap(createTestPointWithIndices(center_point.local.x() + 0.01, center_point.local.y(), neighbor_depth, params, center_point.time - params.frame_dur * 0.5, STATIC));

    // CASE 2: count_close=1, count_closer=0, count_farther=0 -> Should PASS
    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "CASE 2 should pass with this neighbor setup.";

    // CASE 3: count_close=0, count_closer=1, count_farther=0 -> Should PASS (Rule 2 passes)
    // This setup actually passes both, just classifying the neighbor differently.
    // Let's modify to make CASE3 fail Rule 2.
    test_map.depth_map.assign(MAX_2D_N, std::vector<std::shared_ptr<point_soph>>()); // Clear map

    // Add one significantly CLOSER neighbor (using CASE3 max_thr)
    float closer_depth = center_point.vec(2) + params.depth_cons_depth_max_thr3 * 1.5f;
    addPointToMap(createTestPointWithIndices(center_point.local.x() + 0.01, center_point.local.y(), closer_depth, params, center_point.time - params.frame_dur * 0.5, STATIC));
    // Add one significantly FARTHER neighbor (using CASE3 max_thr)
    float farther_depth = center_point.vec(2) - params.depth_cons_depth_max_thr3 * 1.5f;
    addPointToMap(createTestPointWithIndices(center_point.local.x() - 0.01, center_point.local.y(), farther_depth, params, center_point.time - params.frame_dur * 0.5, STATIC));

    // CASE 3: count_close=0, count_closer=1, count_farther=1 -> Should FAIL (Rule 2 fails)
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH))
        << "CASE 3 should fail with mixed closer/farther neighbors based on its specific max_thr.";

    // Optional: Check if CASE2 passes this same setup (it might if max_thr2 is large enough to make both points "close")
    // bool case2_result = ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    // std::cout << "Case 2 result for Case 3 fail setup: " << case2_result << std::endl;

}


// Test Case: Ensure calling with CASE1 throws an exception
TEST_F(ConsistencyChecksTest, DepthConsistency_ThrowsOnCase1) {
    EXPECT_THROW(
        ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION),
        std::invalid_argument
    ) << "Should throw std::invalid_argument when called with CASE1_FALSE_REJECTION.";
}

