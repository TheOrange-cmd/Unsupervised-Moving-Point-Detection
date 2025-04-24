#include "gtest/gtest.h"
#include "point_cloud_utils.h" // Header for interpolation functions
#include "dyn_obj_datatypes.h" // Includes point_soph, DepthMap, V3D, V3F, etc.
#include "config_loader.h"     // Includes DynObjFilterParams
#include <vector>
#include <cmath>
#include <limits>
#include <memory> // For std::make_shared if using std::shared_ptr
#include <boost/make_shared.hpp> // For boost::shared_ptr if using boost
#include <Eigen/Geometry>      // For Eigen::AngleAxisd

// --- Test Fixture ---
class InterpolationTest : public ::testing::Test {
protected:
    // Use constants consistent with dyn_obj_datatypes.h
    static constexpr int TEST_MAX_1D = MAX_1D;
    static constexpr int TEST_MAX_1D_HALF = MAX_1D_HALF;
    static constexpr int TEST_MAX_2D_N = MAX_2D_N;
    static constexpr float TEST_PI = PI_MATH; // Use the same PI constant

    DynObjFilterParams params;
    DepthMap map_info; // Creates the map with size MAX_2D_N

    // Store shared pointers to manage the lifetime of points added to the map
    std::vector<point_soph::Ptr> managed_points;

    // Helper to create a point_soph with global coords and time, calculating projection
    // NOTE: Uses boost::shared_ptr based on dyn_obj_datatypes.h definition
    point_soph::Ptr createManagedTestPoint(const V3D& global_pos, double time, dyn_obj_flg dyn_status = STATIC) {
        // Create a dummy point_soph just to call GetVec - this is slightly awkward
        // A better approach might be a static helper in point_soph or PointCloudUtils
        // if this pattern is common.
        point_soph temp_point;
        temp_point.GetVec(global_pos, params.hor_resolution_max, params.ver_resolution_max);

        // Now create the actual managed point
        auto p_ptr = boost::make_shared<point_soph>();
        p_ptr->glob = global_pos;
        p_ptr->time = time;
        p_ptr->dyn = dyn_status;

        // Copy projection results from temp_point
        p_ptr->vec = temp_point.vec;
        p_ptr->hor_ind = temp_point.hor_ind;
        p_ptr->ver_ind = temp_point.ver_ind;
        p_ptr->position = temp_point.position;

        // Initialize other members if necessary for interpolation logic (e.g., cache if testing old funcs)
        p_ptr->reset(); // Use reset to clear cache arrays etc.

        managed_points.push_back(p_ptr);
        return p_ptr;
    }

    // Helper to add a point (created by createManagedTestPoint) to the map
    void addPointToMap(const point_soph::Ptr& p_ptr) {
        if (!p_ptr) return;

        // Check if indices are valid before adding
        if (p_ptr->hor_ind < 0 || p_ptr->hor_ind >= TEST_MAX_1D ||
            p_ptr->ver_ind < 0 || p_ptr->ver_ind >= TEST_MAX_1D_HALF) {
            // std::cerr << "Warning: Attempting to add point with invalid indices to map." << std::endl;
            return;
        }

        int map_pos = p_ptr->position; // Use pre-calculated position

        if (map_pos >= 0 && map_pos < TEST_MAX_2D_N) {
            map_info.depth_map[map_pos].push_back(p_ptr.get()); // Add raw pointer to map
        } else {
            // std::cerr << "Warning: Calculated map_pos out of bounds: " << map_pos << std::endl;
        }
    }

    void SetUp() override {
        // Initialize default parameters before each test
        // Use values that make index calculation easy if possible
        params.hor_resolution_max = 2.0f * TEST_PI / TEST_MAX_1D; // Approx resolution
        params.ver_resolution_max = TEST_PI / TEST_MAX_1D_HALF;   // Approx resolution

        params.frame_dur = 0.05; // Points within 50ms are "same frame"

        // Interpolation thresholds (adjust as needed for tests)
        params.interp_hor_thr = params.hor_resolution_max * 1.5f; // Allow ~1 neighbor cell away horizontally
        params.interp_ver_thr = params.ver_resolution_max * 1.5f; // Allow ~1 neighbor cell away vertically
        // Calculate derived pixel counts based on thresholds
        params.interp_hor_num = static_cast<int>(std::ceil(params.interp_hor_thr / params.hor_resolution_max));
        params.interp_ver_num = static_cast<int>(std::ceil(params.interp_ver_thr / params.ver_resolution_max));


        // Reset map_info (constructor should init, but Reset is cleaner)
        // map_info.Reset(M3D::Identity(), V3D::Zero(), 0.0, 0); // If Reset is available and safe

        // Clear managed points vector
        managed_points.clear();
    }

    // void TearDown() override {} // Cleanup handled by shared_ptr
};

// --- Test Cases ---

TEST_F(InterpolationTest, BasicSuccessStatic) {
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0); // On x-axis, azimuth=0, elevation=0
    auto p_target = createManagedTestPoint(p_glob, 0.1); // Time 0.1

    // Neighbors (STATIC, different time, forming a triangle around p's projection)
    // Azimuth slightly positive/negative, Elevation slightly positive/negative
    float small_angle = params.hor_resolution_max * 0.5f; // Place neighbors well within threshold
    V3D n1_glob(10.0, 10.0 * tan(small_angle), 0.0); // Azimuth +, Elev 0, Depth ~10
    V3D n2_glob(10.0, -10.0 * tan(small_angle), 10.0 * tan(small_angle)); // Azimuth -, Elev +
    V3D n3_glob(10.0, -10.0 * tan(small_angle), -10.0 * tan(small_angle)); // Azimuth -, Elev -

    auto n1 = createManagedTestPoint(n1_glob, 0.0); // Time 0.0
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    auto n3 = createManagedTestPoint(n3_glob, 0.0);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    // Expected depth should be close to 10.0 (p_target's actual depth)
    float expected_depth = p_target->vec.z();

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::STATIC_ONLY);

    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS);
    // Expect interpolated depth to be very close to the actual depth since neighbors surround it closely
    ASSERT_NEAR(result.depth, expected_depth, 0.1f) << "Interpolated depth mismatch.";
}

TEST_F(InterpolationTest, BasicSuccessAll) {
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Neighbors (mix of STATIC and DYNAMIC)
    float small_angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(small_angle), 0.0); // Static
    V3D n2_glob(10.0, -10.0 * tan(small_angle), 10.0 * tan(small_angle)); // Dynamic
    V3D n3_glob(10.0, -10.0 * tan(small_angle), -10.0 * tan(small_angle)); // Static

    auto n1 = createManagedTestPoint(n1_glob, 0.0, STATIC);
    auto n2 = createManagedTestPoint(n2_glob, 0.0, CASE1); // Mark as dynamic
    auto n3 = createManagedTestPoint(n3_glob, 0.0, STATIC);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    float expected_depth = p_target->vec.z();

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS);
    ASSERT_NEAR(result.depth, expected_depth, 0.1f);
}

TEST_F(InterpolationTest, StaticOnlyIgnoresDynamic) {
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Neighbors (Only 2 STATIC, 1 DYNAMIC)
    float small_angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(small_angle), 0.0); // Static
    V3D n2_glob(10.0, -10.0 * tan(small_angle), 10.0 * tan(small_angle)); // Dynamic
    V3D n3_glob(10.0, -10.0 * tan(small_angle), -10.0 * tan(small_angle)); // Static

    auto n1 = createManagedTestPoint(n1_glob, 0.0, STATIC);
    auto n2 = createManagedTestPoint(n2_glob, 0.0, CASE1); // Dynamic
    auto n3 = createManagedTestPoint(n3_glob, 0.0, STATIC);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    // Expect failure because only 2 static neighbors are available
    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::STATIC_ONLY);

    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS);
}


TEST_F(InterpolationTest, NotEnoughNeighbors) {
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Only two neighbors
    V3D n1_glob(10.0, 0.1, 0.0);
    V3D n2_glob(10.0, -0.1, 0.1);
    auto n1 = createManagedTestPoint(n1_glob, 0.0);
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    addPointToMap(n1);
    addPointToMap(n2);

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS);
}

TEST_F(InterpolationTest, NoValidTriangleCollinear) {
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Three neighbors, but collinear in projection space (same azimuth, different elevation)
    V3D n1_glob(10.0, 0.1, -0.1); // az ~ 0.01
    V3D n2_glob(10.0, 0.1, 0.0);  // az ~ 0.01
    V3D n3_glob(10.0, 0.1, 0.1);  // az ~ 0.01
    auto n1 = createManagedTestPoint(n1_glob, 0.0);
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    auto n3 = createManagedTestPoint(n3_glob, 0.0);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // Expect NO_VALID_TRIANGLE because all neighbors are collinear or form degenerate triangles
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::NO_VALID_TRIANGLE);
}

TEST_F(InterpolationTest, NoValidTriangleOutside) {
    V3D p_glob(10.0, 0.0, 0.0); // Target at origin of projection
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Three neighbors forming a triangle, but p is outside it
    float angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(angle), 0.0);          // Azimuth +, Elev 0
    V3D n2_glob(10.0, 10.0 * tan(angle*2.0f), 0.0);      // Azimuth ++, Elev 0
    V3D n3_glob(10.0, 10.0 * tan(angle), 10.0 * tan(angle)); // Azimuth +, Elev +

    auto n1 = createManagedTestPoint(n1_glob, 0.0);
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    auto n3 = createManagedTestPoint(n3_glob, 0.0);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // Expect NO_VALID_TRIANGLE because p is not inside the triangle formed by n1, n2, n3
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::NO_VALID_TRIANGLE);
}


TEST_F(InterpolationTest, TimeFilter) {
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1); // Time 0.1

    // Neighbors, but all too close in time
    float small_angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(small_angle), 0.0);
    V3D n2_glob(10.0, -10.0 * tan(small_angle), 10.0 * tan(small_angle));
    V3D n3_glob(10.0, -10.0 * tan(small_angle), -10.0 * tan(small_angle));

    // Times are within params.frame_dur (0.05) of p_target's time (0.1)
    auto n1 = createManagedTestPoint(n1_glob, 0.11);
    auto n2 = createManagedTestPoint(n2_glob, 0.09);
    auto n3 = createManagedTestPoint(n3_glob, 0.14);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // Expect failure because all neighbors are filtered by time
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS);
}

TEST_F(InterpolationTest, ProjectionDistanceFilter) {
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1); // Time 0.1

    // Neighbors, but too far in projection space
    float large_angle = params.interp_hor_thr * 1.5f; // Place neighbors outside threshold
    V3D n1_glob(10.0, 10.0 * tan(large_angle), 0.0);
    V3D n2_glob(10.0, -10.0 * tan(large_angle), 10.0 * tan(large_angle));
    V3D n3_glob(10.0, -10.0 * tan(large_angle), -10.0 * tan(large_angle));

    auto n1 = createManagedTestPoint(n1_glob, 0.0); // Time ok
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    auto n3 = createManagedTestPoint(n3_glob, 0.0);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // Expect failure because all neighbors are filtered by projection distance
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS);
}

TEST_F(InterpolationTest, HorizontalWrapAround) {
    // Target point near azimuth = +PI boundary
    // Azimuth PI corresponds roughly to hor_ind = MAX_1D - 1 or 0
    // Let's target slightly less than PI, e.g., index MAX_1D - 2
    float target_azimuth = TEST_PI - params.hor_resolution_max * 1.5f;
    V3D p_glob(10.0 * cos(target_azimuth), 10.0 * sin(target_azimuth), 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);
    ASSERT_TRUE(p_target->hor_ind >= 0 && p_target->hor_ind < TEST_MAX_1D);


    // Neighbor 1: Slightly more positive azimuth (should wrap to index 0 or 1)
    float n1_azimuth = TEST_PI + params.hor_resolution_max * 0.5f; // Should wrap
    V3D n1_glob(11.0 * cos(n1_azimuth), 11.0 * sin(n1_azimuth), 0.0); // Different depth

    // Neighbor 2: Slightly less positive azimuth (near target)
    float n2_azimuth = target_azimuth - params.hor_resolution_max * 0.5f;
    V3D n2_glob(12.0 * cos(n2_azimuth), 12.0 * sin(n2_azimuth), 1.0); // Different depth & elev

    // Neighbor 3: Near target, different elevation
    V3D n3_glob(13.0 * cos(target_azimuth), 13.0 * sin(target_azimuth), -1.0); // Different depth & elev

    auto n1 = createManagedTestPoint(n1_glob, 0.0);
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    auto n3 = createManagedTestPoint(n3_glob, 0.0);

    // Verify neighbor indices (especially n1's wrap-around)
    // std::cout << "p hor_ind: " << p_target->hor_ind << std::endl;
    // std::cout << "n1 hor_ind: " << n1->hor_ind << " (az=" << n1->vec.x() << ")" << std::endl;
    // std::cout << "n2 hor_ind: " << n2->hor_ind << std::endl;
    // std::cout << "n3 hor_ind: " << n3->hor_ind << std::endl;
    ASSERT_TRUE(n1->hor_ind >= 0 && n1->hor_ind < 5); // Expect low index due to wrap
    ASSERT_NEAR(n2->hor_ind, p_target->hor_ind - 1, 1); // Expect index near p
    ASSERT_NEAR(n3->hor_ind, p_target->hor_ind, 1);     // Expect index near p


    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // We expect success because the wrapped neighbor n1 should be found
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS);

    // Calculate expected depth manually using barycentric coords (approximate)
    // This is complex, so maybe just check if the depth is within the range of neighbors
    float min_neighbor_depth = std::min({n1->vec.z(), n2->vec.z(), n3->vec.z()});
    float max_neighbor_depth = std::max({n1->vec.z(), n2->vec.z(), n3->vec.z()});
    ASSERT_GE(result.depth, min_neighbor_depth - 0.1f);
    ASSERT_LE(result.depth, max_neighbor_depth + 0.1f);
}

TEST_F(InterpolationTest, PointOnEdge) {
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Neighbors: n1, n2 form an edge, p lies on it. n3 completes triangle.
    float small_angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(9.0, 9.0 * tan(-small_angle), 0.0); // Depth 9, Azimuth -
    V3D n2_glob(11.0, 11.0 * tan(small_angle), 0.0); // Depth 11, Azimuth +
    // p_target (depth 10, Azimuth 0) should lie on the line between n1 and n2's projections
    V3D n3_glob(10.0, 0.0, 1.0); // Depth 10, Elev + (forms triangle)


    auto n1 = createManagedTestPoint(n1_glob, 0.0);
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    auto n3 = createManagedTestPoint(n3_glob, 0.0);
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS);
    // Since p is exactly halfway between n1 and n2 in projection and they have elev 0,
    // the interpolation should yield the average of their depths (if n1,n2,n3 is chosen)
    // Or if another triangle is chosen, it should still be close to 10.
    // Let's check if it's close to the expected depth of p_target (10.0)
    ASSERT_NEAR(result.depth, p_target->vec.z(), 0.1f) << "Interpolated depth mismatch for point on edge.";
}