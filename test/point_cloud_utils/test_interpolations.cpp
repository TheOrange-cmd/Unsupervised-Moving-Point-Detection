#include "gtest/gtest.h"
#include "point_cloud_utils/point_cloud_utils.h" // Header for interpolation functions
#include "filtering/dyn_obj_datatypes.h" // Includes point_soph, DepthMap, V3D, V3F, etc.
#include "config/config_loader.h"     // Includes DynObjFilterParams
#include <vector>
#include <cmath>
#include <limits>
#include <memory>                // For std::make_shared if using std::shared_ptr
// #include <boost/make_shared.hpp> // For boost::shared_ptr if using boost
#include <Eigen/Geometry>        // For Eigen::AngleAxisd

// --- Test Fixture ---
class InterpolationTest : public ::testing::Test
{
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
    point_soph::Ptr createManagedTestPoint(const V3D &global_pos, double time, dyn_obj_flg dyn_status = STATIC)
    {
        // Create a dummy point_soph just to call GetVec - this is slightly awkward
        // A better approach might be a static helper in point_soph or PointCloudUtils
        // if this pattern is common.
        point_soph temp_point;
        temp_point.GetVec(global_pos, params.hor_resolution_max, params.ver_resolution_max);

        // Now create the actual managed point
        auto p_ptr = std::make_shared<point_soph>();
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
        // Use point's unique global position or time for identification if needed
        std::cout << "[addPointToMap] Processing point az=" << p_ptr->vec.x() << ", el=" << p_ptr->vec.y() << ", depth=" << p_ptr->vec.z() << std::endl;
        std::cout << "[addPointToMap]   Indices: hor=" << p_ptr->hor_ind << ", ver=" << p_ptr->ver_ind << ", pos=" << p_ptr->position << std::endl;
    
        // Check if indices are valid before adding
        if (p_ptr->hor_ind < 0 || p_ptr->hor_ind >= TEST_MAX_1D ||
            p_ptr->ver_ind < 0 || p_ptr->ver_ind >= TEST_MAX_1D_HALF) {
            std::cerr << "[addPointToMap]   ERROR: Invalid indices, skipping add." << std::endl;
            return;
        }
    
        int map_pos = p_ptr->position; // Use pre-calculated position
    
        if (map_pos >= 0 && map_pos < TEST_MAX_2D_N) {
            std::cout << "[addPointToMap]   Adding to map_pos=" << map_pos << std::endl;
            map_info.depth_map[map_pos].push_back(p_ptr.get()); // Add raw pointer to map
            // Verification print:
            std::cout << "[addPointToMap]   Cell map_info.depth_map[" << map_pos << "] now has size: " << map_info.depth_map[map_pos].size() << std::endl;
        } else {
            std::cerr << "[addPointToMap]   ERROR: Calculated map_pos=" << map_pos << " out of bounds [" << 0 << ", " << TEST_MAX_2D_N << "), skipping add." << std::endl;
        }
    }

    void SetUp() override
    {
        // Initialize default parameters before each test
        // Use values that make index calculation easy if possible
        params.hor_resolution_max = 2.0f * TEST_PI / TEST_MAX_1D; // Approx resolution
        params.ver_resolution_max = TEST_PI / TEST_MAX_1D_HALF;   // Approx resolution

        params.frame_dur = 0.05; // Points within 50ms are "same frame"

        // 1. Define the desired grid search extent first
        params.interp_hor_num = 2; // Search +/- 2 cells horizontally
        params.interp_ver_num = 2; // Search +/- 2 cells vertically

        // 2. Calculate thresholds to encompass the search grid + a small margin (e.g., half cell)
        //    This ensures points near the edge of the grid search area can pass the angular check.
        params.interp_hor_thr = (params.interp_hor_num + 0.5f) * params.hor_resolution_max; // (2 + 0.5) * res_h = 2.5 * res_h
        params.interp_ver_thr = (params.interp_ver_num + 0.5f) * params.ver_resolution_max; // (2 + 0.5) * res_v = 2.5 * res_v

        // Reset map_info (constructor should init, but Reset is cleaner)
        // map_info.Reset(M3D::Identity(), V3D::Zero(), 0.0, 0); // If Reset is available and safe

        // Clear managed points vector
        managed_points.clear();
    }

    // void TearDown() override {} // Cleanup handled by shared_ptr
};

// --- Test Cases ---

TEST_F(InterpolationTest, BasicSuccessStatic)
{
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0);                          // On x-axis, azimuth=0, elevation=0
    auto p_target = createManagedTestPoint(p_glob, 0.1); // Time 0.1

    // Neighbors (STATIC, different time, forming a triangle around p's projection)
    // Azimuth slightly positive/negative, Elevation slightly positive/negative
    float small_angle = params.hor_resolution_max * 0.5f;                  // Place neighbors well within threshold
    V3D n1_glob(10.0, 10.0 * tan(small_angle), 0.0);                       // Azimuth +, Elev 0, Depth ~10
    V3D n2_glob(10.0, -10.0 * tan(small_angle), 10.0 * tan(small_angle));  // Azimuth -, Elev +
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

TEST_F(InterpolationTest, BasicSuccessAll)
{
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Neighbors (mix of STATIC and DYNAMIC)
    float small_angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(small_angle), 0.0);                       // Static
    V3D n2_glob(10.0, -10.0 * tan(small_angle), 10.0 * tan(small_angle));  // Dynamic
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

TEST_F(InterpolationTest, StaticOnlyIgnoresDynamic)
{
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Neighbors (Only 2 STATIC, 1 DYNAMIC)
    float small_angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(small_angle), 0.0);                       // Static
    V3D n2_glob(10.0, -10.0 * tan(small_angle), 10.0 * tan(small_angle));  // Dynamic
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

TEST_F(InterpolationTest, NotEnoughNeighbors)
{
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

TEST_F(InterpolationTest, NoValidTriangleCollinear)
{
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

TEST_F(InterpolationTest, NoValidTriangleOutside)
{
    V3D p_glob(10.0, 0.0, 0.0); // Target at origin of projection
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Three neighbors forming a triangle, but p is outside it
    float angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(angle), 0.0);               // Azimuth +, Elev 0
    V3D n2_glob(10.0, 10.0 * tan(angle * 2.0f), 0.0);        // Azimuth ++, Elev 0
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

TEST_F(InterpolationTest, TimeFilter)
{
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

TEST_F(InterpolationTest, ProjectionDistanceFilter)
{
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
    float target_azimuth = TEST_PI - params.hor_resolution_max * 1.5f;
    V3D p_glob(10.0 * cos(target_azimuth), 10.0 * sin(target_azimuth), 0.0); // Target Z=0
    auto p_target = createManagedTestPoint(p_glob, 0.1);
    ASSERT_TRUE(p_target->hor_ind >= 0 && p_target->hor_ind < TEST_MAX_1D);


    // Neighbor 1: Slightly more positive azimuth (should wrap to index 0 or 1)
    float n1_azimuth = TEST_PI + params.hor_resolution_max * 0.5f; // Should wrap
    V3D n1_glob(11.0 * cos(n1_azimuth), 11.0 * sin(n1_azimuth), 0.0); // Keep Z=0

    // Neighbor 2: Slightly less positive azimuth (near target)
    float n2_azimuth = target_azimuth - params.hor_resolution_max * 0.5f;
    // *** CHANGE Z coordinate to be very close to target's Z ***
    V3D n2_glob(12.0 * cos(n2_azimuth), 12.0 * sin(n2_azimuth), 0.1); // Use Z=0.1 (small elevation diff)

    // Neighbor 3: Near target, different elevation (but keep it small)
    // *** CHANGE Z coordinate to be very close to target's Z ***
    V3D n3_glob(13.0 * cos(target_azimuth), 13.0 * sin(target_azimuth), -0.1); // Use Z=-0.1 (small elevation diff)

    auto n1 = createManagedTestPoint(n1_glob, 0.0);
    auto n2 = createManagedTestPoint(n2_glob, 0.0);
    auto n3 = createManagedTestPoint(n3_glob, 0.0);

    // --- Add Debug Prints Again (Optional but Recommended) ---
    std::cout << "DEBUG INFO (Post Z Change): p_target: hor=" << p_target->hor_ind << ", ver=" << p_target->ver_ind << ", pos=" << p_target->position << ", az=" << p_target->vec.x() << ", el=" << p_target->vec.y() << std::endl;
    std::cout << "DEBUG INFO (Post Z Change): n1:       hor=" << n1->hor_ind << ", ver=" << n1->ver_ind << ", pos=" << n1->position << ", az=" << n1->vec.x() << ", el=" << n1->vec.y() << std::endl;
    std::cout << "DEBUG INFO (Post Z Change): n2:       hor=" << n2->hor_ind << ", ver=" << n2->ver_ind << ", pos=" << n2->position << ", az=" << n2->vec.x() << ", el=" << n2->vec.y() << std::endl;
    std::cout << "DEBUG INFO (Post Z Change): n3:       hor=" << n3->hor_ind << ", ver=" << n3->ver_ind << ", pos=" << n3->position << ", az=" << n3->vec.x() << ", el=" << n3->vec.y() << std::endl;
    // --- End Debug Prints ---

    ASSERT_TRUE(n1->hor_ind >= 0 && n1->hor_ind < 5); // Expect low index due to wrap
    // Assert that n2 and n3 are now vertically close to p_target
    ASSERT_NEAR(n2->ver_ind, p_target->ver_ind, params.interp_ver_num + 1); // Should be within search grid
    ASSERT_NEAR(n3->ver_ind, p_target->ver_ind, params.interp_ver_num + 1); // Should be within search grid


    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    // Add a print to verify map contents just before interpolation call
    std::cout << "Map Contents Before Interpolation:" << std::endl;
    for(int i=0; i<TEST_MAX_2D_N; ++i) {
        if (!map_info.depth_map[i].empty()) {
            std::cout << "  Cell " << i << " (hor=" << i/TEST_MAX_1D_HALF << ", ver=" << i%TEST_MAX_1D_HALF << "): " << map_info.depth_map[i].size() << " points" << std::endl;
            // Optionally print point details
            // for(const auto* pt : map_info.depth_map[i]) {
            //     std::cout << "    -> az=" << pt->vec.x() << ", el=" << pt->vec.y() << std::endl;
            // }
        }
    }


    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // We expect success because n1, n2, n3 should now be found
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS); // Line 364
    
    // Calculate expected depth manually using barycentric coords (approximate)
    // This is complex, so maybe just check if the depth is within the range of neighbors
    float min_neighbor_depth = std::min({n1->vec.z(), n2->vec.z(), n3->vec.z()});
    float max_neighbor_depth = std::max({n1->vec.z(), n2->vec.z(), n3->vec.z()});
    ASSERT_GE(result.depth, min_neighbor_depth - 0.1f);
    ASSERT_LE(result.depth, max_neighbor_depth + 0.1f);
}

TEST_F(InterpolationTest, PointOnEdge)
{
    // Target point p
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);

    // Neighbors: n1, n2 form an edge, p lies on it. n3 completes triangle.
    float small_angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(9.0, 9.0 * tan(-small_angle), 0.0);  // Depth 9, Azimuth -
    V3D n2_glob(11.0, 11.0 * tan(small_angle), 0.0); // Depth 11, Azimuth +
    // p_target (depth 10, Azimuth 0) should lie on the line between n1 and n2's projections
    V3D n3_glob(10.0, 0.0, 0.1); // Depth 10, Elev slightly + (forms triangle)

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

TEST_F(InterpolationTest, ThresholdBoundaryExactFilter)
{
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1); // az=0, el=0, time=0.1

    // Neighbors exactly AT the threshold distance
    // Note: Using atan for precise angle control relative to target
    V3D n1_glob(10.0, 10.0 * tan(params.interp_hor_thr), 0.0); // Horizontal threshold
    V3D n2_glob(10.0, 0.0, 10.0 * tan(params.interp_ver_thr)); // Vertical threshold
    V3D n3_glob(10.0, 0.1, 0.1);                              // A valid neighbor (close, different time)

    auto n1 = createManagedTestPoint(n1_glob, 0.0); // time=0.0 (valid time)
    auto n2 = createManagedTestPoint(n2_glob, 0.0); // time=0.0 (valid time)
    auto n3 = createManagedTestPoint(n3_glob, 0.0); // time=0.0 (valid time)
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    // Call findInterpolationNeighbors directly to check filtering
    std::vector<V3F> neighbors = PointCloudUtils::findInterpolationNeighbors(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // Expect only n3 to be kept, as n1 and n2 are exactly on the boundary (>= check filters them)
    ASSERT_EQ(neighbors.size(), 1);

    // Double-check the main function behavior (should fail due to not enough neighbors)
    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS);
}

TEST_F(InterpolationTest, ThresholdBoundaryInside)
{
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1);
    
    // Neighbors just INSIDE the threshold distance, surrounding the target
    float epsilon_angle = 1e-6;
    float az_near_thr = params.interp_hor_thr - epsilon_angle;
    float el_near_thr = params.interp_ver_thr - epsilon_angle;
    
    // n1: Positive az, positive el
    V3D n1_glob(10.0, 10.0 * tan(az_near_thr * 0.5f), 10.0 * tan(el_near_thr * 0.5f));
    // n2: Negative az, positive el
    V3D n2_glob(10.0, 10.0 * tan(-az_near_thr * 0.5f), 10.0 * tan(el_near_thr * 0.5f));
    // n3: Negative az, negative el
    V3D n3_glob(10.0, 10.0 * tan(-az_near_thr * 0.5f), 10.0 * tan(-el_near_thr * 0.5f));
    
    auto n1 = createManagedTestPoint(n1_glob, 0.0); n1->vec[2] = 11.0f; // Give distinct depths
    auto n2 = createManagedTestPoint(n2_glob, 0.0); n2->vec[2] = 12.0f;
    auto n3 = createManagedTestPoint(n3_glob, 0.0); n3->vec[2] = 13.0f;
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);
    
    // Call findInterpolationNeighbors directly - still expect 3
    std::vector<V3F> neighbors = PointCloudUtils::findInterpolationNeighbors(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    ASSERT_EQ(neighbors.size(), 3); // Should still find 3
    
    // Check the main function behavior (should succeed now)
    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    // ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS); // Line 475
    // Add a check for the status before asserting
    if (result.status != PointCloudUtils::InterpolationStatus::SUCCESS) {
        FAIL() << "Expected SUCCESS but got status " << static_cast<int>(result.status);
    }
    ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS);
}

TEST_F(InterpolationTest, TimeFilterBoundary)
{
    V3D p_glob(10.0, 0.0, 0.0);
    double target_time = 0.1;
    auto p_target = createManagedTestPoint(p_glob, target_time); // Target az=0, el=0

    // Neighbors with specific time differences, BUT surrounding the target projection
    float angle = params.hor_resolution_max * 0.5f; // Small angle for positioning

    // n1: Time diff = 0.05 (target_time + frame_dur), Pos: (+az, +el) - Quadrant 1
    V3D n1_glob(10.0, 10.0 * tan(angle), 10.0 * tan(angle));
    auto n1 = createManagedTestPoint(n1_glob, target_time + params.frame_dur);
    n1->vec[2] = 11.0f; // Give distinct depth

    // n2: Time diff = 0.05 (target_time - frame_dur), Pos: (-az, +el) - Quadrant 2
    V3D n2_glob(10.0, 10.0 * tan(-angle), 10.0 * tan(angle));
    auto n2 = createManagedTestPoint(n2_glob, target_time - params.frame_dur);
    n2->vec[2] = 12.0f;

    // n3: Time diff = 0.1 (target_time - 2*frame_dur), Pos: (0, -el) - Negative Elevation Axis
    V3D n3_glob(10.0, 0.0, 10.0 * tan(-angle));
    auto n3 = createManagedTestPoint(n3_glob, target_time - params.frame_dur * 2.0);
    n3->vec[2] = 13.0f;

    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    // Call findInterpolationNeighbors directly to check filtering
    std::vector<V3F> neighbors = PointCloudUtils::findInterpolationNeighbors(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // Expect ALL 3 neighbors to be kept, as time_diff < frame_dur is false for n1 and n2.
    ASSERT_EQ(neighbors.size(), 3); // This should still pass

    // Double-check the main function behavior
    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);

    // Since the 3 neighbors now surround the target projection, interpolation should succeed.
    // ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS); // Line 521
    // Use FAIL() for better debugging message if it fails
    if (result.status != PointCloudUtils::InterpolationStatus::SUCCESS) {
         FAIL() << "TimeFilterBoundary: Expected SUCCESS but got status " << static_cast<int>(result.status)
                << ". Neighbors found: " << neighbors.size();
    }
    // We don't strictly need to check the depth value here, the main point is
    // that SUCCESS is achieved when points at the time boundary are included
    // and form a valid geometric configuration.
}

TEST_F(InterpolationTest, VerticalBoundaries)
{
    // Test near ver_ind = 0
    {
        // Create target point with elevation near the minimum (e.g., slightly negative)
        // We need to calculate global coords that result in ver_ind = 0 or 1
        // Let's aim for elevation slightly below 0, which should map near the middle index if PI range is used.
        // To get ver_ind = 0, we need elevation near -PI/2. Let's use -PI/2 + epsilon.
        float target_el_min = -TEST_PI / 2.0f + params.ver_resolution_max * 0.5f;
        V3D p_glob_min(10.0, 0.0, 10.0 * tan(target_el_min));
        auto p_target_min = createManagedTestPoint(p_glob_min, 0.1);
        ASSERT_LE(p_target_min->ver_ind, 1);

        // Create neighbors surrounding target projection
        float az_offset = params.hor_resolution_max * 0.5f;
        // n1: Same el, zero az offset
        V3D n1_min_glob(10.0, 0.0, 10.0 * tan(target_el_min));
        // n2: el + res, positive az offset
        V3D n2_min_glob(10.0, 10.0 * tan(az_offset), 10.0 * tan(target_el_min + params.ver_resolution_max));
        // n3: el + 2*res, negative az offset
        V3D n3_min_glob(10.0, 10.0 * tan(-az_offset), 10.0 * tan(target_el_min + 2.0f*params.ver_resolution_max));
    
        auto n1_min = createManagedTestPoint(n1_min_glob, 0.0); n1_min->vec[2]=11.f; // Give depths
        auto n2_min = createManagedTestPoint(n2_min_glob, 0.0); n2_min->vec[2]=12.f;
        auto n3_min = createManagedTestPoint(n3_min_glob, 0.0); n3_min->vec[2]=13.f;
        addPointToMap(n1_min);
        addPointToMap(n2_min);
        addPointToMap(n3_min);
    
        PointCloudUtils::InterpolationResult result_min = PointCloudUtils::interpolateDepth(
            *p_target_min, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    
        // ASSERT_EQ(result_min.status, PointCloudUtils::InterpolationStatus::SUCCESS); // Line 541
        if (result_min.status != PointCloudUtils::InterpolationStatus::SUCCESS) {
             FAIL() << "VerticalBoundaries (min): Expected SUCCESS but got status " << static_cast<int>(result_min.status);
        }
        managed_points.clear();
        map_info.Reset(M3D::Identity(), V3D::Zero(), 0.0, 0); // Reset map
    }

    // Test near ver_ind = MAX_1D_HALF - 1
    {
        // Aim for elevation near +PI/2 - epsilon
        float target_el_max = TEST_PI / 2.0f - params.ver_resolution_max * 0.5f;
        V3D p_glob_max(10.0, 0.0, 10.0 * tan(target_el_max));
        auto p_target_max = createManagedTestPoint(p_glob_max, 0.1);
        ASSERT_GE(p_target_max->ver_ind, TEST_MAX_1D_HALF - 2);

        // Create neighbors surrounding target projection
        float az_offset = params.hor_resolution_max * 0.5f;
        // n1: Same el, zero az offset
        V3D n1_max_glob(10.0, 0.0, 10.0 * tan(target_el_max));
        // n2: el - res, positive az offset
        V3D n2_max_glob(10.0, 10.0 * tan(az_offset), 10.0 * tan(target_el_max - params.ver_resolution_max));
        // n3: el - 2*res, negative az offset
        V3D n3_max_glob(10.0, 10.0 * tan(-az_offset), 10.0 * tan(target_el_max - 2.0f*params.ver_resolution_max));
    
        auto n1_max = createManagedTestPoint(n1_max_glob, 0.0); n1_max->vec[2]=11.f;
        auto n2_max = createManagedTestPoint(n2_max_glob, 0.0); n2_max->vec[2]=12.f;
        auto n3_max = createManagedTestPoint(n3_max_glob, 0.0); n3_max->vec[2]=13.f;
        addPointToMap(n1_max);
        addPointToMap(n2_max);
        addPointToMap(n3_max);
    
        PointCloudUtils::InterpolationResult result_max = PointCloudUtils::interpolateDepth(
            *p_target_max, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    
        // ASSERT_EQ(result_max.status, PointCloudUtils::InterpolationStatus::SUCCESS);
         if (result_max.status != PointCloudUtils::InterpolationStatus::SUCCESS) {
             FAIL() << "VerticalBoundaries (max): Expected SUCCESS but got status " << static_cast<int>(result_max.status);
        }
    }
}

TEST_F(InterpolationTest, InvalidTargetIndices)
{
    // Create a point_soph manually with invalid indices
    point_soph invalid_target;
    invalid_target.time = 0.1;
    invalid_target.vec = V3F(0,0,10); // Dummy projection

    // Test invalid horizontal index
    invalid_target.hor_ind = -1;
    invalid_target.ver_ind = TEST_MAX_1D_HALF / 2; // Valid vertical
    invalid_target.position = -1; // Position would also be invalid

    PointCloudUtils::InterpolationResult result_hor = PointCloudUtils::interpolateDepth(
        invalid_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    // findInterpolationNeighbors should return early, leading to not enough neighbors
    ASSERT_EQ(result_hor.status, PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS);

    // Test invalid vertical index
    invalid_target.hor_ind = TEST_MAX_1D / 2; // Valid horizontal
    invalid_target.ver_ind = TEST_MAX_1D_HALF; // Invalid vertical (>= MAX_1D_HALF)
    invalid_target.position = -1;

    PointCloudUtils::InterpolationResult result_ver = PointCloudUtils::interpolateDepth(
        invalid_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    ASSERT_EQ(result_ver.status, PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS);
}


// --- Tests for computeBarycentricDepth Edge Cases ---

TEST_F(InterpolationTest, MultipleValidTriangles)
{
    // Target point p at projection origin
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1); // az=0, el=0

    // Create 4 neighbors forming two overlapping triangles around the target
    float angle = params.hor_resolution_max * 0.5f;
    V3D n1_glob(10.0, 10.0 * tan(angle), 10.0 * tan(angle));   // (+az, +el), Depth ~10
    V3D n2_glob(10.0, 10.0 * tan(-angle), 10.0 * tan(angle));  // (-az, +el), Depth ~10
    V3D n3_glob(10.0, 10.0 * tan(-angle), 10.0 * tan(-angle)); // (-az, -el), Depth ~10
    V3D n4_glob(10.0, 10.0 * tan(angle), 10.0 * tan(-angle));  // (+az, -el), Depth ~10

    // Give them slightly different depths so interpolation matters
    auto n1 = createManagedTestPoint(n1_glob, 0.0); n1->vec[2] = 11.0f; // Depth 11
    auto n2 = createManagedTestPoint(n2_glob, 0.0); n2->vec[2] = 12.0f; // Depth 12
    auto n3 = createManagedTestPoint(n3_glob, 0.0); n3->vec[2] = 13.0f; // Depth 13
    auto n4 = createManagedTestPoint(n4_glob, 0.0); n4->vec[2] = 14.0f; // Depth 14

    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);
    addPointToMap(n4);

    // Potential triangles containing target (az=0, el=0): (n1, n2, n3) and (n1, n3, n4)
    // The code selects based on minimum sum of Manhattan distances of v2, v3 relative to target.
    // Let's analyze: Target T=(0,0)
    // n1=(+a,+a), n2=(-a,+a), n3=(-a,-a), n4=(+a,-a) where a = angle
    // Triangle 1 (v1=n1, v2=n2, v3=n3):
    //   dist(T,n2) = |-a-0| + |+a-0| = 2a
    //   dist(T,n3) = |-a-0| + |-a-0| = 2a
    //   Sum = 4a
    // Triangle 2 (v1=n1, v2=n3, v3=n4):
    //   dist(T,n3) = |-a-0| + |-a-0| = 2a
    //   dist(T,n4) = |+a-0| + |-a-0| = 2a
    //   Sum = 4a
    // Triangle 3 (v1=n2, v2=n3, v3=n4):
    //   dist(T,n3) = |-a-0| + |-a-0| = 2a
    //   dist(T,n4) = |+a-0| + |-a-0| = 2a
    //   Sum = 4a
    // Triangle 4 (v1=n4, v2=n2, v3=n3):
    //   dist(T,n2) = |-a-0| + |+a-0| = 2a
    //   dist(T,n3) = |-a-0| + |-a-0| = 2a
    //   Sum = 4a

    // Hmm, the simple distance metric might yield the same value for multiple triangles
    // in this symmetric case. The implementation iterates i,j,k and updates the best
    // if the current_dist_sum is strictly LESS than overall_min_dist_sum.
    // The first valid triangle found will likely be chosen.
    // Let's assume (n1, n2, n3) is chosen first.
    // Target (0,0) is inside. Calculate barycentric coords (approximate for small angles):
    // Relative vectors: v21=(-2a, 0), v31=(-2a, -2a). TargetOffset=(-a, -a) relative to v1=(a,a)
    // Denom = (-2a)*(-2a) - (-2a)*(0) = 4a^2
    // u = ((-a)*(-2a) - (-a)*(-2a)) / denom = 0 / denom = 0
    // v = ((-a)*(0) - (-a)*(-2a)) / denom = -2a^2 / 4a^2 = -0.5 -> ERROR in manual calc or setup?

    // Let's rethink the geometry or simplify. Place target at (0,0).
    // n1=(0, a), n2=(-a, -a), n3=(a, -a). Depths D1, D2, D3.
    // Target is inside. v21=(-a, -2a), v31=(a, -2a). TargetOffset=(0, -a) relative to v1=(0,a).
    // Denom = (-a)*(-2a) - (a)*(-2a) = 2a^2 + 2a^2 = 4a^2
    // u = (0*(-2a) - (-a)*(a)) / denom = a^2 / 4a^2 = 0.25
    // v = ((-a)*(-a) - 0*(-2a)) / denom = a^2 / 4a^2 = 0.25
    // w = 1 - u - v = 1 - 0.25 - 0.25 = 0.5
    // Interpolated Depth = 0.5*D1 + 0.25*D2 + 0.25*D3

    managed_points.clear();
    map_info.Reset(M3D::Identity(), V3D::Zero(), 0.0, 0);

    // Let's use this simpler setup:
    V3D p_glob_simple(10.0, 0.0, 0.0);
    auto p_target_simple = createManagedTestPoint(p_glob_simple, 0.1); // az=0, el=0
    
    float angle_s = params.ver_resolution_max; // Use vertical resolution for simplicity
    V3D n1s_glob(10.0, 0.0, 10.0 * tan(angle_s));   // (az=0, el=+a), Depth D1=11
    V3D n2s_glob(10.0, 10.0 * tan(-angle_s), 10.0 * tan(-angle_s)); // (az=-a, el=-a), Depth D2=12
    V3D n3s_glob(10.0, 10.0 * tan(angle_s), 10.0 * tan(-angle_s));  // (az=+a, el=-a), Depth D3=13
    // Add a 4th point that forms another triangle but is likely further away
    V3D n4s_glob(10.0, 0.0, 10.0 * tan(2.0f * angle_s)); // (az=0, el=+2a), Depth D4=14
    
    auto n1s = createManagedTestPoint(n1s_glob, 0.0); n1s->vec[2] = 11.0f;
    auto n2s = createManagedTestPoint(n2s_glob, 0.0); n2s->vec[2] = 12.0f;
    auto n3s = createManagedTestPoint(n3s_glob, 0.0); n3s->vec[2] = 13.0f;
    auto n4s = createManagedTestPoint(n4s_glob, 0.0); n4s->vec[2] = 14.0f; // Further away
    
    addPointToMap(n1s);
    addPointToMap(n2s);
    addPointToMap(n3s);
    addPointToMap(n4s); // Add n4s too

    // Triangle (n1s, n2s, n3s) should contain the target (0,0).
    // dist(T, n2s) = |-a-0| + |-a-0| = 2a
    // dist(T, n3s) = |+a-0| + |-a-0| = 2a
    // Sum = 4a (relative to n1s)
    // Triangle (n1s, n2s, n4s) - T is likely on edge or outside.
    // Triangle (n1s, n3s, n4s) - T is likely on edge or outside.
    // Triangle (n2s, n3s, n4s) - T is outside.
    // So, (n1s, n2s, n3s) should be chosen.

    float expected_depth = 0.5f * n1s->vec.z() + 0.25f * n2s->vec.z() + 0.25f * n3s->vec.z();
    // expected_depth = 0.5*11 + 0.25*12 + 0.25*13 = 5.5 + 3.0 + 3.25 = 11.75

    PointCloudUtils::InterpolationResult result = PointCloudUtils::interpolateDepth(
        *p_target_simple, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
    
    // ASSERT_EQ(result.status, PointCloudUtils::InterpolationStatus::SUCCESS); // Check status first
    if (result.status != PointCloudUtils::InterpolationStatus::SUCCESS) {
        FAIL() << "MultipleValidTriangles: Expected SUCCESS but got status " << static_cast<int>(result.status);
    }
    ASSERT_NEAR(result.depth, expected_depth, 0.1f); // Line 704 - Now check depth
}


TEST_F(InterpolationTest, BarycentricEpsilonChecks)
{
    V3D p_glob(10.0, 0.0, 0.0);
    auto p_target = createManagedTestPoint(p_glob, 0.1); // Target az=0, el=0

    float base_angle = params.hor_resolution_max * 0.5f;
    float tiny_offset = PointCloudUtils::BARYCENTRIC_EPSILON / (base_angle + 1e-9); // Offset to control determinant

    // --- Test Collinearity Epsilon ---
    {
        // Case 1: Almost collinear (Determinant slightly > epsilon)
        V3D n1_ac_glob(10.0, 10.0 * tan(-base_angle), 0.0); // (-a, 0)
        V3D n2_ac_glob(10.0, 0.0, 10.0 * tan(tiny_offset)); // (0, tiny_el) - slightly off line
        V3D n3_ac_glob(10.0, 10.0 * tan(base_angle), 0.0);  // (+a, 0)

        auto n1_ac = createManagedTestPoint(n1_ac_glob, 0.0); n1_ac->vec[2]=10.f;
        auto n2_ac = createManagedTestPoint(n2_ac_glob, 0.0); n2_ac->vec[2]=10.f;
        auto n3_ac = createManagedTestPoint(n3_ac_glob, 0.0); n3_ac->vec[2]=10.f;
        addPointToMap(n1_ac); addPointToMap(n2_ac); addPointToMap(n3_ac);

        PointCloudUtils::InterpolationResult result_ac = PointCloudUtils::interpolateDepth(
            *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
        // Should succeed as determinant is > epsilon
        ASSERT_EQ(result_ac.status, PointCloudUtils::InterpolationStatus::SUCCESS);

        managed_points.clear();
        map_info.Reset(M3D::Identity(), V3D::Zero(), 0.0, 0); // Reset map for next case

        // Case 2: Very collinear (Determinant < epsilon)
        V3D n1_vc_glob(10.0, 10.0 * tan(-base_angle), 0.0); // (-a, 0)
        V3D n2_vc_glob(10.0, 0.0, 0.0);                     // (0, 0) - exactly on line
        V3D n3_vc_glob(10.0, 10.0 * tan(base_angle), 0.0);  // (+a, 0)

        auto n1_vc = createManagedTestPoint(n1_vc_glob, 0.0);
        auto n2_vc = createManagedTestPoint(n2_vc_glob, 0.0);
        auto n3_vc = createManagedTestPoint(n3_vc_glob, 0.0);
        addPointToMap(n1_vc); addPointToMap(n2_vc); addPointToMap(n3_vc);

        PointCloudUtils::InterpolationResult result_vc = PointCloudUtils::interpolateDepth(
            *p_target, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
        // Should fail as determinant is ~0 (< epsilon)
        ASSERT_EQ(result_vc.status, PointCloudUtils::InterpolationStatus::NO_VALID_TRIANGLE);

        managed_points.clear();
        map_info.Reset(M3D::Identity(), V3D::Zero(), 0.0, 0); // Reset map for next case
    }

    // --- Test Inside/Outside Epsilon ---
    {
        // Create a triangle
        V3D n1_io_glob(10.0, 0.0, 10.0 * tan(base_angle));   // (0, +a)
        V3D n2_io_glob(10.0, 10.0 * tan(-base_angle), 10.0 * tan(-base_angle)); // (-a, -a)
        V3D n3_io_glob(10.0, 10.0 * tan(base_angle), 10.0 * tan(-base_angle));  // (+a, -a)
        auto n1_io = createManagedTestPoint(n1_io_glob, 0.0);
        auto n2_io = createManagedTestPoint(n2_io_glob, 0.0);
        auto n3_io = createManagedTestPoint(n3_io_glob, 0.0);
        addPointToMap(n1_io); addPointToMap(n2_io); addPointToMap(n3_io);

        // Case 1: Target just outside (one bary coord slightly < BARY_CHECK_EPSILON)
        // Place target slightly below the bottom edge n2-n3 (el = -a - tiny)
        float tiny_el_out = -base_angle - 1e-6; // Adjust epsilon if needed based on BARY_CHECK_EPSILON
        V3D p_glob_out(10.0, 0.0, 10.0 * tan(tiny_el_out));
        auto p_target_out = createManagedTestPoint(p_glob_out, 0.1);

        PointCloudUtils::InterpolationResult result_out = PointCloudUtils::interpolateDepth(
            *p_target_out, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
        ASSERT_EQ(result_out.status, PointCloudUtils::InterpolationStatus::NO_VALID_TRIANGLE);

        // Case 2: Target just inside (all bary coords >= BARY_CHECK_EPSILON)
        // Place target slightly above the bottom edge (el = -a + tiny)
        float tiny_el_in = -base_angle + 1e-7; // Ensure > BARY_CHECK_EPSILON (-1e-5)
        V3D p_glob_in(10.0, 0.0, 10.0 * tan(tiny_el_in));
        auto p_target_in = createManagedTestPoint(p_glob_in, 0.1);
        // Need to remove the old target from managed points if createManagedTestPoint adds it
        managed_points.pop_back(); // Remove p_target_out if added by helper

        PointCloudUtils::InterpolationResult result_in = PointCloudUtils::interpolateDepth(
            *p_target_in, map_info, params, PointCloudUtils::InterpolationNeighborType::ALL_VALID);
        ASSERT_EQ(result_in.status, PointCloudUtils::InterpolationStatus::SUCCESS);
    }
}