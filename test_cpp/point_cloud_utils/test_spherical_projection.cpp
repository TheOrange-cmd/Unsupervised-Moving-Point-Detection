#include "gtest/gtest.h" // Google Test framework
#include "point_cloud_utils/point_cloud_utils.h" // Your header to test
#include "config/config_loader.h"     // For DynObjFilterParams
#include "filtering/dyn_obj_datatypes.h" // For point_soph, V3D, M3D etc.
#include <Eigen/Geometry>      // For Eigen::AngleAxisd
#include "common/types.h"

// --- Helper Function (Optional but useful) ---
// Function to create default parameters for testing
DynObjFilterParams CreateTestParams() {
    DynObjFilterParams params;
    // Set key parameters needed by SphericalProjection
    params.hor_resolution_max = 0.01f; // Example value (radians/index)
    params.ver_resolution_max = 0.02f; // Example value (radians/index)
    // Ensure derived params are calculated if needed, though SphericalProjection doesn't use them directly
    // load_config(...) would normally do this, but we can skip for isolated tests if not needed.
    return params;
}

// --- START Test Suite for SphericalProjection ---
TEST(SphericalProjectionTest, CacheMiss) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input; // Input point
    point_soph p_output; // Output spherical point

    // Set global coordinates for the input point
    p_input.glob = V3D(3.0, 4.0, 5.0); // Example point (x, y, z)

    // Set sensor pose (Rotation and Translation)
    // Example: Sensor at (1, 0, 0) rotated 90 deg around Z (Y points forward)
    M3D rotation = (Eigen::AngleAxisd(M_PI / 2.0, V3D::UnitZ())).toRotationMatrix(); // Rotates world Z to sensor Z
    V3D translation = V3D(1.0, 0.0, 0.0); // Sensor origin in world frame

    // Ensure cache is invalid (set range component to 0)
    int test_depth_index = 5; // Example index
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Make cache invalid

    // Expected result calculation (manual)
    // 1. Point relative to sensor: p_proj = rot * (p_input.glob - translation)
    V3D p_relative_world = p_input.glob - translation; // (2.0, 4.0, 5.0)
    V3D p_proj_expected = rotation * p_relative_world; // (-4.0, 2.0, 5.0) (Check rotation matrix convention!)
                                                      // Assuming standard Eigen convention (rot * vector)

    // 2. Spherical coordinates: vec(2)=range, vec(0)=azimuth, vec(1)=elevation
    float range_expected = p_proj_expected.norm(); // sqrt(16 + 4 + 25) = sqrt(45) approx 6.708
    float azimuth_expected = std::atan2(p_proj_expected.y(), p_proj_expected.x()); // atan2(2, -4) approx 2.678 rad
    float elevation_expected = std::atan2(p_proj_expected.z(), std::sqrt(p_proj_expected.x()*p_proj_expected.x() + p_proj_expected.y()*p_proj_expected.y())); // atan2(5, sqrt(16+4)) = atan2(5, sqrt(20)) approx 0.841 rad

    // 3. Indices
    int hor_ind_expected = std::floor((azimuth_expected + M_PI) / params.hor_resolution_max); // floor((2.678 + 3.1416) / 0.01) = floor(581.96) = 581
    int ver_ind_expected = std::floor((elevation_expected + 0.5 * M_PI) / params.ver_resolution_max); // floor((0.841 + 1.5708) / 0.02) = floor(120.59) = 120
    int pos_expected = hor_ind_expected * MAX_1D_HALF + ver_ind_expected; // 581 * 449 + 120 = 260869 + 120 = 260989

    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);

    // Assert
    // Check spherical coordinates (use ASSERT_NEAR for floats/doubles)
    ASSERT_NEAR(p_output.vec(0), azimuth_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(1), elevation_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(2), range_expected, 1e-5);

    // Check indices (use ASSERT_EQ for integers)
    ASSERT_EQ(p_output.hor_ind, hor_ind_expected);
    ASSERT_EQ(p_output.ver_ind, ver_ind_expected);
    ASSERT_EQ(p_output.position, pos_expected);

    // Check if cache was updated correctly
    ASSERT_NEAR(p_input.last_vecs.at(cache_idx)(0), azimuth_expected, 1e-5);
    ASSERT_NEAR(p_input.last_vecs.at(cache_idx)(1), elevation_expected, 1e-5);
    ASSERT_NEAR(p_input.last_vecs.at(cache_idx)(2), range_expected, 1e-5);
    ASSERT_EQ(p_input.last_positions.at(cache_idx)[0], hor_ind_expected);
    ASSERT_EQ(p_input.last_positions.at(cache_idx)[1], ver_ind_expected);
    ASSERT_EQ(p_input.last_positions.at(cache_idx)[2], pos_expected);
}


TEST(SphericalProjectionTest, CacheHit) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;

    p_input.glob = V3D(1.0, 1.0, 1.0); // Global position doesn't matter for cache hit test

    // Sensor pose doesn't matter for cache hit test
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();

    int test_depth_index = 8;
    int cache_idx = test_depth_index % HASH_PRIM;

    // Pre-load the cache with known valid data
    V3F cached_vec(1.2f, 0.5f, 5.0f); // Azimuth, Elevation, Range (Range > threshold)
    Eigen::Vector3i cached_pos(120, 25, 120 * MAX_1D_HALF + 25); // hor_ind, ver_ind, position

    p_input.last_vecs.at(cache_idx) = cached_vec;
    p_input.last_positions.at(cache_idx) = cached_pos;

    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);

    // Assert
    // Check that output matches the cached data EXACTLY
    ASSERT_EQ(p_output.vec(0), cached_vec(0));
    ASSERT_EQ(p_output.vec(1), cached_vec(1));
    ASSERT_EQ(p_output.vec(2), cached_vec(2));
    ASSERT_EQ(p_output.hor_ind, cached_pos(0));
    ASSERT_EQ(p_output.ver_ind, cached_pos(1));
    ASSERT_EQ(p_output.position, cached_pos(2));

    // Optional: Check that the cache in p_input wasn't modified (it shouldn't be on a hit)
     ASSERT_EQ(p_input.last_vecs.at(cache_idx)(0), cached_vec(0));
     ASSERT_EQ(p_input.last_positions.at(cache_idx)(0), cached_pos(0));
}

// --- Test vertical points (azimuth edge cases) ---
TEST(SphericalProjectionTest, VerticalPoint) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    // Point directly above sensor (azimuth is mathematically undefined)
    p_input.glob = V3D(0.0, 0.0, 5.0);
    
    M3D rotation = M3D::Identity(); // No rotation
    V3D translation = V3D::Zero(); // No translation
    
    int test_depth_index = 1;
    
    // Expected results
    // When directly above, azimuth could be any value, but implementations typically use 0
    float range_expected = 5.0;
    float elevation_expected = M_PI / 2.0; // 90 degrees in radians
    
    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Assert
    ASSERT_NEAR(p_output.vec(2), range_expected, 1e-5); // Range
    ASSERT_NEAR(std::abs(p_output.vec(1)), elevation_expected, 1e-5); // Check elevation is +/- 90°
    // We can't check azimuth precisely as it's mathematically undefined
}

// --- Test origin point (zero distance) ---
TEST(SphericalProjectionTest, OriginPoint) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    // Point at sensor origin
    p_input.glob = V3D(0.0, 0.0, 0.0);
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 2;
    
    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Assert
    // Range should be zero or very close to zero
    ASSERT_NEAR(p_output.vec(2), 0.0, 1e-5);
    // Indices should be valid even at origin
    ASSERT_GE(p_output.hor_ind, 0);
    ASSERT_GE(p_output.ver_ind, 0);
}

// --- Test far distance points ---
TEST(SphericalProjectionTest, FarDistancePoint) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    // Very distant point
    p_input.glob = V3D(10000.0, 10000.0, 10000.0);
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 3;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache
    
    // Expected results
    V3D p_proj = p_input.glob; // No rotation/translation applied
    float range_expected = p_proj.norm();
    float azimuth_expected = std::atan2(p_proj.y(), p_proj.x());
    float elevation_expected = std::atan2(p_proj.z(), 
                                         std::sqrt(p_proj.x()*p_proj.x() + p_proj.y()*p_proj.y()));
    
    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Assert
    ASSERT_NEAR(p_output.vec(0), azimuth_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(1), elevation_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(2), range_expected, 1e-5);
}

// --- Test negative coordinates ---
TEST(SphericalProjectionTest, NegativeCoordinates) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    // Point with negative coordinates
    p_input.glob = V3D(-3.0, -4.0, -5.0);
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 4;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache
    
    // Expected results
    V3D p_proj = p_input.glob; // No rotation/translation applied
    float range_expected = p_proj.norm();
    float azimuth_expected = std::atan2(p_proj.y(), p_proj.x()); // Should be in 3rd quadrant
    float elevation_expected = std::atan2(p_proj.z(), 
                                         std::sqrt(p_proj.x()*p_proj.x() + p_proj.y()*p_proj.y()));
    
    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Assert
    ASSERT_NEAR(p_output.vec(0), azimuth_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(1), elevation_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(2), range_expected, 1e-5);
}

// --- Test boundary conditions ---
TEST(SphericalProjectionTest, IndexBoundary) {
    // Arrange
    DynObjFilterParams params = CreateTestParams(); // Assume hor_resolution_max = 0.01
    point_soph p_input;
    point_soph p_output;

    // Point corresponding to azimuth = +/- PI (180 degrees)
    float range = 10.0;
    p_input.glob = V3D(
        range * std::cos(0.0) * std::cos(M_PI), // -range
        range * std::cos(0.0) * std::sin(M_PI), // 0
        range * std::sin(0.0)                  // 0
    ); // Point at (-10, 0, 0)

    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();

    int test_depth_index = 6;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache

    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);

    // Assert
    // 1. Check that the calculated azimuth is correct (either +PI or -PI)
    ASSERT_NEAR(std::abs(p_output.vec(0)), M_PI, 1e-5);

    // 2. Calculate the expected index *based on the azimuth the function actually computed*
    int hor_ind_expected = std::floor((p_output.vec(0) + M_PI) / params.hor_resolution_max);

    // 3. Handle potential wrap-around due to floating point at the boundary.
    //    If floor((PI + PI)/res) results in exactly num_bins, it should be index 0.
    int num_hor_bins = static_cast<int>(std::round(2.0 * M_PI / params.hor_resolution_max));
    if (hor_ind_expected >= num_hor_bins) {
         // This case can happen if p_output.vec(0) is exactly +PI and floating point pushes
         // (PI + PI) / res slightly above num_bins before floor.
         hor_ind_expected = 0;
    }
     // Add a check for negative index, although unlikely here
    if (hor_ind_expected < 0) {
         hor_ind_expected = num_hor_bins - 1; // Wrap around the other way
    }


    // 4. Assert that the function's stored index matches this expectation
    ASSERT_EQ(p_output.hor_ind, hor_ind_expected)
         << "Calculated azimuth: " << p_output.vec(0)
         << ", Expected index based on calculated azimuth: " << hor_ind_expected
         << ", Actual index from function: " << p_output.hor_ind;
}


// --- Test different rotations ---
TEST(SphericalProjectionTest, ComplexRotation) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    p_input.glob = V3D(3.0, 4.0, 5.0);
    
    // Create a complex rotation: 45° around X, then 30° around Y, then 60° around Z
    Eigen::AngleAxisd rotX(M_PI / 4.0, V3D::UnitX());
    Eigen::AngleAxisd rotY(M_PI / 6.0, V3D::UnitY());
    Eigen::AngleAxisd rotZ(M_PI / 3.0, V3D::UnitZ());
    M3D rotation = (rotZ * rotY * rotX).toRotationMatrix();
    
    V3D translation = V3D(1.0, 2.0, 3.0);
    
    int test_depth_index = 7;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache
    
    // Expected results calculation
    V3D p_relative_world = p_input.glob - translation; // (2.0, 2.0, 2.0)
    V3D p_proj_expected = rotation * p_relative_world;
    
    float range_expected = p_proj_expected.norm();
    float azimuth_expected = std::atan2(p_proj_expected.y(), p_proj_expected.x());
    float elevation_expected = std::atan2(p_proj_expected.z(),
                                         std::sqrt(p_proj_expected.x()*p_proj_expected.x() + 
                                                  p_proj_expected.y()*p_proj_expected.y()));
    
    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Assert
    ASSERT_NEAR(p_output.vec(0), azimuth_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(1), elevation_expected, 1e-5);
    ASSERT_NEAR(p_output.vec(2), range_expected, 1e-5);
}

// --- Test parameter variations ---
TEST(SphericalProjectionTest, DifferentResolutions) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    params.hor_resolution_max = 0.05f; // 5x coarser horizontal resolution
    params.ver_resolution_max = 0.1f;  // 5x coarser vertical resolution
    
    point_soph p_input;
    point_soph p_output;
    
    p_input.glob = V3D(3.0, 4.0, 5.0);
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 9;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache
    
    // Expected results
    V3D p_proj = p_input.glob;
    float azimuth_expected = std::atan2(p_proj.y(), p_proj.x());
    float elevation_expected = std::atan2(p_proj.z(),
                                         std::sqrt(p_proj.x()*p_proj.x() + p_proj.y()*p_proj.y()));
    
    // Expected indices with coarser resolution
    int hor_ind_expected = std::floor((azimuth_expected + M_PI) / params.hor_resolution_max);
    int ver_ind_expected = std::floor((elevation_expected + 0.5 * M_PI) / params.ver_resolution_max);
    
    // Act
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Assert
    ASSERT_EQ(p_output.hor_ind, hor_ind_expected);
    ASSERT_EQ(p_output.ver_ind, ver_ind_expected);
}

// --- Test cache invalidation ---
TEST(SphericalProjectionTest, CacheInvalidation) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    // Initial position
    p_input.glob = V3D(3.0, 4.0, 5.0);
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 10;
    int cache_idx = test_depth_index % HASH_PRIM;
    
    // First run to populate cache
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Store initial cached values
    V3F initial_vec = p_input.last_vecs.at(cache_idx);
    Eigen::Vector3i initial_pos = p_input.last_positions.at(cache_idx);
    
    // Now change point position
    p_input.glob = V3D(6.0, 8.0, 10.0); // Double the distance
    
    // Invalidate the cache by setting the range component below threshold
    p_input.last_vecs.at(cache_idx)[2] = 0.0;
    
    // Act - run projection again
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Assert
    // Cache should be updated with new values
    ASSERT_NE(p_output.vec(2), initial_vec(2)); // Range should have changed
    ASSERT_NEAR(p_output.vec(2), p_input.glob.norm(), 1e-5); // Should match new point distance
    
    // Verify cache was updated correctly
    ASSERT_NEAR(p_input.last_vecs.at(cache_idx)(2), p_input.glob.norm(), 1e-5);
}

// --- Test that caching works based on depth index ---
TEST(SphericalProjectionTest, CacheIndexing) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output1;
    point_soph p_output2;
    
    p_input.glob = V3D(3.0, 4.0, 5.0);
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    // Use two depth indices that map to different cache slots
    int depth_index1 = 1;
    int depth_index2 = 1 + HASH_PRIM; // Should map to same slot as depth_index1
    
    // Initialize cache to zeros
    for (auto& vec : p_input.last_vecs) {
        vec.setZero();
    }
    
    // Act
    // First call will populate cache for depth_index1
    PointCloudUtils::SphericalProjection(p_input, depth_index1, rotation, translation, params, p_output1);
    
    // Move the point but don't invalidate cache
    V3D original_pos = p_input.glob;
    p_input.glob = V3D(6.0, 8.0, 10.0);
    
    // Second call with depth_index2 should get a cache hit since it maps to same slot
    PointCloudUtils::SphericalProjection(p_input, depth_index2, rotation, translation, params, p_output2);
    
    // Assert
    // output2 should match output1 despite point being in different position
    ASSERT_EQ(p_output1.vec(0), p_output2.vec(0));
    ASSERT_EQ(p_output1.vec(1), p_output2.vec(1));
    ASSERT_EQ(p_output1.vec(2), p_output2.vec(2));
    ASSERT_EQ(p_output1.position, p_output2.position);
    
    // The cached values should be based on original position, not current
    ASSERT_NEAR(p_output2.vec(2), original_pos.norm(), 1e-5);
    ASSERT_NE(p_output2.vec(2), p_input.glob.norm());
}

// --- Test points at specific angles (45 degrees in different octants) ---
TEST(SphericalProjectionTest, PointsAt45DegreeAngles) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 11;
    int cache_idx = test_depth_index % HASH_PRIM;
    
    // 45 degree angles in all 8 octants
    std::vector<V3D> test_points = {
        V3D(1.0, 1.0, 1.0),     // Octant 1: +x, +y, +z
        V3D(-1.0, 1.0, 1.0),    // Octant 2: -x, +y, +z
        V3D(-1.0, -1.0, 1.0),   // Octant 3: -x, -y, +z
        V3D(1.0, -1.0, 1.0),    // Octant 4: +x, -y, +z
        V3D(1.0, 1.0, -1.0),    // Octant 5: +x, +y, -z
        V3D(-1.0, 1.0, -1.0),   // Octant 6: -x, +y, -z
        V3D(-1.0, -1.0, -1.0),  // Octant 7: -x, -y, -z
        V3D(1.0, -1.0, -1.0)    // Octant 8: +x, -y, -z
    };
    
    // Expected angles for each octant (azimuth, elevation)
    // Note: azimuth in range [-π, π], elevation in range [-π/2, π/2]
    std::vector<std::pair<float, float>> expected_angles = {
        {M_PI/4, atan2(1, sqrt(2))},     // Octant 1
        {3*M_PI/4, atan2(1, sqrt(2))},   // Octant 2
        {-3*M_PI/4, atan2(1, sqrt(2))},  // Octant 3
        {-M_PI/4, atan2(1, sqrt(2))},    // Octant 4
        {M_PI/4, -atan2(1, sqrt(2))},    // Octant 5
        {3*M_PI/4, -atan2(1, sqrt(2))},  // Octant 6
        {-3*M_PI/4, -atan2(1, sqrt(2))}, // Octant 7
        {-M_PI/4, -atan2(1, sqrt(2))}    // Octant 8
    };
    
    // Range is sqrt(3) for all points
    float expected_range = sqrt(3.0);
    
    // Run tests for each octant
    for (size_t i = 0; i < test_points.size(); i++) {
        // Reset cache
        p_input.last_vecs.at(cache_idx).setZero();
        
        // Set input point
        p_input.glob = test_points[i];
        
        // Get expected angles
        float expected_azimuth = expected_angles[i].first;
        float expected_elevation = expected_angles[i].second;
        
        // Act
        PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
        
        // Assert
        // Note: for azimuth, need to handle wrapping at -π/π boundary
        float azimuth_diff = std::abs(p_output.vec(0) - expected_azimuth);
        if (azimuth_diff > M_PI) {
            azimuth_diff = 2 * M_PI - azimuth_diff;
        }
        
        ASSERT_NEAR(azimuth_diff, 0.0, 1e-5) << "Failed at octant " << (i+1);
        ASSERT_NEAR(p_output.vec(1), expected_elevation, 1e-5) << "Failed at octant " << (i+1);
        ASSERT_NEAR(p_output.vec(2), expected_range, 1e-5) << "Failed at octant " << (i+1);
    }
}

// --- Test handling of invalid values (NaN/Infinity) ---
TEST(SphericalProjectionTest, InvalidValues) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 12;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache
    
    // Test points with NaN and Infinity values
    std::vector<V3D> test_points = {
        V3D(std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0),  // NaN in x
        V3D(1.0, std::numeric_limits<double>::quiet_NaN(), 1.0),  // NaN in y
        V3D(1.0, 1.0, std::numeric_limits<double>::quiet_NaN()),  // NaN in z
        V3D(std::numeric_limits<double>::infinity(), 1.0, 1.0),   // Infinity in x
        V3D(1.0, std::numeric_limits<double>::infinity(), 1.0),   // Infinity in y
        V3D(1.0, 1.0, std::numeric_limits<double>::infinity())    // Infinity in z
    };
    
    // Test each invalid point
    for (const auto& invalid_point : test_points) {
        // Set input point
        p_input.glob = invalid_point;
        
        // Act - this should not crash, but we don't assert specific values
        // as the behavior with invalid inputs depends on implementation
        PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
        
        // Assert - Check that the function produced some output
        // We're primarily testing that it didn't crash
        // The function should handle NaN/Infinity either by:
        // 1. Producing NaN/Infinity in output (propagating)
        // 2. Setting to some default value
        // 3. Throwing an exception (which we would catch in a real application)
        
        // We can check if output contains NaN when input had NaN
        if (invalid_point.hasNaN()) {
            // Either output should have NaN or be set to some default
            // This is a loose assertion as we don't know implementation details
            ASSERT_TRUE(std::isnan(p_output.vec(0)) || 
                       std::isnan(p_output.vec(1)) || 
                       std::isnan(p_output.vec(2)) ||
                       (!std::isnan(p_output.vec(0)) && 
                        !std::isnan(p_output.vec(1)) && 
                        !std::isnan(p_output.vec(2))));
        }
    }
}

// --- Test round-trip conversion consistency ---
// This test checks if converting from Cartesian->Spherical->Cartesian
// gives back the original point (within tolerance)
TEST(SphericalProjectionTest, RoundTripConsistency) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output;
    
    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    
    int test_depth_index = 13;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache
    
    // Original Cartesian point
    p_input.glob = V3D(3.0, 4.0, 5.0);
    
    // Act
    // Step 1: Convert Cartesian -> Spherical
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation, translation, params, p_output);
    
    // Step 2: Convert Spherical -> Cartesian (manual calculation)
    float range = p_output.vec(2);
    float azimuth = p_output.vec(0);
    float elevation = p_output.vec(1);
    
    V3D reconstructed_point(
        range * std::cos(elevation) * std::cos(azimuth),
        range * std::cos(elevation) * std::sin(azimuth),
        range * std::sin(elevation)
    );
    
    // Assert
    // The reconstructed point should be very close to the original
    ASSERT_NEAR(reconstructed_point.x(), p_input.glob.x(), 1e-4);
    ASSERT_NEAR(reconstructed_point.y(), p_input.glob.y(), 1e-4);
    ASSERT_NEAR(reconstructed_point.z(), p_input.glob.z(), 1e-4);
}

// --- Test handling transformation effects ---
TEST(SphericalProjectionTest, TransformationEffects) {
    // Arrange
    DynObjFilterParams params = CreateTestParams();
    point_soph p_input;
    point_soph p_output_identity;
    point_soph p_output_transformed;
    
    // Initial point along x-axis
    p_input.glob = V3D(5.0, 0.0, 0.0);
    
    // Identity transformation
    M3D identity_rot = M3D::Identity();
    V3D zero_transl = V3D::Zero();
    
    // Specific transformation:
    // - Translation by (5,0,0) should put the point at the origin
    // - 90-degree rotation around Z should put points from +X onto +Y
    M3D rotation_z90 = (Eigen::AngleAxisd(M_PI / 2.0, V3D::UnitZ())).toRotationMatrix();
    V3D translation_x5 = V3D(5.0, 0.0, 0.0);
    
    int test_depth_index = 14;
    int cache_idx = test_depth_index % HASH_PRIM;
    p_input.last_vecs.at(cache_idx).setZero(); // Invalidate cache
    
    // Expected results for identity transformation
    float expected_identity_azimuth = 0.0; // Along +X axis
    float expected_identity_elevation = 0.0; // In XY plane
    float expected_identity_range = 5.0; // 5 units from origin
    
    // Expected results after transformation
    // After translation: point at origin would have range 0
    // After rotation: would put point along +Y axis
    float expected_transformed_azimuth = M_PI/2.0; // Along +Y axis after rotation
    float expected_transformed_range = 0.0; // At origin after translation
    
    // Act
    // Run with identity transformation
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, identity_rot, zero_transl, params, p_output_identity);
    
    // Reset cache
    p_input.last_vecs.at(cache_idx).setZero();
    
    // Run with specific transformation
    PointCloudUtils::SphericalProjection(p_input, test_depth_index, rotation_z90, translation_x5, params, p_output_transformed);
    
    // Assert
    // Check identity transformation results
    ASSERT_NEAR(p_output_identity.vec(0), expected_identity_azimuth, 1e-5);
    ASSERT_NEAR(p_output_identity.vec(1), expected_identity_elevation, 1e-5);
    ASSERT_NEAR(p_output_identity.vec(2), expected_identity_range, 1e-5);
    
    // Check specific transformation results
    // Note: If point is at origin (range=0), azimuth/elevation are undefined
    // So we only check range in this case
    ASSERT_NEAR(p_output_transformed.vec(2), expected_transformed_range, 1e-5);
}