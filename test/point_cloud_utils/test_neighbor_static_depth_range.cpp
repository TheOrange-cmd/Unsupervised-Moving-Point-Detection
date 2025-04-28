#include <vector>
#include <limits>   // For std::numeric_limits
#include <cfloat>   // For FLT_MAX
#include "gtest/gtest.h" // Google Test framework
#include "point_cloud_utils/point_cloud_utils.h" // Your header to test
#include "config/config_loader.h"     // For DynObjFilterParams
#include "filtering/dyn_obj_datatypes.h" // For point_soph, V3D, M3D etc.
#include <Eigen/Geometry>      // For Eigen::AngleAxisd

// --- Test Fixture ---
class NeighborDepthTest : public ::testing::Test {
protected:
    // Use constants consistent with dyn_obj_datatypes.h
    // IMPORTANT: Ensure these match the actual MAX values used when compiling point_cloud_utils.cpp
    static constexpr int TEST_MAX_1D = MAX_1D;
    static constexpr int TEST_MAX_1D_HALF = MAX_1D_HALF;
    static constexpr int TEST_MAX_2D_N = MAX_2D_N;

    DynObjFilterParams params;
    DepthMap map_info; // Creates the map with size MAX_2D_N

    inline static point_soph dummy_point_object;

    // Define a deleter that does nothing (since the object is static)
    inline static auto dummy_deleter = [](point_soph* ptr) {
        // No-op: Don't delete the static object!
        (void)ptr; // Avoid unused parameter warning if compiler is strict
    };
    
    // Create a shared_ptr pointing to the static object using the no-op deleter
    // Let's use a more descriptive name like sptr (shared pointer)
    inline static std::shared_ptr<point_soph> dummy_point_sptr{ &dummy_point_object, dummy_deleter };

    // Helper to create a point_soph with specific indices
    point_soph createTestPoint(int hor_idx, int ver_idx) {
        point_soph p;
        p.hor_ind = hor_idx;
        p.ver_ind = ver_idx;
        // Position calculation might not be strictly needed for this test
        // but good practice if other functions use it.
        if (hor_idx >= 0 && hor_idx < TEST_MAX_1D && ver_idx >= 0 && ver_idx < TEST_MAX_1D_HALF) {
             p.position = hor_idx * TEST_MAX_1D_HALF + ver_idx;
        } else {
             p.position = -1; // Indicate invalid index via position if needed
        }
        return p;
    }

    // Helper to simulate adding static depth data to the depth map at specific indices
    void addStaticDataToMap(int hor_idx, int ver_idx, float min_d, float max_d) {
        if (hor_idx < 0 || hor_idx >= TEST_MAX_1D || ver_idx < 0 || ver_idx >= TEST_MAX_1D_HALF) {
            return; // Invalid indices
        }
        int map_pos = hor_idx * TEST_MAX_1D_HALF + ver_idx;
        if (map_pos >= 0 && map_pos < TEST_MAX_2D_N) {
            // Mark cell as non-empty
            if (map_info.depth_map[map_pos].empty()) { // Avoid adding duplicates if called multiple times
                 map_info.depth_map[map_pos].push_back(dummy_point_sptr);
            }
            // Set static depth values
            map_info.min_depth_static[map_pos] = min_d;
            map_info.max_depth_static[map_pos] = max_d;
        }
         // else: Calculated position is out of bounds - indicates issue with TEST_MAX values or calculation
    }

    void SetUp() override {
        // Initialize default parameters before each test
        params.checkneighbor_range = 1; // Default range

        // Reset map_info (constructor should init, but explicit clear might be needed if tests modify heavily)
        // Assuming DepthMap constructor initializes vectors and pointers correctly.
        // If DepthMap::Reset exists and is safe, could call it here.
        // For simplicity, we rely on the fixture recreating map_info each time.
    }

    // void TearDown() override {} // No explicit cleanup needed
};

// --- Test Cases ---

TEST_F(NeighborDepthTest, FindsCorrectDepthsInRange1) {
    params.checkneighbor_range = 1;
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    // Add data to neighbors (including center)
    addStaticDataToMap(center_hor, center_ver, 5.0f, 6.0f);       // Center
    addStaticDataToMap(center_hor - 1, center_ver, 4.0f, 7.0f);   // Left
    addStaticDataToMap(center_hor + 1, center_ver, 6.0f, 8.0f);   // Right
    addStaticDataToMap(center_hor, center_ver - 1, 3.0f, 9.0f);   // Below
    addStaticDataToMap(center_hor, center_ver + 1, 7.0f, 10.0f);  // Above
    addStaticDataToMap(center_hor - 1, center_ver - 1, 2.5f, 6.5f); // Diagonal

    // Add a point outside the range (should be ignored)
    addStaticDataToMap(center_hor + 2, center_ver, 1.0f, 2.0f);

    float min_depth_out = -1.0f; // Use initial values different from expected
    float max_depth_out = -1.0f;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, 2.5f) << "Minimum depth mismatch.";
    ASSERT_FLOAT_EQ(max_depth_out, 10.0f) << "Maximum depth mismatch.";
}

TEST_F(NeighborDepthTest, FindsCorrectDepthsInRange0) {
    params.checkneighbor_range = 0; // Only check the center cell
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    // Add data only to the center
    addStaticDataToMap(center_hor, center_ver, 5.5f, 6.5f);

    // Add data outside the range (should be ignored)
    addStaticDataToMap(center_hor - 1, center_ver, 1.0f, 2.0f);
    addStaticDataToMap(center_hor, center_ver + 1, 8.0f, 9.0f);

    float min_depth_out = -1.0f;
    float max_depth_out = -1.0f;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, 5.5f) << "Minimum depth mismatch for range 0.";
    ASSERT_FLOAT_EQ(max_depth_out, 6.5f) << "Maximum depth mismatch for range 0.";
}

TEST_F(NeighborDepthTest, NoNeighborsFound) {
    params.checkneighbor_range = 1;
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    // Map is empty, no neighbors added

    // Initialize with specific values to check they aren't changed
    float initial_min = 999.0f;
    float initial_max = -999.0f;
    float min_depth_out = initial_min;
    float max_depth_out = initial_max;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, initial_min) << "Min depth changed unexpectedly when no neighbors found.";
    ASSERT_FLOAT_EQ(max_depth_out, initial_max) << "Max depth changed unexpectedly when no neighbors found.";
}

TEST_F(NeighborDepthTest, NeighborsExistButOutsideRange) {
    params.checkneighbor_range = 1;
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    // Add data, but only outside the n=1 range
    addStaticDataToMap(center_hor + 2, center_ver, 1.0f, 2.0f);
    addStaticDataToMap(center_hor - 2, center_ver, 3.0f, 4.0f);
    addStaticDataToMap(center_hor, center_ver + 2, 5.0f, 6.0f);
    addStaticDataToMap(center_hor, center_ver - 2, 7.0f, 8.0f);

    float initial_min = 123.4f;
    float initial_max = 567.8f;
    float min_depth_out = initial_min;
    float max_depth_out = initial_max;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, initial_min) << "Min depth changed when neighbors were outside range.";
    ASSERT_FLOAT_EQ(max_depth_out, initial_max) << "Max depth changed when neighbors were outside range.";
}

TEST_F(NeighborDepthTest, PointAtTopLeftCorner) {
    params.checkneighbor_range = 1;
    int center_hor = 0; // Left edge
    int center_ver = 0; // Bottom edge (ver_ind 0)
    point_soph p = createTestPoint(center_hor, center_ver);

    // Add data to the valid neighbors near the corner
    addStaticDataToMap(0, 0, 10.0f, 11.0f); // Center (corner)
    addStaticDataToMap(1, 0, 12.0f, 13.0f); // Right
    addStaticDataToMap(0, 1, 14.0f, 15.0f); // Above
    addStaticDataToMap(1, 1, 16.0f, 17.0f); // Diagonal Up-Right

    // Add data outside bounds (should be ignored by helper or function)
    addStaticDataToMap(-1, 0, 1.0f, 2.0f);
    addStaticDataToMap(0, -1, 3.0f, 4.0f);

    float min_depth_out = -1.0f;
    float max_depth_out = -1.0f;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, 12.0f) << "Minimum depth mismatch at corner (neighbors only)."; 
    ASSERT_FLOAT_EQ(max_depth_out, 17.0f) << "Maximum depth mismatch at corner (neighbors only)."; 
}

TEST_F(NeighborDepthTest, PointAtBottomRightCornerWithWrap) { // Renamed test
    params.checkneighbor_range = 1;
    int center_hor = TEST_MAX_1D - 1;
    int center_ver = TEST_MAX_1D_HALF - 1;
    point_soph p = createTestPoint(center_hor, center_ver);

    // Neighbors checked: (MAX-2, MAX_HALF-2), (MAX-2, MAX_HALF-1), (MAX-1, MAX_HALF-2)
    // Also checks wrapped indices (0, MAX_HALF-2), (0, MAX_HALF-1) - add data there if needed.
    addStaticDataToMap(center_hor, center_ver, 20.0f, 21.0f);           // Center (ignored by func)
    addStaticDataToMap(center_hor - 1, center_ver, 22.0f, 23.0f);       // Left
    addStaticDataToMap(center_hor, center_ver - 1, 24.0f, 25.0f);       // Below
    addStaticDataToMap(center_hor - 1, center_ver - 1, 26.0f, 27.0f);   // Diagonal Down-Left
    // Add data across wrap boundary if wrap behavior is important for this test
    // addStaticDataToMap(0, center_ver, 30.0f, 31.0f); // Example wrapped neighbor

    float min_depth_out = -1.0f;
    float max_depth_out = -1.0f;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    // Neighbors considered: Left(22), Below(24), Diag(26) (+ wrapped if added)
    ASSERT_FLOAT_EQ(min_depth_out, 22.0f) << "Minimum depth mismatch at corner (neighbors only)."; 
    ASSERT_FLOAT_EQ(max_depth_out, 27.0f) << "Maximum depth mismatch at corner (neighbors only).";
}

TEST_F(NeighborDepthTest, IgnoresEmptyNeighborCells) {
    params.checkneighbor_range = 1;
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    // Add data to *some* neighbors, leave others empty
    addStaticDataToMap(center_hor - 1, center_ver, 4.0f, 7.0f);   // Left (valid)
    // Right cell (center_hor + 1, center_ver) is intentionally left empty
    addStaticDataToMap(center_hor, center_ver - 1, 3.0f, 9.0f);   // Below (valid)
    // Above cell (center_hor, center_ver + 1) is intentionally left empty

    float min_depth_out = -1.0f;
    float max_depth_out = -1.0f;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, 3.0f) << "Minimum depth mismatch when some neighbors are empty.";
    ASSERT_FLOAT_EQ(max_depth_out, 9.0f) << "Maximum depth mismatch when some neighbors are empty.";
}


TEST_F(NeighborDepthTest, HandlesZeroRangeParameter) {
    params.checkneighbor_range = 0; // Explicitly test 0
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    addStaticDataToMap(center_hor, center_ver, 5.5f, 6.5f);
    addStaticDataToMap(center_hor - 1, center_ver, 1.0f, 2.0f); // Should be ignored

    float min_depth_out = -1.0f;
    float max_depth_out = -1.0f;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, 5.5f);
    ASSERT_FLOAT_EQ(max_depth_out, 6.5f);
}

TEST_F(NeighborDepthTest, HandlesNegativeRangeParameter) {
    params.checkneighbor_range = -5; // Should be treated as 0 by the function
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    addStaticDataToMap(center_hor, center_ver, 5.5f, 6.5f);
    addStaticDataToMap(center_hor - 1, center_ver, 1.0f, 2.0f); // Should be ignored

    float min_depth_out = -1.0f;
    float max_depth_out = -1.0f;

    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out, max_depth_out);

    ASSERT_FLOAT_EQ(min_depth_out, 5.5f) << "Min depth wrong for negative range.";
    ASSERT_FLOAT_EQ(max_depth_out, 6.5f) << "Max depth wrong for negative range.";
}

TEST_F(NeighborDepthTest, InvalidInputPointIndices) {
    params.checkneighbor_range = 1;
    // Create point with invalid indices
    point_soph p_invalid_hor = createTestPoint(-5, TEST_MAX_1D_HALF / 2);
    point_soph p_invalid_ver = createTestPoint(TEST_MAX_1D / 2, TEST_MAX_1D_HALF + 10);

    // Add some data to the map (shouldn't be accessed)
    addStaticDataToMap(TEST_MAX_1D / 2, TEST_MAX_1D_HALF / 2, 10.0f, 20.0f);

    float initial_min = 11.1f;
    float initial_max = 22.2f;
    float min_depth_out = initial_min;
    float max_depth_out = initial_max;

    // Test with invalid horizontal index
    PointCloudUtils::findNeighborStaticDepthRange(p_invalid_hor, map_info, params, min_depth_out, max_depth_out);
    ASSERT_FLOAT_EQ(min_depth_out, initial_min) << "Min depth changed for invalid hor_ind.";
    ASSERT_FLOAT_EQ(max_depth_out, initial_max) << "Max depth changed for invalid hor_ind.";

    // Reset outputs and test with invalid vertical index
    min_depth_out = initial_min;
    max_depth_out = initial_max;
    PointCloudUtils::findNeighborStaticDepthRange(p_invalid_ver, map_info, params, min_depth_out, max_depth_out);
    ASSERT_FLOAT_EQ(min_depth_out, initial_min) << "Min depth changed for invalid ver_ind.";
    ASSERT_FLOAT_EQ(max_depth_out, initial_max) << "Max depth changed for invalid ver_ind.";
}

TEST_F(NeighborDepthTest, CorrectInitializationHandling) {
    params.checkneighbor_range = 1;
    int center_hor = TEST_MAX_1D / 2;
    int center_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(center_hor, center_ver);

    // Add one neighbor
    addStaticDataToMap(center_hor + 1, center_ver, 6.0f, 8.0f);   // Right

    // Test 1: Initialize with values larger/smaller than actuals
    float min_depth_out1 = 100.0f;
    float max_depth_out1 = 0.0f;
    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out1, max_depth_out1);
    ASSERT_FLOAT_EQ(min_depth_out1, 6.0f);
    ASSERT_FLOAT_EQ(max_depth_out1, 8.0f);

    // Test 2: Initialize with "zero" (common but potentially problematic if 0 is valid depth)
    float min_depth_out2 = 0.0f;
    float max_depth_out2 = 0.0f;
    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out2, max_depth_out2);
    ASSERT_FLOAT_EQ(min_depth_out2, 6.0f);
    ASSERT_FLOAT_EQ(max_depth_out2, 8.0f);

     // Test 3: Initialize with FLT_MAX / lowest()
    float min_depth_out3 = FLT_MAX;
    float max_depth_out3 = std::numeric_limits<float>::lowest(); // Or just 0.0f if depths are non-negative
    PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_depth_out3, max_depth_out3);
    ASSERT_FLOAT_EQ(min_depth_out3, 6.0f);
    ASSERT_FLOAT_EQ(max_depth_out3, 8.0f);
}