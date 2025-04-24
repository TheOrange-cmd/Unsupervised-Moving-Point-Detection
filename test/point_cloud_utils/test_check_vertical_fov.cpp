#include "gtest/gtest.h" // Google Test framework
#include "point_cloud_utils.h" // header to test
#include "config_loader.h"     // For DynObjFilterParams
#include "dyn_obj_datatypes.h" // For point_soph, V3D, M3D etc.
#include <Eigen/Geometry>      // For Eigen::AngleAxisd
#include <vector>
#include <memory> // For managing point_soph pointers if needed

// --- Test Fixture ---
// Sets up common objects needed for the tests
class VerticalFovTest : public ::testing::Test {
protected:
    // Use constants consistent with dyn_obj_datatypes.h
    // These might need adjustment if your actual values differ significantly,
    // but the relative logic should hold. Using smaller values for testing clarity.
    // IMPORTANT: Ensure these match or are consistent with the actual MAX values
    // used when compiling point_cloud_utils.cpp, otherwise index calculations will fail.
    // If MAX_ values are large, tests might be slow or memory-intensive if DepthMap allocates MAX_2D_N.
    // Consider using smaller, representative values *if* the logic doesn't strictly depend on the exact large number.
    // For this example, we assume the real MAX values are used.
    static constexpr int TEST_MAX_1D = MAX_1D; // e.g., 1257
    static constexpr int TEST_MAX_1D_HALF = MAX_1D_HALF; // e.g., 449
    static constexpr int TEST_MAX_2D_N = MAX_2D_N; // e.g., 564393

    DynObjFilterParams params;
    DepthMap map_info; // Creates the map with size MAX_2D_N

    // Dummy point_soph instance to add to the map vectors.
    // We only care if the vector is empty or not, not the content.
    // Using a static instance avoids repeated allocation/deallocation.
    // Note: If DepthMap stores actual objects, adjust accordingly.
    // If it stores raw pointers, this needs careful memory management.
    // Assuming DepthMap stores std::vector<point_soph*> as per the definition provided.
    // We'll create a dummy object ONCE and use its address. This is okay for testing
    // emptiness, but DO NOT DEREFERENCE this pointer in the actual function.
    inline static point_soph dummy_point_object; // Create one dummy object
    inline static point_soph* dummy_point_ptr = &dummy_point_object; // Pointer to the dummy object

    // Helper to create a point_soph with specific indices
    // Note: The V3F vector part might not be strictly needed if only indices are used
    point_soph createTestPoint(int hor_idx, int ver_idx) {
        point_soph p;
        p.hor_ind = hor_idx;
        p.ver_ind = ver_idx;
        // Calculate position based on indices - crucial for some functions, maybe not this one
        p.position = hor_idx * TEST_MAX_1D_HALF + ver_idx;
        // Other members can remain default
        return p;
    }

    // Helper to simulate adding a point to the depth map at specific indices
    void addPointToMap(int hor_idx, int ver_idx) {
        if (hor_idx < 0 || hor_idx >= TEST_MAX_1D || ver_idx < 0 || ver_idx >= TEST_MAX_1D_HALF) {
            // Invalid indices, don't add
            return;
        }
        int map_pos = hor_idx * TEST_MAX_1D_HALF + ver_idx;
        if (map_pos >= 0 && map_pos < TEST_MAX_2D_N) {
            // Add the dummy pointer to indicate non-emptiness
            map_info.depth_map[map_pos].push_back(dummy_point_ptr);
        } else {
           // Calculated position is out of bounds for the map
           // This indicates an issue with TEST_MAX values or calculation
           // You might want to throw an error or log here in a real test setup
        }
    }

    void SetUp() override {
        // Initialize default parameters before each test
        params.pixel_fov_down = 5; // Default reasonable lower limit
        params.pixel_fov_up = TEST_MAX_1D_HALF - 5; // Default reasonable upper limit

        // Ensure map is clear (DepthMap constructor should handle initial state,
        // but explicit reset might be needed if tests modify it significantly
        // and it's not recreated per test). The fixture recreates map_info per test.
    }

    // void TearDown() override {} // No explicit cleanup needed for these members
};

// --- Test Cases ---

TEST_F(VerticalFovTest, HasSupportBothWays) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = test_ver - 10;
    params.pixel_fov_up = test_ver + 10;

    addPointToMap(test_hor, test_ver - 5); // Add support below
    addPointToMap(test_hor, test_ver + 5); // Add support above

    // Expect false because support exists both ways
    ASSERT_FALSE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return false when support exists above and below within FoV.";
}

TEST_F(VerticalFovTest, HasSupportBelowOnly) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = test_ver - 10;
    params.pixel_fov_up = test_ver + 10;

    addPointToMap(test_hor, test_ver - 5); // Add support below
    // No support above

    // Expect true because support above is missing
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when support exists only below within FoV.";
}

TEST_F(VerticalFovTest, HasSupportAboveOnly) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = test_ver - 10;
    params.pixel_fov_up = test_ver + 10;

    // No support below
    addPointToMap(test_hor, test_ver + 5); // Add support above

    // Expect true because support below is missing
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when support exists only above within FoV.";
}

TEST_F(VerticalFovTest, HasNoSupport) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = test_ver - 10;
    params.pixel_fov_up = test_ver + 10;

    // No support below or above

    // Expect true because support is missing both ways
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when no support exists above or below within FoV.";
}

TEST_F(VerticalFovTest, SupportOutsideFovLimits) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    // Set narrow FoV limits
    params.pixel_fov_down = test_ver - 5;
    params.pixel_fov_up = test_ver + 5;

    // Add support, but outside the narrow limits
    addPointToMap(test_hor, test_ver - 10);
    addPointToMap(test_hor, test_ver + 10);

    // Expect true because the support found is outside the specified FoV
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when support exists but outside the FoV limits.";
}

TEST_F(VerticalFovTest, SupportAtFovLimits) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = test_ver - 10;
    params.pixel_fov_up = test_ver + 10;

    // Add support exactly at the limits
    addPointToMap(test_hor, params.pixel_fov_down);
    addPointToMap(test_hor, params.pixel_fov_up);

    // Expect false because support at the boundary is included
    ASSERT_FALSE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return false when support exists exactly at the FoV limits.";
}


TEST_F(VerticalFovTest, PointAtBottomEdge) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = 0; // Point at the bottom
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = 0; // Limit includes the bottom edge
    params.pixel_fov_up = test_ver + 10;

    // Add support above
    addPointToMap(test_hor, test_ver + 5);
    // Cannot have support below index 0

    // Expect true because support below is impossible/missing
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when point is at bottom edge (no support below possible).";

    // Test case where support above is also missing
    DepthMap empty_map; // Create a new empty map for this sub-case
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, empty_map, params))
        << "Should return true when point is at bottom edge and no support above either.";
}

TEST_F(VerticalFovTest, PointAtTopEdge) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF - 1; // Point at the top
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = test_ver - 10;
    params.pixel_fov_up = TEST_MAX_1D_HALF - 1; // Limit includes the top edge

    // Add support below
    addPointToMap(test_hor, test_ver - 5);
    // Cannot have support above top index

    // Expect true because support above is impossible/missing
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when point is at top edge (no support above possible).";

    // Test case where support below is also missing
    DepthMap empty_map; // Create a new empty map for this sub-case
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, empty_map, params))
        << "Should return true when point is at top edge and no support below either.";
}

TEST_F(VerticalFovTest, FovLimitsClamping) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    // Set limits outside the valid index range [0, MAX_1D_HALF - 1]
    params.pixel_fov_down = -20;
    params.pixel_fov_up = TEST_MAX_1D_HALF + 50;

    // Add support within the valid range, near the actual edges
    addPointToMap(test_hor, 1); // Near bottom
    addPointToMap(test_hor, TEST_MAX_1D_HALF - 2); // Near top

    // Expect false, because the function should clamp the limits to [0, MAX_1D_HALF-1]
    // and find the support within those clamped limits.
    ASSERT_FALSE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return false when FoV limits are outside valid range but support exists within clamped range.";

    // Test again, but only add support below
    DepthMap map_below_only;
    // Need to add the dummy point helper to the fixture or make it static/global if used like this
    // For simplicity, let's reuse the fixture's map and clear it (or rely on fixture setup)
    map_info.Reset(M3D::Identity(), V3D::Zero(), 0.0, 0); // Assuming Reset clears the map vectors
    addPointToMap(test_hor, 1); // Near bottom only
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when FoV limits are clamped and only support below exists.";

}

TEST_F(VerticalFovTest, EmptyMap) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = 0;
    params.pixel_fov_up = TEST_MAX_1D_HALF - 1;

    // map_info is empty by default in the fixture setup (or use DepthMap empty_map;)

    // Expect true because no support exists anywhere
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when the depth map is completely empty.";
}

TEST_F(VerticalFovTest, SupportInDifferentColumn) {
    int test_hor = TEST_MAX_1D / 2;
    int test_ver = TEST_MAX_1D_HALF / 2;
    point_soph p = createTestPoint(test_hor, test_ver);

    params.pixel_fov_down = test_ver - 10;
    params.pixel_fov_up = test_ver + 10;

    // Add support above and below, but in a DIFFERENT horizontal column
    addPointToMap(test_hor + 1, test_ver - 5);
    addPointToMap(test_hor + 1, test_ver + 5);

    // Expect true because support must be in the *same* column (p.hor_ind)
    ASSERT_TRUE(PointCloudUtils::checkVerticalFov(p, map_info, params))
        << "Should return true when support exists but in a different horizontal column.";
}