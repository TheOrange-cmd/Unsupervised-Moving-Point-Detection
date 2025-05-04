#include "gtest/gtest.h"
#include "detection_logic/detection_logic.h" // Where findNeighborCells lives
#include "filtering/dyn_obj_datatypes.h"     // For point_soph, DepthMap
#include "config/config_loader.h"         // For DynObjFilterParams
#include "point_cloud_utils/point_cloud_utils.h" // For SphericalProjection
#include <vector>
#include <utility> // For std::pair
#include <set>     // Useful for comparing results regardless of order

// Helper function to create test parameters with specific map dimensions
DynObjFilterParams createNeighborTestParams(int width, int height, int range) {
    DynObjFilterParams params;
    params.map_width = width;
    params.map_height = height;
    params.checkneighbor_range = range;
    // Fill other necessary params with defaults if SphericalProjection needs them
    params.hor_resolution_min = 0.0; // Example default
    params.hor_resolution_max = 360.0;
    params.ver_resolution_min = -30.0;
    params.ver_resolution_max = 10.0;
    // ... other potentially relevant params ...
    return params;
}

// Define a test fixture if needed (e.g., to hold common params)
class FindNeighborCellsTest : public ::testing::Test {
protected:
    // You can set up common parameters here if many tests use the same ones
    // DynObjFilterParams default_params;
    // virtual void SetUp() override {
    //     default_params = createNeighborTestParams(100, 32, 1); // Example
    // }
};

// Helper to compare vectors of pairs ignoring order
// Converts vector to set for comparison
::testing::AssertionResult CompareNeighborSets(
    const std::vector<std::pair<int, int>>& expected,
    const std::vector<std::pair<int, int>>& actual)
{
    std::set<std::pair<int, int>> expected_set(expected.begin(), expected.end());
    std::set<std::pair<int, int>> actual_set(actual.begin(), actual.end());

    if (expected_set == actual_set) {
        return ::testing::AssertionSuccess();
    } else {
        // Construct a meaningful failure message (optional but helpful)
        ::testing::Message msg;
        msg << "Neighbor sets do not match.\nExpected: { ";
        for(const auto& p : expected_set) msg << "(" << p.first << "," << p.second << ") ";
        msg << "}\nActual: { ";
        for(const auto& p : actual_set) msg << "(" << p.first << "," << p.second << ") ";
        msg << "}";
        return ::testing::AssertionFailure() << msg;
    }
}

// Test Case 1: Basic Center Point, Range 1
TEST_F(FindNeighborCellsTest, CenterPointRange1) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 1); // 100 width, 32 height, range 1
    point_soph p;
    p.ver_ind = 15; // Center vertically
    p.hor_ind = 50; // Center horizontally

    std::vector<std::pair<int, int>> expected = {
        {14, 49}, {14, 50}, {14, 51},
        {15, 49}, {15, 50}, {15, 51},
        {16, 49}, {16, 50}, {16, 51}
    };

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);

    // Use helper for order-independent comparison
    EXPECT_TRUE(CompareNeighborSets(expected, actual));
    ASSERT_EQ(actual.size(), 9); // Ensure correct number of neighbors (3x3)
}

// Test Case 2: Range 0
TEST_F(FindNeighborCellsTest, CenterPointRange0) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 0); // Range 0
    point_soph p;
    p.ver_ind = 15;
    p.hor_ind = 50;

    std::vector<std::pair<int, int>> expected = { {15, 50} }; // Only the center cell

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);

    EXPECT_TRUE(CompareNeighborSets(expected, actual));
    ASSERT_EQ(actual.size(), 1);
}

// Test Case 3: Horizontal Wrap-around (Right Edge)
TEST_F(FindNeighborCellsTest, HorizontalWrapAroundRight) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 1); // Width 100
    point_soph p;
    p.ver_ind = 15;
    p.hor_ind = 99; // Rightmost horizontal index

    std::vector<std::pair<int, int>> expected = {
        {14, 98}, {14, 99}, {14, 0},  // Wraps around
        {15, 98}, {15, 99}, {15, 0},  // Wraps around
        {16, 98}, {16, 99}, {16, 0}   // Wraps around
    };

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);

    EXPECT_TRUE(CompareNeighborSets(expected, actual));
    ASSERT_EQ(actual.size(), 9);
}

// Test Case 4: Horizontal Wrap-around (Left Edge)
TEST_F(FindNeighborCellsTest, HorizontalWrapAroundLeft) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 1); // Width 100
    point_soph p;
    p.ver_ind = 15;
    p.hor_ind = 0; // Leftmost horizontal index

    std::vector<std::pair<int, int>> expected = {
        {14, 99}, {14, 0}, {14, 1}, // Wraps around (99)
        {15, 99}, {15, 0}, {15, 1}, // Wraps around (99)
        {16, 99}, {16, 0}, {16, 1}  // Wraps around (99)
    };

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);

    EXPECT_TRUE(CompareNeighborSets(expected, actual));
    ASSERT_EQ(actual.size(), 9);
}

// Test Case 5: Vertical Clamping (Top Edge)
TEST_F(FindNeighborCellsTest, VerticalClampTop) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 1); // Height 32
    point_soph p;
    p.ver_ind = 0; // Topmost vertical index
    p.hor_ind = 50;

    std::vector<std::pair<int, int>> expected = {
        // Row above (ver_ind = -1) is skipped
        {0, 49}, {0, 50}, {0, 51},
        {1, 49}, {1, 50}, {1, 51}
    };

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);

    EXPECT_TRUE(CompareNeighborSets(expected, actual));
    ASSERT_EQ(actual.size(), 6); // Should be 2x3 grid
}

// Test Case 6: Vertical Clamping (Bottom Edge)
TEST_F(FindNeighborCellsTest, VerticalClampBottom) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 1); // Height 32
    point_soph p;
    p.ver_ind = 31; // Bottommost vertical index
    p.hor_ind = 50;

    std::vector<std::pair<int, int>> expected = {
        {30, 49}, {30, 50}, {30, 51},
        {31, 49}, {31, 50}, {31, 51}
        // Row below (ver_ind = 32) is skipped
    };

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);

    EXPECT_TRUE(CompareNeighborSets(expected, actual));
    ASSERT_EQ(actual.size(), 6); // Should be 2x3 grid
}

// Test Case 7: Corner Case (Top-Left)
TEST_F(FindNeighborCellsTest, CornerTopLeft) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 1);
    point_soph p;
    p.ver_ind = 0;
    p.hor_ind = 0;

    std::vector<std::pair<int, int>> expected = {
        // Row above skipped
        {0, 99}, {0, 0}, {0, 1}, // Horizontal wrap
        {1, 99}, {1, 0}, {1, 1}  // Horizontal wrap
    };

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);

    EXPECT_TRUE(CompareNeighborSets(expected, actual));
    ASSERT_EQ(actual.size(), 6); // Should be 2x3
}


// Test Case 8: Invalid Input Indices
TEST_F(FindNeighborCellsTest, InvalidInputIndices) {
    DynObjFilterParams params = createNeighborTestParams(100, 32, 1);
    point_soph p;
    p.ver_ind = -5; // Invalid vertical
    p.hor_ind = 50;

    std::vector<std::pair<int, int>> actual = DetectionLogic::findNeighborCells(p, params);
    ASSERT_TRUE(actual.empty()); // Expect empty vector for invalid input

    p.ver_ind = 15;
    p.hor_ind = 150; // Invalid horizontal
    actual = DetectionLogic::findNeighborCells(p, params);
    ASSERT_TRUE(actual.empty());
}


// Test Case 9: Integration with SphericalProjection using Full Config
TEST_F(FindNeighborCellsTest, IntegrationSphericalProjectionFullConfig) {
    // Arrange
    DynObjFilterParams params; // Use default constructor first
    const std::string config_filename = "test_full_config.yaml"; // Assumes it's copied to build dir

    // Act 1: Load parameters from the actual config file
    bool load_success = load_config(config_filename, params);

    // Assert 1: Check if config loading was successful
    ASSERT_TRUE(load_success) << "Failed to load config file: " << config_filename;

    // --- Now proceed with the projection and neighbor finding ---

    // Arrange 2: Setup point and projection parameters
    point_soph p_input;
    point_soph p_projected; // Output of projection

    // Define a point in global coordinates (e.g., straight ahead)
    p_input.glob = V3D(10.0, 0.0, 0.0); // 10m ahead

    M3D rotation = M3D::Identity();
    V3D translation = V3D::Zero();
    int dummy_depth_index = 0; // Not relevant for this test's purpose

    // Act 2: Project the point to get indices using loaded params
    // Ensure necessary params for SphericalProjection are loaded correctly
    // (Assuming map_width, map_height, resolutions are in the config)
    PointCloudUtils::SphericalProjection(p_input, dummy_depth_index, rotation, translation, params, p_projected);

    // Assert 2: Basic projection validity (check against loaded dimensions)
    ASSERT_GE(p_projected.ver_ind, 0);
    ASSERT_LT(p_projected.ver_ind, params.map_height);
    ASSERT_GE(p_projected.hor_ind, 0);
    ASSERT_LT(p_projected.hor_ind, params.map_width);

    // Act 3: Find neighbors based on projected indices and loaded params
    // (findNeighborCells will use params.checkneighbor_range from the loaded config)
    std::vector<std::pair<int, int>> actual_neighbors = DetectionLogic::findNeighborCells(p_projected, params);

    // Assert 3: Check neighbor properties based on loaded config
    // Calculate expected size based on the loaded checkneighbor_range
    int expected_size = (2 * params.checkneighbor_range + 1) * (2 * params.checkneighbor_range + 1);
    // Adjust expected size for edge cases if the projected point is near a vertical boundary
    if (p_projected.ver_ind < params.checkneighbor_range ||
        p_projected.ver_ind >= params.map_height - params.checkneighbor_range)
    {
         int vertical_span = std::min(p_projected.ver_ind + params.checkneighbor_range, params.map_height - 1) -
                             std::max(p_projected.ver_ind - params.checkneighbor_range, 0) + 1;
         expected_size = vertical_span * (2 * params.checkneighbor_range + 1);
    }


    ASSERT_EQ(actual_neighbors.size(), expected_size)
        << "Expected " << expected_size << " neighbors based on loaded checkneighbor_range="
        << params.checkneighbor_range << " and projected indices ("
        << p_projected.ver_ind << ", " << p_projected.hor_ind << "), but got "
        << actual_neighbors.size();


    // Check if the center point's index is present
    bool center_found = false;
    for (const auto& cell : actual_neighbors) {
        if (cell.first == p_projected.ver_ind && cell.second == p_projected.hor_ind) {
            center_found = true;
            break;
        }
    }
    ASSERT_TRUE(center_found) << "Center cell index (" << p_projected.ver_ind
                              << ", " << p_projected.hor_ind << ") not found in neighbor list.";

    // Optional: Add a check for a specific known neighbor if the projection is predictable
    // (e.g., for a point straight ahead, hor_ind might be near width/2 or 0/width depending on convention)
    // Example: Check if (p_projected.ver_ind + 1, p_projected.hor_ind) is present (if not on bottom edge)
    if (p_projected.ver_ind < params.map_height - 1) {
         bool neighbor_below_found = false;
         std::pair<int, int> target_neighbor = {p_projected.ver_ind + 1, p_projected.hor_ind};
         for (const auto& cell : actual_neighbors) {
             if (cell == target_neighbor) {
                 neighbor_below_found = true;
                 break;
             }
         }
         ASSERT_TRUE(neighbor_below_found) << "Expected neighbor below not found.";
    }
}