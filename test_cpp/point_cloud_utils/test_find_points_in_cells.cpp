#include <gtest/gtest.h>
#include "point_cloud_utils/point_cloud_utils.h" // Your header to test
#include "config/config_loader.h"     // For DynObjFilterParams
#include "filtering/dyn_obj_datatypes.h" // For point_soph, V3D, M3D etc.
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory> // Include for std::shared_ptr

// Helper to create a simple point_soph with a unique ID (using depth for simplicity)
point_soph createTestPoint(int id, int hor_ind, int ver_ind) {
    point_soph p; // Create a default point_soph
    p.vec.z() = static_cast<float>(id); // Store ID in depth for easy identification
    p.hor_ind = hor_ind;
    p.ver_ind = ver_ind;
    // Calculate position based on indices (optional but good for consistency)
    p.position = hor_ind * MAX_1D_HALF + ver_ind;
    // Set other fields if necessary for equality comparison or sorting
    // For instance, ensure 'vec' is somewhat reasonable if other components are used
    p.vec(0) = 0.0f; // Example azimuth
    p.vec(1) = 0.0f; // Example elevation

    return p; // <--- ADD THIS LINE
}

void assertPointVectorsEqualById(
    const std::vector<std::shared_ptr<point_soph>>& actual_const_ref, // Pass by const reference
    const std::vector<std::shared_ptr<point_soph>>& expected_const_ref) // Pass by const reference
{
    // --- Make local copies for sorting ---
    auto actual = actual_const_ref;
    auto expected = expected_const_ref;
    // --- End local copies ---

    // Sort by ID (stored in depth) by dereferencing pointers
    auto sort_crit = [](const std::shared_ptr<point_soph>& a, const std::shared_ptr<point_soph>& b) {
        // Handle potential nullptrs if they could exist
        if (!a && !b) return false; // Both null, treat as equal for sorting
        if (!a) return true;  // Nulls sort before non-nulls
        if (!b) return false; // Non-null sorts after null
        return a->vec.z() < b->vec.z(); // Dereference with ->
    };

    // Sort the local copies
    std::sort(actual.begin(), actual.end(), sort_crit);
    std::sort(expected.begin(), expected.end(), sort_crit);

    // Compare the sorted local copies
    ASSERT_EQ(actual.size(), expected.size()) << "Vector sizes differ.";

    for (size_t i = 0; i < actual.size(); ++i) {
        // Check that both pointers are valid before dereferencing
        ASSERT_TRUE(actual[i]) << "Actual pointer is null at sorted index " << i;
        ASSERT_TRUE(expected[i]) << "Expected pointer is null at sorted index " << i;
        // Compare relevant fields by dereferencing
        EXPECT_EQ(actual[i]->vec.z(), expected[i]->vec.z())
            << "Point ID (depth) mismatch at sorted index " << i;
        // Add more checks if needed (e.g., coordinates)
        // EXPECT_EQ(actual[i]->hor_ind, expected[i]->hor_ind) << "Mismatch at index " << i;
    }
}

class FindPointsInCellsTest : public ::testing::Test {
protected:
    DepthMap test_map;
    const size_t map_size = MAX_1D * MAX_1D_HALF;

    void SetUp() override {
        test_map.depth_map.resize(map_size);
        // Setup already correctly uses make_shared from the previous fix
        test_map.depth_map[0].push_back(std::make_shared<point_soph>(createTestPoint(1, 0, 0)));
        int cell_idx_10 = 10;
        test_map.depth_map[cell_idx_10].push_back(std::make_shared<point_soph>(createTestPoint(2, 0, 10)));
        test_map.depth_map[cell_idx_10].push_back(std::make_shared<point_soph>(createTestPoint(3, 0, 10)));
        int cell_idx_500 = 500;
        test_map.depth_map[cell_idx_500].push_back(std::make_shared<point_soph>(createTestPoint(4, 1, 51)));
        int last_cell_idx = map_size - 1;
        test_map.depth_map[last_cell_idx].push_back(std::make_shared<point_soph>(createTestPoint(5, MAX_1D - 1, MAX_1D_HALF - 1)));
    }
};

// --- FIX: Update EXPECTED values in tests to be vectors of shared_ptr ---

TEST_F(FindPointsInCellsTest, EmptyInputIndices) {
    std::vector<int> indices = {};
    // Result type is now vector<shared_ptr<point_soph>>
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    ASSERT_TRUE(result.empty());
}

TEST_F(FindPointsInCellsTest, SingleValidIndexEmptyCell) {
    std::vector<int> indices = {1};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    ASSERT_TRUE(result.empty());
}

TEST_F(FindPointsInCellsTest, SingleValidIndexOnePoint) {
    std::vector<int> indices = {0};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    // Expected must now be vector<shared_ptr<point_soph>>
    std::vector<std::shared_ptr<point_soph>> expected = {
        std::make_shared<point_soph>(createTestPoint(1, 0, 0)) // Use make_shared
    };
    assertPointVectorsEqualById(result, expected);
}

TEST_F(FindPointsInCellsTest, SingleValidIndexMultiplePoints) {
    std::vector<int> indices = {10};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    std::vector<std::shared_ptr<point_soph>> expected = {
        std::make_shared<point_soph>(createTestPoint(2, 0, 10)),
        std::make_shared<point_soph>(createTestPoint(3, 0, 10))
    };
    assertPointVectorsEqualById(result, expected);
}

TEST_F(FindPointsInCellsTest, MultipleValidIndices) {
    std::vector<int> indices = {1, 500, 0, 10};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    std::vector<std::shared_ptr<point_soph>> expected = {
        std::make_shared<point_soph>(createTestPoint(4, 1, 51)),
        std::make_shared<point_soph>(createTestPoint(1, 0, 0)),
        std::make_shared<point_soph>(createTestPoint(2, 0, 10)),
        std::make_shared<point_soph>(createTestPoint(3, 0, 10))
    };
    assertPointVectorsEqualById(result, expected);
}

TEST_F(FindPointsInCellsTest, IndexOutOfBoundsNegative) {
    std::vector<int> indices = {-1, 10};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    std::vector<std::shared_ptr<point_soph>> expected = {
        std::make_shared<point_soph>(createTestPoint(2, 0, 10)),
        std::make_shared<point_soph>(createTestPoint(3, 0, 10))
    };
    assertPointVectorsEqualById(result, expected);
}

TEST_F(FindPointsInCellsTest, IndexOutOfBoundsTooLarge) {
    std::vector<int> indices = {0, static_cast<int>(map_size)};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    std::vector<std::shared_ptr<point_soph>> expected = {
        std::make_shared<point_soph>(createTestPoint(1, 0, 0))
    };
    assertPointVectorsEqualById(result, expected);
}

TEST_F(FindPointsInCellsTest, MixedValidAndInvalidIndices) {
    std::vector<int> indices = {-5, 0, 501, static_cast<int>(map_size) + 10, 10, -2};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    std::vector<std::shared_ptr<point_soph>> expected = {
        std::make_shared<point_soph>(createTestPoint(1, 0, 0)),
        std::make_shared<point_soph>(createTestPoint(2, 0, 10)),
        std::make_shared<point_soph>(createTestPoint(3, 0, 10))
    };
    assertPointVectorsEqualById(result, expected);
}

TEST_F(FindPointsInCellsTest, LastCellIndex) {
    std::vector<int> indices = {static_cast<int>(map_size - 1)};
    auto result = PointCloudUtils::findPointsInCells(indices, test_map);
    std::vector<std::shared_ptr<point_soph>> expected = {
        std::make_shared<point_soph>(createTestPoint(5, MAX_1D - 1, MAX_1D_HALF - 1))
    };
    assertPointVectorsEqualById(result, expected);
}