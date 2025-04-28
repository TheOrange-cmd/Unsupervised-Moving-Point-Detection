#include <gtest/gtest.h>
#include "point_cloud_utils/point_cloud_utils.h" // Your header to test
#include "config/config_loader.h"     // For DynObjFilterParams
#include "filtering/dyn_obj_datatypes.h" // For point_soph, V3D, M3D etc.
#include <vector>
#include <unordered_set> // Useful for comparing results regardless of order
#include <algorithm>     // For std::sort if needed, though set comparison is better
#include <cmath>         // For std::abs

// Define constants consistent with other tests if possible
static constexpr int TEST_MAX_1D = MAX_1D;
static constexpr int TEST_MAX_1D_HALF = MAX_1D_HALF;
static constexpr int TEST_MAX_2D_N = MAX_2D_N;

// Helper to create a simple point_soph with just indices set
point_soph createTestPointIndices(int hor_ind, int ver_ind) {
    point_soph p;
    p.hor_ind = hor_ind;
    p.ver_ind = ver_ind;
    // Other members (vec, time, etc.) are not used by findNeighborCells
    return p;
}

// Helper to calculate the expected 1D index
int getIndex(int hor, int ver) {
    if (hor < 0 || hor >= TEST_MAX_1D || ver < 0 || ver >= TEST_MAX_1D_HALF) {
        return -1; // Indicate invalid index outside bounds
    }
    return hor * TEST_MAX_1D_HALF + ver;
}

// Helper to compare results using unordered_set (order-independent)
void assertVectorsContainSameElements(const std::vector<int>& actual, const std::vector<int>& expected) {
    std::unordered_set<int> actual_set(actual.begin(), actual.end());
    std::unordered_set<int> expected_set(expected.begin(), expected.end());
    ASSERT_EQ(actual_set.size(), actual.size()) << "Actual vector contains duplicates.";
    ASSERT_EQ(expected_set.size(), expected.size()) << "Expected vector contains duplicates.";
    ASSERT_EQ(actual_set, expected_set);
}


class FindNeighborCellsTest : public ::testing::Test {
protected:
    // You can add setup/teardown logic here if needed
};

// --- Test Cases ---

TEST_F(FindNeighborCellsTest, RangeZeroIncludeCenter) {
    point_soph p = createTestPointIndices(100, 50); // Point in the middle
    int hor_range = 0;
    int ver_range = 0;
    bool include_center = true;
    bool wrap_horizontal = false;

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected = { getIndex(100, 50) };
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeZeroExcludeCenter) {
    point_soph p = createTestPointIndices(100, 50);
    int hor_range = 0;
    int ver_range = 0;
    bool include_center = false; // Exclude
    bool wrap_horizontal = false;

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected = {}; // Expect empty
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneIncludeCenterNoWrapMiddle) {
    point_soph p = createTestPointIndices(100, 50); // Point well within bounds
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false;

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    for (int h = 99; h <= 101; ++h) {
        for (int v = 49; v <= 51; ++v) {
            expected.push_back(getIndex(h, v));
        }
    }
    ASSERT_EQ(result.size(), 9); // Should be 3x3 grid
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneExcludeCenterNoWrapMiddle) {
    point_soph p = createTestPointIndices(100, 50);
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = false; // Exclude
    bool wrap_horizontal = false;

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    for (int h = 99; h <= 101; ++h) {
        for (int v = 49; v <= 51; ++v) {
            if (h == 100 && v == 50) continue; // Skip center
            expected.push_back(getIndex(h, v));
        }
    }
    ASSERT_EQ(result.size(), 8); // Should be 3x3 grid minus center
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneIncludeCenterWrapLeftEdge) {
    point_soph p = createTestPointIndices(0, 50); // Left edge
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = true; // Enable wrap

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = -1 (wraps to MAX-1), 0, 1
    // v = 49, 50, 51
    expected.push_back(getIndex(TEST_MAX_1D - 1, 49));
    expected.push_back(getIndex(TEST_MAX_1D - 1, 50));
    expected.push_back(getIndex(TEST_MAX_1D - 1, 51));
    expected.push_back(getIndex(0, 49));
    expected.push_back(getIndex(0, 50)); // Center
    expected.push_back(getIndex(0, 51));
    expected.push_back(getIndex(1, 49));
    expected.push_back(getIndex(1, 50));
    expected.push_back(getIndex(1, 51));

    ASSERT_EQ(result.size(), 9);
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneExcludeCenterWrapLeftEdge) {
    point_soph p = createTestPointIndices(0, 50); // Left edge
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = false; // Exclude
    bool wrap_horizontal = true; // Enable wrap

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    expected.push_back(getIndex(TEST_MAX_1D - 1, 49));
    expected.push_back(getIndex(TEST_MAX_1D - 1, 50));
    expected.push_back(getIndex(TEST_MAX_1D - 1, 51));
    expected.push_back(getIndex(0, 49));
    // Skip center (0, 50)
    expected.push_back(getIndex(0, 51));
    expected.push_back(getIndex(1, 49));
    expected.push_back(getIndex(1, 50));
    expected.push_back(getIndex(1, 51));

    ASSERT_EQ(result.size(), 8);
    assertVectorsContainSameElements(result, expected);
}


TEST_F(FindNeighborCellsTest, RangeOneIncludeCenterWrapRightEdge) {
    point_soph p = createTestPointIndices(TEST_MAX_1D - 1, 50); // Right edge
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = true; // Enable wrap

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = MAX-2, MAX-1, MAX (wraps to 0)
    // v = 49, 50, 51
    expected.push_back(getIndex(TEST_MAX_1D - 2, 49));
    expected.push_back(getIndex(TEST_MAX_1D - 2, 50));
    expected.push_back(getIndex(TEST_MAX_1D - 2, 51));
    expected.push_back(getIndex(TEST_MAX_1D - 1, 49));
    expected.push_back(getIndex(TEST_MAX_1D - 1, 50)); // Center
    expected.push_back(getIndex(TEST_MAX_1D - 1, 51));
    expected.push_back(getIndex(0, 49)); // Wrapped
    expected.push_back(getIndex(0, 50)); // Wrapped
    expected.push_back(getIndex(0, 51)); // Wrapped

    ASSERT_EQ(result.size(), 9);
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneNoWrapNearTopEdge) {
    point_soph p = createTestPointIndices(100, 0); // Top edge
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false; // No wrap

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = 99, 100, 101
    // v = -1 (invalid), 0, 1
    expected.push_back(getIndex(99, 0));
    expected.push_back(getIndex(99, 1));
    expected.push_back(getIndex(100, 0)); // Center
    expected.push_back(getIndex(100, 1));
    expected.push_back(getIndex(101, 0));
    expected.push_back(getIndex(101, 1));

    ASSERT_EQ(result.size(), 6); // Should be 3x2 grid
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneNoWrapNearBottomEdge) {
    point_soph p = createTestPointIndices(100, TEST_MAX_1D_HALF - 1); // Bottom edge
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false; // No wrap

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = 99, 100, 101
    // v = MAX_HALF-2, MAX_HALF-1, MAX_HALF (invalid)
    expected.push_back(getIndex(99, TEST_MAX_1D_HALF - 2));
    expected.push_back(getIndex(99, TEST_MAX_1D_HALF - 1));
    expected.push_back(getIndex(100, TEST_MAX_1D_HALF - 2));
    expected.push_back(getIndex(100, TEST_MAX_1D_HALF - 1)); // Center
    expected.push_back(getIndex(101, TEST_MAX_1D_HALF - 2));
    expected.push_back(getIndex(101, TEST_MAX_1D_HALF - 1));

    ASSERT_EQ(result.size(), 6); // Should be 3x2 grid
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneNoWrapTopLeftCorner) {
    point_soph p = createTestPointIndices(0, 0); // Top-left corner
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false; // No wrap

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = -1(invalid), 0, 1
    // v = -1(invalid), 0, 1
    expected.push_back(getIndex(0, 0)); // Center
    expected.push_back(getIndex(0, 1));
    expected.push_back(getIndex(1, 0));
    expected.push_back(getIndex(1, 1));

    ASSERT_EQ(result.size(), 4); // Should be 2x2 grid
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, RangeOneWrapTopLeftCorner) {
    point_soph p = createTestPointIndices(0, 0); // Top-left corner
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = true; // Wrap

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = -1(wraps to MAX-1), 0, 1
    // v = -1(invalid), 0, 1
    expected.push_back(getIndex(TEST_MAX_1D - 1, 0)); // Wrapped
    expected.push_back(getIndex(TEST_MAX_1D - 1, 1)); // Wrapped
    expected.push_back(getIndex(0, 0)); // Center
    expected.push_back(getIndex(0, 1));
    expected.push_back(getIndex(1, 0));
    expected.push_back(getIndex(1, 1));

    ASSERT_EQ(result.size(), 6); // Should be 3x2 grid effectively
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, AsymmetricRangeNoWrap) {
    point_soph p = createTestPointIndices(100, 50);
    int hor_range = 2; // Wider horizontal range
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false;

    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = 98, 99, 100, 101, 102
    // v = 49, 50, 51
    for (int h = 98; h <= 102; ++h) {
        for (int v = 49; v <= 51; ++v) {
            expected.push_back(getIndex(h, v));
        }
    }
    ASSERT_EQ(result.size(), 15); // Should be 5x3 grid
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, NegativeRangeIsZeroRange) {
    point_soph p = createTestPointIndices(100, 50);
    int hor_range = -1; // Negative range
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false;

    // Note: forEachNeighborCell clamps negative range in its loop bounds effectively making it 0
    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected;
    // h = 100 (since hor_range <= 0)
    // v = 49, 50, 51
    expected.push_back(getIndex(100, 49));
    expected.push_back(getIndex(100, 50)); // Center
    expected.push_back(getIndex(100, 51));

    ASSERT_EQ(result.size(), 3); // Should be 1x3 grid
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, InvalidCenterIndexHorizontal) {
    // Create a point with an invalid horizontal index
    point_soph p = createTestPointIndices(TEST_MAX_1D, 50); // hor_ind >= MAX_1D
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false;

    // forEachNeighborCell should handle this gracefully and not call the lambda
    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected = {}; // Expect empty
    ASSERT_TRUE(result.empty());
    assertVectorsContainSameElements(result, expected);
}

TEST_F(FindNeighborCellsTest, InvalidCenterIndexVertical) {
    // Create a point with an invalid vertical index
    point_soph p = createTestPointIndices(100, TEST_MAX_1D_HALF); // ver_ind >= MAX_1D_HALF
    int hor_range = 1;
    int ver_range = 1;
    bool include_center = true;
    bool wrap_horizontal = false;

    // forEachNeighborCell should handle this gracefully and not call the lambda
    std::vector<int> result = PointCloudUtils::findNeighborCells(p, hor_range, ver_range, include_center, wrap_horizontal);

    std::vector<int> expected = {}; // Expect empty
    ASSERT_TRUE(result.empty());
    assertVectorsContainSameElements(result, expected);
}