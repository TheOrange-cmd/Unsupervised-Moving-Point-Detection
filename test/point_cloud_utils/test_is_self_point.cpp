#include "gtest/gtest.h" // Google Test framework
#include "point_cloud_utils.h" // Your header to test
#include "config_loader.h"     // For DynObjFilterParams
#include "dyn_obj_datatypes.h" // For point_soph, V3D, M3D etc.
#include <Eigen/Geometry>      // For Eigen::AngleAxisd

// Define constants for testing
const int DATASET_SELF_CHECK = 0;
const int DATASET_OTHER = 1;

// Helper to create points easily
V3D p(float x, float y, float z) { return V3D(x, y, z); }

TEST(SelfPointCheckTest, NonApplicableDataset) {
    // A point that would be inside Box 1 if dataset was 0
    V3D point_in_box1 = p(-1.0, -1.5, -0.5);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_in_box1,  DATASET_OTHER))
        << "Should return false for non-zero dataset ID";

    // A point outside any box
    V3D point_outside = p(0.0, 0.0, 0.0);
     ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_outside,  DATASET_OTHER))
        << "Should return false for non-zero dataset ID";
}

TEST(SelfPointCheckTest, DatasetZero_OutsideAllBoxes) {
    V3D point_outside = p(0.0, 0.0, 0.0); // Origin is not in any box
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_outside,  DATASET_SELF_CHECK));

    V3D point_far_away = p(100.0, 100.0, 100.0);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_far_away,  DATASET_SELF_CHECK));
}

// --- Tests for each Box when dataset == 0 ---

TEST(SelfPointCheckTest, DatasetZero_InsideBox1) {
    V3D point_center = p(-0.8, -1.35, -0.525); // Center of box 1 approx
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_center,  DATASET_SELF_CHECK));
}

TEST(SelfPointCheckTest, DatasetZero_InsideBox2) {
    V3D point_center = p(-1.3, 1.3, -0.575); // Center of box 2 approx
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_center,  DATASET_SELF_CHECK));
}

TEST(SelfPointCheckTest, DatasetZero_InsideBox3) {
    V3D point_center = p(1.55, -1.1, -0.7); // Center of box 3 approx
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_center,  DATASET_SELF_CHECK));
}

TEST(SelfPointCheckTest, DatasetZero_InsideBox4) {
    V3D point_center = p(2.525, -0.525, -0.95); // Center of box 4 approx
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_center,  DATASET_SELF_CHECK));
}

TEST(SelfPointCheckTest, DatasetZero_InsideBox5) {
    V3D point_center = p(2.525, 0.525, -0.95); // Center of box 5 approx
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_center,  DATASET_SELF_CHECK));
}

// --- Tests for Boundaries (Exclusive) ---

TEST(SelfPointCheckTest, DatasetZero_BoundaryBox1) {
    // Point exactly on min x boundary of box 1
    V3D point_on_boundary = p(-1.2, -1.5, -0.5);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_on_boundary,  DATASET_SELF_CHECK));
    // Point exactly on max y boundary of box 1
    point_on_boundary = p(-1.0, -1.0, -0.5);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_on_boundary,  DATASET_SELF_CHECK));
    // ... add more boundary checks for other faces/boxes as needed ...
}

// --- Test points slightly outside ---
TEST(SelfPointCheckTest, DatasetZero_SlightlyOutsideBox1) {
    // Just below min x
    V3D point_outside = p(-1.21, -1.5, -0.5);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_outside,  DATASET_SELF_CHECK));
     // Just above max x
    point_outside = p(-0.39, -1.5, -0.5);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_outside,  DATASET_SELF_CHECK));
    // ... add more for other dimensions/boxes ...
}