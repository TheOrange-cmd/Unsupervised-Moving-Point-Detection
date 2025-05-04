// test_cpp/point_cloud_utils/test_is_self_point.cpp
#include "gtest/gtest.h"
#include "point_cloud_utils/point_cloud_utils.h" // Header for the function under test
#include "config/config_loader.h"     // For DynObjFilterParams
#include "filtering/dyn_obj_datatypes.h" // For V3D

namespace {
    // Helper to create points easily
    V3D p(float x, float y, float z) { return V3D(x, y, z); }
} // anonymous namespace

// Create a fixture to hold default params for tests
class SelfPointCheckTest : public ::testing::Test {
protected:
    DynObjFilterParams test_params;

    void SetUp() override {
        // Define the self-box used for testing here
        test_params.self_x_b = -1.5; // Back
        test_params.self_x_f =  2.0; // Front
        test_params.self_y_r = -0.8; // Right
        test_params.self_y_l =  0.8; // Left
    }
};

TEST_F(SelfPointCheckTest, PointInsideBox) {
    V3D point_center = p(0.0, 0.0, 0.0); // Origin
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_center, test_params));

    V3D point_front_left = p(1.9, 0.7, 10.0); // Near front-left corner (Z ignored)
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_front_left, test_params));

    V3D point_back_right = p(-1.4, -0.7, -5.0); // Near back-right corner (Z ignored)
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_back_right, test_params));
}

TEST_F(SelfPointCheckTest, PointOutsideBox) {
    V3D point_too_far_front = p(2.1, 0.0, 0.0);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_too_far_front, test_params));

    V3D point_too_far_back = p(-1.6, 0.0, 0.0);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_too_far_back, test_params));

    V3D point_too_far_left = p(0.0, 0.9, 0.0);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_too_far_left, test_params));

    V3D point_too_far_right = p(0.0, -0.9, 0.0);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_too_far_right, test_params));

    V3D point_far_away = p(100.0, 100.0, 100.0);
    ASSERT_FALSE(PointCloudUtils::isSelfPoint(point_far_away, test_params));
}

TEST_F(SelfPointCheckTest, PointOnBoundary) {
    // Test points exactly on the boundary (inclusive check >=, <=)
    V3D point_on_back_boundary = p(test_params.self_x_b, 0.0, 0.0);
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_on_back_boundary, test_params));

    V3D point_on_front_boundary = p(test_params.self_x_f, 0.0, 0.0);
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_on_front_boundary, test_params));

    V3D point_on_left_boundary = p(0.0, test_params.self_y_l, 0.0);
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_on_left_boundary, test_params));

    V3D point_on_right_boundary = p(0.0, test_params.self_y_r, 0.0);
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(point_on_right_boundary, test_params));

    // Test corners
    V3D corner_bl = p(test_params.self_x_b, test_params.self_y_l, 0.0);
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(corner_bl, test_params));
    V3D corner_fr = p(test_params.self_x_f, test_params.self_y_r, 0.0);
    ASSERT_TRUE(PointCloudUtils::isSelfPoint(corner_fr, test_params));
}