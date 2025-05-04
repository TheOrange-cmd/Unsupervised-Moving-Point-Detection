#include "gtest/gtest.h"
#include "point_cloud_utils/point_cloud_utils.h"
#include "config/config_loader.h"
#include "filtering/dyn_obj_datatypes.h" // For V3D
#include <cmath> // For std::fabs

namespace {
    // Helper to create points easily
    V3D p(float x, float y, float z) { return V3D(x, y, z); }
} // anonymous namespace

// Create a fixture to hold default params for tests
class InvalidPointCheckTest : public ::testing::Test {
    protected:
        DynObjFilterParams test_params;
    
        void SetUp() override {
            // Use defaults from config_loader.h constructor initially
            // We will override specific ones in tests
            test_params = DynObjFilterParams(); // Get defaults
    
            // Set specific values relevant for these tests
            test_params.blind_dis = 0.2f;
            test_params.invalid_box_x_half_width = 0.5f; // Example box dimensions
            test_params.invalid_box_y_half_width = 1.0f;
            test_params.invalid_box_z_half_width = 0.5f;
        }
    };

// --- Tests for Blind Distance ---

TEST_F(InvalidPointCheckTest, PointTooClose) {
    V3D point_inside = p(0.1f, 0.0f, 0.0f); // norm = 0.1 < blind_dis (0.2)
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point_inside, test_params))
        << "Point inside blind distance should always be invalid.";
    
    // Also test with box check enabled, should still be invalid due to distance
    test_params.enable_invalid_box_check = true;
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point_inside, test_params)) 
        << "Point inside blind distance should be invalid even if box check is enabled.";
}

TEST_F(InvalidPointCheckTest, PointAtBlindDistance) {
    V3D point_on_boundary = p(1.0, 0.0, 0.0); // Distance 1.0 == 1.0
    // squaredNorm (1.0) is NOT < blind_distance^2 (1.0)
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_on_boundary, test_params))
        << "Point exactly at blind distance boundary should be valid.";

    // Test with box enabled (point is outside the box)
    test_params.enable_invalid_box_check = true;
     ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_on_boundary, test_params))
        << "Point at blind distance boundary should be valid even if box check is enabled (and point is outside box).";
}

TEST_F(InvalidPointCheckTest, PointOutsideBlindDistance) {
    V3D point_outside = p(1.5, 0.0, 0.0); // Distance 1.5 > 1.0
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside, test_params))
         << "Point clearly outside blind distance should be valid (box check disabled).";

    // Test with box enabled (point is outside the box)
    test_params.enable_invalid_box_check = true;
     ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside, test_params))
        << "Point clearly outside blind distance should be valid (box check enabled, point outside box).";
}

// --- Tests for Invalid Box ---

TEST_F(InvalidPointCheckTest, BoxCheckDisabled_PointInBoxRegion) {
    // Point outside blind distance, but would be inside the box if enabled
    V3D point_in_box_region = p(0.05, 0.5, 0.05); // norm > 1.0
    point_in_box_region = point_in_box_region.normalized() * 1.1f; // Ensure > blind_dis
    ASSERT_GT(point_in_box_region.norm(), test_params.blind_dis); // Sanity check

    test_params.enable_invalid_box_check = false; // Explicitly disable
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_in_box_region, test_params))
        << "Point in box region should be VALID when box check is disabled.";
}

// Test case for when the box check is enabled
TEST_F(InvalidPointCheckTest, BoxCheckEnabled) {
    test_params.enable_invalid_box_check = true; // Enable the check

    // Case 1: Point inside blind distance (should be invalid)
    V3D point_too_close = p(0.1f, 0.0f, 0.0f); // norm = 0.1 < blind_dis
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point_too_close, test_params))
        << "Point inside blind distance should always be invalid.";

    // Case 2: Point outside blind distance AND inside the invalid box (should be invalid)
    V3D point_in_box = p(0.4f, 0.9f, -0.4f); // norm > blind_dis, |x|<0.5, |y|<1.0, |z|<0.5
    ASSERT_GT(point_in_box.norm(), test_params.blind_dis); // Verify outside blind
    ASSERT_LT(std::fabs(point_in_box.x()), test_params.invalid_box_x_half_width); // Verify inside box X
    ASSERT_LT(std::fabs(point_in_box.y()), test_params.invalid_box_y_half_width); // Verify inside box Y
    ASSERT_LT(std::fabs(point_in_box.z()), test_params.invalid_box_z_half_width); // Verify inside box Z
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point_in_box, test_params))
        << "Point outside blind distance but inside the box should be INVALID when box check is enabled.";

    // Case 3: Point outside blind distance AND outside the invalid box (should be valid)
    V3D point_outside_box_x = p(0.6f, 0.5f, 0.0f); // norm > blind_dis, |x| > 0.5
    ASSERT_GT(point_outside_box_x.norm(), test_params.blind_dis); // Verify outside blind
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside_box_x, test_params))
        << "Point outside blind distance and outside the box (X) should be VALID when box check is enabled.";

    V3D point_outside_box_y = p(0.3f, 1.1f, 0.0f); // norm > blind_dis, |y| > 1.0
    ASSERT_GT(point_outside_box_y.norm(), test_params.blind_dis); // Verify outside blind
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside_box_y, test_params))
        << "Point outside blind distance and outside the box (Y) should be VALID when box check is enabled.";

    V3D point_far_away = p(10.0f, 10.0f, 10.0f); // Clearly outside both
     ASSERT_GT(point_far_away.norm(), test_params.blind_dis); // Verify outside blind
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_far_away, test_params))
        << "Far away point should be VALID when box check is enabled.";
}

TEST_F(InvalidPointCheckTest, BoxCheckEnabled_PointOutsideBoxX) {
    // Point outside blind distance, outside box (X too large)
    V3D point_outside_box = p(0.6f, 0.5f, 0.0f); // |x|>=0.5, norm=sqrt(0.36+0.25) > 0.2
    ASSERT_GT(point_outside_box.norm(), test_params.blind_dis); // Sanity check
    ASSERT_GE(std::fabs(point_outside_box.x()), test_params.invalid_box_x_half_width); // Check >= 0.5
    ASSERT_LT(std::fabs(point_outside_box.y()), test_params.invalid_box_y_half_width); // Check < 1.0 (inside Y)
    ASSERT_LT(std::fabs(point_outside_box.z()), test_params.invalid_box_z_half_width); // Check < 0.5 (inside Z)

    test_params.enable_invalid_box_check = true; // Enable the check
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside_box, test_params))
        << "Point outside blind distance AND outside the box (X) should be VALID when box check is enabled.";
}

TEST_F(InvalidPointCheckTest, BoxCheckEnabled_PointOutsideBoxY) {
    // Point outside blind distance, outside box (Y too large)
    V3D point_outside_box = p(0.05, 1.1, 0.05); // y >= invalid_box_y_half_width (1.0)
    point_outside_box = point_outside_box.normalized() * 1.1f; // Ensure > blind_dis
    ASSERT_GT(point_outside_box.norm(), test_params.blind_dis); // Sanity check
    ASSERT_GE(std::fabs(point_outside_box.y()), test_params.invalid_box_y_half_width);

    test_params.enable_invalid_box_check = true; // Enable the check
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside_box, test_params))
        << "Point outside blind distance AND outside the box (Y) should be VALID when box check is enabled.";
}

TEST_F(InvalidPointCheckTest, BoxCheckEnabled_PointOutsideBoxZ) {
    // Point outside blind distance, outside box (Z too large)
    V3D point_outside_box = p(0.1f, 0.5f, 0.6f); // |z|>=0.5, norm=sqrt(0.01+0.25+0.36) > 0.2
    ASSERT_GT(point_outside_box.norm(), test_params.blind_dis); // Sanity check
    ASSERT_LT(std::fabs(point_outside_box.x()), test_params.invalid_box_x_half_width); // Check < 0.5 (inside X)
    ASSERT_LT(std::fabs(point_outside_box.y()), test_params.invalid_box_y_half_width); // Check < 1.0 (inside Y)
    ASSERT_GE(std::fabs(point_outside_box.z()), test_params.invalid_box_z_half_width); // Check >= 0.5

    test_params.enable_invalid_box_check = true; // Enable the check
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside_box, test_params))
        << "Point outside blind distance AND outside the box (Z) should be VALID when box check is enabled.";
}

TEST_F(InvalidPointCheckTest, BoxCheckEnabled_PointInsideBlindAndBox) {
    // Point inside blind distance AND inside the box region
    V3D point_inside_both = p(0.05f, 0.1f, 0.05f); // norm=sqrt(0.0025+0.01+0.0025)=sqrt(0.015) approx 0.12 < 0.2
    ASSERT_LT(point_inside_both.norm(), test_params.blind_dis); // Sanity check: norm < 0.2
    // Check it's also inside the box bounds (it is: |x|<0.5, |y|<1.0, |z|<0.5)
    ASSERT_LT(std::fabs(point_inside_both.x()), test_params.invalid_box_x_half_width);
    ASSERT_LT(std::fabs(point_inside_both.y()), test_params.invalid_box_y_half_width);
    ASSERT_LT(std::fabs(point_inside_both.z()), test_params.invalid_box_z_half_width);

    test_params.enable_invalid_box_check = true; // Enable the check
    // Function should return true because blind distance check comes first
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point_inside_both, test_params))
        << "Point inside blind distance AND inside the box should be INVALID (due to blind distance).";
}