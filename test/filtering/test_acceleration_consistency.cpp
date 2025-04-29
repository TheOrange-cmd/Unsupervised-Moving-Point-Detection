// file: test/test_acceleration_consistency.cpp
// (Or add these tests to an existing file like test_consistency_checks.cpp)

#include <gtest/gtest.h>
#include "filtering/consistency_checks.h" // Header with checkAccelerationLimit
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "test/test_consistency_helpers.hpp"
#include <cmath>      // For std::fabs
#include <stdexcept>  // For std::invalid_argument

// Use a specific fixture or reuse/inherit from ConsistencyChecksTest
// Make sure the SetUp loads the config file correctly.
class AccelerationConsistencyTest : public ConsistencyChecksTest {
protected:
    // Inherits params, test_map, SetUp() etc. from ConsistencyChecksTest

    void SetUp() override {
        // Ensure base class SetUp is called to load the config
        ConsistencyChecksTest::SetUp();

        // Verify that the expected parameters were loaded correctly from the config
        // This makes the tests more robust against config file changes.
        ASSERT_FLOAT_EQ(params.acc_thr2, 7.0f)
            << "Test setup error: Expected acc_thr2=7.0 in " << config_path;
        ASSERT_FLOAT_EQ(params.acc_thr3, 15.0f)
            << "Test setup error: Expected acc_thr3=15.0 in " << config_path;
    }
};

// --- Test Cases for checkAccelerationLimit ---

TEST_F(AccelerationConsistencyTest, Case2_PlausibleAcceleration) {
    float v1 = 5.0f;
    float v2 = 5.5f; // Change = 0.5 m/s
    double delta_t_centers = 0.1; // Time delta = 0.1s
    // Implied acceleration = 0.5 / 0.1 = 5.0 m/s^2
    // Case 2 threshold = 7.0 m/s^2
    ASSERT_LT(5.0f, params.acc_thr2); // Verify test logic

    EXPECT_TRUE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(AccelerationConsistencyTest, Case3_PlausibleAcceleration) {
    float v1 = 5.0f;
    float v2 = 6.0f; // Change = 1.0 m/s
    double delta_t_centers = 0.1; // Time delta = 0.1s
    // Implied acceleration = 1.0 / 0.1 = 10.0 m/s^2
    // Case 3 threshold = 15.0 m/s^2
    ASSERT_LT(10.0f, params.acc_thr3); // Verify test logic

    EXPECT_TRUE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(AccelerationConsistencyTest, Case2_ExceedsThreshold) {
    float v1 = 5.0f;
    float v2 = 5.8f; // Change = 0.8 m/s
    double delta_t_centers = 0.1; // Time delta = 0.1s
    // Implied acceleration = 0.8 / 0.1 = 8.0 m/s^2
    // Case 2 threshold = 7.0 m/s^2
    ASSERT_GT(8.0f, params.acc_thr2); // Verify test logic

    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(AccelerationConsistencyTest, Case3_ExceedsThreshold) {
    float v1 = 5.0f;
    float v2 = 7.0f; // Change = 2.0 m/s
    double delta_t_centers = 0.1; // Time delta = 0.1s
    // Implied acceleration = 2.0 / 0.1 = 20.0 m/s^2
    // Case 3 threshold = 15.0 m/s^2
    ASSERT_GT(20.0f, params.acc_thr3); // Verify test logic

    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(AccelerationConsistencyTest, Case2_BoundaryCheck_JustOverThreshold) {
    float v1 = 5.0f;
    double delta_t_centers = 0.1;
    float threshold = params.acc_thr2; // 7.0f
    float allowed_change = threshold * static_cast<float>(delta_t_centers); // 0.7f

    // Calculate v2 such that the change |v2-v1| is *slightly larger* than the allowed change
    float small_epsilon = 1e-5f; // A small value to push it over the boundary
    float v2 = v1 + allowed_change + small_epsilon; // 5.0f + 0.7f + 0.00001f = 5.70001f

    // Verify the calculation ensures the condition is mathematically false
    ASSERT_GT(std::fabs(v2 - v1), allowed_change);

    // The function should return false because the change exceeds the threshold
    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(AccelerationConsistencyTest, Case3_BoundaryCheck_JustOverThreshold) {
    float v1 = 5.0f;
    double delta_t_centers = 0.1;
    float threshold = params.acc_thr3; // 15.0f
    float allowed_change = threshold * static_cast<float>(delta_t_centers); // 1.5f

    // Calculate v2 such that the change |v2-v1| is *slightly larger* than the allowed change
    float small_epsilon = 1e-5f; // A small value
    float v2 = v1 + allowed_change + small_epsilon; // 5.0f + 1.5f + 0.00001f = 6.50001f

    // Verify the calculation ensures the condition is mathematically false
    ASSERT_GT(std::fabs(v2 - v1), allowed_change);

    // The function should return false because the change exceeds the threshold
    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(AccelerationConsistencyTest, Case2_BoundaryCheck_JustUnderThreshold) {
    float v1 = 5.0f;
    double delta_t_centers = 0.1;
    float threshold = params.acc_thr2; // 7.0f
    float allowed_change = threshold * static_cast<float>(delta_t_centers); // 0.7f

    // Calculate v2 such that the change |v2-v1| is *slightly smaller* than the allowed change
    float small_epsilon = 1e-6f; // Use a slightly smaller epsilon than the difference check might resolve
    float v2 = v1 + allowed_change - small_epsilon; // 5.0f + 0.7f - 0.000001f = 5.699999f

    // Verify the calculation ensures the condition is mathematically true
    // Use ASSERT_LT with tolerance or check relative difference if needed for robustness
    ASSERT_LT(std::fabs(v2 - v1), allowed_change);

    // The function should return true because the change is within the threshold
    EXPECT_TRUE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}
TEST_F(AccelerationConsistencyTest, ZeroTimeDelta_IdenticalVelocities) {
    float v1 = 5.0f;
    float v2 = 5.0f;
    double delta_t_centers = 0.0;

    // Should use the epsilon_vel check and pass
    EXPECT_TRUE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    EXPECT_TRUE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(AccelerationConsistencyTest, ZeroTimeDelta_DifferentVelocities) {
    float v1 = 5.0f;
    float v2 = 5.001f; // Difference > epsilon_vel (default 1e-4)
    double delta_t_centers = 0.0;

    // Should use the epsilon_vel check and fail
    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(AccelerationConsistencyTest, NearZeroTimeDelta_SmallVelocityDifference) {
    float v1 = 5.0f;
    float v2 = 5.00001f; // Small difference
    double delta_t_centers = 1e-7; // Less than epsilon_time (1e-6)

    // Should use the epsilon_vel check. |v2-v1| = 1e-5 < epsilon_vel (1e-4). Should pass.
    EXPECT_TRUE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}


TEST_F(AccelerationConsistencyTest, NegativeTimeDelta) {
    float v1 = 5.0f;
    float v2 = 5.5f; // |v2-v1| = 0.5
    double delta_t_centers = -0.1;
    // The check becomes: 0.5 < threshold * (-0.1)
    // Since threshold is positive, the right side is negative.
    // Comparing a positive float (0.5) to a negative float should always be false.
    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Check with negative time delta unexpectedly passed for Case 2.";
    EXPECT_FALSE(checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH))
        << "Check with negative time delta unexpectedly passed for Case 3.";
    // NOTE: This test passes with the current code, but highlights that negative delta_t
    // isn't explicitly prevented and leads to potentially unintuitive results.
    // Consider adding an assertion `assert(time_delta_between_velocity_centers >= 0.0)`
    // or returning false if delta is negative in checkAccelerationLimit if needed.
}


TEST_F(AccelerationConsistencyTest, InvalidCheckTypeThrows) {
    float v1 = 5.0f;
    float v2 = 5.5f;
    double delta_t_centers = 0.1;

    // Expect std::invalid_argument when using a type not handled in the switch
    EXPECT_THROW(
        checkAccelerationLimit(v1, v2, delta_t_centers, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION),
        std::invalid_argument
    );
}