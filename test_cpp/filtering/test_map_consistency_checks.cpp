#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <string>
#include <iostream> // For potential debug output

// Headers for the code being tested and its dependencies
#include "filtering/consistency_checks.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "point_cloud_utils/point_cloud_utils.h" // For types used internally by consistency_checks
#include "test/test_consistency_helpers.hpp"

// --- Test Cases for Map Consistency ---

TEST_F(ConsistencyChecksTest, PointInsideSelfBox) {
    // Modify center_point's local coordinates to be inside the self-box defined by params
    center_point.local.x() = (params.self_x_b + params.self_x_f) / 2.0;
    center_point.local.y() = (params.self_y_r + params.self_y_l) / 2.0;
    center_point.local.z() = 0.0; // Z doesn't matter for the check

    // Add some arbitrary points to the map (shouldn't affect the outcome)
    addInterpolationTriangle(center_point, center_point.vec(2), center_point.vec(2), center_point.vec(2));

    // Check consistency for all cases - should return false due to self-box
    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION)) << "Case 1 failed self-box check";
    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, 2)) << "Case 2 failed self-box check";
    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH, 2)) << "Case 3 failed self-box check";
}

// --- Case 1 Tests (STATIC_ONLY neighbors) ---

TEST_F(ConsistencyChecksTest, Case1_InterpolationSucceedsAndPasses) {
    center_point.vec(2) = 15.0f; // Set depth for center point
    float neighbor_depth = 15.1f; // Close depth
    float expected_interp_depth = neighbor_depth; // Approx

    // Add a valid triangle of STATIC neighbors with depths close to center_point
    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, DynObjLabel::STATIC, -params.frame_dur * 2); // Old static points

    // Calculate expected threshold for Case 1
    float threshold = params.interp_thr1;
     if (center_point.vec(2) > params.interp_start_depth1) {
         threshold += ((center_point.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
     }
    // Assume center_point is not distorted for this test

    ASSERT_LT(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be within threshold";

    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be consistent when interpolation passes threshold.";
}

TEST_F(ConsistencyChecksTest, Case1_InterpolationSucceedsButFailsThreshold) {
    center_point.vec(2) = 15.0f;
    float neighbor_depth = 17.0f; // Far depth
    float expected_interp_depth = neighbor_depth; // Approx

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, DynObjLabel::STATIC, -params.frame_dur * 2);

    // Calculate expected threshold for Case 1
    float threshold = params.interp_thr1;
     if (center_point.vec(2) > params.interp_start_depth1) {
         threshold += ((center_point.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
     }

    ASSERT_GE(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be outside threshold";

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be inconsistent when interpolation result is outside threshold.";
}

TEST_F(ConsistencyChecksTest, Case1_InterpolationFailsNotEnoughNeighbors) {
    center_point.vec(2) = 15.0f;

    // Add only one STATIC neighbor
    point_soph n1 = createTestPointWithIndices(center_point.local.x()+1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*2, DynObjLabel::STATIC);
    n1.vec(2) = 15.1f; // Close depth, but shouldn't matter
    addPointToMap(n1);

    // Add one close static point *conceptually* for the old logic check
    // The new logic should ignore this if interpolation fails.
    // point_soph close_neighbor = createTestPointWithIndices(center_point.local.x()+0.1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*2, STATIC);
    // close_neighbor.vec(2) = 15.05f; // Very close
    // addPointToMap(close_neighbor); // Add if you want to be extra sure it doesn't interfere

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be inconsistent when interpolation fails (not enough neighbors), even if a close static point exists.";
}

TEST_F(ConsistencyChecksTest, Case1_InterpolationFailsBadTriangle) {
    center_point.vec(2) = 15.0f;

    // Add 3 STATIC neighbors, but make them collinear (same elevation, different azimuth)
    float depth = 15.1f;
    float az_offset = params.hor_resolution_max * 1.5f;
    V3D p_local = center_point.local;

    V3D p1_local = p_local; p1_local.x() -= p_local.norm() * az_offset; p1_local = p1_local.normalized() * depth;
    V3D p2_local = p_local;                                              p2_local = p2_local.normalized() * depth;
    V3D p3_local = p_local; p3_local.x() += p_local.norm() * az_offset; p3_local = p3_local.normalized() * depth;

    point_soph n1 = createTestPointWithIndices(p1_local.x(), p1_local.y(), p1_local.z(), params, center_point.time - params.frame_dur*2, DynObjLabel::STATIC);
    point_soph n2 = createTestPointWithIndices(p2_local.x(), p2_local.y(), p2_local.z(), params, center_point.time - params.frame_dur*2, DynObjLabel::STATIC);
    point_soph n3 = createTestPointWithIndices(p3_local.x(), p3_local.y(), p3_local.z(), params, center_point.time - params.frame_dur*2, DynObjLabel::STATIC);

    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be inconsistent when interpolation fails (bad triangle).";
}

TEST_F(ConsistencyChecksTest, Case1_DistortionEnlargement) {
    // Requires dataset 0 and is_distort = true
    if (params.dataset != 0) {
        GTEST_SKIP() << "Skipping distortion test because dataset is not 0";
    }
    if (params.enlarge_distort <= 1.0f) {
         GTEST_SKIP() << "Skipping distortion test because enlarge_distort is not > 1.0";
    }

    center_point = createTestPointWithIndices(20.0, 5.0, 0.0, params, 0.1, DynObjLabel::STATIC, true); // is_distort = true
    center_point.vec(2) = 15.0f;
    float neighbor_depth = 15.5f; // Choose a depth difference
    float expected_interp_depth = neighbor_depth; // Approx

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, DynObjLabel::STATIC, -params.frame_dur * 2);

    // Calculate threshold WITHOUT distortion
    float threshold_no_distort = params.interp_thr1;
     if (center_point.vec(2) > params.interp_start_depth1) {
         threshold_no_distort += ((center_point.vec(2) - params.interp_start_depth1) * params.interp_kp1 + params.interp_kd1);
     }
    // Calculate threshold WITH distortion
    float threshold_with_distort = threshold_no_distort * params.enlarge_distort;

    float actual_diff = std::fabs(expected_interp_depth - center_point.vec(2));

    // Ensure the chosen depth difference falls between the two thresholds
    ASSERT_GE(actual_diff, threshold_no_distort) << "Test setup error: Depth diff should be >= non-distorted threshold.";
    ASSERT_LT(actual_diff, threshold_with_distort) << "Test setup error: Depth diff should be < distorted threshold.";

    // Expect TRUE because the enlarged threshold should allow it
    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION))
        << "Point should be consistent due to distortion enlargement.";
}


// --- Case 2 Tests (ALL_VALID neighbors, older map) ---

TEST_F(ConsistencyChecksTest, Case2_InterpolationSucceedsAndPasses) {
    center_point.vec(2) = 25.0f;
    float neighbor_depth = 25.2f; // Close depth
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 3; // Simulate map being 3 steps older

    // Add a valid triangle - can include DYNAMIC points if they are old enough
    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, DynObjLabel::UNCERTAIN, -params.frame_dur * (map_diff + 1)); // Old enough dynamic points

    float threshold = params.interp_thr2 * static_cast<float>(map_diff);

    ASSERT_LT(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be within threshold";

    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, map_diff))
        << "Case 2: Point should be consistent when interpolation passes threshold.";
}

TEST_F(ConsistencyChecksTest, Case2_InterpolationFailsThreshold) {
    center_point.vec(2) = 25.0f;
    float neighbor_depth = 28.0f; // Far depth
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 3;

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, DynObjLabel::STATIC, -params.frame_dur * (map_diff + 1));

    float threshold = params.interp_thr2 * static_cast<float>(map_diff);

    ASSERT_GE(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be outside threshold";

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, map_diff))
        << "Case 2: Point should be inconsistent when interpolation fails threshold.";
}

TEST_F(ConsistencyChecksTest, Case2_InterpolationFailsNotEnoughValidNeighbors) {
    center_point.vec(2) = 25.0f;
    int map_diff = 3;

    // Add only one OLD STATIC neighbor
    point_soph n1 = createTestPointWithIndices(center_point.local.x()+1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*(map_diff+1), DynObjLabel::STATIC);
    n1.vec(2) = 25.1f;
    addPointToMap(n1);
    // Add one RECENT DYNAMIC neighbor (should be ignored by findInterpolationNeighbors for ALL_VALID if time diff < frame_dur)
    point_soph n2 = createTestPointWithIndices(center_point.local.x()-1, center_point.local.y(), center_point.local.z(), params, center_point.time - params.frame_dur*0.5, DynObjLabel::UNCERTAIN);
    n2.vec(2) = 25.1f;
    addPointToMap(n2);

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH, map_diff))
        << "Case 2: Point should be inconsistent when interpolation fails (not enough valid neighbors).";
}


// --- Case 3 Tests (ALL_VALID neighbors, older map) ---

TEST_F(ConsistencyChecksTest, Case3_InterpolationSucceedsAndPasses) {
    center_point.vec(2) = 30.0f;
    // Adjust depth slightly to be clearly within threshold
    float neighbor_depth = 30.19f; // WAS 30.2f
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 4;

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, DynObjLabel::STATIC, -params.frame_dur * (map_diff + 1));

    float threshold = params.interp_thr3 * static_cast<float>(map_diff);

    // This assertion should now pass
    ASSERT_LT(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be within threshold";

    EXPECT_TRUE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH, map_diff))
        << "Case 3: Point should be consistent when interpolation passes threshold.";
}

TEST_F(ConsistencyChecksTest, Case3_InterpolationFailsThreshold) {
    center_point.vec(2) = 30.0f;
    float neighbor_depth = 33.0f; // Far depth
    float expected_interp_depth = neighbor_depth; // Approx
    int map_diff = 4;

    addInterpolationTriangle(center_point, neighbor_depth, neighbor_depth, neighbor_depth, DynObjLabel::STATIC, -params.frame_dur * (map_diff + 1));

    float threshold = params.interp_thr3 * static_cast<float>(map_diff);

    ASSERT_GE(std::fabs(expected_interp_depth - center_point.vec(2)), threshold) << "Test setup error: Expected depth diff should be outside threshold";

    EXPECT_FALSE(ConsistencyChecks::checkMapConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH, map_diff))
        << "Case 3: Point should be inconsistent when interpolation fails threshold.";
}
