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
#include <iomanip>  // For std::fixed, std::setprecision

// Helper for consistent float printing in tests
#define PRINT_FLOAT(val) std::fixed << std::setprecision(5) << (val)

// Test Case: No suitable neighbors found (empty map)
TEST_F(ConsistencyChecksTest, DepthConsistency_NoNeighbors) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_NoNeighbors" << std::endl;
    std::cout << "[TEST DEBUG] Center Point: H=" << center_point.hor_ind << " V=" << center_point.ver_ind
              << " D=" << PRINT_FLOAT(center_point.vec(2)) << " T=" << PRINT_FLOAT(center_point.time) << std::endl;
    std::cout << "[TEST DEBUG] Map is empty." << std::endl;
    std::cout << "[TEST DEBUG] Expecting FALSE." << std::endl;
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should return false when no neighbors exist in the map.";
}

// Test Case: Neighbors exist but are filtered out (time, angle, status)
TEST_F(ConsistencyChecksTest, DepthConsistency_NoSuitableNeighbors) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_NoSuitableNeighbors" << std::endl;
    std::cout << "[TEST DEBUG] Center Point: H=" << center_point.hor_ind << " V=" << center_point.ver_ind
              << " D=" << PRINT_FLOAT(center_point.vec(2)) << " T=" << PRINT_FLOAT(center_point.time) << std::endl;

    // Add neighbors outside the time window
    auto p_time_out = createTestPointWithIndices(center_point.local.x() + 0.1, center_point.local.y(), center_point.local.z(), params, center_point.time + params.frame_dur * 2.0, STATIC);
    addPointToMap(p_time_out);
    std::cout << "[TEST DEBUG] Added time-out neighbor: D=" << PRINT_FLOAT(p_time_out.vec(2)) << " T=" << PRINT_FLOAT(p_time_out.time) << " Status=" << p_time_out.dyn << std::endl;

    // Add neighbors outside the angular threshold (assuming default thresholds are not huge)
    auto p_angle_out = createTestPointWithIndices(center_point.local.x() + 5.0, center_point.local.y() + 5.0, center_point.local.z(), params, center_point.time, STATIC);
    addPointToMap(p_angle_out);
     std::cout << "[TEST DEBUG] Added angle-out neighbor: D=" << PRINT_FLOAT(p_angle_out.vec(2)) << " T=" << PRINT_FLOAT(p_angle_out.time) << " Status=" << p_angle_out.dyn
               << " Az=" << PRINT_FLOAT(p_angle_out.vec(0)) << " El=" << PRINT_FLOAT(p_angle_out.vec(1)) << std::endl;

    // Add non-static neighbor
    auto p_non_static = createTestPointWithIndices(center_point.local.x(), center_point.local.y(), center_point.local.z(), params, center_point.time, UNCERTAIN); // Assuming UNCERTAIN is not STATIC
    addPointToMap(p_non_static);
    std::cout << "[TEST DEBUG] Added non-static neighbor: D=" << PRINT_FLOAT(p_non_static.vec(2)) << " T=" << PRINT_FLOAT(p_non_static.time) << " Status=" << p_non_static.dyn << std::endl;

    std::cout << "[TEST DEBUG] Expecting FALSE." << std::endl;
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should return false when neighbors exist but none meet the criteria (time, angle, static).";
}

// Test Case: Pass - Only one close static neighbor (Rule 1 skipped, Rule 2 passes trivially)
TEST_F(ConsistencyChecksTest, DepthConsistency_PassOneCloseNeighbor) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_PassOneCloseNeighbor" << std::endl;
    center_point.vec(2) = 20.0f;
    float neighbor_depth = 20.1f;
    float depth_diff = center_point.vec(2) - neighbor_depth;
    float max_thr = params.depth_cons_depth_max_thr2;
    ASSERT_LT(std::fabs(depth_diff), max_thr);
    std::cout << "[TEST DEBUG] Center Point: H=" << center_point.hor_ind << " V=" << center_point.ver_ind
              << " D=" << PRINT_FLOAT(center_point.vec(2)) << " T=" << PRINT_FLOAT(center_point.time) << std::endl;
    std::cout << "[TEST DEBUG] Max Thr (Case 2): " << PRINT_FLOAT(max_thr) << std::endl;

    // Add neighbor in the *same cell* (delta 0,0)
    point_soph added_neighbor = addNeighborInRelativeCell(center_point, test_map, params, 0, 0, neighbor_depth, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added neighbor: H=" << added_neighbor.hor_ind << " V=" << added_neighbor.ver_ind
              << " D=" << PRINT_FLOAT(added_neighbor.vec(2)) << " T=" << PRINT_FLOAT(added_neighbor.time) << " Status=" << added_neighbor.dyn
              << " (Depth Diff = " << PRINT_FLOAT(depth_diff) << ")" << std::endl;

    std::cout << "[TEST DEBUG] Expecting TRUE (Rule 1 skipped, Rule 2 passes)." << std::endl;
    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass with only one close static neighbor.";
}

// Test Case: Pass - Multiple close neighbors, average diff below threshold, no significant diffs
TEST_F(ConsistencyChecksTest, DepthConsistency_PassAvgDiffBelowThreshold) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_PassAvgDiffBelowThreshold" << std::endl;
    center_point.vec(2) = 20.0f;
    float current_depth_threshold = std::max(params.depth_cons_depth_thr2, params.k_depth2 * center_point.vec(2));
    float max_thr = params.depth_cons_depth_max_thr2;
    float depth1 = 20.0f + current_depth_threshold * 0.4f; // diff = -0.4*thr
    float depth2 = 20.0f - current_depth_threshold * 0.3f; // diff = +0.3*thr
    float depth3 = 20.0f + current_depth_threshold * 0.2f; // diff = -0.2*thr
    float expected_avg_abs_diff = (std::fabs(center_point.vec(2) - depth1) + std::fabs(center_point.vec(2) - depth2) + std::fabs(center_point.vec(2) - depth3)) / 2.0f; // (3-1)
    ASSERT_LT(expected_avg_abs_diff, current_depth_threshold);
    ASSERT_LT(std::fabs(center_point.vec(2) - depth1), max_thr);
    ASSERT_LT(std::fabs(center_point.vec(2) - depth2), max_thr);
    ASSERT_LT(std::fabs(center_point.vec(2) - depth3), max_thr);

    std::cout << "[TEST DEBUG] Center Point: D=" << PRINT_FLOAT(center_point.vec(2)) << std::endl;
    std::cout << "[TEST DEBUG] Depth Thr (Rule 1): " << PRINT_FLOAT(current_depth_threshold) << std::endl;
    std::cout << "[TEST DEBUG] Max Thr (Categorization): " << PRINT_FLOAT(max_thr) << std::endl;
    std::cout << "[TEST DEBUG] Expected Avg Abs Diff (close): " << PRINT_FLOAT(expected_avg_abs_diff) << std::endl;

    auto n1 = addNeighborInRelativeCell(center_point, test_map, params, 0, 0, depth1, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added N1: D=" << PRINT_FLOAT(n1.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n1.vec(2)) << std::endl;
    auto n2 = addNeighborInRelativeCell(center_point, test_map, params, 1, 0, depth2, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added N2: D=" << PRINT_FLOAT(n2.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n2.vec(2)) << std::endl;
    auto n3 = addNeighborInRelativeCell(center_point, test_map, params, 0, 1, depth3, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added N3: D=" << PRINT_FLOAT(n3.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n3.vec(2)) << std::endl;

    std::cout << "[TEST DEBUG] Expecting TRUE (Rule 1 passes, Rule 2 passes)." << std::endl;
    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass when average depth diff of close neighbors is below threshold.";
}

// Test Case: Fail - Multiple close neighbors, average diff ABOVE threshold
TEST_F(ConsistencyChecksTest, DepthConsistency_FailAvgDiffAboveThreshold) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_FailAvgDiffAboveThreshold" << std::endl;
    center_point.vec(2) = 20.0f;
    float current_depth_threshold = std::max(params.depth_cons_depth_thr2, params.k_depth2 * center_point.vec(2));
    float max_thr = params.depth_cons_depth_max_thr2;
    float depth1 = 20.0f + current_depth_threshold * 1.1f; // diff = -1.1*thr
    float depth2 = 20.0f - current_depth_threshold * 1.2f; // diff = +1.2*thr
    float expected_avg_abs_diff = (std::fabs(center_point.vec(2) - depth1) + std::fabs(center_point.vec(2) - depth2)) / 1.0f; // (2-1)
    ASSERT_GT(expected_avg_abs_diff, current_depth_threshold);
    ASSERT_LT(std::fabs(center_point.vec(2) - depth1), max_thr);
    ASSERT_LT(std::fabs(center_point.vec(2) - depth2), max_thr);

    std::cout << "[TEST DEBUG] Center Point: D=" << PRINT_FLOAT(center_point.vec(2)) << std::endl;
    std::cout << "[TEST DEBUG] Depth Thr (Rule 1): " << PRINT_FLOAT(current_depth_threshold) << std::endl;
    std::cout << "[TEST DEBUG] Max Thr (Categorization): " << PRINT_FLOAT(max_thr) << std::endl;
    std::cout << "[TEST DEBUG] Expected Avg Abs Diff (close): " << PRINT_FLOAT(expected_avg_abs_diff) << std::endl;

    // *** MODIFIED: Use addNeighborInRelativeCell ***
    // Place neighbors in adjacent cells to ensure they are found
    auto n1 = addNeighborInRelativeCell(center_point, test_map, params, 1, 0, depth1, center_point.time - params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added N1: H=" << n1.hor_ind << " V=" << n1.ver_ind << " D=" << PRINT_FLOAT(n1.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n1.vec(2)) << std::endl;
    auto n2 = addNeighborInRelativeCell(center_point, test_map, params, -1, 0, depth2, center_point.time - params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added N2: H=" << n2.hor_ind << " V=" << n2.ver_ind << " D=" << PRINT_FLOAT(n2.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n2.vec(2)) << std::endl;

    std::cout << "[TEST DEBUG] Expecting FALSE (Rule 1 fails)." << std::endl;
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should fail when average depth diff of close neighbors is above threshold.";
}

// Test Case: Pass - Rule 1 passes (or skipped), Rule 2 passes (only significantly FARTHER neighbors)
TEST_F(ConsistencyChecksTest, DepthConsistency_PassOnlyFartherNeighbors) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_PassOnlyFartherNeighbors" << std::endl;
    center_point.vec(2) = 20.0f;
    float max_thr = params.depth_cons_depth_max_thr2;
    std::cout << "[TEST DEBUG] Center Point: D=" << PRINT_FLOAT(center_point.vec(2)) << std::endl;
    std::cout << "[TEST DEBUG] Max Thr (Categorization): " << PRINT_FLOAT(max_thr) << std::endl;

    // Add one close neighbor
    auto n_close = addNeighborInRelativeCell(center_point, test_map, params, 0, 0, 20.0f, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Close N: D=" << PRINT_FLOAT(n_close.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_close.vec(2)) << std::endl;

    // Add neighbors significantly FARTHER than p (p.depth > neighbor.depth)
    float farther_depth1 = center_point.vec(2) - max_thr * 1.5f;
    float farther_depth2 = center_point.vec(2) - max_thr * 2.0f;
    auto n_far1 = addNeighborInRelativeCell(center_point, test_map, params, -1, 0, farther_depth1, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Farther N1: D=" << PRINT_FLOAT(n_far1.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_far1.vec(2)) << std::endl;
    auto n_far2 = addNeighborInRelativeCell(center_point, test_map, params, 0, -1, farther_depth2, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Farther N2: D=" << PRINT_FLOAT(n_far2.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_far2.vec(2)) << std::endl;

    std::cout << "[TEST DEBUG] Expecting TRUE (Rule 1 passes/skipped, Rule 2 passes - only farther)." << std::endl;
    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass when significantly different neighbors are all farther.";
}

// Test Case: Pass - Rule 1 passes (or skipped), Rule 2 passes (only significantly CLOSER neighbors)
TEST_F(ConsistencyChecksTest, DepthConsistency_PassOnlyCloserNeighbors) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_PassOnlyCloserNeighbors" << std::endl;
    center_point.vec(2) = 20.0f;
    float max_thr = params.depth_cons_depth_max_thr2;
    std::cout << "[TEST DEBUG] Center Point: D=" << PRINT_FLOAT(center_point.vec(2)) << std::endl;
    std::cout << "[TEST DEBUG] Max Thr (Categorization): " << PRINT_FLOAT(max_thr) << std::endl;

    // Add one close neighbor
    auto n_close = addNeighborInRelativeCell(center_point, test_map, params, 0, 0, 20.0f, -params.frame_dur * 0.5, STATIC);
     std::cout << "[TEST DEBUG] Added Close N: D=" << PRINT_FLOAT(n_close.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_close.vec(2)) << std::endl;

    // Add neighbors significantly CLOSER than p (p.depth < neighbor.depth)
    float closer_depth1 = center_point.vec(2) + max_thr * 1.5f;
    float closer_depth2 = center_point.vec(2) + max_thr * 2.0f;
    auto n_close1 = addNeighborInRelativeCell(center_point, test_map, params, 1, 1, closer_depth1, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Closer N1: D=" << PRINT_FLOAT(n_close1.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_close1.vec(2)) << std::endl;
    auto n_close2 = addNeighborInRelativeCell(center_point, test_map, params, 0, 2, closer_depth2, -params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Closer N2: D=" << PRINT_FLOAT(n_close2.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_close2.vec(2)) << std::endl;

    std::cout << "[TEST DEBUG] Expecting TRUE (Rule 1 passes/skipped, Rule 2 passes - only closer)." << std::endl;
    EXPECT_TRUE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should pass when significantly different neighbors are all closer.";
}

// Test Case: Fail - Rule 1 passes (or skipped), Rule 2 fails (MIXED closer/farther neighbors)
TEST_F(ConsistencyChecksTest, DepthConsistency_FailMixedCloserFartherNeighbors) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_FailMixedCloserFartherNeighbors" << std::endl;
    center_point.vec(2) = 20.0f;
    float max_thr = params.depth_cons_depth_max_thr2;
    std::cout << "[TEST DEBUG] Center Point: D=" << PRINT_FLOAT(center_point.vec(2)) << std::endl;
    std::cout << "[TEST DEBUG] Max Thr (Categorization): " << PRINT_FLOAT(max_thr) << std::endl;

    // *** MODIFIED: Use addNeighborInRelativeCell ***
    // Add one close neighbor (in the same cell)
    auto n_close = addNeighborInRelativeCell(center_point, test_map, params, 0, 0, 20.0f, center_point.time - params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Close N: H=" << n_close.hor_ind << " V=" << n_close.ver_ind << " D=" << PRINT_FLOAT(n_close.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_close.vec(2)) << std::endl;

    // Add one significantly FARTHER neighbor (in an adjacent cell)
    float farther_depth = center_point.vec(2) - max_thr * 1.5f;
    auto n_far = addNeighborInRelativeCell(center_point, test_map, params, -1, 0, farther_depth, center_point.time - params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Farther N: H=" << n_far.hor_ind << " V=" << n_far.ver_ind << " D=" << PRINT_FLOAT(n_far.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_far.vec(2)) << std::endl;

    // Add one significantly CLOSER neighbor (in another adjacent cell)
    float closer_depth = center_point.vec(2) + max_thr * 1.5f;
    auto n_closer = addNeighborInRelativeCell(center_point, test_map, params, 0, 1, closer_depth, center_point.time - params.frame_dur * 0.5, STATIC);
    std::cout << "[TEST DEBUG] Added Closer N: H=" << n_closer.hor_ind << " V=" << n_closer.ver_ind << " D=" << PRINT_FLOAT(n_closer.vec(2)) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - n_closer.vec(2)) << std::endl;

    std::cout << "[TEST DEBUG] Expecting FALSE (Rule 1 passes/skipped, Rule 2 fails - mixed)." << std::endl;
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH))
        << "Should fail when there is a mix of significantly closer and farther neighbors.";
}

// Test Case: Ensure CASE3 uses its specific parameters (simple check)
TEST_F(ConsistencyChecksTest, DepthConsistency_UsesCase3Params) {
     std::cout << "[TEST DEBUG] Running DepthConsistency_UsesCase3Params" << std::endl;
    // ... (skip check remains the same) ...
    if (params.depth_cons_depth_max_thr2 <= params.depth_cons_depth_max_thr3) {
        GTEST_SKIP() << "Skipping test: depth_cons_depth_max_thr2 is not greater than depth_cons_depth_max_thr3 in config.";
    }
    std::cout << "[TEST DEBUG] Max Thr Case 2: " << PRINT_FLOAT(params.depth_cons_depth_max_thr2) << std::endl;
    std::cout << "[TEST DEBUG] Max Thr Case 3: " << PRINT_FLOAT(params.depth_cons_depth_max_thr3) << std::endl;

    center_point.vec(2) = 20.0f;
    std::cout << "[TEST DEBUG] Center Point: D=" << PRINT_FLOAT(center_point.vec(2)) << std::endl;

    // Clear map before adding points for the specific Case 3 check
    test_map.depth_map.assign(MAX_2D_N, std::vector<std::shared_ptr<point_soph>>());
    std::cout << "[TEST DEBUG] Cleared map for Case 3 specific check." << std::endl;

    // Add one significantly CLOSER neighbor (using CASE3 max_thr)
    float closer_depth = center_point.vec(2) + params.depth_cons_depth_max_thr3 * 1.5f;
    auto n_closer = addPointToMap(createTestPointWithIndices(center_point.local.x() + 0.01, center_point.local.y(), closer_depth, params, center_point.time - params.frame_dur * 0.5, STATIC));
    std::cout << "[TEST DEBUG] Added Closer N (for Case 3): D=" << PRINT_FLOAT(closer_depth) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - closer_depth) << std::endl;

    // Add one significantly FARTHER neighbor (using CASE3 max_thr)
    float farther_depth = center_point.vec(2) - params.depth_cons_depth_max_thr3 * 1.5f;
    auto n_far = addPointToMap(createTestPointWithIndices(center_point.local.x() - 0.01, center_point.local.y(), farther_depth, params, center_point.time - params.frame_dur * 0.5, STATIC));
    std::cout << "[TEST DEBUG] Added Farther N (for Case 3): D=" << PRINT_FLOAT(farther_depth) << " Diff=" << PRINT_FLOAT(center_point.vec(2) - farther_depth) << std::endl;

    std::cout << "[TEST DEBUG] Expecting FALSE for CASE 3 (Rule 2 fails - mixed)." << std::endl;
    EXPECT_FALSE(ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH))
        << "CASE 3 should fail with mixed closer/farther neighbors based on its specific max_thr.";

    // Optional check for Case 2 (as before)
    std::cout << "[TEST DEBUG] Checking same setup with CASE 2 parameters..." << std::endl;
    bool case2_result = ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    std::cout << "[TEST DEBUG] Case 2 result for Case 3 fail setup: " << (case2_result ? "TRUE" : "FALSE") << std::endl;
    // Potentially add an EXPECT_TRUE or EXPECT_FALSE for case2_result if the outcome is predictable and desired for the test.
}


// Test Case: Ensure calling with CASE1 throws an exception
TEST_F(ConsistencyChecksTest, DepthConsistency_ThrowsOnCase1) {
    std::cout << "[TEST DEBUG] Running DepthConsistency_ThrowsOnCase1" << std::endl;
    std::cout << "[TEST DEBUG] Expecting std::invalid_argument." << std::endl;
    EXPECT_THROW(
        ConsistencyChecks::checkDepthConsistency(center_point, test_map, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION),
        std::invalid_argument
    ) << "Should throw std::invalid_argument when called with CASE1_FALSE_REJECTION.";
}