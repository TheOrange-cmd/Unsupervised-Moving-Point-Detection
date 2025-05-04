// file: test/filtering/test_occlusion_check_relationship.cpp

#include <gtest/gtest.h>
#include "test/test_consistency_helpers.hpp" // Base fixture and standalone helpers
#include "filtering/consistency_checks.h"    // Function under test
#include "filtering/dyn_obj_datatypes.h"
#include <cmath>
#include <stdexcept>
#include <limits>

// Fixture specific to checkOcclusionRelationship tests
class OcclusionRelationshipTest : public ConsistencyChecksTest {
protected:
    point_soph p_occluder; // Potential occluder (closer, later)
    point_soph p_occluded; // Potential occluded (farther, earlier)

    void SetUp() override {
        // Run base setup first (loads config into params, clears map)
        ConsistencyChecksTest::SetUp();

        // --- Default point setup (can be overridden in tests) ---
        // Occluder: Closer, later time
        p_occluder = createTestPointWithIndices(10.0f, 0.0f, 1.0f, params, 0.1); // time=0.1
        p_occluder.vec(2) = 10.0f; // Depth = 10m
        p_occluder.local = V3D(10.0, 0.0, 1.0); // Set local coords for self-check

        // Occluded: Farther, earlier time
        p_occluded = createTestPointWithIndices(10.0f, 0.0f, 1.0f, params, 0.0); // time=0.0
        p_occluded.vec(2) = 12.0f; // Depth = 12m (farther) - will be adjusted in tests
        p_occluded.local = V3D(12.0, 0.0, 1.2); // Set different local coords
        p_occluded.dyn = DynObjLabel::STATIC; // Default to valid static

        // Make angles identical by default for angular tests
        p_occluded.vec(0) = p_occluder.vec(0);
        p_occluded.vec(1) = p_occluder.vec(1);

        // --- Assert necessary params are loaded (add more as needed) ---
        ASSERT_NE(params.occ_hor_thr2, 0.0f);
        ASSERT_NE(params.occ_ver_thr2, 0.0f);
        ASSERT_NE(params.occ_depth_thr2, 0.0f); // Base offset Case 2
        ASSERT_NE(params.v_min_thr2, 0.0f);
        ASSERT_NE(params.occ_hor_thr3, 0.0f);
        ASSERT_NE(params.occ_ver_thr3, 0.0f);
        ASSERT_NE(params.map_cons_depth_thr3, 0.0f); // Base offset Case 3
        ASSERT_NE(params.v_min_thr3, 0.0f);
        // Add asserts for k_depth_max_thr*, d_depth_max_thr*, cutoff_value, self_*, dataset if crucial
        ASSERT_GE(params.cutoff_value, 0.0f);
        ASSERT_GT(params.enlarge_distort, 0.0f); // Should be >= 1.0 ideally
    }

    // No specific helpers needed here anymore, calculateOcclusionDepthThreshold is standalone
};


// --- Basic Pass Cases ---
TEST_F(OcclusionRelationshipTest, Case2_Pass_Basic) {
    // Default setup should pass basic checks for Case 2 if thresholds are reasonable
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold)) << "Threshold calculation failed (NaN)";
    ASSERT_GT(threshold, 0.0f); // Ensure threshold calculation is valid and positive
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f; // Ensure depth passes

    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case3_Pass_Basic) {
    // Default setup should pass basic checks for Case 3 if thresholds are reasonable
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
    ASSERT_FALSE(std::isnan(threshold)) << "Threshold calculation failed (NaN)";
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f; // Ensure depth passes

    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

// --- Failure Cases ---

TEST_F(OcclusionRelationshipTest, Fail_InvalidOccludedStatus) {
    p_occluded.dyn = DynObjLabel::INVALID;
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

// TEST_F(OcclusionRelationshipTest, Fail_DistortionDataset0_Occluded) {
//     int original_dataset = params.dataset;
//     params.dataset = 0;
//     p_occluded.is_distort = true;
//     EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
//     EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
//     params.dataset = original_dataset; // Restore
// }

// TEST_F(OcclusionRelationshipTest, Fail_DistortionDataset0_Occluder) {
//     int original_dataset = params.dataset;
//     params.dataset = 0;
//     p_occluder.is_distort = true;
//     // Adjust depth threshold calculation for distortion effect if needed for other tests
//     float threshold_case2 = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
//     float threshold_case3 = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
//     ASSERT_FALSE(std::isnan(threshold_case2));
//     ASSERT_FALSE(std::isnan(threshold_case3));
//     p_occluded.vec(2) = p_occluder.vec(2) + std::max(threshold_case2, threshold_case3) + 0.1f; // Ensure depth passes *without* distortion check

//     // Now check the function which *should* apply the distortion check internally
//     EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
//     EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
//     params.dataset = original_dataset; // Restore
// }

TEST_F(OcclusionRelationshipTest, Pass_DistortionNotDataset0) {
    int original_dataset = params.dataset;
    params.dataset = 1; // Or any non-zero value
    p_occluder.is_distort = true;
    // Adjust depth to pass threshold for Case 2 (distortion effect ignored)
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;

    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    params.dataset = original_dataset; // Restore
}


TEST_F(OcclusionRelationshipTest, Fail_SelfOcclusion_Occluder) {
    // Place occluder inside self box (adjust coords based on params.self_*)
    ASSERT_LT(params.self_x_b, params.self_x_f);
    ASSERT_LT(params.self_y_r, params.self_y_l);
    p_occluder.local = V3D(params.self_x_b + 0.1, params.self_y_r + 0.1, 1.0); // Use valid Z
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Fail_SelfOcclusion_Occluded) {
    // Place occluded inside self box
    ASSERT_LT(params.self_x_b, params.self_x_f);
    ASSERT_LT(params.self_y_r, params.self_y_l);
    p_occluded.local = V3D(params.self_x_b + 0.1, params.self_y_r + 0.1, 1.2); // Use valid Z
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Fail_TimeDeltaZero) {
    p_occluder.time = p_occluded.time; // Same time
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Fail_TimeDeltaNegative) {
    p_occluder.time = p_occluded.time - 0.01; // Occluder earlier than occluded
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

// --- Depth Threshold Tests ---
TEST_F(OcclusionRelationshipTest, Case2_Fail_DepthJustBelowThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold - 0.001f; // Just below
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case2_Pass_DepthJustAboveThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.001f; // Just above
    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case3_Fail_DepthJustBelowThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold - 0.001f; // Just below
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case3_Pass_DepthJustAboveThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.001f; // Just above
    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

// --- Angular Threshold Tests ---
TEST_F(OcclusionRelationshipTest, Case2_Fail_AngleHorJustAboveThreshold) {
    // Ensure depth passes first
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    // Set angle difference just outside threshold
    p_occluded.vec(0) = p_occluder.vec(0) + params.occ_hor_thr2 + 0.0001f;
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case2_Pass_AngleHorJustBelowThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    p_occluded.vec(0) = p_occluder.vec(0) + params.occ_hor_thr2 - 0.0001f;
    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case2_Fail_AngleVerJustAboveThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    p_occluded.vec(1) = p_occluder.vec(1) + params.occ_ver_thr2 + 0.0001f;
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case2_Pass_AngleVerJustBelowThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    p_occluded.vec(1) = p_occluder.vec(1) + params.occ_ver_thr2 - 0.0001f;
    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH));
}

// --- Case 3 Angular Tests ---
TEST_F(OcclusionRelationshipTest, Case3_Fail_AngleHorJustAboveThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    p_occluded.vec(0) = p_occluder.vec(0) + params.occ_hor_thr3 + 0.0001f;
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case3_Pass_AngleHorJustBelowThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
     ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    p_occluded.vec(0) = p_occluder.vec(0) + params.occ_hor_thr3 - 0.0001f;
    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case3_Fail_AngleVerJustAboveThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
     ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    p_occluded.vec(1) = p_occluder.vec(1) + params.occ_ver_thr3 + 0.0001f;
    EXPECT_FALSE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

TEST_F(OcclusionRelationshipTest, Case3_Pass_AngleVerJustBelowThreshold) {
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
     ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;
    p_occluded.vec(1) = p_occluder.vec(1) + params.occ_ver_thr3 - 0.0001f;
    EXPECT_TRUE(ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE3_OCCLUDED_SEARCH));
}

// --- Invalid Type ---
TEST_F(OcclusionRelationshipTest, Fail_InvalidCheckType) {
    // Ensure depth passes for Case 2 first (arbitrary valid case)
    float threshold = calculateOcclusionDepthThreshold(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE2_OCCLUDER_SEARCH);
    ASSERT_FALSE(std::isnan(threshold));
    ASSERT_GT(threshold, 0.0f);
    p_occluded.vec(2) = p_occluder.vec(2) + threshold + 0.1f;

    EXPECT_THROW(
        ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, ConsistencyChecks::ConsistencyCheckType::CASE1_FALSE_REJECTION),
        std::invalid_argument
    );
    // Also test the default case in the switch
     EXPECT_THROW(
        ConsistencyChecks::checkOcclusionRelationship(p_occluder, p_occluded, params, static_cast<ConsistencyChecks::ConsistencyCheckType>(99)), // Invalid enum value
        std::invalid_argument
    );
}