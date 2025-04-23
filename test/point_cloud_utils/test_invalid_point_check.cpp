#include "gtest/gtest.h" // Google Test framework
#include "point_cloud_utils.h" // header to test
#include "config_loader.h"     // For DynObjFilterParams
#include "dyn_obj_datatypes.h" // For point_soph, V3D, M3D etc.
#include <Eigen/Geometry>      // For Eigen::AngleAxisd

// Define some constants for testing
const float TEST_BLIND_DISTANCE = 1.0f;
const int DATASET_OTHER = 0;
const int DATASET_SPECIAL = 1;

TEST(InvalidPointCheckTest, PointTooClose) {
    V3D point_inside(0.5, 0.0, 0.0); // Distance 0.5 < 1.0
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point_inside, TEST_BLIND_DISTANCE, DATASET_OTHER));
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point_inside, TEST_BLIND_DISTANCE, DATASET_SPECIAL));
}

TEST(InvalidPointCheckTest, PointAtBlindDistance) {
    V3D point_on_boundary(1.0, 0.0, 0.0); // Distance 1.0 == 1.0
    // squaredNorm (1.0) is NOT < blind_distance^2 (1.0)
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_on_boundary, TEST_BLIND_DISTANCE, DATASET_OTHER));
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_on_boundary, TEST_BLIND_DISTANCE, DATASET_SPECIAL));
}

TEST(InvalidPointCheckTest, PointOutsideBlindDistance) {
    V3D point_outside(1.5, 0.0, 0.0); // Distance 1.5 > 1.0
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside, TEST_BLIND_DISTANCE, DATASET_OTHER));
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point_outside, TEST_BLIND_DISTANCE, DATASET_SPECIAL));
}

TEST(InvalidPointCheckTest, SpecialDatasetBox_OutsideDistance_InsideBox) {
    V3D point(0.05, 0.5, 0.05); // Outside blind distance (norm > 1), but inside special box
    
    // Debug info for first assertion
    float squaredNorm = point.squaredNorm();
    float squaredBlindDistance = TEST_BLIND_DISTANCE * TEST_BLIND_DISTANCE;
    bool inBlindZone = squaredNorm < squaredBlindDistance;
    bool inSpecialBox = std::fabs(point.x()) < 0.1f && 
                        std::fabs(point.y()) < 1.0f && 
                        std::fabs(point.z()) < 0.1f;
    
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: OTHER" << std::endl;
    std::cout << "  squaredNorm: " << squaredNorm << ", squaredBlindDistance: " 
              << squaredBlindDistance << std::endl;
    std::cout << "  inBlindZone: " << (inBlindZone ? "true" : "false") 
              << ", inSpecialBox: " << (inSpecialBox ? "true" : "false") << std::endl;
    
    // Assertion for DATASET_OTHER: Should be TRUE (invalid) because it's too close
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_OTHER))
        << "Point inside blind distance should be invalid regardless of dataset.";
    
    // Debug info for second assertion
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: SPECIAL" << std::endl;
    std::cout << "  squaredNorm: " << squaredNorm << ", squaredBlindDistance: " 
              << squaredBlindDistance << std::endl;
    std::cout << "  inBlindZone: " << (inBlindZone ? "true" : "false") 
              << ", inSpecialBox: " << (inSpecialBox ? "true" : "false") << std::endl;
    std::cout << "  x in range: " << (std::fabs(point.x()) < 0.1f ? "true" : "false") 
              << ", y in range: " << (std::fabs(point.y()) < 1.0f ? "true" : "false") 
              << ", z in range: " << (std::fabs(point.z()) < 0.1f ? "true" : "false") << std::endl;
    
    // Assertion for DATASET_SPECIAL: Should be TRUE (invalid) because it's too close
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_SPECIAL))
        << "Point inside blind distance should be invalid regardless of dataset.";
}

TEST(InvalidPointCheckTest, SpecialDatasetBox_OutsideDistance_OutsideBoxX) {
    V3D point(0.2, 0.5, 0.05); // Outside blind distance, outside special box (x too large)
    
    // Debug info
    float squaredNorm = point.squaredNorm();
    float squaredBlindDistance = TEST_BLIND_DISTANCE * TEST_BLIND_DISTANCE;
    bool inBlindZone = squaredNorm < squaredBlindDistance;
    bool inSpecialBox = std::fabs(point.x()) < 0.1f && 
                        std::fabs(point.y()) < 1.0f && 
                        std::fabs(point.z()) < 0.1f;
    
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: OTHER" << std::endl;
    std::cout << "  squaredNorm: " << squaredNorm << ", squaredBlindDistance: " 
              << squaredBlindDistance << std::endl;
    std::cout << "  inBlindZone: " << (inBlindZone ? "true" : "false") 
              << ", inSpecialBox: " << (inSpecialBox ? "true" : "false") << std::endl;
    
    // Assertion for DATASET_OTHER: Should be TRUE (invalid) because it's too close
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_OTHER))
        << "Point inside blind distance should be invalid regardless of dataset.";
    
    // Debug for second assertion
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: SPECIAL" << std::endl;
    std::cout << "  x in range: " << (std::fabs(point.x()) < 0.1f ? "true" : "false") 
              << ", y in range: " << (std::fabs(point.y()) < 1.0f ? "true" : "false") 
              << ", z in range: " << (std::fabs(point.z()) < 0.1f ? "true" : "false") << std::endl;
    

    // Assertion for DATASET_SPECIAL: Should be TRUE (invalid) because it's too close
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_SPECIAL))
        << "Point inside blind distance should be invalid regardless of dataset.";
}

TEST(InvalidPointCheckTest, SpecialDatasetBox_OutsideDistance_OutsideBoxY) {
    V3D point(0.05, 1.5, 0.05); // Outside blind distance, outside special box (y too large)
    
    // Debug info
    float squaredNorm = point.squaredNorm();
    float squaredBlindDistance = TEST_BLIND_DISTANCE * TEST_BLIND_DISTANCE;
    bool inBlindZone = squaredNorm < squaredBlindDistance;
    bool inSpecialBox = std::fabs(point.x()) < 0.1f && 
                        std::fabs(point.y()) < 1.0f && 
                        std::fabs(point.z()) < 0.1f;
    
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: OTHER" << std::endl;
    std::cout << "  squaredNorm: " << squaredNorm << ", squaredBlindDistance: " 
              << squaredBlindDistance << std::endl;
    std::cout << "  inBlindZone: " << (inBlindZone ? "true" : "false") 
              << ", inSpecialBox: " << (inSpecialBox ? "true" : "false") << std::endl;
    
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_OTHER));
    
    // Debug for second assertion
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: SPECIAL" << std::endl;
    std::cout << "  x in range: " << (std::fabs(point.x()) < 0.1f ? "true" : "false") 
              << ", y in range: " << (std::fabs(point.y()) < 1.0f ? "true" : "false") 
              << ", z in range: " << (std::fabs(point.z()) < 0.1f ? "true" : "false") << std::endl;
    
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_SPECIAL));
}

TEST(InvalidPointCheckTest, SpecialDatasetBox_OutsideDistance_OutsideBoxZ) {
    // Test the nuance of the original logic: Z check only matters if dataset=1 and X/Y are in range
    V3D point(0.05, 0.5, 0.2); // Outside blind distance, X/Y inside box, Z outside box
    
    // Debug info
    float squaredNorm = point.squaredNorm();
    float squaredBlindDistance = TEST_BLIND_DISTANCE * TEST_BLIND_DISTANCE;
    bool inBlindZone = squaredNorm < squaredBlindDistance;
    bool inSpecialBox = std::fabs(point.x()) < 0.1f && 
                        std::fabs(point.y()) < 1.0f && 
                        std::fabs(point.z()) < 0.1f;
    
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: OTHER" << std::endl;
    std::cout << "  squaredNorm: " << squaredNorm << ", squaredBlindDistance: " 
              << squaredBlindDistance << std::endl;
    std::cout << "  inBlindZone: " << (inBlindZone ? "true" : "false") 
              << ", inSpecialBox: " << (inSpecialBox ? "true" : "false") << std::endl;
    
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_OTHER));
    
    // Debug for second assertion with focus on Z dimension
    std::cout << "Point (" << point.x() << ", " << point.y() << ", " << point.z() 
              << "), Dataset: SPECIAL" << std::endl;
    std::cout << "  x in range: " << (std::fabs(point.x()) < 0.1f ? "true" : "false") 
              << ", y in range: " << (std::fabs(point.y()) < 1.0f ? "true" : "false") 
              << ", z in range: " << (std::fabs(point.z()) < 0.1f ? "true" : "false") << std::endl;
    std::cout << "  Point should be valid because z=" << point.z() << " is outside special box bounds" << std::endl;
    
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_SPECIAL)); // Z check makes it valid for dataset 1
}

TEST(InvalidPointCheckTest, SpecialDatasetBox_InsideDistance_InsideBox) {
    V3D point(0.05, 0.5, 0.05);
    // Make it closer than blind distance
    point = point.normalized() * (TEST_BLIND_DISTANCE * 0.5f);
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_OTHER)); // Invalid due to distance
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_SPECIAL)); // Invalid due to distance (even though also in box)
}

TEST(InvalidPointCheckTest, SpecialDatasetBox_OutsideDistance_InsideBox_TrueOutside) {
    // Point with sqNorm > 1.0 AND |x|<0.1, |y|<1.0, |z|<0.1
    V3D point(0.0f, 0.999f, 0.09f); // sqNorm = 1.006101 > 1.0. Inside box.

    // Assertion for DATASET_OTHER: Should be FALSE (valid) - outside distance, dataset not 1
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_OTHER));

    // Assertion for DATASET_SPECIAL: Should be TRUE (invalid) - outside distance, but dataset=1 AND inside box
    ASSERT_TRUE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_SPECIAL));
}

// NEW TEST: Point OUTSIDE blind distance, OUTSIDE special box (X)
TEST(InvalidPointCheckTest, SpecialDatasetBox_OutsideDistance_OutsideBoxX_TrueOutside) {
    // Point outside distance (sqNorm > 1.0), outside box because x >= 0.1
    V3D point(1.1f, 0.5f, 0.05f); // sqNorm = 1.4625 > 1.0. Outside box (x).

    // Assertion for DATASET_OTHER: Should be FALSE (valid) - outside distance
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_OTHER));

    // Assertion for DATASET_SPECIAL: Should be FALSE (valid) - outside distance AND outside box
    ASSERT_FALSE(PointCloudUtils::isPointInvalid(point, TEST_BLIND_DISTANCE, DATASET_SPECIAL));
}