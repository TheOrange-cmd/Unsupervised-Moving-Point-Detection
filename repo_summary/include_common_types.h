// file: common/types.h

/**
 * @file types.h
 * @brief Defines common type aliases used throughout the project, primarily using Eigen and PCL.
 */

#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#define _USE_MATH_DEFINES

/** @brief PCL point type definition including XYZ coordinates, intensity, and normal vector. */
typedef pcl::PointXYZINormal PointType;

/** @brief PCL point cloud definition using PointType. */
typedef pcl::PointCloud<PointType> PointCloudXYZI;

/** @brief Standard vector container for PointType, using Eigen's aligned allocator for fixed-size vectorizable Eigen types. */
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;

/** @brief Eigen 2D float vector alias. */
typedef Eigen::Vector2f V2F;
/** @brief Eigen 3D double vector alias. */
typedef Eigen::Vector3d V3D;
/** @brief Eigen 3x3 double matrix alias. */
typedef Eigen::Matrix3d M3D;
/** @brief Eigen 3D float vector alias. */
typedef Eigen::Vector3f V3F;
/** @brief Eigen 3x3 float matrix alias. */
typedef Eigen::Matrix3f M3F;

// --- Eigen Matrix/Vector Macros (Consider replacing with explicit typedefs for clarity) ---

/** @brief Macro for defining an Eigen double matrix of size (a)x(b). */
#define MD(a,b)  Eigen::Matrix<double, (a), (b)>
/** @brief Macro for defining an Eigen double column vector of size (a). */
#define VD(a)    Eigen::Matrix<double, (a), 1>
/** @brief Macro for defining an Eigen float matrix of size (a)x(b). */
#define MF(a,b)  Eigen::Matrix<float, (a), (b)>
/** @brief Macro for defining an Eigen float column vector of size (a). */
#define VF(a)    Eigen::Matrix<float, (a), 1>

// --- Configuration for Detailed Logging ---
// Set DEBUG_POINT_IDX to a specific point's original_index to log only that point.
// Set DEBUG_FRAME_SEQ_ID to a specific frame's seq_id to log only within that frame (requires DEBUG_POINT_IDX != -1).
// Set LOG_FREQUENCY > 0 to log every Nth point (only active if DEBUG_POINT_IDX == -1).

// *** SET YOUR DEBUG VALUES HERE ***
const int DEBUG_POINT_IDX = -1;    // <-- SET TO 0 for your specific case
const int DEBUG_FRAME_SEQ_ID = -1; // <-- SET TO 2 for your specific case
// const int DEBUG_POINT_IDX = -1;    // Example: Disable specific point/frame logging
// const int DEBUG_FRAME_SEQ_ID = -1; // Example: Disable specific point/frame logging
const int LOG_FREQUENCY = 1000;    // Fallback if specific point disabled
namespace PointCloudUtils {
    inline uint64_t g_current_logging_seq_id = 0;
}

#endif