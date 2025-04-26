
// file: include/dataloader/dataloader_types.h

/**
 * @file dataloader_types.h
 * @brief Defines common structs used specifically for the dataloader module. 
 * Dataset specific loaders *must* use these structs to ensure consistent handling of the data within the algorithm. 
 */
 

#include <vector>
#include <string>
#include <Eigen/Dense> // Or your preferred linear algebra library

// Basic point structure
struct PointXYZI {
    float x, y, z, intensity;
    // double timestamp; // Optional: Add if needed later, NuScenes .bin doesn't store per-point time
    // uint16_t ring_index; // Optional: Add if needed later
};

// Structure to hold a point cloud from a single sensor sweep
struct PointCloudData {
    double timestamp;       // Timestamp of the sweep (e.g., from sample_data)
    std::string frame_id;   // Coordinate frame the points are in (e.g., "lidar_top", "ego_vehicle")
    std::vector<PointXYZI> points;
};

// Structure for poses (ego vehicle or sensor calibration)
// Using Eigen for rotation (Quaternion) and translation (Vector3d)
struct Pose {
    double timestamp;       // Timestamp associated with this pose
    std::string frame_id;   // The frame this pose is defined in (e.g., "map")
    std::string child_frame_id; // The frame being defined by this pose (e.g., "ego_vehicle")
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation; // w, x, y, z order is common but be consistent!

    // Helper to get transformation matrix
    Eigen::Matrix4d getTransformMatrix() const {
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3,3>(0,0) = rotation.toRotationMatrix();
        transform.block<3,1>(0,3) = translation;
        return transform;
    }
};

// Could add structs for Annotations, CameraImages etc. later