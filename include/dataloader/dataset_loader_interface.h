#ifndef DATA_LOADER_INTERFACE_H
#define DATA_LOADER_INTERFACE_H

#include <memory> // For std::unique_ptr

class DatasetLoader {
public:
    virtual ~DatasetLoader() = default; // Important: Virtual destructor

    // --- Methods for Simplified Loader (Focus on these first) ---

    // Load Lidar point cloud for a specific sensor at a specific time instance
    // Inputs might be file paths initially, later tokens/timestamps
    virtual PointCloudData loadLidarData(const std::string& lidar_data_path) = 0;

    // Load Ego Pose at a specific time instance
    virtual Pose loadEgoPose(const std::string& ego_pose_path) = 0;

    // Load Sensor Calibration data (pose of sensor relative to ego)
    virtual Pose loadSensorCalibration(const std::string& calibration_path) = 0;

    // --- Methods for Full Loader (Implement Later) ---

    // Get available scene names/identifiers
    // virtual std::vector<std::string> getSceneIdentifiers() = 0;

    // Get sample tokens/timestamps for a given scene
    // virtual std::vector<std::string> getSampleIdentifiers(const std::string& scene_id) = 0;

    // Get data paths based on identifiers (internal helper or part of interface)
    // virtual std::string getLidarPath(const std::string& sample_id, const std::string& sensor_name) = 0;
    // virtual std::string getEgoPosePath(const std::string& sample_id) = 0;
    // ... etc.

    // Methods to load data based on sample/sensor identifiers
    // virtual PointCloudData loadLidarDataById(const std::string& sample_id, const std::string& sensor_name) = 0;
    // virtual Pose loadEgoPoseById(const std::string& sample_id) = 0;
    // virtual Pose loadSensorCalibrationById(const std::string& sample_id, const std::string& sensor_name) = 0;

};

#endif // DATA_LOADER_INTERFACE_H