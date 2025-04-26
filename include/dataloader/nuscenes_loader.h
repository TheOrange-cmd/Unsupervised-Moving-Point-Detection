// file: include/dataloader/nuscenes_loader.h

/**
 * @file nuscenes_loader.h
 * @brief Concrete implementation of the DatasetLoader interface for the NuScenes dataset.
 *        Declares the NuScenesLoader class. Implementations are in nuscenes_loader.cpp.
 */

 #ifndef NUSCENES_LOADER_H
 #define NUSCENES_LOADER_H
 
 #include "dataloader_interface.h" // Use the correct relative path if needed
 #include "dataloader_types.h"     // Include the type definitions
 
 // Forward declare json class to avoid including the full json header here if possible,
 // ONLY if you only use json objects by reference or pointer in the header.
 // If you use json objects directly (by value) as members or return types
 // in the header, you MUST include <nlohmann/json.hpp>.
 // In this case, we don't use it directly in the header, so forward declaration is okay.
 namespace nlohmann { template<typename T, typename S, typename ...Args> class basic_json; using json = basic_json<std::map, std::vector, std::string, bool, std::int64_t, std::uint64_t, double, std::allocator, adl_serializer>; }
 // Alternatively, just include it if forward declaration is complex/unwanted:
 // #include <nlohmann/json.hpp>
 
 
 class NuScenesLoader : public DatasetLoader {
 public:
     /**
      * @brief Constructor. Can be extended later to take dataset root path, version etc.
      */
     NuScenesLoader() = default;
 
     // --- Declare the overridden methods ---
 
     /**
      * @brief Loads Lidar point cloud data from a NuScenes .bin file.
      * @param lidar_data_path Path to the specific .bin file.
      * @return PointCloudData struct containing points in the SENSOR's coordinate frame.
      * @throws std::runtime_error if the file cannot be opened or read correctly.
      */
     PointCloudData loadLidarData(const std::string& lidar_data_path) override;
 
     /**
      * @brief Loads ego vehicle pose data from a NuScenes ego_pose JSON file.
      * @param ego_pose_path Path to the specific ego_pose JSON file.
      * @return Pose struct representing the ego vehicle's pose in the MAP frame.
      * @throws std::runtime_error if the file cannot be opened or parsed.
      * @throws nlohmann::json::parse_error if JSON parsing fails.
      */
     Pose loadEgoPose(const std::string& ego_pose_path) override;
 
     /**
      * @brief Loads sensor calibration data from a NuScenes calibrated_sensor JSON file.
      * @param calibration_path Path to the specific calibrated_sensor JSON file.
      * @return Pose struct representing the sensor's pose in the EGO VEHICLE frame.
      * @throws std::runtime_error if the file cannot be opened or parsed.
      * @throws nlohmann::json::parse_error if JSON parsing fails.
      */
     Pose loadSensorCalibration(const std::string& calibration_path) override;
 
     // --- Declare Full Loader Methods Later ---
     // ...
 };
 
 #endif // NUSCENES_LOADER_H