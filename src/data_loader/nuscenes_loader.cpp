// file: src/dataloader/nuscenes_loader.cpp

/**
 * @file nuscenes_loader.cpp
 * @brief Implements the NuScenesLoader class methods for loading data from the NuScenes dataset.
 */

 #include "dataloader/nuscenes_loader.h" // Include the header for the class declaration
 #include <fstream>
 #include <stdexcept> // For std::runtime_error
 #include <vector>
 #include <nlohmann/json.hpp> // Include the full JSON library HERE
 
 // Use the json alias within this cpp file
 using json = nlohmann::json;
 
 // --- Method Implementations ---
 
 PointCloudData NuScenesLoader::loadLidarData(const std::string& lidar_data_path) {
     PointCloudData pc_data;
     pc_data.frame_id = "lidar_top"; // Assuming LIDAR_TOP for now - TODO: Make configurable?
     // NuScenes .bin: Each point is [x, y, z, intensity, ring_index(optional)] floats.
     // Assuming 4 floats (x, y, z, intensity) per point. Check NuScenes docs if 5 is possible.
     const size_t point_dimension = 4; // x, y, z, intensity
 
     std::ifstream file(lidar_data_path, std::ios::binary);
     if (!file) {
         throw std::runtime_error("Cannot open Lidar file: " + lidar_data_path);
     }
 
     // Get file size
     file.seekg(0, std::ios::end);
     std::streampos file_size = file.tellg();
     file.seekg(0, std::ios::beg);
 
     // Check if file size is valid for the point dimension
     size_t num_bytes = static_cast<size_t>(file_size);
     if (num_bytes == 0) {
         // Handle empty file case - return empty PointCloudData or throw?
         // Returning empty is often safer.
         pc_data.timestamp = 0.0; // Placeholder timestamp
         return pc_data;
     }
     if (num_bytes % (point_dimension * sizeof(float)) != 0) {
             throw std::runtime_error("Lidar file size mismatch: " + lidar_data_path + ". Size " + std::to_string(num_bytes) + " not divisible by " + std::to_string(point_dimension * sizeof(float)));
     }
     size_t num_points = num_bytes / (point_dimension * sizeof(float));
     pc_data.points.resize(num_points);
 
     // Read all points at once for efficiency
     // Using a unique_ptr for automatic memory management is safer than raw vector
     // though a vector on the stack might be fine if num_points isn't excessively large.
     // std::vector<float> buffer(num_points * point_dimension); // Stack/Heap depending on size
     auto buffer = std::make_unique<float[]>(num_points * point_dimension); // Heap allocated buffer
 
     file.read(reinterpret_cast<char*>(buffer.get()), num_bytes);
     if (!file) {
             throw std::runtime_error("Error reading Lidar file: " + lidar_data_path + ". Read " + std::to_string(file.gcount()) + " bytes out of " + std::to_string(num_bytes));
     }
 
     // #pragma omp parallel for // Optional: Parallelize if num_points is large and worth the overhead
     for (size_t i = 0; i < num_points; ++i) {
         pc_data.points[i].x = buffer[i * point_dimension + 0];
         pc_data.points[i].y = buffer[i * point_dimension + 1];
         pc_data.points[i].z = buffer[i * point_dimension + 2];
         pc_data.points[i].intensity = buffer[i * point_dimension + 3];
     }
 
     // Timestamp needs to be added from corresponding sample_data JSON
     // This function currently doesn't have access to that info.
     // The caller might need to set it, or loadLidarData needs more context.
     pc_data.timestamp = 0.0; // Placeholder - CRITICAL TO FIX LATER
 
     // IMPORTANT: NuScenes Lidar data is in the SENSOR's coordinate frame.
     return pc_data;
 }
 
 Pose NuScenesLoader::loadEgoPose(const std::string& ego_pose_path) {
     std::ifstream f(ego_pose_path);
         if (!f) throw std::runtime_error("Cannot open ego pose file: " + ego_pose_path);
 
     json data;
     try {
         data = json::parse(f);
     } catch (const json::parse_error& e) {
         throw std::runtime_error("Failed to parse ego pose JSON: " + ego_pose_path + " - " + e.what());
     }
 
     Pose pose;
     try {
         pose.timestamp = data.at("timestamp").get<double>() / 1e6; // Convert microseconds to seconds
         pose.frame_id = "map"; // NuScenes ego pose is map -> ego
         pose.child_frame_id = "ego_vehicle"; // TODO: Make configurable?
 
         const auto& translation_json = data.at("translation");
         pose.translation = Eigen::Vector3d(
             translation_json.at(0).get<double>(),
             translation_json.at(1).get<double>(),
             translation_json.at(2).get<double>()
         );
 
         const auto& rotation_json = data.at("rotation");
         // NuScenes uses [w, x, y, z] order for quaternions
         pose.rotation = Eigen::Quaterniond(
             rotation_json.at(0).get<double>(), // w
             rotation_json.at(1).get<double>(), // x
             rotation_json.at(2).get<double>(), // y
             rotation_json.at(3).get<double>()  // z
         ).normalized(); // Ensure it's a unit quaternion
 
     } catch (const json::out_of_range& e) {
          throw std::runtime_error("Missing expected field in ego pose JSON: " + ego_pose_path + " - " + e.what());
     } catch (const json::type_error& e) {
          throw std::runtime_error("Type error in ego pose JSON field: " + ego_pose_path + " - " + e.what());
     }
 
     return pose;
 }
 
 Pose NuScenesLoader::loadSensorCalibration(const std::string& calibration_path) {
         std::ifstream f(calibration_path);
         if (!f) throw std::runtime_error("Cannot open calibration file: " + calibration_path);
 
     json data;
      try {
         data = json::parse(f);
     } catch (const json::parse_error& e) {
         throw std::runtime_error("Failed to parse calibration JSON: " + calibration_path + " - " + e.what());
     }
 
     Pose pose;
      try {
         pose.timestamp = 0.0; // Calibration is static, timestamp not really relevant
         pose.frame_id = "ego_vehicle"; // NuScenes calibration is ego -> sensor
         pose.child_frame_id = "lidar_top"; // TODO: Assume sensor name from context or filename? Make configurable?
 
         const auto& translation_json = data.at("translation");
         pose.translation = Eigen::Vector3d(
             translation_json.at(0).get<double>(),
             translation_json.at(1).get<double>(),
             translation_json.at(2).get<double>()
         );
 
         const auto& rotation_json = data.at("rotation");
         pose.rotation = Eigen::Quaterniond(
             rotation_json.at(0).get<double>(), // w
             rotation_json.at(1).get<double>(), // x
             rotation_json.at(2).get<double>(), // y
             rotation_json.at(3).get<double>()  // z
         ).normalized();
 
     } catch (const json::out_of_range& e) {
             throw std::runtime_error("Missing expected field in calibration JSON: " + calibration_path + " - " + e.what());
     } catch (const json::type_error& e) {
             throw std::runtime_error("Type error in calibration JSON field: " + calibration_path + " - " + e.what());
     }
 
     return pose;
 }
 
 // --- Implement Full Loader Methods Later ---
 // ...