// file: src/config_loader.cpp

/**
 * @file config_loader.cpp
 * @brief Implements the functionality for loading dynamic object filter parameters from a YAML file.
 *
 * This file contains the implementation of the load_config function, which reads
 * parameters from a YAML file specified by the filename, populates a DynObjFilterParams
 * struct, and calculates derived parameters needed for the filtering algorithms.
 */

#define _USE_MATH_DEFINES

#include "config/config_loader.h"

#include <fstream>
#include <iostream>
#include <cmath> // For std::ceil, std::floor, std::round

// #include "filtering/dyn_obj_datatypes.h" // For PI_MATH definition (though it should ideally be in a common math header)
 
 // Note: The loadParam template function is defined in the header (config_loader.h)
 
 /**
  * @brief Loads dynamic object filter parameters from a specified YAML file.
  *
  * Opens and parses the YAML file using YAML::LoadFile. It expects parameters
  * to be nested under a top-level key named 'dyn_obj'. It iterates through all
  * expected parameters, calling the loadParam helper function for each. If any
  * parameter fails to load (due to type mismatch or other YAML errors), it flags
  * the loading process as unsuccessful but continues trying to load others.
  *
  * After attempting to load all base parameters, it checks if the loading was
  * successful so far. If yes, it proceeds to calculate derived parameters like
  * pixel counts based on angular thresholds and resolutions, and FOV limits in
  * terms of pixel indices. It performs checks to prevent division by zero during
  * these calculations.
  *
  * @param filename The path to the YAML configuration file.
  * @param[out] params The DynObjFilterParams struct to be populated. Parameters
  *             are initialized with defaults (from the struct's constructor) and
  *             overwritten by values found in the file. If a parameter is missing
  *             in the file, its default value is retained.
  * @return true if the file was opened, parsed, all required base parameters were
  *         loaded or defaulted without critical errors, and derived parameters
  *         were calculated successfully.
  * @return false if the file could not be opened, a critical YAML parsing error
  *         occurred, essential base parameters (like resolutions) failed to load
  *         or were invalid, or an error occurred during derived parameter calculation.
  */
 bool load_config(const std::string& filename, DynObjFilterParams& params) {
   std::ifstream fin(filename);
   if (!fin.is_open()) {
    throw std::runtime_error("Error: Could not open config file: " + filename);
   }
   // Close the ifstream, YAML::LoadFile handles file reading
   fin.close();
 
   bool success = true;  // Track overall success
 
   try {
    YAML::Node config = YAML::LoadFile(filename);
    YAML::Node dyn_obj_node =
        config["dyn_obj"];  // Assuming params are under 'dyn_obj' group

    if (!dyn_obj_node) {
    std::cerr << "Error: 'dyn_obj' group not found in config file: "
                << filename << ". Using default parameters." << std::endl;
    // Still return true, as we can proceed with default parameters.
    // If defaults are not acceptable, return false here.
    // Let's assume defaults are okay for now.
    return true; // Or return false if 'dyn_obj' group is mandatory
    }

    // --- Load Base Parameters ---
    // Use loadParam for each parameter. `success` becomes false if any loadParam returns false.
    // Buffer and processing parameters
    success &= loadParam(dyn_obj_node, "buffer_delay", params.buffer_delay);
    success &= loadParam(dyn_obj_node, "buffer_size", params.buffer_size);
    success &= loadParam(dyn_obj_node, "points_num_perframe", params.points_num_perframe);
    success &= loadParam(dyn_obj_node, "history_length", params.history_length);
    success &= loadParam(dyn_obj_node, "depth_map_dur", params.depth_map_dur);
    success &= loadParam(dyn_obj_node, "max_depth_map_num", params.max_depth_map_num);
    success &= loadParam(dyn_obj_node, "max_pixel_points", params.max_pixel_points);
    success &= loadParam(dyn_obj_node, "frame_dur", params.frame_dur);
    success &= loadParam(dyn_obj_node, "dataset", params.dataset);
    success &= loadParam(dyn_obj_node, "buffer_dur", params.buffer_dur);
    success &= loadParam(dyn_obj_node, "point_index", params.point_index);

    // Field of view and self-filtering parameters
    success &= loadParam(dyn_obj_node, "self_x_f", params.self_x_f);
    success &= loadParam(dyn_obj_node, "self_x_b", params.self_x_b);
    success &= loadParam(dyn_obj_node, "self_y_l", params.self_y_l);
    success &= loadParam(dyn_obj_node, "self_y_r", params.self_y_r);
    success &= loadParam(dyn_obj_node, "blind_dis", params.blind_dis);
    success &= loadParam(dyn_obj_node, "fov_up", params.fov_up);
    success &= loadParam(dyn_obj_node, "fov_down", params.fov_down);
    success &= loadParam(dyn_obj_node, "fov_cut", params.fov_cut);
    success &= loadParam(dyn_obj_node, "fov_left", params.fov_left);
    success &= loadParam(dyn_obj_node, "fov_right", params.fov_right);

    // Neighbor check parameters
    success &= loadParam(dyn_obj_node, "checkneighbor_range",
                        params.checkneighbor_range);
    success &= loadParam(dyn_obj_node, "stop_object_detect",
                        params.stop_object_detect);

    // Case 1 parameters
    success &= loadParam(dyn_obj_node, "depth_thr1", params.depth_thr1);
    success &= loadParam(dyn_obj_node, "enter_min_thr1", params.enter_min_thr1);
    success &= loadParam(dyn_obj_node, "enter_max_thr1", params.enter_max_thr1);
    success &= loadParam(dyn_obj_node, "map_cons_depth_thr1",
                        params.map_cons_depth_thr1);
    success &=
        loadParam(dyn_obj_node, "map_cons_hor_thr1", params.map_cons_hor_thr1);
    success &=
        loadParam(dyn_obj_node, "map_cons_ver_thr1", params.map_cons_ver_thr1);
    success &=
        loadParam(dyn_obj_node, "map_cons_hor_dis1", params.map_cons_hor_dis1);
    success &=
        loadParam(dyn_obj_node, "map_cons_ver_dis1", params.map_cons_ver_dis1);
    success &= loadParam(dyn_obj_node, "depth_cons_depth_thr1",
                        params.depth_cons_depth_thr1);
    success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr1",
                        params.depth_cons_depth_max_thr1);
    success &= loadParam(dyn_obj_node, "depth_cons_hor_thr1",
                        params.depth_cons_hor_thr1);
    success &= loadParam(dyn_obj_node, "depth_cons_ver_thr1",
                        params.depth_cons_ver_thr1);
    success &= loadParam(dyn_obj_node, "enlarge_z_thr1", params.enlarge_z_thr1);
    success &= loadParam(dyn_obj_node, "enlarge_angle", params.enlarge_angle);
    success &= loadParam(dyn_obj_node, "enlarge_depth", params.enlarge_depth);
    success &=
        loadParam(dyn_obj_node, "occluded_map_thr1", params.occluded_map_thr1);
    success &=
        loadParam(dyn_obj_node, "case1_interp_en", params.case1_interp_en);
    success &=
        loadParam(dyn_obj_node, "k_depth_min_thr1", params.k_depth_min_thr1);
    success &=
        loadParam(dyn_obj_node, "d_depth_min_thr1", params.d_depth_min_thr1);
    success &=
        loadParam(dyn_obj_node, "k_depth_max_thr1", params.k_depth_max_thr1);
    success &=
        loadParam(dyn_obj_node, "d_depth_max_thr1", params.d_depth_max_thr1);
    
 
    // Case 2 parameters
    success &= loadParam(dyn_obj_node, "v_min_thr2", params.v_min_thr2);
    success &= loadParam(dyn_obj_node, "acc_thr2", params.acc_thr2);
    success &= loadParam(dyn_obj_node, "map_cons_depth_thr2",
                        params.map_cons_depth_thr2);
    success &=
        loadParam(dyn_obj_node, "map_cons_hor_thr2", params.map_cons_hor_thr2);
    success &=
        loadParam(dyn_obj_node, "map_cons_ver_thr2", params.map_cons_ver_thr2);
    success &= loadParam(dyn_obj_node, "occ_depth_thr2", params.occ_depth_thr2);
    success &= loadParam(dyn_obj_node, "occ_hor_thr2", params.occ_hor_thr2);
    success &= loadParam(dyn_obj_node, "occ_ver_thr2", params.occ_ver_thr2);
    success &= loadParam(dyn_obj_node, "depth_cons_depth_thr2",
                        params.depth_cons_depth_thr2);
    success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr2",
                        params.depth_cons_depth_max_thr2);
    success &= loadParam(dyn_obj_node, "depth_cons_hor_thr2",
                        params.depth_cons_hor_thr2);
    success &= loadParam(dyn_obj_node, "depth_cons_ver_thr2",
                        params.depth_cons_ver_thr2);
    success &= loadParam(dyn_obj_node, "k_depth2", params.k_depth2);
    success &= loadParam(dyn_obj_node, "occluded_times_thr2",
                        params.occluded_times_thr2);
    success &=
        loadParam(dyn_obj_node, "case2_interp_en", params.case2_interp_en);
    success &=
        loadParam(dyn_obj_node, "k_depth_max_thr2", params.k_depth_max_thr2);
    success &=
        loadParam(dyn_obj_node, "d_depth_max_thr2", params.d_depth_max_thr2);

    // Case 3 parameters
    success &= loadParam(dyn_obj_node, "v_min_thr3", params.v_min_thr3);
    success &= loadParam(dyn_obj_node, "acc_thr3", params.acc_thr3);
    success &= loadParam(dyn_obj_node, "map_cons_depth_thr3",
                        params.map_cons_depth_thr3);
    success &=
        loadParam(dyn_obj_node, "map_cons_hor_thr3", params.map_cons_hor_thr3);
    success &=
        loadParam(dyn_obj_node, "map_cons_ver_thr3", params.map_cons_ver_thr3);
    success &= loadParam(dyn_obj_node, "occ_depth_thr3", params.occ_depth_thr3);
    success &= loadParam(dyn_obj_node, "occ_hor_thr3", params.occ_hor_thr3);
    success &= loadParam(dyn_obj_node, "occ_ver_thr3", params.occ_ver_thr3);
    success &= loadParam(dyn_obj_node, "depth_cons_depth_thr3",
                        params.depth_cons_depth_thr3);
    success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr3",
                        params.depth_cons_depth_max_thr3);
    success &= loadParam(dyn_obj_node, "depth_cons_hor_thr3",
                        params.depth_cons_hor_thr3);
    success &= loadParam(dyn_obj_node, "depth_cons_ver_thr3",
                        params.depth_cons_ver_thr3);
    success &= loadParam(dyn_obj_node, "k_depth3", params.k_depth3);
    success &= loadParam(dyn_obj_node, "occluding_times_thr3",
                        params.occluding_times_thr3);
    success &=
        loadParam(dyn_obj_node, "case3_interp_en", params.case3_interp_en);
    success &=
        loadParam(dyn_obj_node, "k_depth_max_thr3", params.k_depth_max_thr3);
    success &=
        loadParam(dyn_obj_node, "d_depth_max_thr3", params.d_depth_max_thr3);

    // Interpolation parameters
    success &= loadParam(dyn_obj_node, "interp_hor_thr", params.interp_hor_thr);
    success &= loadParam(dyn_obj_node, "interp_ver_thr", params.interp_ver_thr);
    success &= loadParam(dyn_obj_node, "interp_thr1", params.interp_thr1);
    success &=
        loadParam(dyn_obj_node, "interp_static_max", params.interp_static_max);
    success &= loadParam(dyn_obj_node, "interp_start_depth1",
                        params.interp_start_depth1);
    success &= loadParam(dyn_obj_node, "interp_kp1", params.interp_kp1);
    success &= loadParam(dyn_obj_node, "interp_kd1", params.interp_kd1);
    success &= loadParam(dyn_obj_node, "interp_thr2", params.interp_thr2);
    success &= loadParam(dyn_obj_node, "interp_thr3", params.interp_thr3);

    // other found params during refactor phase 2
    success &= loadParam(dyn_obj_node, "enlarge_distort", params.enlarge_distort);

    // Other configuration parameters
    success &= loadParam(dyn_obj_node, "dyn_filter_en", params.dyn_filter_en);
    success &= loadParam(dyn_obj_node, "debug_en", params.debug_en);
    success &= loadParam(dyn_obj_node, "laserCloudSteadObj_accu_limit",
                        params.laserCloudSteadObj_accu_limit);
    success &=
        loadParam(dyn_obj_node, "voxel_filter_size", params.voxel_filter_size);
    success &=
        loadParam(dyn_obj_node, "cluster_coupled", params.cluster_coupled);
    success &= loadParam(dyn_obj_node, "cluster_future", params.cluster_future);
    success &= loadParam(dyn_obj_node, "Cluster_cluster_extend_pixel",
                        params.Cluster_cluster_extend_pixel);
    success &= loadParam(dyn_obj_node, "Cluster_cluster_min_pixel_number",
                        params.Cluster_cluster_min_pixel_number);
    success &= loadParam(dyn_obj_node, "Cluster_thrustable_thresold",
                        params.Cluster_thrustable_thresold);
    success &= loadParam(dyn_obj_node, "Cluster_Voxel_revolusion",
                        params.Cluster_Voxel_revolusion);
    success &=
        loadParam(dyn_obj_node, "Cluster_debug_en", params.Cluster_debug_en);
    success &=
        loadParam(dyn_obj_node, "Cluster_out_file", params.Cluster_out_file);
    success &= loadParam(dyn_obj_node, "cutoff_value", params.cutoff_value);
    // Ensure resolutions are loaded before derived calculations
    bool res_h_ok = loadParam(dyn_obj_node, "hor_resolution_max", params.hor_resolution_max);
    bool res_v_ok = loadParam(dyn_obj_node, "ver_resolution_max", params.ver_resolution_max);
    success &= res_h_ok;
    success &= res_v_ok;
    success &= loadParam(dyn_obj_node, "frame_id", params.frame_id);
    success &= loadParam(dyn_obj_node, "time_file", params.time_file);
    success &= loadParam(dyn_obj_node, "time_breakdown_file",
                        params.time_breakdown_file);


    // --- Check if base parameters were loaded successfully before calculating derived ones ---
    if (!success) {
    std::cerr << "Warning: Some base parameters could not be loaded correctly from "
                "config file: "
                << filename << ". Derived parameters might be incorrect or defaults used." << std::endl;
    // Decide if this is a fatal error. If resolutions failed, it likely is.
    if (!res_h_ok || !res_v_ok) {
        std::cerr << "Error: Essential resolution parameters failed to load." << std::endl;
        return false;
    }
    // If other non-critical params failed, we might continue, but success should remain false.
    }

    // --- Calculate Derived Parameters ---
    // Check for division by zero before calculating pixel counts
    if (params.hor_resolution_max <= 0 || params.ver_resolution_max <= 0) {
    throw std::invalid_argument("Error: hor_resolution_max or ver_resolution_max is zero or negative. Cannot calculate derived parameters.");
    }

    // Interpolation pixel counts (based on angle thresholds and resolution)
    // Use ceil to ensure the range covers the threshold angle
    params.interp_hor_num = static_cast<int>(
        std::ceil(std::fabs(params.interp_hor_thr) / params.hor_resolution_max));
    params.interp_ver_num = static_cast<int>(
        std::ceil(std::fabs(params.interp_ver_thr) / params.ver_resolution_max));

    // FOV pixel indices
    // Convert FOV angles from degrees to radians
    const float fov_up_rad = params.fov_up * M_PI / 180.0f;
    const float fov_down_rad = params.fov_down * M_PI / 180.0f; // Often negative
    const float fov_cut_rad = params.fov_cut * M_PI / 180.0f;
    const float fov_left_rad = params.fov_left * M_PI / 180.0f;
    const float fov_right_rad = params.fov_right * M_PI / 180.0f; // Often negative or wraps

    // Vertical FOV: Convert elevation angle to vertical index.
    // Elevation angle range is typically [-pi/2, +pi/2]. Index = floor((elevation + pi/2) / resolution)
    // Add a small epsilon to handle angles exactly at bin boundaries if needed.
    params.pixel_fov_up = static_cast<int>(
        std::floor((fov_up_rad + 0.5f * M_PI) / params.ver_resolution_max));
    params.pixel_fov_down = static_cast<int>(
        std::floor((fov_down_rad + 0.5f * M_PI) / params.ver_resolution_max));
    params.pixel_fov_cut = static_cast<int>(
        std::floor((fov_cut_rad + 0.5f * M_PI) / params.ver_resolution_max));

    // Clamp vertical indices to be within valid range [0, MAX_1D_HALF - 1] if necessary
    // (Requires MAX_1D_HALF definition, potentially from dyn_obj_datatypes.h)
    // params.pixel_fov_up = std::max(0, std::min(params.pixel_fov_up, MAX_1D_HALF - 1));
    // params.pixel_fov_down = std::max(0, std::min(params.pixel_fov_down, MAX_1D_HALF - 1));
    // params.pixel_fov_cut = std::max(0, std::min(params.pixel_fov_cut, MAX_1D_HALF - 1));


    // Horizontal FOV: Convert azimuth angle to horizontal index.
    // Azimuth angle range is [-pi, +pi]. Index = floor((azimuth + pi) / resolution)
    params.pixel_fov_left = static_cast<int>(
        std::floor((fov_left_rad + M_PI) / params.hor_resolution_max));
    params.pixel_fov_right = static_cast<int>(
        std::floor((fov_right_rad + M_PI) / params.hor_resolution_max));

    // Handle potential wrap-around for horizontal indices if needed, depending on usage.
    // The indices calculated here might represent boundaries in a non-wrapped space.
    // Clamping might also be needed if MAX_1D is defined.
    // params.pixel_fov_left = std::max(0, std::min(params.pixel_fov_left, MAX_1D - 1));
    // params.pixel_fov_right = std::max(0, std::min(params.pixel_fov_right, MAX_1D - 1));


    // Max pointers needed based on map history duration and frame rate
    if (params.frame_dur <= 0) {
    throw std::invalid_argument("Error: frame_dur is zero or negative. Cannot calculate max_pointers_num.");
    }
    // Calculate total time span covered by maps + buffer delay
    double total_time_span = params.max_depth_map_num * params.depth_map_dur + params.buffer_delay;
    // Calculate number of frames within this span and add 1 for safety/current frame
    params.max_pointers_num = static_cast<int>(std::ceil(total_time_span / params.frame_dur)) + 1;

    // Calculate depth consistency check parameters
    params.depth_cons_hor_num2 = ceil(params.depth_cons_hor_thr2/params.ver_resolution_max);
    params.depth_cons_ver_num2 = ceil(params.depth_cons_ver_thr2/params.ver_resolution_max);
    params.depth_cons_hor_num3 = ceil(params.depth_cons_hor_thr3/params.ver_resolution_max);
    params.depth_cons_ver_num3 = ceil(params.depth_cons_ver_thr3/params.ver_resolution_max);

    // Calculate occlusion consistency check parameters
    params.occ_hor_num2 = ceil(params.occ_hor_thr2/params.hor_resolution_max);
    params.occ_ver_num2 = ceil(params.occ_ver_thr2/params.ver_resolution_max);
    params.occ_hor_num3 = ceil(params.occ_hor_thr3/params.hor_resolution_max);
    params.occ_ver_num3 = ceil(params.occ_ver_thr3/params.ver_resolution_max);
 
   } catch (const YAML::Exception& e) {
    throw std::runtime_error("Error parsing YAML file: " + filename + " - " + e.what());
   } catch (const std::exception& e) {
     // Catch other potential standard exceptions during calculations
    throw std::runtime_error("Standard exception during config loading or processing: " + std::string(e.what()));
   } catch (...) {
     // Catch any other unexpected exceptions
    throw std::runtime_error("Unknown exception during config loading or processing.");
   }
 
   // Return true only if all loading AND calculations were successful (or defaults were acceptable)
   // The 'success' variable tracks issues during loadParam calls.
   // Additional checks within this function return false directly on critical errors.
   return success;
 }