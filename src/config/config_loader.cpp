/**
 * @file config_loader.cpp
 * @brief Implements the functionality for loading dynamic object filter parameters from a YAML file.
 */

// Define _USE_MATH_DEFINES before including cmath for M_PI on some platforms (e.g., MSVC)
#define _USE_MATH_DEFINES
#include <cmath> // For std::ceil, std::floor, std::fabs, M_PI

#include "config/config_loader.h" // Header for DynObjFilterParams and loadParam template

#include <fstream>   // For std::ifstream
#include <iostream>  // For std::cerr
#include <stdexcept> // For std::runtime_error, std::invalid_argument

// Note: The loadParam template function is defined in the header (config_loader.h)

bool load_config(const std::string& filename, DynObjFilterParams& params) {
    // params object comes in with default values already set by its constructor

    std::ifstream fin(filename);
    if (!fin.is_open()) {
        // Use std::runtime_error for file opening issues
        throw std::runtime_error("Error: Could not open config file: " + filename);
    }
    // Close the ifstream, YAML::LoadFile handles file reading itself
    fin.close();

    bool base_load_success = true; // Track success specifically for base parameter loading

    try {
        YAML::Node config = YAML::LoadFile(filename);
        YAML::Node dyn_obj_node = config["dyn_obj"]; // Parameters are under 'dyn_obj'

        if (!dyn_obj_node) {
            std::cerr << "Warning: 'dyn_obj' group not found in config file: "
                      << filename << ". Using default parameters provided by the constructor." << std::endl;
            // Proceeding with defaults is often acceptable, so return true.
            // If the 'dyn_obj' group MUST exist, return false here.
            return true;
        }

        // --- Load Base Parameters (Grouped matching the header) ---
        // Use loadParam for each. base_load_success becomes false if any loadParam returns false.

        // --- General & Dataset ---
        base_load_success &= loadParam(dyn_obj_node, "dataset", params.dataset);
        base_load_success &= loadParam(dyn_obj_node, "frame_id", params.frame_id);
        base_load_success &= loadParam(dyn_obj_node, "dyn_filter_en", params.dyn_filter_en);

        // --- Buffering & Timing ---
        base_load_success &= loadParam(dyn_obj_node, "buffer_delay", params.buffer_delay);
        base_load_success &= loadParam(dyn_obj_node, "buffer_size", params.buffer_size);
        base_load_success &= loadParam(dyn_obj_node, "history_length", params.history_length);
        base_load_success &= loadParam(dyn_obj_node, "depth_map_dur", params.depth_map_dur);
        base_load_success &= loadParam(dyn_obj_node, "max_depth_map_num", params.max_depth_map_num);
        base_load_success &= loadParam(dyn_obj_node, "frame_dur", params.frame_dur);
        base_load_success &= loadParam(dyn_obj_node, "buffer_dur", params.buffer_dur);
        base_load_success &= loadParam(dyn_obj_node, "points_num_perframe", params.points_num_perframe);

        // --- Sensor Characteristics & FOV ---
        // Load resolutions separately to check them before derived calculations
        bool res_h_ok = loadParam(dyn_obj_node, "hor_resolution_max", params.hor_resolution_max);
        bool res_v_ok = loadParam(dyn_obj_node, "ver_resolution_max", params.ver_resolution_max);
        base_load_success &= res_h_ok;
        base_load_success &= res_v_ok;
        base_load_success &= loadParam(dyn_obj_node, "fov_up", params.fov_up);
        base_load_success &= loadParam(dyn_obj_node, "fov_down", params.fov_down);
        base_load_success &= loadParam(dyn_obj_node, "fov_cut", params.fov_cut);
        base_load_success &= loadParam(dyn_obj_node, "fov_left", params.fov_left);
        base_load_success &= loadParam(dyn_obj_node, "fov_right", params.fov_right);

        // --- Point Filtering (Invalid / Self) ---
        base_load_success &= loadParam(dyn_obj_node, "blind_dis", params.blind_dis);
        base_load_success &= loadParam(dyn_obj_node, "enable_invalid_box_check", params.enable_invalid_box_check);
        base_load_success &= loadParam(dyn_obj_node, "invalid_box_x_half_width", params.invalid_box_x_half_width);
        base_load_success &= loadParam(dyn_obj_node, "invalid_box_y_half_width", params.invalid_box_y_half_width);
        base_load_success &= loadParam(dyn_obj_node, "invalid_box_z_half_width", params.invalid_box_z_half_width);
        base_load_success &= loadParam(dyn_obj_node, "self_x_f", params.self_x_f);
        base_load_success &= loadParam(dyn_obj_node, "self_x_b", params.self_x_b);
        base_load_success &= loadParam(dyn_obj_node, "self_y_l", params.self_y_l);
        base_load_success &= loadParam(dyn_obj_node, "self_y_r", params.self_y_r);

        // --- Depth Map & Grid ---
        base_load_success &= loadParam(dyn_obj_node, "max_pixel_points", params.max_pixel_points);

        // --- Neighbor Check Parameters ---
        base_load_success &= loadParam(dyn_obj_node, "checkneighbor_range", params.checkneighbor_range);

        // --- Stopped Object Detection ---
        base_load_success &= loadParam(dyn_obj_node, "stop_object_detect", params.stop_object_detect);
        base_load_success &= loadParam(dyn_obj_node, "laserCloudSteadObj_accu_limit", params.laserCloudSteadObj_accu_limit);

        // --- Case 1 Parameters (Appearing) ---
        base_load_success &= loadParam(dyn_obj_node, "depth_thr1", params.depth_thr1);
        base_load_success &= loadParam(dyn_obj_node, "enter_min_thr1", params.enter_min_thr1);
        base_load_success &= loadParam(dyn_obj_node, "enter_max_thr1", params.enter_max_thr1);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_depth_thr1", params.map_cons_depth_thr1);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_hor_thr1", params.map_cons_hor_thr1);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_ver_thr1", params.map_cons_ver_thr1);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_hor_dis1", params.map_cons_hor_dis1);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_ver_dis1", params.map_cons_ver_dis1);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_depth_thr1", params.depth_cons_depth_thr1);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr1", params.depth_cons_depth_max_thr1);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_hor_thr1", params.depth_cons_hor_thr1);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_ver_thr1", params.depth_cons_ver_thr1);
        base_load_success &= loadParam(dyn_obj_node, "k_depth_min_thr1", params.k_depth_min_thr1);
        base_load_success &= loadParam(dyn_obj_node, "d_depth_min_thr1", params.d_depth_min_thr1);
        base_load_success &= loadParam(dyn_obj_node, "k_depth_max_thr1", params.k_depth_max_thr1);
        base_load_success &= loadParam(dyn_obj_node, "d_depth_max_thr1", params.d_depth_max_thr1);
        base_load_success &= loadParam(dyn_obj_node, "enlarge_z_thr1", params.enlarge_z_thr1);
        base_load_success &= loadParam(dyn_obj_node, "enlarge_angle", params.enlarge_angle);
        base_load_success &= loadParam(dyn_obj_node, "enlarge_depth", params.enlarge_depth);
        base_load_success &= loadParam(dyn_obj_node, "occluded_map_thr1", params.occluded_map_thr1);
        base_load_success &= loadParam(dyn_obj_node, "case1_interp_en", params.case1_interp_en);

        // --- Case 2 Parameters (Occluding) ---
        base_load_success &= loadParam(dyn_obj_node, "v_min_thr2", params.v_min_thr2);
        base_load_success &= loadParam(dyn_obj_node, "acc_thr2", params.acc_thr2);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_depth_thr2", params.map_cons_depth_thr2);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_hor_thr2", params.map_cons_hor_thr2);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_ver_thr2", params.map_cons_ver_thr2);
        base_load_success &= loadParam(dyn_obj_node, "occ_depth_thr2", params.occ_depth_thr2);
        base_load_success &= loadParam(dyn_obj_node, "occ_hor_thr2", params.occ_hor_thr2);
        base_load_success &= loadParam(dyn_obj_node, "occ_ver_thr2", params.occ_ver_thr2);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_depth_thr2", params.depth_cons_depth_thr2);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr2", params.depth_cons_depth_max_thr2);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_hor_thr2", params.depth_cons_hor_thr2);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_ver_thr2", params.depth_cons_ver_thr2);
        base_load_success &= loadParam(dyn_obj_node, "k_depth2", params.k_depth2);
        base_load_success &= loadParam(dyn_obj_node, "k_depth_max_thr2", params.k_depth_max_thr2);
        base_load_success &= loadParam(dyn_obj_node, "d_depth_max_thr2", params.d_depth_max_thr2);
        base_load_success &= loadParam(dyn_obj_node, "occluded_times_thr2", params.occluded_times_thr2);
        base_load_success &= loadParam(dyn_obj_node, "case2_interp_en", params.case2_interp_en);

        // --- Case 3 Parameters (Disoccluded) ---
        base_load_success &= loadParam(dyn_obj_node, "v_min_thr3", params.v_min_thr3);
        base_load_success &= loadParam(dyn_obj_node, "acc_thr3", params.acc_thr3);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_depth_thr3", params.map_cons_depth_thr3);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_hor_thr3", params.map_cons_hor_thr3);
        base_load_success &= loadParam(dyn_obj_node, "map_cons_ver_thr3", params.map_cons_ver_thr3);
        base_load_success &= loadParam(dyn_obj_node, "occ_depth_thr3", params.occ_depth_thr3);
        base_load_success &= loadParam(dyn_obj_node, "occ_hor_thr3", params.occ_hor_thr3);
        base_load_success &= loadParam(dyn_obj_node, "occ_ver_thr3", params.occ_ver_thr3);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_depth_thr3", params.depth_cons_depth_thr3);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr3", params.depth_cons_depth_max_thr3);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_hor_thr3", params.depth_cons_hor_thr3);
        base_load_success &= loadParam(dyn_obj_node, "depth_cons_ver_thr3", params.depth_cons_ver_thr3);
        base_load_success &= loadParam(dyn_obj_node, "k_depth3", params.k_depth3);
        base_load_success &= loadParam(dyn_obj_node, "k_depth_max_thr3", params.k_depth_max_thr3);
        base_load_success &= loadParam(dyn_obj_node, "d_depth_max_thr3", params.d_depth_max_thr3);
        base_load_success &= loadParam(dyn_obj_node, "occluding_times_thr3", params.occluding_times_thr3);
        base_load_success &= loadParam(dyn_obj_node, "case3_interp_en", params.case3_interp_en);

        // --- Interpolation Parameters ---
        base_load_success &= loadParam(dyn_obj_node, "interp_hor_thr", params.interp_hor_thr);
        base_load_success &= loadParam(dyn_obj_node, "interp_ver_thr", params.interp_ver_thr);
        base_load_success &= loadParam(dyn_obj_node, "interp_thr1", params.interp_thr1);
        base_load_success &= loadParam(dyn_obj_node, "interp_static_max", params.interp_static_max);
        base_load_success &= loadParam(dyn_obj_node, "interp_start_depth1", params.interp_start_depth1);
        base_load_success &= loadParam(dyn_obj_node, "interp_kp1", params.interp_kp1);
        base_load_success &= loadParam(dyn_obj_node, "interp_kd1", params.interp_kd1);
        base_load_success &= loadParam(dyn_obj_node, "interp_thr2", params.interp_thr2);
        base_load_success &= loadParam(dyn_obj_node, "interp_thr3", params.interp_thr3);

        // --- Clustering Parameters ---
        base_load_success &= loadParam(dyn_obj_node, "cluster_coupled", params.cluster_coupled);
        base_load_success &= loadParam(dyn_obj_node, "cluster_future", params.cluster_future);
        base_load_success &= loadParam(dyn_obj_node, "Cluster_cluster_extend_pixel", params.Cluster_cluster_extend_pixel);
        base_load_success &= loadParam(dyn_obj_node, "Cluster_cluster_min_pixel_number", params.Cluster_cluster_min_pixel_number);
        base_load_success &= loadParam(dyn_obj_node, "Cluster_thrustable_thresold", params.Cluster_thrustable_thresold);
        base_load_success &= loadParam(dyn_obj_node, "Cluster_Voxel_resolution", params.Cluster_Voxel_resolution); // Fixed typo
        base_load_success &= loadParam(dyn_obj_node, "Cluster_debug_en", params.Cluster_debug_en);
        base_load_success &= loadParam(dyn_obj_node, "Cluster_out_file", params.Cluster_out_file);

        // --- Debugging & Misc ---
        base_load_success &= loadParam(dyn_obj_node, "debug_en", params.debug_en);
        base_load_success &= loadParam(dyn_obj_node, "time_file", params.time_file);
        base_load_success &= loadParam(dyn_obj_node, "time_breakdown_file", params.time_breakdown_file);
        base_load_success &= loadParam(dyn_obj_node, "point_index", params.point_index);
        base_load_success &= loadParam(dyn_obj_node, "enlarge_distort", params.enlarge_distort);
        base_load_success &= loadParam(dyn_obj_node, "cutoff_value", params.cutoff_value);
        base_load_success &= loadParam(dyn_obj_node, "voxel_filter_size", params.voxel_filter_size);


        // --- Check if base parameters were loaded successfully before calculating derived ones ---
        if (!base_load_success) {
            std::cerr << "Warning: Some base parameters failed to load correctly from config file: "
                      << filename << ". Using defaults for failed parameters. Derived parameters might be incorrect." << std::endl;
            // Decide if this is fatal. Failure to load resolutions IS fatal.
            if (!res_h_ok || !res_v_ok) {
                std::cerr << "Error: Essential resolution parameters (hor_resolution_max or ver_resolution_max) failed to load. Cannot proceed." << std::endl;
                return false; // Cannot calculate derived params without resolutions
            }
            // If other non-critical params failed, we might continue, but the overall function should indicate failure.
        }

        // --- Calculate Derived Parameters ---
        // Ensure essential parameters for calculations are valid
        if (params.hor_resolution_max <= 0.0f) {
             throw std::invalid_argument("Error: hor_resolution_max must be positive. Check config file.");
        }
        if (params.ver_resolution_max <= 0.0f) {
            throw std::invalid_argument("Error: ver_resolution_max must be positive. Check config file.");
        }
        if (params.frame_dur <= 0.0) {
            throw std::invalid_argument("Error: frame_dur must be positive. Check config file.");
        }

        // Interpolation pixel counts (number of pixels spanning the angle threshold)
        // Use ceil to ensure the range fully covers the threshold angle.
        params.interp_hor_num = static_cast<int>(std::ceil(std::fabs(params.interp_hor_thr) / params.hor_resolution_max));
        params.interp_ver_num = static_cast<int>(std::ceil(std::fabs(params.interp_ver_thr) / params.ver_resolution_max));

        // FOV pixel indices
        // Convert FOV angles from degrees to radians for calculations
        const float fov_up_rad = params.fov_up * M_PI / 180.0f;
        const float fov_down_rad = params.fov_down * M_PI / 180.0f; // Often negative
        const float fov_cut_rad = params.fov_cut * M_PI / 180.0f;
        const float fov_left_rad = params.fov_left * M_PI / 180.0f;
        const float fov_right_rad = params.fov_right * M_PI / 180.0f; // Often negative or wraps

        // Vertical FOV: Convert elevation angle to vertical index.
        // Assuming elevation angle 'alpha' maps to index 'v' roughly as: v = floor((alpha - min_elevation) / ver_resolution)
        // Or, if using spherical coords where elevation is angle from xy-plane [-pi/2, pi/2]: v = floor((alpha + pi/2) / ver_resolution)
        // Using the second convention here, consistent with GetVec potentially:
        params.pixel_fov_up = static_cast<int>(std::floor((fov_up_rad + 0.5f * M_PI) / params.ver_resolution_max));
        params.pixel_fov_down = static_cast<int>(std::floor((fov_down_rad + 0.5f * M_PI) / params.ver_resolution_max));
        params.pixel_fov_cut = static_cast<int>(std::floor((fov_cut_rad + 0.5f * M_PI) / params.ver_resolution_max));
        // Note: Clamping these indices might be necessary later depending on MAX_1D_HALF definition and usage.

        // Horizontal FOV: Convert azimuth angle to horizontal index.
        // Assuming azimuth angle 'beta' maps to index 'h' roughly as: h = floor((beta - min_azimuth) / hor_resolution)
        // Or, if using spherical coords where azimuth is angle from x-axis [-pi, +pi]: h = floor((beta + pi) / hor_resolution)
        // Using the second convention here:
        params.pixel_fov_left = static_cast<int>(std::floor((fov_left_rad + M_PI) / params.hor_resolution_max));
        params.pixel_fov_right = static_cast<int>(std::floor((fov_right_rad + M_PI) / params.hor_resolution_max));
        // Note: Clamping or handling wrap-around for horizontal indices might be necessary depending on MAX_1D definition and usage.

        // Max pointers needed based on map history duration and frame rate
        // Calculate total time span covered by maps + buffer delay
        double total_time_span = params.max_depth_map_num * params.depth_map_dur + params.buffer_delay;
        // Calculate number of frames within this span and add 1 for safety/current frame
        params.max_pointers_num = static_cast<int>(std::ceil(total_time_span / params.frame_dur)) + 1;

        // Calculate depth consistency check pixel counts
        // *** FIX: Use hor_resolution_max for horizontal thresholds ***
        params.depth_cons_hor_num2 = static_cast<int>(std::ceil(std::fabs(params.depth_cons_hor_thr2) / params.hor_resolution_max));
        params.depth_cons_ver_num2 = static_cast<int>(std::ceil(std::fabs(params.depth_cons_ver_thr2) / params.ver_resolution_max));
        params.depth_cons_hor_num3 = static_cast<int>(std::ceil(std::fabs(params.depth_cons_hor_thr3) / params.hor_resolution_max));
        params.depth_cons_ver_num3 = static_cast<int>(std::ceil(std::fabs(params.depth_cons_ver_thr3) / params.ver_resolution_max));

        // Calculate occlusion consistency check pixel counts
        // *** FIX: Use hor_resolution_max for horizontal thresholds ***
        params.occ_hor_num2 = static_cast<int>(std::ceil(std::fabs(params.occ_hor_thr2) / params.hor_resolution_max));
        params.occ_ver_num2 = static_cast<int>(std::ceil(std::fabs(params.occ_ver_thr2) / params.ver_resolution_max));
        params.occ_hor_num3 = static_cast<int>(std::ceil(std::fabs(params.occ_hor_thr3) / params.hor_resolution_max));
        params.occ_ver_num3 = static_cast<int>(std::ceil(std::fabs(params.occ_ver_thr3) / params.ver_resolution_max));

        // Add more derived parameter calculations here if needed...


    } catch (const YAML::Exception& e) {
        // Catch YAML parsing specific errors
        throw std::runtime_error("Error parsing YAML file: " + filename + " - " + e.what());
    } catch (const std::invalid_argument& e) {
        // Catch errors from our explicit checks (e.g., division by zero)
        throw std::runtime_error("Invalid parameter value found during config processing: " + std::string(e.what()));
    } catch (const std::exception& e) {
        // Catch other potential standard exceptions during calculations
        throw std::runtime_error("Standard exception during config loading or processing: " + std::string(e.what()));
    } catch (...) {
        // Catch any other unexpected exceptions
        throw std::runtime_error("Unknown exception during config loading or processing for file: " + filename);
    }

    // Return the success status determined during base parameter loading.
    // If base_load_success is false, it means some defaults were used due to loading errors,
    // but we might still proceed if critical parameters (like resolutions) were okay.
    // The calling code might want to check specific parameters if their defaults are unacceptable.
    return base_load_success;
}