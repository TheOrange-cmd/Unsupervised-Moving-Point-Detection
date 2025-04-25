// file: src/config_loader.cpp

#include "config/config_loader.h"
#include <iostream>
#include <fstream>
#include "filtering/dyn_obj_datatypes.h"

bool load_config(const std::string& filename, DynObjFilterParams& params) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: Could not open config file: " << filename << std::endl;
        return false;
    }
    bool success = true; // Track overall success

    try {
        YAML::Node config = YAML::LoadFile(filename);
        YAML::Node dyn_obj_node = config["dyn_obj"]; // Assuming params are under 'dyn_obj' group

        if (!dyn_obj_node) {
             std::cerr << "Error: 'dyn_obj' group not found in config file: " << filename << std::endl;
             return false;
        }
        // Buffer and processing parameters
        success &= loadParam(dyn_obj_node, "buffer_delay", params.buffer_delay);
        success &= loadParam(dyn_obj_node, "buffer_size", params.buffer_size);
        success &= loadParam(dyn_obj_node, "points_num_perframe", params.points_num_perframe);
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
        success &= loadParam(dyn_obj_node, "checkneighbor_range", params.checkneighbor_range);
        success &= loadParam(dyn_obj_node, "stop_object_detect", params.stop_object_detect);
        
        // Case 1 parameters
        success &= loadParam(dyn_obj_node, "depth_thr1", params.depth_thr1);
        success &= loadParam(dyn_obj_node, "enter_min_thr1", params.enter_min_thr1);
        success &= loadParam(dyn_obj_node, "enter_max_thr1", params.enter_max_thr1);
        success &= loadParam(dyn_obj_node, "map_cons_depth_thr1", params.map_cons_depth_thr1);
        success &= loadParam(dyn_obj_node, "map_cons_hor_thr1", params.map_cons_hor_thr1);
        success &= loadParam(dyn_obj_node, "map_cons_ver_thr1", params.map_cons_ver_thr1);
        success &= loadParam(dyn_obj_node, "map_cons_hor_dis1", params.map_cons_hor_dis1);
        success &= loadParam(dyn_obj_node, "map_cons_ver_dis1", params.map_cons_ver_dis1);
        success &= loadParam(dyn_obj_node, "depth_cons_depth_thr1", params.depth_cons_depth_thr1);
        success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr1", params.depth_cons_depth_max_thr1);
        success &= loadParam(dyn_obj_node, "depth_cons_hor_thr1", params.depth_cons_hor_thr1);
        success &= loadParam(dyn_obj_node, "depth_cons_ver_thr1", params.depth_cons_ver_thr1);
        success &= loadParam(dyn_obj_node, "enlarge_z_thr1", params.enlarge_z_thr1);
        success &= loadParam(dyn_obj_node, "enlarge_angle", params.enlarge_angle);
        success &= loadParam(dyn_obj_node, "enlarge_depth", params.enlarge_depth);
        success &= loadParam(dyn_obj_node, "occluded_map_thr1", params.occluded_map_thr1);
        success &= loadParam(dyn_obj_node, "case1_interp_en", params.case1_interp_en);
        success &= loadParam(dyn_obj_node, "k_depth_min_thr1", params.k_depth_min_thr1);
        success &= loadParam(dyn_obj_node, "d_depth_min_thr1", params.d_depth_min_thr1);
        success &= loadParam(dyn_obj_node, "k_depth_max_thr1", params.k_depth_max_thr1);
        success &= loadParam(dyn_obj_node, "d_depth_max_thr1", params.d_depth_max_thr1);
        
        // Case 2 parameters
        success &= loadParam(dyn_obj_node, "v_min_thr2", params.v_min_thr2);
        success &= loadParam(dyn_obj_node, "acc_thr2", params.acc_thr2);
        success &= loadParam(dyn_obj_node, "map_cons_depth_thr2", params.map_cons_depth_thr2);
        success &= loadParam(dyn_obj_node, "map_cons_hor_thr2", params.map_cons_hor_thr2);
        success &= loadParam(dyn_obj_node, "map_cons_ver_thr2", params.map_cons_ver_thr2);
        success &= loadParam(dyn_obj_node, "occ_depth_thr2", params.occ_depth_thr2);
        success &= loadParam(dyn_obj_node, "occ_hor_thr2", params.occ_hor_thr2);
        success &= loadParam(dyn_obj_node, "occ_ver_thr2", params.occ_ver_thr2);
        success &= loadParam(dyn_obj_node, "depth_cons_depth_thr2", params.depth_cons_depth_thr2);
        success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr2", params.depth_cons_depth_max_thr2);
        success &= loadParam(dyn_obj_node, "depth_cons_hor_thr2", params.depth_cons_hor_thr2);
        success &= loadParam(dyn_obj_node, "depth_cons_ver_thr2", params.depth_cons_ver_thr2);
        success &= loadParam(dyn_obj_node, "k_depth2", params.k_depth2);
        success &= loadParam(dyn_obj_node, "occluded_times_thr2", params.occluded_times_thr2);
        success &= loadParam(dyn_obj_node, "case2_interp_en", params.case2_interp_en);
        success &= loadParam(dyn_obj_node, "k_depth_max_thr2", params.k_depth_max_thr2);
        success &= loadParam(dyn_obj_node, "d_depth_max_thr2", params.d_depth_max_thr2);
        
        // Case 3 parameters
        success &= loadParam(dyn_obj_node, "v_min_thr3", params.v_min_thr3);
        success &= loadParam(dyn_obj_node, "acc_thr3", params.acc_thr3);
        success &= loadParam(dyn_obj_node, "map_cons_depth_thr3", params.map_cons_depth_thr3);
        success &= loadParam(dyn_obj_node, "map_cons_hor_thr3", params.map_cons_hor_thr3);
        success &= loadParam(dyn_obj_node, "map_cons_ver_thr3", params.map_cons_ver_thr3);
        success &= loadParam(dyn_obj_node, "occ_depth_thr3", params.occ_depth_thr3);
        success &= loadParam(dyn_obj_node, "occ_hor_thr3", params.occ_hor_thr3);
        success &= loadParam(dyn_obj_node, "occ_ver_thr3", params.occ_ver_thr3);
        success &= loadParam(dyn_obj_node, "depth_cons_depth_thr3", params.depth_cons_depth_thr3);
        success &= loadParam(dyn_obj_node, "depth_cons_depth_max_thr3", params.depth_cons_depth_max_thr3);
        success &= loadParam(dyn_obj_node, "depth_cons_hor_thr3", params.depth_cons_hor_thr3);
        success &= loadParam(dyn_obj_node, "depth_cons_ver_thr3", params.depth_cons_ver_thr3);
        success &= loadParam(dyn_obj_node, "k_depth3", params.k_depth3);
        success &= loadParam(dyn_obj_node, "occluding_times_thr3", params.occluding_times_thr3);
        success &= loadParam(dyn_obj_node, "case3_interp_en", params.case3_interp_en);
        success &= loadParam(dyn_obj_node, "k_depth_max_thr3", params.k_depth_max_thr3);
        success &= loadParam(dyn_obj_node, "d_depth_max_thr3", params.d_depth_max_thr3);
        
        // Interpolation parameters
        success &= loadParam(dyn_obj_node, "interp_hor_thr", params.interp_hor_thr);
        success &= loadParam(dyn_obj_node, "interp_ver_thr", params.interp_ver_thr);
        success &= loadParam(dyn_obj_node, "interp_thr1", params.interp_thr1);
        success &= loadParam(dyn_obj_node, "interp_static_max", params.interp_static_max);
        success &= loadParam(dyn_obj_node, "interp_start_depth1", params.interp_start_depth1);
        success &= loadParam(dyn_obj_node, "interp_kp1", params.interp_kp1);
        success &= loadParam(dyn_obj_node, "interp_kd1", params.interp_kd1);
        success &= loadParam(dyn_obj_node, "interp_thr2", params.interp_thr2);
        success &= loadParam(dyn_obj_node, "interp_thr3", params.interp_thr3);
        
        // Other configuration parameters
        success &= loadParam(dyn_obj_node, "dyn_filter_en", params.dyn_filter_en);
        success &= loadParam(dyn_obj_node, "debug_en", params.debug_en);
        success &= loadParam(dyn_obj_node, "laserCloudSteadObj_accu_limit", params.laserCloudSteadObj_accu_limit);
        success &= loadParam(dyn_obj_node, "voxel_filter_size", params.voxel_filter_size);
        success &= loadParam(dyn_obj_node, "cluster_coupled", params.cluster_coupled);
        success &= loadParam(dyn_obj_node, "cluster_future", params.cluster_future);
        success &= loadParam(dyn_obj_node, "Cluster_cluster_extend_pixel", params.Cluster_cluster_extend_pixel);
        success &= loadParam(dyn_obj_node, "Cluster_cluster_min_pixel_number", params.Cluster_cluster_min_pixel_number);
        success &= loadParam(dyn_obj_node, "Cluster_thrustable_thresold", params.Cluster_thrustable_thresold);
        success &= loadParam(dyn_obj_node, "Cluster_Voxel_revolusion", params.Cluster_Voxel_revolusion);
        success &= loadParam(dyn_obj_node, "Cluster_debug_en", params.Cluster_debug_en);
        success &= loadParam(dyn_obj_node, "Cluster_out_file", params.Cluster_out_file);
        success &= loadParam(dyn_obj_node, "hor_resolution_max", params.hor_resolution_max);
        success &= loadParam(dyn_obj_node, "ver_resolution_max", params.ver_resolution_max);
        success &= loadParam(dyn_obj_node, "frame_id", params.frame_id);
        success &= loadParam(dyn_obj_node, "time_file", params.time_file);
        success &= loadParam(dyn_obj_node, "time_breakdown_file", params.time_breakdown_file);
            
        // Check if base parameters were loaded successfully before calculating derived ones
        if (!success) {
            std::cerr << "Error: Some base parameters could not be loaded from config file: " << filename << std::endl;
            return false;
        }

        // --- Calculate Derived Parameters ---
        // Check for division by zero before calculating pixel counts
        if (params.hor_resolution_max <= 0 || params.ver_resolution_max <= 0) {
             std::cerr << "Error: hor_resolution_max or ver_resolution_max is zero or negative. Cannot calculate derived parameters." << std::endl;
             return false;
        }

        // Interpolation pixel counts (based on angle thresholds and resolution)
        params.interp_hor_num = static_cast<int>(std::ceil(params.interp_hor_thr / params.hor_resolution_max));
        params.interp_ver_num = static_cast<int>(std::ceil(params.interp_ver_thr / params.ver_resolution_max));

        // FOV pixel indices (convert degrees to radians first)
        // Vertical FOV: Angle relative to xy-plane -> Spherical elevation angle -> Index
        // Elevation angle = atan2(z, sqrt(x^2+y^2)). 0 is xy-plane, +pi/2 is +z axis.
        // Index = floor((elevation_angle + PI_MATH / 2.0) / ver_resolution_max)
        params.pixel_fov_up = static_cast<int>(std::floor((params.fov_up * PI_MATH / 180.0 + 0.5 * PI_MATH) / params.ver_resolution_max));
        params.pixel_fov_down = static_cast<int>(std::floor((params.fov_down * PI_MATH / 180.0 + 0.5 * PI_MATH) / params.ver_resolution_max)); // fov_down is likely negative degrees
        params.pixel_fov_cut = static_cast<int>(std::floor((params.fov_cut * PI_MATH / 180.0 + 0.5 * PI_MATH) / params.ver_resolution_max));

        // Horizontal FOV: Angle relative to x-axis -> Spherical azimuth angle -> Index
        // Azimuth angle = atan2(y, x). Range [-pi, pi].
        // Index = floor((azimuth_angle + PI_MATH) / hor_resolution_max)
        params.pixel_fov_left = static_cast<int>(std::floor((params.fov_left * PI_MATH / 180.0 + PI_MATH) / params.hor_resolution_max));
        params.pixel_fov_right = static_cast<int>(std::floor((params.fov_right * PI_MATH / 180.0 + PI_MATH) / params.hor_resolution_max)); // fov_right might be negative degrees

        // Max pointers needed based on map history duration and frame rate
        if (params.frame_dur <= 0) {
            std::cerr << "Error: frame_dur is zero or negative. Cannot calculate max_pointers_num." << std::endl;
            return false;
        }
        params.max_pointers_num = static_cast<int>(std::round((params.max_depth_map_num * params.depth_map_dur + params.buffer_delay) / params.frame_dur)) + 1;


        // --- Optional: Calculate other derived numbers if needed later ---
        // params.map_cons_hor_num1 = static_cast<int>(std::ceil(params.map_cons_hor_thr1 / params.hor_resolution_max));
        // params.map_cons_ver_num1 = static_cast<int>(std::ceil(params.map_cons_ver_thr1 / params.ver_resolution_max));
        // ... etc ...


    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing YAML file: " << filename << " - " << e.what() << std::endl;
        return false;
    }

    // Return true only if all loading AND calculations were successful
    return success;
}