#include "config_loader.h"
#include <iostream>
#include <fstream>

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
        // Load General/Check parameters
        success &= loadParam(dyn_obj_node, "checkneighbor_range", params.checkneighbor_range);
        success &= loadParam(dyn_obj_node, "stop_object_detect", params.stop_object_detect);
        
        // Load dataset and buffer Params
        success &= loadParam(dyn_obj_node, "dataset", params.dataset);
        success &= loadParam(dyn_obj_node, "buffer_delay", params.buffer_delay);
        success &= loadParam(dyn_obj_node, "buffer_size", params.buffer_size);
        success &= loadParam(dyn_obj_node, "points_num_perframe", params.points_num_perframe);
        success &= loadParam(dyn_obj_node, "depth_map_dur", params.depth_map_dur);
        success &= loadParam(dyn_obj_node, "max_depth_map_num", params.max_depth_map_num);
        success &= loadParam(dyn_obj_node, "max_pixel_points", params.max_pixel_points);
        success &= loadParam(dyn_obj_node, "frame_dur", params.frame_dur);
        success &= loadParam(dyn_obj_node, "buffer_dur", params.buffer_dur);

        // Load self-vehicle dimensions and FOV settings
        success &= loadParam(dyn_obj_node, "self_x_f", params.self_x_f);
        success &= loadParam(dyn_obj_node, "self_x_b", params.self_x_b);
        success &= loadParam(dyn_obj_node, "self_y_l", params.self_y_l);    
        success &= loadParam(dyn_obj_node, "self_y_r", params.self_y_r);
        success &= loadParam(dyn_obj_node, "blind_dis", params.blind_dis);
        success &= loadParam(dyn_obj_node, "fov_up", params.fov_up);
        success &= loadParam(dyn_obj_node, "fov_down", params.fov_down);
        success &= loadParam(dyn_obj_node, "fov_left", params.fov_left);
        success &= loadParam(dyn_obj_node, "fov_right", params.fov_right);
        success &= loadParam(dyn_obj_node, "fov_cut", params.fov_cut);

        // Load case 1 parameters
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

        // Load case 2 parameters   
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
        success &= loadParam(dyn_obj_node, "occluded_times_thr2", params.occluded_times_thr2);
        success &= loadParam(dyn_obj_node, "k_depth2", params.k_depth2);
        success &= loadParam(dyn_obj_node, "case2_interp_en", params.case2_interp_en);
        success &= loadParam(dyn_obj_node, "k_depth_max_thr2", params.k_depth_max_thr2);
        success &= loadParam(dyn_obj_node, "d_depth_max_thr2", params.d_depth_max_thr2);

        // Load case 3 parameters
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
        success &= loadParam(dyn_obj_node, "occluding_times_thr3", params.occluding_times_thr3);
        success &= loadParam(dyn_obj_node, "k_depth3", params.k_depth3);
        success &= loadParam(dyn_obj_node, "case3_interp_en", params.case3_interp_en);
        success &= loadParam(dyn_obj_node, "k_depth_max_thr3", params.k_depth_max_thr3);
        success &= loadParam(dyn_obj_node, "d_depth_max_thr3", params.d_depth_max_thr3);

        // Load interpolation parameters
        success &= loadParam(dyn_obj_node, "interp_hor_thr", params.interp_hor_thr);
        success &= loadParam(dyn_obj_node, "interp_ver_thr", params.interp_ver_thr);
        success &= loadParam(dyn_obj_node, "interp_thr1", params.interp_thr1);
        success &= loadParam(dyn_obj_node, "interp_static_max", params.interp_static_max);    
        success &= loadParam(dyn_obj_node, "interp_start_depth1", params.interp_start_depth1);
        success &= loadParam(dyn_obj_node, "interp_kp1", params.interp_kp1);
        success &= loadParam(dyn_obj_node, "interp_kd1", params.interp_kd1);
        success &= loadParam(dyn_obj_node, "interp_bg", params.interp_bg);
        success &= loadParam(dyn_obj_node, "interp_thr2", params.interp_thr2);
        success &= loadParam(dyn_obj_node, "interp_thr3", params.interp_thr3);

        // Load Steady point cloud accumulation parameters
        success &= loadParam(dyn_obj_node, "laserCloudSteadObj_accu_limit", params.laserCloudSteadObj_accu_limit);
        success &= loadParam(dyn_obj_node, "voxel_filter_size", params.voxel_filter_size);

        // Load debug and cluster parameters
        success &= loadParam(dyn_obj_node, "point_index", params.point_index);
        success &= loadParam(dyn_obj_node, "debug_x", params.debug_x);
        success &= loadParam(dyn_obj_node, "debug_y", params.debug_y);
        success &= loadParam(dyn_obj_node, "debug_z", params.debug_z);
        success &= loadParam(dyn_obj_node, "cluster_coupled", params.cluster_coupled);
        success &= loadParam(dyn_obj_node, "cluster_future", params.cluster_future);
        success &= loadParam(dyn_obj_node, "cluster_extend_pixel", params.cluster_extend_pixel);
        success &= loadParam(dyn_obj_node, "cluster_min_pixel_number", params.cluster_min_pixel_number);
        success &= loadParam(dyn_obj_node, "cluster_Voxel_revolusion", params.cluster_Voxel_revolusion);
        success &= loadParam(dyn_obj_node, "cluster_thrustable_thresold", params.cluster_thrustable_thresold);
        success &= loadParam(dyn_obj_node, "cluster_debug_en", params.cluster_debug_en);
        success &= loadParam(dyn_obj_node, "cluster_out_file", params.cluster_out_file);

        // Load Output/Frame ID settings
        success &= loadParam(dyn_obj_node, "frame_id", params.frame_id);
        success &= loadParam(dyn_obj_node, "time_file", params.time_file);
        success &= loadParam(dyn_obj_node, "time_breakdown_file", params.time_breakdown_file);
        
        // Load filter and debug flags
        success &= loadParam(dyn_obj_node, "dyn_filter_en", params.dyn_filter_en);
        success &= loadParam(dyn_obj_node, "debug_publish", params.debug_publish);
        // Load resolution parameters
        success &= loadParam(dyn_obj_node, "ver_resolution_max", params.ver_resolution_max);
        success &= loadParam(dyn_obj_node, "hor_resolution_max", params.hor_resolution_max);
            
        // Check if all parameters were loaded successfully
        if (!success) {
            std::cerr << "Error: Some parameters could not be loaded from config file: " << filename << std::endl;
            return false;
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing YAML file: " << filename << " - " << e.what() << std::endl;
        return false;
    }

    return success;
}