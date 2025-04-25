// file: include/config_loader.h

#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h> // Include YAML-CPP header
#include <type_traits> // For std::is_same
#include <iostream>

// Structure to hold ALL parameters from the YAML file
struct DynObjFilterParams {

    // Buffer and processing parameters
    double buffer_delay;
    int buffer_size;
    int points_num_perframe;
    double depth_map_dur;
    int max_depth_map_num;
    int max_pixel_points;
    double frame_dur;
    int dataset;
    
    // Field of view and self-filtering parameters
    float self_x_f;
    float self_x_b;
    float self_y_l;
    float self_y_r;
    float blind_dis;
    float fov_up;
    float fov_down;
    float fov_cut;
    float fov_left;
    float fov_right;
    
    // Neighbor check parameters
    int checkneighbor_range;
    bool stop_object_detect;
    
    // Case 1 parameters
    float depth_thr1;
    float enter_min_thr1;
    float enter_max_thr1;
    float map_cons_depth_thr1;
    float map_cons_hor_thr1;
    float map_cons_ver_thr1;
    float map_cons_hor_dis1;
    float map_cons_ver_dis1;
    float depth_cons_depth_thr1;
    float depth_cons_depth_max_thr1;
    float depth_cons_hor_thr1;
    float depth_cons_ver_thr1;
    float enlarge_z_thr1;
    float enlarge_angle;
    float enlarge_depth;
    int occluded_map_thr1;
    bool case1_interp_en;
    float k_depth_min_thr1;
    float d_depth_min_thr1;
    float k_depth_max_thr1;
    float d_depth_max_thr1;
    
    // Case 2 parameters
    float v_min_thr2;
    float acc_thr2;
    float map_cons_depth_thr2;
    float map_cons_hor_thr2;
    float map_cons_ver_thr2;
    float occ_depth_thr2;
    float occ_hor_thr2;
    float occ_ver_thr2;
    float depth_cons_depth_thr2;
    float depth_cons_depth_max_thr2;
    float depth_cons_hor_thr2;
    float depth_cons_ver_thr2;
    float k_depth2;
    int occluded_times_thr2;
    bool case2_interp_en;
    float k_depth_max_thr2;
    float d_depth_max_thr2;
    
    // Case 3 parameters
    float v_min_thr3;
    float acc_thr3;
    float map_cons_depth_thr3;
    float map_cons_hor_thr3;
    float map_cons_ver_thr3;
    float occ_depth_thr3;
    float occ_hor_thr3;
    float occ_ver_thr3;
    float depth_cons_depth_thr3;
    float depth_cons_depth_max_thr3;
    float depth_cons_hor_thr3;
    float depth_cons_ver_thr3;
    float k_depth3;
    int occluding_times_thr3;
    bool case3_interp_en;
    float k_depth_max_thr3;
    float d_depth_max_thr3;
    
    // Interpolation parameters
    float interp_hor_thr;
    float interp_ver_thr;
    float interp_thr1;
    float interp_static_max;
    float interp_start_depth1;
    float interp_kp1;
    float interp_kd1;
    float interp_thr2;
    float interp_thr3;
    
    // Other configuration parameters
    bool dyn_filter_en;
    bool debug_en;
    int laserCloudSteadObj_accu_limit;
    float voxel_filter_size;
    bool cluster_coupled;
    bool cluster_future;
    int Cluster_cluster_extend_pixel;
    int Cluster_cluster_min_pixel_number;
    float Cluster_thrustable_thresold;
    float Cluster_Voxel_revolusion;
    bool Cluster_debug_en;
    std::string Cluster_out_file;
    float hor_resolution_max;
    float ver_resolution_max;
    float buffer_dur;
    int point_index;
    std::string frame_id;
    std::string time_file;
    std::string time_breakdown_file;

    // --- Derived Parameters ---
    int interp_hor_num = 0; // Derived pixel count for interpolation horizontal search
    int interp_ver_num = 0; // Derived pixel count for interpolation vertical search
    int pixel_fov_up = 0;   // Derived pixel index for vertical FOV up limit
    int pixel_fov_down = 0; // Derived pixel index for vertical FOV down limit
    int pixel_fov_cut = 0;  // Derived pixel index for vertical FOV cut limit (if used)
    int pixel_fov_left = 0; // Derived pixel index for horizontal FOV left limit
    int pixel_fov_right = 0;// Derived pixel index for horizontal FOV right limit
    int max_pointers_num = 0; // Derived number of point_soph buffers needed

    // Constructor to set default values (copied from original code, dynobjfilter.c DynObjFilter::init())
    DynObjFilterParams() :
        buffer_delay(0.1),
        buffer_size(300000),
        points_num_perframe(150000),
        depth_map_dur(0.2),
        max_depth_map_num(5),
        max_pixel_points(50),
        frame_dur(0.1),
        dataset(0),
        self_x_f(0.15f),
        self_x_b(0.15f),
        self_y_l(0.15f),
        self_y_r(0.5f),
        blind_dis(0.15f),
        fov_up(0.15f),
        fov_down(0.15f),
        fov_cut(0.15f),
        fov_left(180.0f),
        fov_right(-180.0f),
        checkneighbor_range(1),
        stop_object_detect(false),
        depth_thr1(0.15f),
        enter_min_thr1(0.15f),
        enter_max_thr1(0.15f),
        map_cons_depth_thr1(0.5f),
        map_cons_hor_thr1(0.01f),
        map_cons_ver_thr1(0.01f),
        map_cons_hor_dis1(0.2f),
        map_cons_ver_dis1(0.1f),
        depth_cons_depth_thr1(0.5f),
        depth_cons_depth_max_thr1(0.5f),
        depth_cons_hor_thr1(0.02f),
        depth_cons_ver_thr1(0.01f),
        enlarge_z_thr1(0.05f),
        enlarge_angle(2.0f),
        enlarge_depth(3.0f),
        occluded_map_thr1(3),
        case1_interp_en(false),
        k_depth_min_thr1(0.0f),
        d_depth_min_thr1(0.15f),
        k_depth_max_thr1(0.0f),
        d_depth_max_thr1(0.15f),
        v_min_thr2(0.5f),
        acc_thr2(1.0f),
        map_cons_depth_thr2(0.15f),
        map_cons_hor_thr2(0.02f),
        map_cons_ver_thr2(0.01f),
        occ_depth_thr2(0.15f),
        occ_hor_thr2(0.02f),
        occ_ver_thr2(0.01f),
        depth_cons_depth_thr2(0.5f),
        depth_cons_depth_max_thr2(0.5f),
        depth_cons_hor_thr2(0.02f),
        depth_cons_ver_thr2(0.01f),
        k_depth2(0.005f),
        occluded_times_thr2(3),
        case2_interp_en(false),
        k_depth_max_thr2(0.0f),
        d_depth_max_thr2(0.15f),
        v_min_thr3(0.5f),
        acc_thr3(1.0f),
        map_cons_depth_thr3(0.15f),
        map_cons_hor_thr3(0.02f),
        map_cons_ver_thr3(0.01f),
        occ_depth_thr3(0.15f),
        occ_hor_thr3(0.02f),
        occ_ver_thr3(0.01f),
        depth_cons_depth_thr3(0.5f),
        depth_cons_depth_max_thr3(0.5f),
        depth_cons_hor_thr3(0.02f),
        depth_cons_ver_thr3(0.01f),
        k_depth3(0.005f),
        occluding_times_thr3(3),
        case3_interp_en(false),
        k_depth_max_thr3(0.0f),
        d_depth_max_thr3(0.15f),
        interp_hor_thr(0.01f),
        interp_ver_thr(0.01f),
        interp_thr1(1.0f),
        interp_static_max(10.0f),
        interp_start_depth1(20.0f),
        interp_kp1(0.1f),
        interp_kd1(1.0f),
        interp_thr2(0.15f),
        interp_thr3(0.15f),
        dyn_filter_en(true),
        debug_en(true),
        laserCloudSteadObj_accu_limit(5),
        voxel_filter_size(0.1f),
        cluster_coupled(false),
        cluster_future(false),
        Cluster_cluster_extend_pixel(2),
        Cluster_cluster_min_pixel_number(4),
        Cluster_thrustable_thresold(0.3f),
        Cluster_Voxel_revolusion(0.3f),
        Cluster_debug_en(false),
        Cluster_out_file(""),
        hor_resolution_max(0.0025f),
        ver_resolution_max(0.0025f),
        buffer_dur(0.1f),
        point_index(0),
        frame_id("camera_init"),
        time_file(""),
        time_breakdown_file("")
    {
    }

}; 

// Function to load parameters from a YAML file
bool load_config(const std::string& filename, DynObjFilterParams& params);
 
template <typename T>
bool loadParam(const YAML::Node& parentNode, const std::string& param_name, T& param_var) {
    try {
        const YAML::Node paramNode = parentNode[param_name];
        if (!paramNode) {
             // Parameter not found, leave the default value in param_var untouched
             // std::cerr << "Warning: Parameter '" << param_name << "' not found. Using default value." << std::endl; // Optional warning
             return true; // Return true because using the default is acceptable
        }
        param_var = paramNode.as<T>();
        return true; // Success
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading parameter '" << param_name << "': " << e.what() << ". Default value might be used." << std::endl;
        return false; // Error during loading/conversion is still an error
    }
}

#endif

