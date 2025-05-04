// file: config/config_loader.h

/**
 * @file config_loader.h
 * @brief Defines structures and functions for loading dynamic object filter parameters from a YAML configuration file.
 *
 * This header declares the DynObjFilterParams struct, which holds all configuration
 * parameters (both loaded and derived), and the functions necessary to parse a YAML
 * file and populate this struct. It utilizes the yaml-cpp library for parsing.
 */

#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H


#include "common/types.h"
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h> // Include YAML-CPP header
#include <type_traits> // For std::is_same
#include <iostream>
 
/**
 * @struct DynObjFilterParams
 * @brief Holds all configuration parameters for the dynamic object filtering process.
 *
 * This structure aggregates parameters loaded directly from the YAML configuration file
 * as well as parameters derived from the loaded values (e.g., pixel counts based on
 * angular resolutions and thresholds). It includes default values for all parameters,
 * which are used if a parameter is missing in the config file.
 */
struct DynObjFilterParams {

    // === Loaded Parameters ===

    // --- Buffer and processing parameters ---
    double buffer_delay;            /**< @brief Delay in seconds before processing points in the buffer. */
    int buffer_size;                /**< @brief Maximum number of points to store in the buffer. */
    int points_num_perframe;        /**< @brief Expected approximate number of points per frame/scan. */
    int history_length;             /**< @brief Number of scans to keep in the buffer for analysis*/
    double depth_map_dur;           /**< @brief Duration in seconds that each depth map represents. */
    int max_depth_map_num;          /**< @brief Maximum number of historical depth maps to keep. */
    int max_pixel_points;           /**< @brief Maximum number of points allowed in a single pixel/cell of the depth map. */
    double frame_dur;               /**< @brief Expected duration in seconds of a single frame/scan. */
    int dataset;                    /**< @brief Identifier for the dataset being processed (e.g., 0, 1), used for dataset-specific logic. */
    double buffer_dur;              /**< @brief Duration in seconds related to buffering (potentially redundant with buffer_delay?). */
    int point_index;                /**< @brief Index related to point processing (purpose needs clarification). */

    // --- Field of view and self-filtering parameters ---
    float self_x_f;                 /**< @brief Forward distance threshold for self-filtering along the x-axis. */
    float self_x_b;                 /**< @brief Backward distance threshold for self-filtering along the x-axis. */
    float self_y_l;                 /**< @brief Left distance threshold for self-filtering along the y-axis. */
    float self_y_r;                 /**< @brief Right distance threshold for self-filtering along the y-axis. */
    float blind_dis;                /**< @brief Minimum distance threshold; points closer than this are considered invalid. */
    float fov_up;                   /**< @brief Upper vertical field of view limit (degrees). */
    float fov_down;                 /**< @brief Lower vertical field of view limit (degrees, often negative). */
    float fov_cut;                  /**< @brief Vertical field of view cut-off limit (degrees, purpose needs clarification). */
    float fov_left;                 /**< @brief Left horizontal field of view limit (degrees). */
    float fov_right;                /**< @brief Right horizontal field of view limit (degrees, often negative or wraps around). */

    // --- Neighbor check parameters ---
    int checkneighbor_range;        /**< @brief Half-size of the square neighborhood (in pixels) to check around a point. (e.g., 1 means 3x3). */
    bool stop_object_detect;        /**< @brief Flag to enable/disable stopped object detection logic. */

    // --- Case 1 parameters (Related to newly detected/appearing objects) ---
    float depth_thr1;               /**< @brief Depth threshold for Case 1 checks. */
    float enter_min_thr1;           /**< @brief Minimum threshold for entering Case 1 state. */
    float enter_max_thr1;           /**< @brief Maximum threshold for entering Case 1 state. */
    float map_cons_depth_thr1;      /**< @brief Depth consistency threshold against map for Case 1. */
    float map_cons_hor_thr1;        /**< @brief Horizontal angular consistency threshold against map for Case 1 (radians). */
    float map_cons_ver_thr1;        /**< @brief Vertical angular consistency threshold against map for Case 1 (radians). */
    float map_cons_hor_dis1;        /**< @brief Horizontal distance consistency threshold against map for Case 1 (meters?). */
    float map_cons_ver_dis1;        /**< @brief Vertical distance consistency threshold against map for Case 1 (meters?). */
    float depth_cons_depth_thr1;    /**< @brief Depth consistency threshold between points for Case 1. */
    float depth_cons_depth_max_thr1;/**< @brief Maximum depth consistency threshold between points for Case 1. */
    float depth_cons_hor_thr1;      /**< @brief Horizontal angular consistency threshold between points for Case 1 (radians). */
    float depth_cons_ver_thr1;      /**< @brief Vertical angular consistency threshold between points for Case 1 (radians). */
    float enlarge_z_thr1;           /**< @brief Z-axis enlargement threshold for Case 1. */
    float enlarge_angle;            /**< @brief Angular enlargement threshold (degrees?). */
    float enlarge_depth;            /**< @brief Depth enlargement threshold. */
    int occluded_map_thr1;          /**< @brief Occlusion count threshold based on map checks for Case 1. */
    bool case1_interp_en;           /**< @brief Enable/disable interpolation specifically for Case 1 points. */
    float k_depth_min_thr1;         /**< @brief Proportional factor for minimum depth threshold calculation in Case 1. */
    float d_depth_min_thr1;         /**< @brief Constant offset for minimum depth threshold calculation in Case 1. */
    float k_depth_max_thr1;         /**< @brief Proportional factor for maximum depth threshold calculation in Case 1. */
    float d_depth_max_thr1;         /**< @brief Constant offset for maximum depth threshold calculation in Case 1. */

    // --- Case 2 parameters (Related to a new occlusion) ---
    float v_min_thr2;               /**< @brief Minimum velocity threshold for Case 2. */
    float acc_thr2;                 /**< @brief Acceleration threshold for Case 2. */
    float map_cons_depth_thr2;      /**< @brief Depth consistency threshold against map for Case 2. */
    float map_cons_hor_thr2;        /**< @brief Horizontal angular consistency threshold against map for Case 2 (radians). */
    float map_cons_ver_thr2;        /**< @brief Vertical angular consistency threshold against map for Case 2 (radians). */
    float occ_depth_thr2;           /**< @brief Occlusion depth threshold for Case 2. */
    float occ_hor_thr2;             /**< @brief Occlusion horizontal angular threshold for Case 2 (radians). */
    float occ_ver_thr2;             /**< @brief Occlusion vertical angular threshold for Case 2 (radians). */
    float depth_cons_depth_thr2;    /**< @brief Depth consistency threshold between points for Case 2. */
    float depth_cons_depth_max_thr2;/**< @brief Maximum depth consistency threshold between points for Case 2. */
    float depth_cons_hor_thr2;      /**< @brief Horizontal angular consistency threshold between points for Case 2 (radians). */
    float depth_cons_ver_thr2;      /**< @brief Vertical angular consistency threshold between points for Case 2 (radians). */
    float k_depth2;                 /**< @brief Proportional factor for depth threshold calculation in Case 2. */
    int occluded_times_thr2;        /**< @brief Occluded times count threshold for Case 2. */
    bool case2_interp_en;           /**< @brief Enable/disable interpolation specifically for Case 2 points. */
    float k_depth_max_thr2;         /**< @brief Proportional factor for maximum depth threshold calculation in Case 2. */
    float d_depth_max_thr2;         /**< @brief Constant offset for maximum depth threshold calculation in Case 2. */


    // --- Case 3 parameters (Related to an old occlusion that is now visible) ---
    float v_min_thr3;               /**< @brief Minimum velocity threshold for Case 3. */
    float acc_thr3;                 /**< @brief Acceleration threshold for Case 3. */
    float map_cons_depth_thr3;      /**< @brief Depth consistency threshold against map for Case 3. */
    float map_cons_hor_thr3;        /**< @brief Horizontal angular consistency threshold against map for Case 3 (radians). */
    float map_cons_ver_thr3;        /**< @brief Vertical angular consistency threshold against map for Case 3 (radians). */
    float occ_depth_thr3;           /**< @brief Occlusion depth threshold for Case 3. */
    float occ_hor_thr3;             /**< @brief Occlusion horizontal angular threshold for Case 3 (radians). */
    float occ_ver_thr3;             /**< @brief Occlusion vertical angular threshold for Case 3 (radians). */
    float depth_cons_depth_thr3;    /**< @brief Depth consistency threshold between points for Case 3. */
    float depth_cons_depth_max_thr3;/**< @brief Maximum depth consistency threshold between points for Case 3. */
    float depth_cons_hor_thr3;      /**< @brief Horizontal angular consistency threshold between points for Case 3 (radians). */
    float depth_cons_ver_thr3;      /**< @brief Vertical angular consistency threshold between points for Case 3 (radians). */
    float k_depth3;                 /**< @brief Proportional factor for depth threshold calculation in Case 3. */
    int occluding_times_thr3;       /**< @brief Occluding times count threshold for Case 3. */
    bool case3_interp_en;           /**< @brief Enable/disable interpolation specifically for Case 3 points. */
    float k_depth_max_thr3;         /**< @brief Proportional factor for maximum depth threshold calculation in Case 3. */
    float d_depth_max_thr3;         /**< @brief Constant offset for maximum depth threshold calculation in Case 3. */

    // --- Interpolation parameters ---
    float interp_hor_thr;           /**< @brief Horizontal angular threshold (radians) for selecting interpolation neighbors. */
    float interp_ver_thr;           /**< @brief Vertical angular threshold (radians) for selecting interpolation neighbors. */
    float interp_thr1;              /**< @brief Interpolation threshold 1 (purpose needs clarification, maybe related to Case 1?). */
    float interp_static_max;        /**< @brief Maximum depth value allowed for static interpolation results. */
    float interp_start_depth1;      /**< @brief Depth threshold to start applying specific interpolation logic (Case 1?). */
    float interp_kp1;               /**< @brief Proportional factor for interpolation threshold calculation (Case 1?). */
    float interp_kd1;               /**< @brief Constant offset for interpolation threshold calculation (Case 1?). */
    float interp_thr2;              /**< @brief Interpolation threshold 2 (purpose needs clarification, maybe related to Case 2?). */
    float interp_thr3;              /**< @brief Interpolation threshold 3 (purpose needs clarification, maybe related to Case 3?). */

    // Parameters found during refactoring
    // Parameters related to consistency checks:
    float enlarge_distort;
    int cutoff_value;

    // --- Other configuration parameters ---
    bool dyn_filter_en;             /**< @brief Master enable/disable flag for the entire dynamic object filter. */
    bool debug_en;                  /**< @brief Enable/disable debug output or logging. */
    int laserCloudSteadObj_accu_limit; /**< @brief Accumulation limit related to steady object point clouds. */
    float voxel_filter_size;        /**< @brief Voxel grid size for downsampling or other voxel-based operations. */
    bool cluster_coupled;           /**< @brief Flag related to clustering coupling. */
    bool cluster_future;            /**< @brief Flag related to future state in clustering. */
    int Cluster_cluster_extend_pixel;/**< @brief Pixel extension range for clustering. */
    int Cluster_cluster_min_pixel_number; /**< @brief Minimum number of pixels required to form a cluster. */
    float Cluster_thrustable_thresold; /**< @brief Trustworthiness threshold for clusters. */
    float Cluster_Voxel_revolusion; /**< @brief Voxel resolution specifically for clustering (typo: should be resolution?). */
    bool Cluster_debug_en;          /**< @brief Enable/disable debug output specifically for clustering. */
    std::string Cluster_out_file;   /**< @brief Output file path for clustering results. */
    float hor_resolution_max;       /**< @brief Maximum horizontal angular resolution of the sensor (radians per pixel/bin). */
    float ver_resolution_max;       /**< @brief Maximum vertical angular resolution of the sensor (radians per pixel/bin). */
    std::string frame_id;           /**< @brief Default frame ID for output messages or coordinate transforms. */
    std::string time_file;          /**< @brief File path for logging timing information. */
    std::string time_breakdown_file;/**< @brief File path for logging detailed timing breakdown. */

    // === Derived Parameters (Calculated after loading) ===
    int interp_hor_num = 0;         /**< @brief Derived pixel count for interpolation horizontal search range based on interp_hor_thr. */
    int interp_ver_num = 0;         /**< @brief Derived pixel count for interpolation vertical search range based on interp_ver_thr. */
    int pixel_fov_up = 0;           /**< @brief Derived pixel index corresponding to the upper vertical FOV limit (fov_up). */
    int pixel_fov_down = 0;         /**< @brief Derived pixel index corresponding to the lower vertical FOV limit (fov_down). */
    int pixel_fov_cut = 0;          /**< @brief Derived pixel index corresponding to the vertical FOV cut limit (fov_cut). */
    int pixel_fov_left = 0;         /**< @brief Derived pixel index corresponding to the left horizontal FOV limit (fov_left). */
    int pixel_fov_right = 0;        /**< @brief Derived pixel index corresponding to the right horizontal FOV limit (fov_right). */
    int max_pointers_num = 0;       /**< @brief Derived maximum number of point_soph pointer buffers needed based on map history and frame rate. */
    int depth_cons_ver_num2 = 0;     
    int depth_cons_ver_num3 = 0;
    int depth_cons_hor_num2 = 0;     
    int depth_cons_hor_num3 = 0;
    int occ_hor_num2 = 0;
    int occ_ver_num2 = 0;
    int occ_hor_num3 = 0;
    int occ_ver_num3 = 0;


    /**
     * @brief Construct a new Dyn Obj Filter Params object with default values.
     *
     * Initializes all parameters to predefined default values. These defaults are used
     * if the corresponding parameter is not found in the loaded configuration file.
     */
    DynObjFilterParams() :
    buffer_delay(0.1),
    buffer_size(300000),
    points_num_perframe(150000),
    history_length(5),
    depth_map_dur(0.2),
    max_depth_map_num(5),
    max_pixel_points(50),
    frame_dur(0.1),
    dataset(0),
    buffer_dur(0.1f), // Added default based on usage in config_loader.cpp
    point_index(0), // Added default based on usage in config_loader.cpp
    self_x_f(0.15f),
    self_x_b(0.15f),
    self_y_l(0.15f),
    self_y_r(0.5f),
    blind_dis(0.15f),
    fov_up(0.15f), // Note: Default FOV values seem small (0.15 degrees)
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
    enlarge_distort(4.0),
    cutoff_value(0),
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
    hor_resolution_max(0.0025f), // Radians
    ver_resolution_max(0.0025f), // Radians
    
    frame_id("camera_init"),
    time_file(""),
    time_breakdown_file("")        
    {
        // Derived parameters are initialized to 0 or empty string by default member initialization
        // and calculated in load_config.
    }

};
 
/**
 * @brief Loads dynamic object filter parameters from a specified YAML file.
 *
 * Parses the YAML file, extracts parameters under the 'dyn_obj' group,
 * populates the provided DynObjFilterParams struct, and calculates derived parameters.
 * Handles file opening errors, YAML parsing errors, and missing parameters (uses defaults).
 *
 * @param filename The path to the YAML configuration file.
 * @param[out] params The DynObjFilterParams struct to be populated.
 * @return true if the configuration was loaded and parsed successfully (including calculation of derived parameters), false otherwise.
 */
bool load_config(const std::string& filename, DynObjFilterParams& params);

/**
 * @brief Template helper function to load a single parameter from a YAML node.
 *
 * Attempts to extract a parameter with the given name from the parent YAML node
 * and convert it to the specified type T. If the parameter is not found, it
 * silently leaves the target variable unchanged (preserving its default value)
 * and returns true. If a YAML parsing or type conversion error occurs, it prints
 * an error message to std::cerr and returns false.
 *
 * @tparam T The data type of the parameter to load.
 * @param parentNode The parent YAML node (e.g., the 'dyn_obj' node).
 * @param param_name The name of the parameter key within the parent node.
 * @param[out] param_var The variable where the loaded parameter value will be stored.
 * @return true if the parameter was loaded successfully OR if it was not found (using default), false if a loading/conversion error occurred.
 */
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
 
#endif // CONFIG_LOADER_H