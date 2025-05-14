/**
 * @file config_loader.h
 * @brief Defines structures and functions for loading dynamic object filter parameters from a YAML configuration file.
 */

 #ifndef CONFIG_LOADER_H
 #define CONFIG_LOADER_H
 
 #include "common/types.h" // Includes M3D, V3D etc. (needed by users of the struct potentially)
 #include <string>
 #include <vector> // Potentially needed if params include vectors in the future
 #include <yaml-cpp/yaml.h> // Include YAML-CPP header required for load_config signature
 
 // Note: <iostream> and <type_traits> removed as they are not needed in the header itself.
 
 /**
  * @struct DynObjFilterParams
  * @brief Holds all configuration parameters for the dynamic object filtering process.
  */
 struct DynObjFilterParams {
 
     // === Loaded Parameters ===
 
     // --- General & Dataset ---
     int dataset;                    /**< @brief Identifier for the dataset (0:kitti, 1:nuscenes, etc.), used for dataset-specific logic. */
     std::string frame_id;           /**< @brief Default frame ID for output messages or coordinate transforms. */
     bool dyn_filter_en;             /**< @brief Master enable/disable flag for the entire dynamic object filter. */
 
     // --- Buffering & Timing ---
     double buffer_delay;            /**< @brief Delay in seconds before processing points in the buffer (purpose needs clarification vs buffer_dur). */
     int buffer_size;                /**< @brief Maximum number of points to store in the buffer (purpose needs clarification - point buffer or frame buffer?). */
     int history_length;             /**< @brief Number of scans (ScanFrames) to keep in the ring buffer for analysis. */
     double depth_map_dur;           /**< @brief Duration in seconds that each depth map represents. New map created after this time. */
     int max_depth_map_num;          /**< @brief Maximum number of historical depth maps to keep in the list. */
     double frame_dur;               /**< @brief Expected duration in seconds of a single frame/scan (used for calculations?). */
     double buffer_dur;              /**< @brief Duration in seconds related to buffering (potentially redundant with buffer_delay?). */
     int points_num_perframe;        /**< @brief Expected approximate number of points per frame/scan (used for reservations?). */
 
     // --- Sensor Characteristics & FOV ---
     float hor_resolution_max;       /**< @brief Maximum horizontal angular resolution of the sensor (radians per pixel/bin). */
     float ver_resolution_max;       /**< @brief Maximum vertical angular resolution of the sensor (radians per pixel/bin). */
     float fov_up;                   /**< @brief Upper vertical field of view limit (degrees). */
     float fov_down;                 /**< @brief Lower vertical field of view limit (degrees, often negative). */
     float fov_cut;                  /**< @brief Vertical field of view cut-off limit (degrees, purpose needs clarification - maybe related to projection?). */
     float fov_left;                 /**< @brief Left horizontal field of view limit (degrees). */
     float fov_right;                /**< @brief Right horizontal field of view limit (degrees, often negative or wraps around). */
 
     // --- Point Filtering (Invalid / Self) ---
     float blind_dis;                /**< @brief Minimum distance threshold; points closer than this are considered invalid. */
     bool enable_invalid_box_check;  /**< @brief Enable check for an additional invalid box region near origin. */
     float invalid_box_x_half_width; /**< @brief Half-width of the invalid box along X (if enabled). */
     float invalid_box_y_half_width; /**< @brief Half-width of the invalid box along Y (if enabled). */
     float invalid_box_z_half_width; /**< @brief Half-width of the invalid box along Z (if enabled). */
     float self_x_f;                 /**< @brief Forward distance threshold for self-filtering along the x-axis. */
     float self_x_b;                 /**< @brief Backward distance threshold for self-filtering along the x-axis. */
     float self_y_l;                 /**< @brief Left distance threshold for self-filtering along the y-axis. */
     float self_y_r;                 /**< @brief Right distance threshold for self-filtering along the y-axis. */
 
     // --- Depth Map & Grid ---
     int max_pixel_points;           /**< @brief Maximum number of points allowed in a single pixel/cell of the depth map. */
 
     // --- Neighbor Check Parameters ---
     int checkneighbor_range;        /**< @brief Half-size of the square neighborhood (in pixels) to check around a point. (e.g., 1 means 3x3). */
 
     // --- Stopped Object Detection ---
     bool stop_object_detect;        /**< @brief Flag to enable/disable stopped object detection logic. */
     int laserCloudSteadObj_accu_limit; /**< @brief Accumulation limit related to steady object point clouds. */
 
     // --- Case 1 Parameters (Appearing) ---
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
     float k_depth_min_thr1;         /**< @brief Proportional factor for minimum depth threshold calculation in Case 1. */
     float d_depth_min_thr1;         /**< @brief Constant offset for minimum depth threshold calculation in Case 1. */
     float k_depth_max_thr1;         /**< @brief Proportional factor for maximum depth threshold calculation in Case 1. */
     float d_depth_max_thr1;         /**< @brief Constant offset for maximum depth threshold calculation in Case 1. */
     float enlarge_z_thr1;           /**< @brief Z-axis enlargement threshold for Case 1. */
     float enlarge_angle;            /**< @brief Angular enlargement threshold (degrees?). */
     float enlarge_depth;            /**< @brief Depth enlargement threshold. */
     int occluded_map_thr1;          /**< @brief Occlusion count threshold based on map checks for Case 1. */
     int occluded_map_thr2;          /**< @brief Occlusion count threshold based on map checks for Case 1 when few static neighbors are found. */
     bool case1_interp_en;           /**< @brief Enable/disable interpolation specifically for Case 1 points. */
 
     // --- Case 2 Parameters (Occluding) ---
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
     float k_depth_max_thr2;         /**< @brief Proportional factor for maximum depth threshold calculation in Case 2. */
     float d_depth_max_thr2;         /**< @brief Constant offset for maximum depth threshold calculation in Case 2. */
     int occluded_times_thr2;        /**< @brief Occluded times count threshold for Case 2. */
     bool case2_interp_en;           /**< @brief Enable/disable interpolation specifically for Case 2 points. */
 
     // --- Case 3 Parameters (Disoccluded) ---
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
     float k_depth_max_thr3;         /**< @brief Proportional factor for maximum depth threshold calculation in Case 3. */
     float d_depth_max_thr3;         /**< @brief Constant offset for maximum depth threshold calculation in Case 3. */
     int occluding_times_thr3;       /**< @brief Occluding times count threshold for Case 3. */
     bool case3_interp_en;           /**< @brief Enable/disable interpolation specifically for Case 3 points. */
 
     // --- Interpolation Parameters ---
     float interp_hor_thr;           /**< @brief Horizontal angular threshold (radians) for selecting interpolation neighbors. */
     float interp_ver_thr;           /**< @brief Vertical angular threshold (radians) for selecting interpolation neighbors. */
     float interp_thr1;              /**< @brief Interpolation threshold 1 (purpose needs clarification, maybe related to Case 1?). */
     float interp_static_max;        /**< @brief Maximum depth value allowed for static interpolation results. */
     float interp_start_depth1;      /**< @brief Depth threshold to start applying specific interpolation logic (Case 1?). */
     float interp_kp1;               /**< @brief Proportional factor for interpolation threshold calculation (Case 1?). */
     float interp_kd1;               /**< @brief Constant offset for interpolation threshold calculation (Case 1?). */
     float interp_thr2;              /**< @brief Interpolation threshold 2 (purpose needs clarification, maybe related to Case 2?). */
     float interp_thr3;              /**< @brief Interpolation threshold 3 (purpose needs clarification, maybe related to Case 3?). */
 
     // --- Clustering Parameters ---
     bool cluster_coupled;           /**< @brief Flag related to clustering coupling. */
     bool cluster_future;            /**< @brief Flag related to future state in clustering. */
     int Cluster_cluster_extend_pixel;/**< @brief Pixel extension range for clustering. */
     int Cluster_cluster_min_pixel_number; /**< @brief Minimum number of pixels required to form a cluster. */
     float Cluster_thrustable_thresold; /**< @brief Trustworthiness threshold for clusters. */
     float Cluster_Voxel_resolution; /**< @brief Voxel resolution specifically for clustering (fixed typo). */
     bool Cluster_debug_en;          /**< @brief Enable/disable debug output specifically for clustering. */
     std::string Cluster_out_file;   /**< @brief Output file path for clustering results. */
 
     // --- Debugging & Misc ---
    //  bool debug_en;                              /**< @brief Enable/disable general debug output or logging. */
    std::string log_level;                          /**< @brief Default logging level (e.g., "trace", "debug", "info", "warn", "error", "critical", "off"). */
    std::map<std::string, std::string> log_levels;  /**< @brief Optional map to override log levels for specific named loggers (e.g., {"Filter": "debug", "Consistency": "trace"}). */
     std::string time_file;                         /**< @brief File path for logging timing information. */
     std::string time_breakdown_file;               /**< @brief File path for logging detailed timing breakdown. */
     int point_index;                               /**< @brief Index related to point processing (purpose needs clarification, maybe debug?). */
     float enlarge_distort;                         /**< @brief Distortion enlargement factor (purpose needs clarification). */
     int cutoff_value;                              /**< @brief A cutoff value (purpose needs clarification). */
     float voxel_filter_size;                       /**< @brief Voxel grid size for downsampling or other voxel-based operations (if used). */
 
     // === Derived Parameters (Calculated after loading in load_config) ===
     int interp_hor_num = 0;         /**< @brief Derived pixel count for interpolation horizontal search range based on interp_hor_thr. */
     int interp_ver_num = 0;         /**< @brief Derived pixel count for interpolation vertical search range based on interp_ver_thr. */
     int pixel_fov_up = 0;           /**< @brief Derived pixel index corresponding to the upper vertical FOV limit (fov_up). */
     int pixel_fov_down = 0;         /**< @brief Derived pixel index corresponding to the lower vertical FOV limit (fov_down). */
     int pixel_fov_cut = 0;          /**< @brief Derived pixel index corresponding to the vertical FOV cut limit (fov_cut). */
     int pixel_fov_left = 0;         /**< @brief Derived pixel index corresponding to the left horizontal FOV limit (fov_left). */
     int pixel_fov_right = 0;        /**< @brief Derived pixel index corresponding to the right horizontal FOV limit (fov_right). */
     int max_pointers_num = 0;       /**< @brief Derived maximum number of point_soph pointer buffers needed based on map history and frame rate. */
     int depth_cons_ver_num2 = 0;    /**< @brief Derived pixel count for vertical depth consistency check (Case 2). */
     int depth_cons_ver_num3 = 0;    /**< @brief Derived pixel count for vertical depth consistency check (Case 3). */
     int depth_cons_hor_num2 = 0;    /**< @brief Derived pixel count for horizontal depth consistency check (Case 2). */
     int depth_cons_hor_num3 = 0;    /**< @brief Derived pixel count for horizontal depth consistency check (Case 3). */
     int occ_hor_num2 = 0;           /**< @brief Derived pixel count for horizontal occlusion check (Case 2). */
     int occ_ver_num2 = 0;           /**< @brief Derived pixel count for vertical occlusion check (Case 2). */
     int occ_hor_num3 = 0;           /**< @brief Derived pixel count for horizontal occlusion check (Case 3). */
     int occ_ver_num3 = 0;           /**< @brief Derived pixel count for vertical occlusion check (Case 3). */
 
 
     /**
      * @brief Construct a new DynObjFilterParams object with default values.
      * Initializes all parameters to predefined default values, matching declaration order.
      */
     DynObjFilterParams() :
        // --- General & Dataset ---
        dataset(0),
        frame_id("camera_init"),
        dyn_filter_en(true),
        // --- Buffering & Timing ---
        buffer_delay(0.1),
        buffer_size(300000),
        history_length(5),
        depth_map_dur(0.2),
        max_depth_map_num(5),
        frame_dur(0.1),
        buffer_dur(0.1f),
        points_num_perframe(150000),
        // --- Sensor Characteristics & FOV ---
        hor_resolution_max(0.0025f), // Radians (~0.14 deg) - Default seems reasonable
        ver_resolution_max(0.0025f), // Radians (~0.14 deg) - Default seems reasonable
        fov_up(10.0f), // Degrees - Changed default to be more realistic
        fov_down(-30.0f), // Degrees - Changed default to be more realistic
        fov_cut(0.0f), // Degrees - Default 0?
        fov_left(180.0f), // Degrees
        fov_right(-180.0f), // Degrees
        // --- Point Filtering (Invalid / Self) ---
        blind_dis(0.2f), // Default from your yaml
        enable_invalid_box_check(false), // Default from your yaml (was true before)
        invalid_box_x_half_width(0.1f),
        invalid_box_y_half_width(1.0f),
        invalid_box_z_half_width(0.1f),
        self_x_f(2.2f), // Default from your yaml
        self_x_b(-1.2f), // Default from your yaml
        self_y_l(0.7f), // Default from your yaml
        self_y_r(-0.7f), // Default from your yaml
        // --- Depth Map & Grid ---
        max_pixel_points(50),
        // --- Neighbor Check Parameters ---
        checkneighbor_range(1),
        // --- Stopped Object Detection ---
        stop_object_detect(false),
        laserCloudSteadObj_accu_limit(5),
        // --- Case 1 Parameters (Appearing) ---
        depth_thr1(0.15f),
        enter_min_thr1(1.0f), // Default from your yaml
        enter_max_thr1(0.3f), // Default from your yaml
        map_cons_depth_thr1(0.3f), // Default from your yaml
        map_cons_hor_thr1(0.01f),
        map_cons_ver_thr1(0.03f), // Default from your yaml
        map_cons_hor_dis1(0.15f), // Default from your yaml
        map_cons_ver_dis1(0.15f), // Default from your yaml
        depth_cons_depth_thr1(0.5f),
        depth_cons_depth_max_thr1(1.0f), // Default from your yaml
        depth_cons_hor_thr1(0.02f),
        depth_cons_ver_thr1(0.01f),
        k_depth_min_thr1(0.0f), // Default seems ok
        d_depth_min_thr1(0.15f), // Default seems ok
        k_depth_max_thr1(0.0f), // Default seems ok
        d_depth_max_thr1(0.15f), // Default seems ok
        enlarge_z_thr1(-2.5f), // Default from your yaml
        enlarge_angle(2.0f),
        enlarge_depth(3.0f),
        occluded_map_thr1(2), // Default from your yaml
        occluded_map_thr2(3), // Default from your yaml
        case1_interp_en(true), // Default from your yaml
        // --- Case 2 Parameters (Occluding) ---
        v_min_thr2(1.0f), // Default from your yaml
        acc_thr2(7.0f), // Default from your yaml
        map_cons_depth_thr2(0.2f), // Default from your yaml
        map_cons_hor_thr2(0.01f), // Default from your yaml
        map_cons_ver_thr2(0.03f), // Default from your yaml
        occ_depth_thr2(10.15f), // Default from your yaml - Note: Large value!
        occ_hor_thr2(0.01f), // Default from your yaml
        occ_ver_thr2(0.04f), // Default from your yaml
        depth_cons_depth_thr2(0.1f), // Default from your yaml
        depth_cons_depth_max_thr2(0.5f),
        depth_cons_hor_thr2(0.01f), // Default from your yaml
        depth_cons_ver_thr2(0.03f), // Default from your yaml
        k_depth2(0.005f),
        k_depth_max_thr2(0.0f), // Default seems ok
        d_depth_max_thr2(0.15f), // Default seems ok
        occluded_times_thr2(2), // Default from your yaml
        case2_interp_en(false),
        // --- Case 3 Parameters (Disoccluded) ---
        v_min_thr3(0.5f),
        acc_thr3(15.0f), // Default from your yaml
        map_cons_depth_thr3(0.2f), // Default from your yaml
        map_cons_hor_thr3(0.01f), // Default from your yaml
        map_cons_ver_thr3(0.03f), // Default from your yaml
        occ_depth_thr3(0.15f),
        occ_hor_thr3(0.01f), // Default from your yaml
        occ_ver_thr3(0.04f), // Default from your yaml
        depth_cons_depth_thr3(0.3f), // Default from your yaml
        depth_cons_depth_max_thr3(1.0f), // Default from your yaml
        depth_cons_hor_thr3(0.01f), // Default from your yaml
        depth_cons_ver_thr3(0.03f), // Default from your yaml
        k_depth3(0.005f),
        k_depth_max_thr3(0.0f), // Default seems ok
        d_depth_max_thr3(0.15f), // Default seems ok
        occluding_times_thr3(2), // Default from your yaml
        case3_interp_en(false),
        // --- Interpolation Parameters ---
        interp_hor_thr(0.015f), // Default from your yaml
        interp_ver_thr(0.06f), // Default from your yaml
        interp_thr1(1.5f), // Default from your yaml
        interp_static_max(10.0f),
        interp_start_depth1(15.0f), // Default from your yaml
        interp_kp1(0.15f), // Default from your yaml
        interp_kd1(1.5f), // Default from your yaml
        interp_thr2(0.25f), // Default from your yaml
        interp_thr3(0.05f), // Default from your yaml
        // --- Clustering Parameters ---
        cluster_coupled(true), // Default from your yaml
        cluster_future(true), // Default from your yaml
        Cluster_cluster_extend_pixel(3), // Default from your yaml
        Cluster_cluster_min_pixel_number(3), // Default from your yaml
        Cluster_thrustable_thresold(0.3f),
        Cluster_Voxel_resolution(0.3f), // Fixed typo
        Cluster_debug_en(false),
        Cluster_out_file(""),
        // --- Debugging & Misc ---
        log_level("info"),
        time_file(""),
        time_breakdown_file(""),
        point_index(-1), // Default from your yaml
        enlarge_distort(4.0), // Default seems ok
        cutoff_value(0), // Default seems ok
        voxel_filter_size(0.1f) // Default seems ok
         // Derived parameters are default initialized by their declaration
     {
         // Constructor body is empty, all initialization is done in the list above
         // or by default member initializers for derived params.
     }
 
 }; // End struct DynObjFilterParams
 
 /**
  * @brief Loads dynamic object filter parameters from a specified YAML file.
  * Parses the YAML file, extracts parameters under the 'dyn_obj' group,
  * populates the provided DynObjFilterParams struct, and calculates derived parameters.
  * Handles file opening errors, YAML parsing errors, and missing parameters (uses defaults).
  * @param filename The path to the YAML configuration file.
  * @param[out] params The DynObjFilterParams struct to be populated.
  * @return true if the configuration was loaded and parsed successfully, false otherwise.
  */
 bool load_config(const std::string& filename, DynObjFilterParams& params);
 
 /**
  * @brief Template helper function to load a single parameter from a YAML node.
  * Attempts to extract a parameter with the given name from the parent YAML node.
  * If not found, it leaves the target variable unchanged (preserving default) and returns true.
  * If a YAML parsing or type conversion error occurs, it prints an error message and returns false.
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
             // Parameter not found, leave the default value in param_var untouched.
             // Optionally add a warning here if desired, but often using defaults is intended.
             // std::cerr << "Warning: Parameter '" << param_name << "' not found in config. Using default value." << std::endl;
             return true; // Return true because using the default is acceptable
         }
         param_var = paramNode.as<T>();
         return true; // Success
     } catch (const YAML::Exception& e) {
         // Log error if loading/conversion fails
         std::cerr << "Error loading parameter '" << param_name << "': " << e.what() << ". Check config file format/type. Default value might be used." << std::endl;
         return false; // Indicate that an error occurred during loading attempt
     }
 }
 
 #endif // CONFIG_LOADER_H