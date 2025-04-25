Phase 1: Isolate and Test DynObjFilter without ROS

    Setup & Baseline:
        Create a new branch or copy of the project for refactoring.
        Ensure the original ROS version builds and runs correctly (this is your baseline).
        Set up a new CMake project (CMakeLists.txt) that does not use catkin or find ROS packages. It should find Eigen, PCL (if needed by DynObjFilter itself, though it seems less reliant than DynObjCluster), and C++17 (for std::execution::par).

    Remove ROS Includes & Basic Types (within DynObjFilter files):
        Go through DynObjFilter.h, DynObjFilter.cpp, and any directly related helper files (like RingBuffer, DepthMap, point_soph).
        Remove #include <ros/ros.h> and other ROS headers.
        Replace ros::Time with std::chrono::time_point<std::chrono::high_resolution_clock> or double for timestamps. Be consistent.
        Replace ROS logging (ROS_INFO, etc.) with std::cout or a simple logging library like spdlog.
        Goal: Get DynObjFilter.cpp and its direct, non-ROS dependencies to compile using your new non-ROS CMake setup. It won't link or run yet.

    Create a Standalone main Entry Point:
        Create a new main.cpp file (or similar).
        Include DynObjFilter.h.
        Instantiate DynObjFilter: std::shared_ptr<DynObjFilter> dyn_filter(new DynObjFilter());.
        Goal: Have a basic executable structure that includes the filter class.

    Replace ROS Parameter Loading:
        The DynObjFilter::init(ros::NodeHandle& nh) function uses nh.param<...>().
        Modify DynObjFilter::init to accept parameters differently, e.g., via a struct, a configuration file path, or individual arguments.
        In your main.cpp, implement a simple configuration loader (e.g., read a basic text file, JSON, or YAML using libraries like nlohmann/json or yaml-cpp) to load the parameters needed by DynObjFilter.
        Call the modified dyn_filter->init(...) with the loaded parameters.
        Goal: Initialize DynObjFilter without ROS NodeHandle.

    Replace ROS Callbacks with Direct Calls (Mock Data):
        The original code uses PointsCallback and OdomCallback to receive data and then calls DynObjFilter::filter via a timer.
        In your main.cpp, create a simple loop.
        Inside the loop, create mock input data:
            A pcl::PointCloud<PointType>::Ptr (e.g., feats_undistort). Populate it with a few dummy points.
            An Eigen::Matrix3d rot_end (e.g., identity).
            An Eigen::Vector3d pos_end (e.g., zero).
            A double scan_end_time (e.g., incrementing).
        Directly call dyn_filter->filter(feats_undistort, rot_end, pos_end, scan_end_time);.
        Goal: Run the core filtering logic on demand with controlled inputs, completely independent of ROS topics/callbacks. Test if it runs without crashing.

    Replace ROS Output with Simple Verification:
        The DynObjFilter::filter function populates member point clouds (laserCloudDynObj_world, laserCloudSteadObj). The publish_dyn function (called later) used ROS publishers.
        Remove the publish_dyn call for now, or modify it to not use ROS publishers.
        After calling dyn_filter->filter, access the resulting point clouds (dyn_filter->laserCloudDynObj_world, dyn_filter->laserCloudSteadObj).
        Add simple checks: Print the number of dynamic/static points found. Optionally, save these clouds to PCD files using PCL for visual inspection (pcl::io::savePCDFileASCII).
        Goal: Verify that the filter produces some output based on the mock input, without using ROS publishers.

Phase 2: Integrate nuScenes Data Loading

    nuScenes Data Input Pipeline:
        In your main.cpp, replace the mock data creation.
        Option A (Python Helper): Write a small Python script using nuscenes-devkit to iterate through scenes/sweeps, load LiDAR paths and ego poses, and print them or save them to an intermediate file that C++ can easily read.
        Option B (C++ JSON/File IO): Use a C++ JSON library (nlohmann/json) to parse the relevant nuScenes .json files (sample.json, sample_data.json, ego_pose.json).
            Iterate through scenes and samples (sample.json).
            For each sample, get the LIDAR_TOP data token (sample_data.json).
            Get the corresponding point cloud file path (sample_data.json).
            Get the ego_pose token (sample_data.json).
            Get the ego pose (translation, rotation quaternion, timestamp) from ego_pose.json. Convert quaternion to Eigen::Matrix3d.
            Use PCL's file I/O (pcl::io::loadPCDFile or a custom loader if needed for .pcd.bin) to load the point cloud data into a pcl::PointCloud<PointType>::Ptr. Note: nuScenes point clouds are often x, y, z, intensity, ring_index. Ensure PointType matches or adapt loading.
        Feed the loaded point cloud and ego_pose (as rot_end, pos_end, scan_end_time) into dyn_filter->filter in your loop.
        Goal: Run the standalone filter on actual nuScenes data. Test with a single scene first, then multiple.

    Handling ego_pose:
        The DynObjFilter::filter function signature already takes const M3D & rot_end, const V3D & pos_end, const double & scan_end_time.
        This maps directly to the data you get from ego_pose.json for the corresponding LiDAR sample_data timestamp. No complex conversion is needed here compared to integrating raw IMU data.
        Goal: Confirm that the ego_pose data is correctly extracted and passed to the filter function.

Phase 3: Refinements and Future Work Prep

    Testing and Debugging: Thoroughly test the output against the baseline ROS version if possible, or by visual inspection of saved PCD files. Debug any discrepancies.

    Offline Processing (Past/Future):
        The current DynObjFilter already uses a RingBuffer (buffer) and DepthMap history (depth_map_list). This structure is inherently suitable for offline processing.
        Since you're loading data sequentially offline, you can easily load multiple future and past scans/poses into memory if needed.
        Modifications for past/future tracking would involve:
            Potentially increasing the size/duration managed by RingBuffer and depth_map_list.
            Adjusting the logic within Case1/2/3 or adding new logic to explicitly query future/past depth maps or point states.
        The refactored structure allows these algorithmic changes without fighting ROS infrastructure.

    Re-integrate Clustering (Later): Once DynObjFilter is stable standalone, you can apply the same refactoring steps (remove ROS, replace params, replace callbacks, replace publishers) to DynObjCluster and call it after DynObjFilter::filter, passing the necessary point clouds and tags.
