// filtering/dyn_obj_filter.h

/**
 * @file dyn_obj_filter.h
 * @brief Defines the DynObjFilter class for detecting dynamic points in lidar scans.
*/

#ifndef DYN_OBJ_FILTER_H
#define DYN_OBJ_FILTER_H

#include <string>
#include <vector>
#include <deque>
#include <memory> // For std::shared_ptr
#include <utility> // For std::pair
#include <cstddef> // For size_t
#include <cstdint> // For uint64_t

#include "config/config_loader.h"        // includes DynObjFilterParams
#include "common/dyn_obj_datatypes.h" // Includes DepthMap, point_soph etc.
#include "filtering/ring_buffer.h"       // Includes ScanFrame definition
#include "common/types.h"                // Includes M3D, V3D, PointCloudXYZI (via PCL includes)

// Forward declare PCL type if not fully included via common/types.h
namespace pcl { template<typename T> class PointCloud; struct PointXYZI; }


class DynObjFilter {
    public:
        /**
         * @brief Constructor: Initializes the filter with loaded configuration parameters.
         * @param config_params The configuration parameters loaded from YAML.
         */
        explicit DynObjFilter(const DynObjFilterParams& config_params);

        /**
         * @brief Adds a new lidar scan (frame) to the internal buffer for processing.
         *        Triggers the internal processing of buffered frames if the sequence is contiguous.
         * @param cloud Shared pointer to the point cloud data (pcl::PointCloud<pcl::PointXYZI>).
         * @param pose The sensor pose (Eigen::Isometry3d) in the world frame corresponding to the scan's timestamp.
         * @param timestamp The timestamp (e.g., seconds since epoch) of the scan.
         */
        void addScan(const ScanFrame::PointCloudPtr& cloud, const ScanFrame::PoseType& pose, double timestamp);

        // --- Getters ---

        /**
         * @brief Gets detailed information about points processed for a specific frame sequence ID.
         * @param seq_id The sequence ID of the frame whose processed points are requested.
         * @return A vector of ProcessedPointInfo structs for the requested frame.
         *         Returns an empty vector if the seq_id doesn't match the last processed frame.
         */
        std::vector<ProcessedPointInfo> getProcessedPointsInfo(uint64_t seq_id) const;


    
    /**
     * @brief Gets the sequence ID that will be assigned to the *next* scan added.
     * Useful for logging in wrappers before the scan is fully processed.
     * @return The next sequence ID.
     */
    uint64_t get_next_scan_seq_id() const { return next_scan_seq_id_; }
        
    /**
     * @brief Returns the current number of DepthMap objects stored.
     * @return The number of depth maps in the internal list.
     */
        size_t get_depth_map_count() const;

        /**
     * @brief Returns the sequence ID of the last scan frame successfully processed.
     * @return The sequence ID, or the initial value (e.g., uint64_t(-1)) if none processed yet.
     */
        uint64_t get_last_processed_seq_id() const;

        /**
     * @brief Returns the capacity of the internal scan history buffer.
     * @return The capacity of the scan frame ring buffer.
     */
        size_t get_scan_buffer_capacity() const;

        /**
     * @brief Returns the current number of scans stored in the internal history buffer.
     * @return The current size of the scan frame ring buffer.
     */
        size_t get_scan_buffer_size() const;

        /**
     * @brief Returns the total number of points stored across all cells in a specific historical depth map.
     * @param map_absolute_index The absolute index (`map_index`) of the desired depth map.
     * @return The total point count in the map, or -1 if the map index is not found.
     */
        int get_map_total_point_count(int map_absolute_index) const;

        // --- Placeholder ---
        // Kept for compatibility during transition, potentially removable later.
        std::vector<DynObjLabel> placeholder_labeling(
            const ScanFrame::PointCloudPtr& cloud
        );


    private:
        // --- Helper Struct for processSinglePoint ---
        /** @brief Holds the results of processing a single raw point. */
        struct ProcessSinglePointResult {
            std::shared_ptr<point_soph> point_to_map = nullptr; /**< Pointer to the processed point if it should be added to the map, nullptr otherwise. */
            ProcessedPointInfo info;                            /**< Information about the processed point for external use. */
            bool processed_successfully = false;                /**< Flag indicating if basic processing (creation, validity) succeeded. */
        };


        // --- Core Processing Logic ---

        /**
         * @brief Processes frames stored in the scan_history_buffer_ that are sequential to the last processed frame.
         * Iterates through the buffer, finds the next frame (seq_id == last_processed_seq_id_ + 1),
         * processes each point in that frame using `processSinglePoint`, collects the results,
         * and updates the depth maps using `updateDepthMaps`.
         */
        void processBufferedFrames();

        /**
         * @brief Processes a single raw point from an input scan frame.
         * Performs validity checks, creates a `point_soph` object, populates its fields,
         * performs self-check, calls consistency checks (currently `checkAppearingPoint`),
         * assigns a label (`DynObjLabel`), and populates the `ProcessedPointInfo` struct.
         * @param raw_point The raw input point (pcl::PointXYZI).
         * @param original_index The index of the raw point in the original scan cloud.
         * @param current_frame The ScanFrame object containing the raw point, pose, and timestamp.
         * @return ProcessSinglePointResult Containing the processed point (if valid for map), info, and success status.
         */
        ProcessSinglePointResult processSinglePoint(
            const pcl::PointXYZI& raw_point,
            size_t original_index,
            const ScanFrame& current_frame);

        /**
         * @brief Updates the historical depth maps (`depth_map_list_`) based on the current frame's time.
         * Handles map creation, rotation based on `params_.depth_map_dur` and `params_.max_depth_map_num`.
         * Calls `addPointsToMap` to insert the processed points into the appropriate target map.
         * @param points_to_add Vector of shared pointers to the `point_soph` objects processed from the current frame that are valid for map insertion.
         * @param current_frame_info The ScanFrame corresponding to the `points_to_add` (needed for pose/time).
         */
        void updateDepthMaps(
            const std::vector<std::shared_ptr<point_soph>>& points_to_add,
            const ScanFrame& current_frame_info);

        /**
         * @brief Adds a collection of processed points to a specific depth map grid and updates statistics.
         * Projects each point relative to the map's frame to get the correct grid index (`position`).
         * Adds the point pointer to the cell if space allows (`params_.max_pixel_points`).
         * Updates the map's statistics (`min/max_depth_all`, `min/max_depth_static`).
         * @param map The DepthMap object (passed by reference) to add points to.
         * @param points_to_add Vector of shared pointers to the `point_soph` objects to add.
         */
        void addPointsToMap(
            DepthMap& map,
            const std::vector<std::shared_ptr<point_soph>>& points_to_add);


        // --- Consistency Checks ---

        /**
         * @brief Checks if a point qualifies as APPEARING (Case 1).
         * Iterates through historical maps. For each map, projects the point `p` into that map's frame.
         * Checks for inconsistency based on projection validity, self-occlusion, and comparison
         * with interpolated static surfaces in the historical map.
         * @param p The point_soph object (passed by non-const ref as projection may update its cache).
         * @return True if the point is deemed APPEARING based on `params_.occluded_map_thr1`, false otherwise.
         */
        bool checkAppearingPoint(point_soph& p);

        // Placeholder for future consistency checks
        // bool checkOccludingPoint(point_soph& p);
        // bool checkDisoccludedPoint(point_soph& p);


        // --- Helper Functions ---

        /**
         * @brief Prepares a point for consistency checking against a specific historical map.
         * Projects the world point (`p_world`) into the map's spherical coordinate system
         * and calculates its local coordinates relative to the map's pose. Populates the
         * output `p_map_frame` with these map-relative values.
         * @param p_world The original point_soph object with valid global coordinates (`glob`).
         * @param map The target historical DepthMap.
         * @param[out] p_map_frame The point_soph object to be populated with coordinates and indices relative to `map`.
         */
        void setupPointForMapCheck(
            const point_soph& p_world,
            const DepthMap& map,
            point_soph& p_map_frame
        );


        // --- Member Variables ---
        const DynObjFilterParams params_;             // Holds configuration settings.
        RingBuffer<ScanFrame> scan_history_buffer_;   // Buffers incoming raw scans.
        std::deque<DepthMap::Ptr> depth_map_list_;    // Stores historical processed depth maps.
        int map_index_;                               // Monotonically increasing index for created depth maps.

        // State tracking for processing the ring buffer
        uint64_t next_scan_seq_id_;                   // Sequence ID for the *next* incoming scan.
        uint64_t last_processed_seq_id_;              // Sequence ID of the *last* scan successfully processed.

        // Storage for last processed frame details (used by getProcessedPointsInfo)
        std::vector<ProcessedPointInfo> last_processed_frame_info_;

    };

#endif // DYN_OBJ_FILTER_H