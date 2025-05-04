// filtering/dyn_obj_filter.h (Minimal for now)

#ifndef DYN_OBJ_FILTER_H
#define DYN_OBJ_FILTER_H

#include <string>
#include <vector>
#include <deque>
#include <memory> // For std::shared_ptr
#include <utility> // For std::pair
#include <cstddef> // For size_t

#include "config/config_loader.h"        // includes DynObjFilterParams
#include "filtering/dyn_obj_datatypes.h" // Includes DepthMap, point_soph etc.
#include "filtering/ring_buffer.h"       // Includes ScanFrame definition 
#include "common/types.h"                // Includes M3D, V3D, PointCloudXYZI (via PCL includes)

// Forward declare RingBuffer if it's in its own header
// template <typename T> class RingBuffer;

class DynObjFilter {
    public:
        /**
         * @brief Constructor: Initializes the filter with loaded configuration parameters.
         * @param config_params The configuration parameters loaded from YAML.
         */
        explicit DynObjFilter(const DynObjFilterParams& config_params);
    
        /**
         * @brief Adds a new lidar scan (frame) to the internal buffer for processing.
         *        Triggers the internal processing of buffered frames.
         * @param cloud Shared pointer to the point cloud data (pcl::PointCloud<pcl::PointXYZI>).
         * @param pose The sensor pose (Eigen::Isometry3d) in the world frame corresponding to the scan's timestamp.
         * @param timestamp The timestamp (e.g., seconds since epoch) of the scan.
         */
        void addScan(const ScanFrame::PointCloudPtr& cloud, const ScanFrame::PoseType& pose, double timestamp);
    
        // Placeholder labeling method - kept for compatibility during transition
        std::vector<DynObjLabel> placeholder_labeling(
            const ScanFrame::PointCloudPtr& cloud
        );

        /**
         * @brief Checks if a point qualifies as APPEARING (Case 1).
         * @param p The point_soph object (potentially modified with cached projections).
         * @return True if the point is appearing, false otherwise.
         */
        bool checkAppearingPoint(point_soph& p);
    
        // getters below this point

        /**
         * @brief Gets detailed information about points processed for a specific frame sequence ID.
         * @param seq_id The sequence ID of the frame whose processed points are requested.
         * @return A vector of ProcessedPointInfo structs for the requested frame.
         *         Returns an empty vector if the seq_id doesn't match the last processed frame.
         */
        std::vector<ProcessedPointInfo> getProcessedPointsInfo(uint64_t seq_id) const;

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
        * @brief Returns the current number of points stored in frame with the requested index the internal history buffer.
        * @return The current size of the requested frame inside the buffer.
        */
        int get_map_total_point_count(int map_absolute_index) const;
    
    
    private:
        /**
         * @brief Processes frames stored in the scan_history_buffer_ that haven't been processed yet.
         *        This is where point classification (Case 1/2/3 logic) will occur.
         */
        void processBufferedFrames();
    
        /**
         * @brief Updates the historical depth maps (depth_map_list_) with processed points.
         *        Handles map creation, rotation, and adding points to the grid.
         * @param processed_points Vector of shared pointers to the point_soph objects processed from a single frame.
         * @param source_frame The ScanFrame from which the processed_points originated (needed for pose/time).
         */
        void updateDepthMaps(const std::vector<std::shared_ptr<point_soph>>& processed_points, const ScanFrame& source_frame);
    
        // --- Member Variables ---
        const DynObjFilterParams params_;             // Holds configuration settings.
        RingBuffer<ScanFrame> scan_history_buffer_;   // Buffers incoming raw scans.
        std::deque<DepthMap::Ptr> depth_map_list_;    // Stores historical processed depth maps.
        int map_index_;                               // Index counter for depth maps.
    
        // State tracking for processing the ring buffer
        size_t processed_scan_count_ = 0; // Tracks how many scans *from the start* have been processed
                                          // Needs careful handling if buffer wraps around.
                                          // Using absolute count assumes buffer doesn't wrap *unnoticed*.
                                          // A sequence ID in ScanFrame might be more robust.
        uint64_t next_scan_seq_id_ = 0;   // Sequence ID for incoming scans
        uint64_t last_processed_seq_id_ = -1; // Sequence ID of the last scan processed

        // --- Storage for last processed frame details ---
        std::vector<ProcessedPointInfo> last_processed_frame_info_;
        // uint64_t last_processed_frame_seq_id_ = static_cast<uint64_t>(-1); // Initialize to sentinel
    
    };
    
    #endif // DYN_OBJ_FILTER_H
    