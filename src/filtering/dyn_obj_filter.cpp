// src/filtering/DynObjFilter.cpp (Minimal for now)

#include <iostream> // For stub output
#include <cmath>    // For floor, round etc. in init
#include <vector>
#include <stdexcept> // For potential errors
#include <algorithm>  // For std::min, std::max

#include "filtering/dyn_obj_filter.h"
#include "point_cloud_utils/point_cloud_utils.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h" // Include to get DynObjFilterParams definition


// constructor
DynObjFilter::DynObjFilter(const DynObjFilterParams& config_params) :
    params_(config_params), // Store the loaded parameters
    scan_history_buffer_(params_.history_length > 0 ? params_.history_length : 1), // Ensure capacity > 0
    map_index_(-1),
    next_scan_seq_id_(0),
    last_processed_seq_id_(-1)
{
    // Reserve space for depth maps if desired (optional optimization)
    // depth_map_list_.reserve(params_.max_depth_map_num);
    std::cout << "DynObjFilter constructed. History length: " << params_.history_length << std::endl;
    std::cout << "  RingBuffer capacity: " << scan_history_buffer_.capacity() << std::endl;
    std::cout << "DynObjFilter initialization complete." << std::endl;
}

// // Method to add a new scan
// void DynObjFilter::addScan(const ScanFrame::PointCloudPtr& cloud, const ScanFrame::PoseType& pose, double timestamp) {
//     if (!cloud) {
//         std::cerr << "[DynObjFilter::addScan] Warning: Received null point cloud. Skipping." << std::endl;
//         return;
//     }

//     // Create a ScanFrame with the next sequence ID
//     uint64_t current_seq_id = next_scan_seq_id_++;
//     ScanFrame current_frame(timestamp, cloud, pose, current_seq_id);

//     std::cout << "[DynObjFilter::addScan] Adding scan Timestamp: " << timestamp
//               << ", SeqID: " << current_seq_id
//               << ", Points: " << (cloud ? cloud->size() : 0) << std::endl;

//     // Add the frame to the ring buffer (moves the frame)
//     scan_history_buffer_.add(std::move(current_frame));

//     // Trigger processing of any newly added frames in the buffer
//     processBufferedFrames();
// }

// // Skeleton for processing buffered frames
// void DynObjFilter::processBufferedFrames() {
//     std::cout << "[DynObjFilter::processBufferedFrames] Checking buffer. Size=" << scan_history_buffer_.size()
//               << ", LastProcessedID=" << last_processed_seq_id_ << std::endl;

//     bool processed_something = false;
//     // Iterate through the buffer from oldest to newest
//     for (size_t i = 0; i < scan_history_buffer_.size(); ++i) {
//         try {
//             // Get a const reference to avoid accidental modification here
//             const ScanFrame& frame = scan_history_buffer_.get(i); // Get by relative age (0=oldest)

//             // Check if this frame's sequence ID is newer than the last one processed
//             if (frame.seq_id > last_processed_seq_id_) {
//                 std::cout << "  - Processing Frame SeqID: " << frame.seq_id
//                           << ", Timestamp: " << frame.timestamp << std::endl;

//                 // --- Skeleton for future steps ---
//                 // 1. Point Processing Loop (Parallel):
//                 //    - Create point_soph objects for points in 'frame.cloud'
//                 std::vector<std::shared_ptr<point_soph>> processed_points; // Placeholder
//                 processed_points.reserve(frame.cloud ? frame.cloud->size() : 0);
//                 //    - Perform validity, self, and dynamic checks (checkAppearingPoint etc.)
//                 //    - Populate 'processed_points' vector with shared_ptrs to processed point_soph
//                 //      (For skeleton, maybe just create dummy point_soph objects)
//                 if (frame.cloud) { // Simulate processing
//                     for(size_t pt_idx = 0; pt_idx < frame.cloud->size(); ++pt_idx) {
//                          // Dummy processing: create basic point_soph and add shared_ptr
//                          // In reality, this involves transforms and checks
//                          auto dummy_soph = std::make_shared<point_soph>();
//                          dummy_soph->time = frame.timestamp;
//                          // dummy_soph->glob = ... calculate ...
//                          dummy_soph->dyn = STATIC; // Default for skeleton
//                          if (pt_idx % 100 == 0) { // Simulate some dynamic points for testing map updates
//                              if (pt_idx % 300 == 0) dummy_soph->dyn = CASE1;
//                              else if (pt_idx % 300 == 100) dummy_soph->dyn = CASE2;
//                              else dummy_soph->dyn = CASE3;
//                          }
//                          processed_points.push_back(dummy_soph);
//                     }
//                 }

//                 // 2. Update Depth Maps with results
//                 updateDepthMaps(processed_points, frame);

//                 // 3. Update the ID of the last processed frame
//                 last_processed_seq_id_ = frame.seq_id;
//                 processed_something = true;

//             } else {
//                  // This frame (and older ones) already processed
//                  // std::cout << "  - Skipping Frame SeqID: " << frame.seq_id << " (already processed)" << std::endl;
//             }

//         } catch (const std::out_of_range& e) {
//             // Should not happen if loop bounds are correct, but good practice
//             std::cerr << "[DynObjFilter::processBufferedFrames] Error accessing ring buffer: " << e.what() << std::endl;
//             break; // Stop processing if buffer access fails
//         }
//     }
//      if (!processed_something) {
//          std::cout << "[DynObjFilter::processBufferedFrames] No new frames to process." << std::endl;
//      }
// }

// // Skeleton for updating depth maps
// void DynObjFilter::updateDepthMaps(const std::vector<std::shared_ptr<point_soph>>& processed_points, const ScanFrame& source_frame) {
//     std::cout << "[DynObjFilter::updateDepthMaps] Updating maps for Frame SeqID: " << source_frame.seq_id
//               << ", Timestamp: " << source_frame.timestamp << std::endl;

//     // --- 1. Manage Depth Map List (Rotation/Creation) ---
//     bool map_list_updated = false;
//     if (depth_map_list_.empty() || (source_frame.timestamp - depth_map_list_.back()->time) >= params_.depth_map_dur - params_.frame_dur / 2.0) {
//         map_index_++; // Increment global map index
//         std::cout << "  - Creating/Rotating DepthMap. New map_index: " << map_index_ << std::endl;

//         if (depth_map_list_.size() == params_.max_depth_map_num) {
//             std::cout << "    - Rotating map list (Max size reached)." << std::endl;
//             // Get pointer to the oldest map (front)
//             DepthMap::Ptr oldest_map = depth_map_list_.front();
//             // Reset it with the new frame's info
//             oldest_map->Reset(source_frame.sensor_pose.rotation().transpose(), // Pass rotation (world to sensor)
//                               source_frame.sensor_pose.translation(),         // Pass translation (world to sensor)
//                               source_frame.timestamp,
//                               map_index_);
//             // Move it to the back
//             depth_map_list_.pop_front();
//             depth_map_list_.push_back(oldest_map); // Add reused map to the back
//         } else {
//             std::cout << "    - Adding new map to list." << std::endl;
//             // Create a new map
//             depth_map_list_.push_back(std::make_shared<DepthMap>(
//                 source_frame.sensor_pose.rotation().transpose(), // Pass rotation (world to sensor)
//                 source_frame.sensor_pose.translation(),         // Pass translation (world to sensor)
//                 source_frame.timestamp,
//                 map_index_
//             ));
//         }
//         map_list_updated = true;
//     } else {
//          std::cout << "  - Adding points to existing latest DepthMap (Index: " << depth_map_list_.back()->map_index << ")" << std::endl;
//     }

//     // --- 2. Add Processed Points to the Latest Map ---
//     if (depth_map_list_.empty()) {
//         std::cerr << "  - Warning: Depth map list is empty, cannot add points." << std::endl;
//         return;
//     }

//     auto& latest_map = depth_map_list_.back();

//     std::cout << "  - Adding " << processed_points.size() << " processed points to map index " << latest_map->map_index << "." << std::endl;

//     // (Deferred Implementation for Step 2)
//     // Loop through 'processed_points':
//     //   - Project point_soph into 'latest_map' frame using SphericalProjection (use cache)
//     //   - Check validity of projection (indices, position)
//     //   - If valid and cell not full:
//     //     - latest_map->depth_map[pos].push_back(point_ptr); // Store the shared_ptr
//     //     - Update latest_map->min_depth_all, max_depth_all, etc. based on projected depth and dyn status

//     if (map_list_updated) {
//         std::cout << "[DynObjFilter::updateDepthMaps] Depth map list size: " << depth_map_list_.size() << std::endl;
//     }
// }

// Implementation of the placeholder labeling method
// all points are labeled to one of the dynamic classes based on their index % 7 (for the 7 classes). 
// This method is only used to verify the data pipeline from numpy - pcl - DynObjFilter - pcl - numpy
// without depending on having an actual working classifier. 
std::vector<DynObjLabel> DynObjFilter::placeholder_labeling(
    const ScanFrame::PointCloudPtr& cloud)
{
    std::vector<DynObjLabel> labels;
    if (!cloud) {
        std::cerr << "Warning: placeholder_labeling received a null point cloud." << std::endl;
        return labels; // Return empty vector
    }

    size_t num_points = cloud->size();
    labels.reserve(num_points); // Pre-allocate space

    std::cout << "  DynObjFilter::placeholder_labeling processing " << num_points << " points..." << std::endl;

    for (size_t i = 0; i < num_points; ++i) {
        // Assign label based on index modulo 7
        DynObjLabel label = static_cast<DynObjLabel>(i % 7);
        labels.push_back(label);
    }

    std::cout << "  DynObjFilter::placeholder_labeling finished." << std::endl;
    return labels;
}

// Implement the real process_scan method later...
