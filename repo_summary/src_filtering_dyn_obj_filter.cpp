/**
 * @file dyn_obj_filter.cpp
 * @brief Implements core parts of the DynObjFilter class: constructor, scan adding, getters.
 */
#include "filtering/dyn_obj_filter.h"

#include <vector>
#include <stdexcept>
#include <iomanip> // For std::fixed, std::setprecision in logging

// Only include headers needed by functions *in this file*
#include "common/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "filtering/ring_buffer.h" // Needed for ScanFrame definition used in addScan

// Logging
#include "common/logging_setup.h" 
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

// --- Constructor ---
DynObjFilter::DynObjFilter(const DynObjFilterParams& config_params) :
    params_(config_params),
    scan_history_buffer_(params_.history_length > 0 ? params_.history_length : 1),
    map_index_(-1),
    next_scan_seq_id_(0),
    last_processed_seq_id_(static_cast<uint64_t>(-1))
{
    // Use the "Filter_Core" logger
    setup_logging(params_); // Setup logging using the loaded parameters
    auto logger = spdlog::get("Filter_Core");
    if (logger) {
        // No need to check for logger existence here anymore if setup_logging worked,
        // but keeping it doesn't hurt.
        logger->info("DynObjFilter constructed. History length: {}", params_.history_length);
        logger->info("  RingBuffer capacity: {}", scan_history_buffer_.capacity());
        logger->info("DynObjFilter initialization complete."); // This log should now appear
    } else {
        // This block should ideally not be reached anymore.
        // Keep it for robustness or change to an error if logging is mandatory.
        std::cerr << "CRITICAL ERROR: Logger 'Filter_Core' not found even after calling setup_logging()." << std::endl;
    }
}

// --- Method to add a new scan ---
void DynObjFilter::addScan(const ScanFrame::PointCloudPtr& cloud, const ScanFrame::PoseType& pose, double timestamp) {
    // Use the "Filter_Core" logger
    auto logger = spdlog::get("Filter_Core");
    

    if (!cloud) {
        if (logger) logger->warn("[addScan] Received null point cloud. Skipping.");
        else std::cerr << "[DynObjFilter::addScan] Warning: Received null point cloud. Skipping." << std::endl;
        return;
    }

    uint64_t current_seq_id = next_scan_seq_id_++;
    ScanFrame current_frame(timestamp, cloud, pose, current_seq_id);

    if (logger) {
        // Log at debug level - info might be too noisy for every scan added
        logger->debug("[addScan] Adding scan Timestamp: {:.6f}, SeqID: {}, Points: {}",
                     timestamp, current_seq_id, cloud->size());
    }

    scan_history_buffer_.add(std::move(current_frame));
    processBufferedFrames(); // Process immediately after adding
}

// --- Getters ---

std::vector<ProcessedPointInfo> DynObjFilter::getProcessedPointsInfo(uint64_t seq_id) const {
    // Use the "Filter_Core" logger for warnings
    auto logger = spdlog::get("Filter_Core");
    if (seq_id == last_processed_seq_id_) {
        // No logging needed for successful get
        return last_processed_frame_info_; // Return a copy
    } else {
        if (logger) logger->warn("[getProcessedPointsInfo] Requested seq_id {} does not match last processed seq_id {}. Returning empty vector.",
                                seq_id, last_processed_seq_id_);
        return {}; // Return empty vector
    }
}

int DynObjFilter::get_map_total_point_count(int map_absolute_index) const {
    // Use the "Filter_Core" logger for warnings
    auto logger = spdlog::get("Filter_Core");
    for (const auto& map_ptr : depth_map_list_) {
        if (map_ptr && map_ptr->map_index == map_absolute_index) { // Add null check for map_ptr
            int total_points = 0;
            // Ensure depth_map itself is valid (though unlikely to be invalid if map_ptr is valid)
            if (!map_ptr->depth_map.empty()) {
                 for (const auto& cell : map_ptr->depth_map) {
                     total_points += cell.size();
                 }
            }
            return total_points;
        }
    }
    if (logger) logger->warn("[get_map_total_point_count] Map with index {} not found.", map_absolute_index);
    return -1; // Not found
}

size_t DynObjFilter::get_depth_map_count() const {
    return depth_map_list_.size();
}

uint64_t DynObjFilter::get_last_processed_seq_id() const {
    return last_processed_seq_id_;
}

size_t DynObjFilter::get_scan_buffer_capacity() const {
    return scan_history_buffer_.capacity();
}

size_t DynObjFilter::get_scan_buffer_size() const {
    return scan_history_buffer_.size();
}

// --- Placeholder Labeling (If kept) ---
std::vector<DynObjLabel> DynObjFilter::placeholder_labeling(
    const ScanFrame::PointCloudPtr& cloud)
{
    // Use the "Filter_Core" logger
    auto logger = spdlog::get("Filter_Core");
    std::vector<DynObjLabel> labels;
    if (!cloud) {
        if (logger) logger->warn("[placeholder_labeling] Received a null point cloud.");
        return labels;
    }
    size_t num_points = cloud->size();
    labels.reserve(num_points);
    if (logger) logger->debug("[placeholder_labeling] Processing {} points...", num_points);
    for (size_t i = 0; i < num_points; ++i) {
        labels.push_back(static_cast<DynObjLabel>(i % 7)); // Cycle through labels
    }
    if (logger) logger->debug("[placeholder_labeling] Finished.");
    return labels;
}