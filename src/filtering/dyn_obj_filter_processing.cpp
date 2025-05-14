/**
 * @file dyn_obj_filter_processing.cpp
 * @brief Implements the frame and point processing logic for DynObjFilter.
 */
#include "filtering/dyn_obj_filter.h" // Need full class definition

#include <vector>
#include <memory> // For std::make_shared
#include <exception> // For std::exception

// Include necessary utility headers
#include "point_cloud_utils/point_validity.h"
#include "point_cloud_utils/projection_utils.h" // Needed for GetVec in processSinglePoint
#include "point_cloud_utils/logging_context.h"   // <-- ADDED THIS INCLUDE
#include "point_cloud_utils/logging_utils.h"     // <-- ADDED THIS INCLUDE (for checks within this function if needed)

// Logging
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

// --- Process Buffered Frames ---
void DynObjFilter::processBufferedFrames() {
    // Use "Filter_Core" for high-level frame processing status
    auto core_logger = spdlog::get("Filter_Core");
    bool processed_a_frame = false;

    for (size_t buf_idx = 0; buf_idx < scan_history_buffer_.size(); ++buf_idx) {
        const ScanFrame& frame_to_check = scan_history_buffer_.get(buf_idx);

        if (frame_to_check.seq_id == last_processed_seq_id_ + 1) {
            if (core_logger) {
                core_logger->info("Processing frame seq_id: {}", frame_to_check.seq_id);
            }
            processed_a_frame = true;

            if (!frame_to_check.cloud) {
                if (core_logger) core_logger->warn("Frame {} has null cloud pointer. Skipping processing.", frame_to_check.seq_id);
                last_processed_seq_id_ = frame_to_check.seq_id;
                break;
            }

            std::vector<std::shared_ptr<point_soph>> points_for_map;
            std::vector<ProcessedPointInfo> current_frame_info_vec;
            points_for_map.reserve(frame_to_check.cloud->size());
            current_frame_info_vec.reserve(frame_to_check.cloud->size());

            // Loop through raw points, calling helper
            for (size_t i = 0; i < frame_to_check.cloud->size(); ++i) {
                const auto& raw_point = frame_to_check.cloud->points[i];
                // Call processSinglePoint which now handles TLS context internally
                ProcessSinglePointResult result = processSinglePoint(raw_point, i, frame_to_check);

                if (result.processed_successfully) {
                    current_frame_info_vec.push_back(result.info);
                    if (result.point_to_map) {
                        points_for_map.push_back(result.point_to_map);
                    }
                }
            }

            // Use "Filter_Processing" logger for point count summary
            auto proc_logger = spdlog::get("Filter_Processing");
            if (proc_logger) {
                 proc_logger->debug("Processed {} points for frame {}. Found {} valid points for map.",
                             current_frame_info_vec.size(), frame_to_check.seq_id, points_for_map.size());
            }

            // Update Depth Maps
            updateDepthMaps(points_for_map, frame_to_check);

            // Update state
            last_processed_seq_id_ = frame_to_check.seq_id;
            last_processed_frame_info_ = std::move(current_frame_info_vec);

            break; // Processed the sequential frame
        }
    }
     if (!processed_a_frame && !scan_history_buffer_.empty() && core_logger) {
         core_logger->debug("No sequential frame found to process. Last processed: {}, Buffer front: {}",
                      last_processed_seq_id_, scan_history_buffer_.newest().seq_id);
     }
}

// --- Process Single Point ---
DynObjFilter::ProcessSinglePointResult DynObjFilter::processSinglePoint(
    const pcl::PointXYZI& raw_point,
    size_t original_index, // This is the index we need to assign
    const ScanFrame& current_frame)
{
    // Use the "Filter_Processing" logger for detailed point info
    auto logger = spdlog::get("Filter_Processing");
    ProcessSinglePointResult result;
    result.processed_successfully = false;
    // Initialize info struct's index early, though it might be overwritten later
    result.info.original_index = original_index;

    // 1. Basic Validity Check
    V3D local_point_coords(raw_point.x, raw_point.y, raw_point.z);
    if (PointCloudUtils::isPointInvalid(local_point_coords, params_)) {
        // Use helper for conditional logging (though this is usually just trace/debug)
        if (PointCloudUtils::should_log_point_details(logger)) { // Check if needed for this level
             logger->trace("Point {} skipped: Invalid (too close or in invalid box).", original_index);
        }
        // Return result with processed_successfully = false
        return result;
    }

    // 2. Create and Populate point_soph
    std::shared_ptr<point_soph> p_soph = nullptr; // Use shared_ptr
    try {
        p_soph = std::make_shared<point_soph>(); // Create on heap

        // Populate members from raw point and frame data
        p_soph->time = current_frame.timestamp;
        p_soph->intensity = raw_point.intensity;
        p_soph->local = local_point_coords;
        p_soph->glob = current_frame.sensor_pose * p_soph->local; // Transform local to global
        p_soph->GetVec(p_soph->local, params_.hor_resolution_max, params_.ver_resolution_max); // Calculate spherical coords/indices from local
        p_soph->reset(); // Reset counters and caches
        p_soph->dyn = DynObjLabel::STATIC; // Default label
        p_soph->raw_curvature = 0.0f; // Assuming no curvature info for now
        p_soph->is_distort = false; // Assuming no distortion for now

        // *** ADD THE ASSIGNMENT HERE ***
        p_soph->original_index = original_index;

    } catch (const std::exception& e) {
        if (logger) logger->error("Exception during point_soph creation/population for point {}: {}", original_index, e.what());
        // Return result with processed_successfully = false
        return result;
    }

    // Ensure p_soph was successfully created (though make_shared throws on allocation failure)
    if (!p_soph) {
         if (logger) logger->error("Failed to allocate point_soph for point {}.", original_index);
         return result;
    }

    // --- Set Thread-Local Logging Context for this scope ---
    // Place this *after* p_soph is successfully created and populated
    PointCloudUtils::LoggingContextSetter log_context(current_frame.seq_id, *p_soph);
    // --- Context is now set for any function called below ---


    // 3. Self Point Check (using local coordinates)
    if (PointCloudUtils::isSelfPoint(p_soph->local, params_)) {
        p_soph->dyn = DynObjLabel::SELF;
        // Use helper for conditional logging
        if (PointCloudUtils::should_log_point_details(logger)) {
            logger->trace("Point {} labeled SELF.", original_index);
        }
        // Decide if SELF points should be added to the map for visualization/consistency checks
        // result.point_to_map = p_soph; // Example: Exclude SELF from map
    }
    // 4. Dynamic Filtering Checks (if enabled)
    else if (params_.dyn_filter_en) {
        // if (logger) { // "Filter_Processing" logger
        //     logger->info("Point {} (Frame {}): Entering dyn_filter_en block, preparing for checkAppearingPoint.",
        //                  p_soph->original_index, current_frame.seq_id);
        // }
        // 4.a Case 1 Check (Appearing)
        // Note: checkAppearingPoint uses the "Filter_Consistency" logger internally
        // It now has access to p_soph->original_index for conditional logging via TLS
        if (checkAppearingPoint(*p_soph)) { // Pass by reference as before
            p_soph->dyn = DynObjLabel::APPEARING;
            // Use helper for conditional logging
            if (PointCloudUtils::should_log_point_details(logger)) {
                logger->trace("Point {} labeled APPEARING by checkAppearingPoint.", original_index);
            }
            result.point_to_map = p_soph; // Appearing points are usually added to the map
        }
        // 4.b Placeholder for Future Checks (Case 2: Occluding, Case 3: Occluded)
        // else if (checkOccludingPoint(*p_soph)) {
        //     p_soph->dyn = DynObjLabel::OCCLUDING;
        //     if (logger) logger->trace("Point {} labeled OCCLUDING.", original_index);
        //     result.point_to_map = p_soph;
        // }
        // else if (checkOccludedPoint(*p_soph)) {
        //     p_soph->dyn = DynObjLabel::OCCLUDED;
        //     if (logger) logger->trace("Point {} labeled OCCLUDED.", original_index);
        //     // Occluded points might not go to the map, or might with a special status
        // }
        else {
            // If no dynamic check labels it, it remains STATIC
             // Use helper for conditional logging
             if (PointCloudUtils::should_log_point_details(logger)) {
                 logger->trace("Point {} remains STATIC after checks.", original_index);
             }
             result.point_to_map = p_soph; // Static points go to the map
        }
    }
    // 5. Dynamic Filtering Disabled
    else {
         // If filtering is off, treat non-SELF points as STATIC and add to map
         if (p_soph->dyn != DynObjLabel::SELF) {
             p_soph->dyn = DynObjLabel::STATIC; // Ensure label is STATIC if not SELF
             result.point_to_map = p_soph;
             // Use helper for conditional logging
             if (PointCloudUtils::should_log_point_details(logger)) {
                 logger->trace("Point {} labeled STATIC (dyn_filter_en=false).", original_index);
             }
         }
    }

    // 6. Populate ProcessedPointInfo for Python binding
    // Ensure all fields are populated correctly from the final state of p_soph
    result.info.original_index = p_soph->original_index; // Use value from p_soph
    result.info.label = p_soph->dyn;
    result.info.local_x = p_soph->local.x();
    result.info.local_y = p_soph->local.y();
    result.info.local_z = p_soph->local.z();
    result.info.global_x = p_soph->glob.x();
    result.info.global_y = p_soph->glob.y();
    result.info.global_z = p_soph->glob.z();
    result.info.intensity = p_soph->intensity;
    result.info.grid_pos = p_soph->position; // Linearized grid position
    result.info.spherical_azimuth = p_soph->vec(0);
    result.info.spherical_elevation = p_soph->vec(1);
    result.info.spherical_depth = p_soph->vec(2); // Range/Depth

    result.processed_successfully = true;
    return result;
} // --- log_context destructor runs here, automatically restoring previous TLS values ---