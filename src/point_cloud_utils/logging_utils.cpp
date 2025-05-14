// src/point_cloud_utils/logging_utils.cpp
#include "point_cloud_utils/logging_utils.h"
#include "point_cloud_utils/logging_context.h" // Include the context header
#include "common/dyn_obj_datatypes.h"          // Include point_soph definition
#include <spdlog/spdlog.h>
#include <cstdint> // Include if not already present
#include "common/types.h"
// Anonymous namespace for constants controlling the logging
namespace {


} // anonymous namespace


namespace PointCloudUtils {

// Implementation reads thread_local variables
bool should_log_point_details(const std::shared_ptr<spdlog::logger>& logger)
{
    if (!logger) return false;

    // If logger's level is TRACE, log everything for now for debugging this issue
    if (logger->should_log(spdlog::level::trace)) {
        return true; // TEMPORARY OVERRIDE
    }

    // Get context from thread-local storage
    const point_soph* p_ptr = g_current_logging_point_ptr;
    uint64_t current_seq_id = g_current_logging_seq_id;

    // Check if context is valid
    if (!p_ptr || current_seq_id == INVALID_SEQ_ID) {
        // Context not set, cannot perform detailed check.
        // Optionally log a warning here ONCE if this happens unexpectedly.
        return false;
    }

    // Apply Debug/Sampling Logic using context from TLS
    bool log_this = false;
    if constexpr (DEBUG_POINT_IDX == -1)
    {
        // Frequency-based logging mode
        if constexpr (LOG_FREQUENCY > 0) {
            log_this = (p_ptr->original_index % LOG_FREQUENCY == 0);
        }
    }
    else
    {
        // Specific point/frame logging mode
        bool point_match = (p_ptr->original_index == static_cast<size_t>(DEBUG_POINT_IDX));
        bool frame_match = true;

        if constexpr (DEBUG_FRAME_SEQ_ID != -1) {
            frame_match = (current_seq_id == static_cast<uint64_t>(DEBUG_FRAME_SEQ_ID));
        }
        log_this = point_match && frame_match;
    }

    return log_this;
}

} // namespace PointCloudUtils