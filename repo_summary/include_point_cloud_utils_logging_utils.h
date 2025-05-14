// include/point_cloud_utils/logging_utils.h
#ifndef POINT_CLOUD_UTILS_LOGGING_UTILS_H
#define POINT_CLOUD_UTILS_LOGGING_UTILS_H

#include <spdlog/spdlog.h>
#include <memory> // For std::shared_ptr

namespace PointCloudUtils {

/**
 * @brief Determines if detailed trace logging should be performed based on thread-local context.
 *
 * Reads thread-local variables (g_current_logging_point_ptr, g_current_logging_seq_id)
 * set by LoggingContextSetter and applies debug flags or frequency logic.
 *
 * @param logger The logger instance being used (used to check for existence and level).
 * @return True if detailed logging should occur, False otherwise.
 */
bool should_log_point_details(
    const std::shared_ptr<spdlog::logger>& logger
);

} // namespace PointCloudUtils

#endif // POINT_CLOUD_UTILS_LOGGING_UTILS_H