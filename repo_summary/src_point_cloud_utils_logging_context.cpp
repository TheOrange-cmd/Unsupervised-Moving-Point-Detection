// src/point_cloud_utils/logging_context.cpp
#include "point_cloud_utils/logging_context.h"
#include "logging/logging_management.h"         // Use CoreLogging namespace
#include "logging/logging_config.h"           // For CoreLoggingConfig::g_log_directory_path
#include "common/dyn_obj_datatypes.h"
#include <spdlog/common.h>

namespace PointCloudUtils {

// --- Thread-Local Storage Variables (Definition) ---
// thread_local uint64_t g_current_logging_seq_id = INVALID_SEQ_ID;
thread_local const point_soph* g_current_logging_point_ptr = nullptr;


// --- RAII Helper Class Implementation ---
LoggingContextSetter::LoggingContextSetter(uint64_t seq_id, const point_soph& p)
    : context_was_set_(false)
{
    previous_seq_id_ = g_current_logging_seq_id;
    previous_point_ptr_ = g_current_logging_point_ptr;

    g_current_logging_seq_id = seq_id;
    g_current_logging_point_ptr = &p;

    context_was_set_ = true;

    if (seq_id != previous_seq_id_ && seq_id != INVALID_SEQ_ID) {
        if (!CoreLoggingConfig::g_log_directory_path.empty()) {
            CoreLogging::switch_file_sink_for_sequence(
                seq_id,
                CoreLoggingConfig::g_log_directory_path, // Get path from global config
                spdlog::level::trace // Or make this configurable
            );
        } else {
            // Log a warning if directory is not set, perhaps only once
            static bool dir_warning_logged = false;
            if (!dir_warning_logged) {
                spdlog::warn("[LoggingContextSetter] Log directory not set. Per-sequence file logging disabled.");
                dir_warning_logged = true;
            }
        }
    }
}

LoggingContextSetter::~LoggingContextSetter()
{
    if (context_was_set_) {
        PointCloudUtils::g_current_logging_seq_id = previous_seq_id_;
        g_current_logging_point_ptr = previous_point_ptr_;
    }
}

} // namespace PointCloudUtils