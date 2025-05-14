// src/point_cloud_utils/logging_context.cpp
#include "point_cloud_utils/logging_context.h"
#include "logging/logging_management.h"     // For CoreLogging::switch_file_sink_for_sequence
#include "common/logging_setup.h"           // For CoreLoggingConfig (needs to provide the globals)
#include "common/dyn_obj_datatypes.h"       // For point_soph
#include <spdlog/spdlog.h>                  // For spdlog::warn

// Note: Ensure "common/logging_setup.h" correctly declares the CoreLoggingConfig globals
// or include a specific "logging_config.h" if you created one for CoreLoggingConfig.

namespace PointCloudUtils {

// --- Thread-Local Storage Variables (Definition) ---
thread_local uint64_t g_current_logging_seq_id = INVALID_SEQ_ID;
thread_local const point_soph* g_current_logging_point_ptr = nullptr;


// --- RAII Helper Class Implementation ---
LoggingContextSetter::LoggingContextSetter(uint64_t seq_id, const point_soph& p)
    : previous_seq_id_(g_current_logging_seq_id), // Initialize members
      previous_point_ptr_(g_current_logging_point_ptr),
      context_was_set_(false)
{
    // Set the new context for the current thread
    g_current_logging_seq_id = seq_id;
    g_current_logging_point_ptr = &p;
    context_was_set_ = true; // Mark that context was set by this instance

    // Check if the sequence ID has changed and is valid
    if (seq_id != previous_seq_id_ && seq_id != INVALID_SEQ_ID) {
        // Check if the log directory path has been configured
        if (!CoreLoggingConfig::g_log_directory_path.empty()) {
            // Switch the file sink using the globally configured directory and file log level
            CoreLogging::switch_file_sink_for_sequence(
                seq_id,
                CoreLoggingConfig::g_log_directory_path,             // Get path from global config
                CoreLoggingConfig::g_default_configured_file_log_level // USE THE CONFIGURED LEVEL
            );
        } else {
            // Log a warning if the directory is not set (e.g., if params.log_directory was empty)
            // This warning will only be logged once due to the static flag.
            static bool dir_warning_logged = false;
            if (!dir_warning_logged) {
                // Use spdlog::warn or a default logger if available and setup
                // If logging isn't fully up, this might not appear, or use std::cerr
                spdlog::warn("[LoggingContextSetter] Log directory (CoreLoggingConfig::g_log_directory_path) is not set. "
                             "Per-sequence file logging will be disabled for sequence {}.", seq_id);
                dir_warning_logged = true;
            }
        }
    }
}

LoggingContextSetter::~LoggingContextSetter()
{
    // Restore the previous logging context if this instance set it
    if (context_was_set_) {
        g_current_logging_seq_id = previous_seq_id_;
        g_current_logging_point_ptr = previous_point_ptr_;
    }
}

} // namespace PointCloudUtils