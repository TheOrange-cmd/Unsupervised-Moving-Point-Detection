#ifndef CORE_LOGGING_MANAGEMENT_H
#define CORE_LOGGING_MANAGEMENT_H

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/common.h> // For spdlog::level::level_enum
#include <string>
#include <vector>
#include <memory> // For std::shared_ptr

namespace CoreLogging { // Changed namespace

    // Call this once during setup_logging to tell the management system which loggers should use rotated files.
    void initialize_managed_logger_names(const std::vector<std::string>& names);

    // Call this when the sequence ID changes.
    // log_directory_path: Absolute path to the directory where sequence logs should be stored.
    // file_log_level: Minimum level for messages to be written to this sequence's file.
    void switch_file_sink_for_sequence(
        uint64_t seq_id,
        const std::string& log_directory_path, // Added parameter for directory
        spdlog::level::level_enum file_log_level
    );

    // Call this at the end of the program to ensure logs are flushed.
    void flush_and_close_current_file_sink();

} // namespace CoreLogging

#endif // CORE_LOGGING_MANAGEMENT_H