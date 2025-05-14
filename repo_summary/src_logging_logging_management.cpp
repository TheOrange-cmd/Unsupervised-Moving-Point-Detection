#include "logging/logging_management.h" // Adjusted include path if necessary
#include <spdlog/spdlog.h>
#include <algorithm> // For std::remove_if
#include <filesystem> // For creating directories (C++17)

// If not C++17 or filesystem is not available, you might need OS-specific directory creation
// or ensure the directory exists beforehand.
// For example, on POSIX: #include <sys/stat.h> #include <sys/types.h>

namespace CoreLogging { // Changed namespace

    static std::shared_ptr<spdlog::sinks::basic_file_sink_mt> g_current_sequence_file_sink = nullptr;
    static std::string g_current_sequence_file_path = ""; // Store full path now
    static std::vector<std::string> g_managed_logger_names;

    void initialize_managed_logger_names(const std::vector<std::string>& names) {
        g_managed_logger_names = names;
    }

    void switch_file_sink_for_sequence(
        uint64_t seq_id,
        const std::string& log_directory_path,
        spdlog::level::level_enum file_log_level)
    {
        if (log_directory_path.empty()) {
            spdlog::error("[LoggingManagement] Log directory path is empty. Cannot create sequence log file.");
            return;
        }

        std::filesystem::path dir_path(log_directory_path);
        std::string filename_only = "run_log_seq_" + std::to_string(seq_id) + ".txt";
        std::filesystem::path full_new_path = dir_path / filename_only;
        std::string new_file_path_str = full_new_path.string();


        if (new_file_path_str == g_current_sequence_file_path && g_current_sequence_file_sink != nullptr) {
            return; // Idempotent if path and sink are already set
        }

        // Ensure the directory exists
        try {
            if (!std::filesystem::exists(dir_path)) {
                if (std::filesystem::create_directories(dir_path)) {
                    spdlog::info("[LoggingManagement] Created log directory: {}", dir_path.string());
                } else {
                    spdlog::error("[LoggingManagement] Failed to create log directory: {}. Check permissions.", dir_path.string());
                    return; // Cannot proceed without directory
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            spdlog::error("[LoggingManagement] Filesystem error while checking/creating directory {}: {}", dir_path.string(), e.what());
            return;
        }


        if (g_current_sequence_file_sink) {
            g_current_sequence_file_sink->flush();
            for (const auto& logger_name : g_managed_logger_names) {
                auto logger = spdlog::get(logger_name);
                if (logger) {
                    auto& sinks_vec = logger->sinks();
                    sinks_vec.erase(std::remove_if(sinks_vec.begin(), sinks_vec.end(),
                                                   [&](const spdlog::sink_ptr& s_ptr) {
                                                       return s_ptr == g_current_sequence_file_sink;
                                                   }),
                                    sinks_vec.end());
                }
            }
            g_current_sequence_file_sink.reset();
        }

        try {
            g_current_sequence_file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(new_file_path_str, true); // true to truncate
            g_current_sequence_file_sink->set_level(file_log_level);
            // Optional: Set a pattern for the file sink
            // g_current_sequence_file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] [tid %t] %v");

            g_current_sequence_file_path = new_file_path_str;

            for (const auto& logger_name : g_managed_logger_names) {
                auto logger = spdlog::get(logger_name);
                if (logger) {
                    logger->sinks().push_back(g_current_sequence_file_sink);
                    // Ensure logger's level is appropriate (as discussed before)
                }
            }
            spdlog::info("[LoggingManagement] File logging switched to: '{}'. Min level for this file: {}", new_file_path_str, spdlog::level::to_string_view(file_log_level));

        } catch (const spdlog::spdlog_ex& ex) {
            spdlog::error("[LoggingManagement] Failed to create or switch file sink to '{}': {}", new_file_path_str, ex.what());
            g_current_sequence_file_sink = nullptr;
            g_current_sequence_file_path = "";
        }
    }

    void flush_and_close_current_file_sink() {
        if (g_current_sequence_file_sink) {
            g_current_sequence_file_sink->flush();
            for (const auto& logger_name : g_managed_logger_names) {
                auto logger = spdlog::get(logger_name);
                if (logger) {
                     auto& sinks_vec = logger->sinks();
                    sinks_vec.erase(std::remove_if(sinks_vec.begin(), sinks_vec.end(),
                                                   [&](const spdlog::sink_ptr& s_ptr) {
                                                       return s_ptr == g_current_sequence_file_sink;
                                                   }),
                                    sinks_vec.end());
                }
            }
            g_current_sequence_file_sink.reset();
            spdlog::info("[LoggingManagement] Current sequence file sink ('{}') flushed and closed.", g_current_sequence_file_path);
            g_current_sequence_file_path = "";
        }
    }

} // namespace CoreLogging