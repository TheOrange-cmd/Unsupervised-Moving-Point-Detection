// src/common/logging_setup.cpp
#include "common/logging_setup.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
// "config/config_loader.h" is included via "common/logging_setup.h" for DynObjFilterParams
#include "logging/logging_management.h" // For CoreLogging::initialize_managed_logger_names etc.
#include <vector>
#include <string>
#include <memory>
#include <iostream> // For std::cerr

// Define the global variables declared in the header
namespace CoreLoggingConfig {
    std::string g_log_directory_path = ""; // Default to empty, will be set from params
    spdlog::level::level_enum g_default_configured_file_log_level = spdlog::level::info; // Default, will be set from params
}

// Definition of setup_logging
void setup_logging(const DynObjFilterParams& params) {
    // 1. Set global logging configuration from params
    CoreLoggingConfig::g_log_directory_path = params.log_directory;
    CoreLoggingConfig::g_default_configured_file_log_level = params.default_file_log_level; // This is the parsed enum

    try {
        // 2. Determine base level for console and default for loggers
        // params.log_level is the string (e.g., "info", "debug")
        spdlog::level::level_enum base_console_level = spdlog::level::info; // Fallback
        try {
            base_console_level = spdlog::level::from_str(params.log_level);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[Logging Setup] Invalid base log_level string: '" << params.log_level
                      << "'. Using 'info'. Error: " << e.what() << std::endl;
            // base_console_level remains 'info'
        }

        // 3. Create and configure the console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(base_console_level); // Use the configured base level for the console sink

        // 4. Initialize managed logger names for per-sequence file switching
        std::vector<std::string> managed_logger_names = {
            "Utils_Grid", "Utils_Interpolation", "Utils_Projection", "Utils_Validity",
            "Filter_Core", "Filter_Processing", "Filter_Map", "Filter_Consistency", "Consistency", "Bindings"
            // Add other loggers that should have per-sequence file output
        };
        CoreLogging::initialize_managed_logger_names(managed_logger_names);

        // 5. Set spdlog's global level (acts as a cap if stricter than logger's level)
        // Usually, you want individual loggers to control their output, and sinks to filter.
        // Setting a global level can be useful but sometimes confusing.
        // If base_console_level is 'trace', all loggers can potentially log trace.
        // If base_console_level is 'info', even if a logger is set to 'trace', global level might restrict it.
        // For flexibility, let's ensure it's at least as verbose as the most verbose logger could be,
        // or simply set it to trace to allow all levels, and let loggers/sinks control.
        // Or, more commonly, set it to the base_console_level.
        spdlog::set_level(base_console_level); // Global level filter

        // 6. Create/configure each logger
        for (const auto& name : managed_logger_names) {
            auto logger = spdlog::get(name);
            if (logger == nullptr) {
                logger = std::make_shared<spdlog::logger>(name); // Create logger
                spdlog::register_logger(logger);
            }

            // Clear existing sinks (if any, e.g. from previous setup) and add the console sink
            logger->sinks().clear();
            logger->sinks().push_back(console_sink);

            // Determine the logger's specific level
            spdlog::level::level_enum final_logger_level = base_console_level; // Default to base console level
            std::string level_source_msg = "base_console_level";

            auto it = params.log_levels.find(name); // Check for override in params.log_levels map
            if (it != params.log_levels.end()) {
                try {
                    final_logger_level = spdlog::level::from_str(it->second);
                    level_source_msg = "config_override";
                } catch (const std::invalid_argument& e) {
                    spdlog::error("[Logging Setup] Invalid log level string '{}' for logger '{}' in 'log_levels' map. Using base_console_level ('{}'). Error: {}",
                                  it->second, name, spdlog::level::to_string_view(base_console_level), e.what());
                    // final_logger_level remains base_console_level
                }
            }
            logger->set_level(final_logger_level); // Set the logger's own level

            // Informative log about this specific logger's setup (using default logger or a setup logger)
            // spdlog::info("[Logging Setup] Logger '{}' level set to '{}' (from {}). Console sink active.",
            //              name, spdlog::level::to_string_view(final_logger_level), level_source_msg);
        }

        // 7. Set global flush policy
        spdlog::flush_on(spdlog::level::warn); // Flush logs automatically for warnings and above

        // 8. Final informational message
        spdlog::info("[Logging Setup] Logging initialized. Console base level: '{}'. Default per-sequence file level: '{}'. Log directory: '{}'",
                     spdlog::level::to_string_view(base_console_level),
                     spdlog::level::to_string_view(CoreLoggingConfig::g_default_configured_file_log_level),
                     CoreLoggingConfig::g_log_directory_path.empty() ? "[Not Set/Empty]" : CoreLoggingConfig::g_log_directory_path);

    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "[Logging Setup] Log initialization failed (spdlog exception): " << ex.what() << std::endl;
    } catch (const std::invalid_argument& e) { // Catch from spdlog::level::from_str if params.log_level was bad
         std::cerr << "[Logging Setup] Log initialization failed (invalid level string): " << e.what() << std::endl;
    } catch (const std::exception& e) {
         std::cerr << "[Logging Setup] Log initialization failed (general exception): " << e.what() << std::endl;
    } catch (...) {
         std::cerr << "[Logging Setup] Log initialization failed (unknown exception)." << std::endl;
    }
}

// Definition of shutdown_logging (ensure it's declared in logging_setup.h)
void shutdown_logging() {
    spdlog::info("[Logging Shutdown] Flushing and closing current file sink if active.");
    CoreLogging::flush_and_close_current_file_sink(); // Use CoreLogging namespace
    spdlog::info("[Logging Shutdown] Shutting down spdlog.");
    spdlog::shutdown();
}