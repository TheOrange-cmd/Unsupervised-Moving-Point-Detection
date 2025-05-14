// src/common/logging_setup.cpp
#include "common/logging_setup.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include "config/config_loader.h" // For DynObjFilterParams
#include "logging/logging_management.h" // Use the new path/namespace
#include <vector>
#include <string>
#include <memory>

namespace CoreLoggingConfig { // A small namespace for config storage
    std::string g_log_directory_path = ""; // Default to empty
}

// Definition of setup_logging
void setup_logging(const DynObjFilterParams& params) { // params now provides log_dir
    CoreLoggingConfig::g_log_directory_path = "/home/drugge/staff-umbrella/TeamHolgerResearch/drugge/mdet_logs"; // replace later using params!
    try {
        auto base_level = spdlog::level::from_str(params.log_level);

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::from_str("info"));

        std::vector<std::string> managed_logger_names = {
            "Utils_Grid", "Utils_Interpolation", "Utils_Projection", "Utils_Validity",
            "Filter_Core", "Filter_Processing", "Filter_Map", "Filter_Consistency", "Consistency"
            // Add other loggers that should have per-sequence file output
        };
        CoreLogging::initialize_managed_logger_names(managed_logger_names); // Use new namespace

        spdlog::set_level(base_level);

        for (const auto& name : managed_logger_names) { // Or a combined list of all loggers
            auto logger = spdlog::get(name);
            if (logger == nullptr) {
                logger = std::make_shared<spdlog::logger>(name, console_sink);
                spdlog::register_logger(logger);
                // spdlog::debug("Logger '{}' created with console sink.", name);
            } else {
                logger->sinks().clear();
                logger->sinks().push_back(console_sink);
                // spdlog::debug("Logger '{}' re-assigned console sink.", name);
            }

            spdlog::level::level_enum final_logger_level = base_level;
            std::string level_source = "base";
            auto it = params.log_levels.find(name);
            if (it != params.log_levels.end()) {
                try {
                    final_logger_level = spdlog::level::from_str(it->second);
                    level_source = "override";
                } catch (const std::invalid_argument& e) {
                    spdlog::error("Invalid log level '{}' for logger '{}'. Using base '{}'.",
                                  it->second, name, params.log_level);
                }
            }
            logger->set_level(final_logger_level);
            spdlog::info("Logger '{}' level set to '{}' (from {}). Console output active.", name, spdlog::level::to_string_view(final_logger_level), level_source);
        }

        spdlog::flush_on(spdlog::level::warn);
        spdlog::info("Logging setup complete. Base console level: '{}'. Per-sequence file logging configured. Log directory: '{}'",
                     params.log_level, CoreLoggingConfig::g_log_directory_path); // Log the directory path

    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed (spdlog exception): " << ex.what() << std::endl;
    } catch (const std::invalid_argument& e) {
         std::cerr << "Log initialization failed (invalid level string): " << e.what() << std::endl;
    } catch (const std::exception& e) {
         std::cerr << "Log initialization failed (general exception): " << e.what() << std::endl;
    } catch (...) {
         std::cerr << "Log initialization failed (unknown exception)." << std::endl;
    }
}

// Definition of shutdown_logging
void shutdown_logging() {
    CoreLogging::flush_and_close_current_file_sink(); // Use new namespace
    spdlog::shutdown();
}

