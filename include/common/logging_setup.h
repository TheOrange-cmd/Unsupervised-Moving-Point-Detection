// common/logging_setup.h
#ifndef COMMON_LOGGING_SETUP_H // Guard against multiple inclusions
#define COMMON_LOGGING_SETUP_H

#include "config/config_loader.h" // For DynObjFilterParams
#include <spdlog/common.h>       // For spdlog::level::level_enum
#include <string>

// Forward declaration if setup_logging is used by other headers, otherwise not strictly needed here.
// struct DynObjFilterParams;

void setup_logging(const DynObjFilterParams& params);
void shutdown_logging(); 

// Namespace for globally accessible logging configuration values
namespace CoreLoggingConfig {
    extern std::string g_log_directory_path;                             // To be set by setup_logging
    extern spdlog::level::level_enum g_default_configured_file_log_level; // To be set by setup_logging
}

#endif // COMMON_LOGGING_SETUP_H