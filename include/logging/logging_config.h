#ifndef CORE_LOGGING_CONFIG_H
#define CORE_LOGGING_CONFIG_H

#include <string>

namespace CoreLoggingConfig {
    extern std::string g_log_directory_path;
    extern spdlog::level::level_enum g_default_configured_file_log_level; // New
}

#endif // CORE_LOGGING_CONFIG_H