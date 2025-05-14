// test_cpp/common/test_environment.cpp
#include "common/test_environment.h"
#include "config/config_loader.h" // To load config for logging setup
#include "common/logging_setup.h"      // To call the actual setup function
#include <iostream>             // For initial error output if logging setup fails
#include <string>

// Define a default path for a test configuration file
// Assumes the test executable's working directory is set correctly by CMake/CTest
// (We configured this in CMakeLists.txt using WORKING_DIRECTORY)
const std::string DEFAULT_TEST_CONFIG_PATH = "config/test_full_config.yaml"; // Or just test_full_config.yaml

void GlobalTestEnvironment::SetUp() {
    std::cout << "[ GTest Environment ] Setting up global test environment..." << std::endl;

    DynObjFilterParams params; // Create params with constructor defaults

    // Try to load the default test config to potentially override log levels
    try {
        // Use the load_config function (make sure it's accessible)
        if (load_config(DEFAULT_TEST_CONFIG_PATH, params)) {
            std::cout << "[ GTest Environment ] Loaded logging levels from: "
                      << DEFAULT_TEST_CONFIG_PATH << std::endl;
        } else {
            std::cerr << "[ GTest Environment ] Warning: Failed to load "
                      << DEFAULT_TEST_CONFIG_PATH << " (returned false). Using default log levels." << std::endl;
            // Proceed with default params from constructor
        }
    } catch (const std::exception& e) {
        // Catch file not found or parsing errors
        std::cerr << "[ GTest Environment ] Warning: Exception loading "
                  << DEFAULT_TEST_CONFIG_PATH << ": " << e.what()
                  << ". Using default log levels." << std::endl;
        // Proceed with default params from constructor
    }

    // Now, setup logging using the (potentially config-loaded) parameters
    try {
        setup_logging(params); // Call the same setup function used in main.cpp
        std::cout << "[ GTest Environment ] Logging setup complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ GTest Environment ] FATAL: Logging setup failed during SetUp: " << e.what() << std::endl;
        // If logging is critical for tests, maybe abort?
        // exit(1); // Or handle more gracefully
    }
    std::cout << "[ GTest Environment ] Global setup finished." << std::endl;
}

void GlobalTestEnvironment::TearDown() {
    std::cout << "[ GTest Environment ] Tearing down global test environment..." << std::endl;
    // Optional: Shutdown spdlog if necessary (usually not required unless flushing files)
    // spdlog::shutdown();
    std::cout << "[ GTest Environment ] Teardown finished." << std::endl;
}

// --- Crucial Step: Register the Environment ---
// This line tells GTest to use our custom environment.
// It needs to be linked into the test executable. Placing it here is fine.
::testing::Environment* const g_test_env = ::testing::AddGlobalTestEnvironment(new GlobalTestEnvironment);