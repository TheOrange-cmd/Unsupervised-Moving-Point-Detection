// file: src/main.cpp

#include <iostream>
#include <string>
#include <vector>
#include "config_loader.h" // Include the config loader header

int main(int argc, char* argv[]) {
    // --- Argument Parsing ---
    if (argc != 2) { // Expect exactly one argument: the config file path
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return 1;
    }
    std::string config_path = argv[1];
    std::cout << "Attempting to load config file: " << config_path << std::endl;

    // --- Load Configuration ---
    DynObjFilterParams params; // Create an instance of the parameters struct
    if (!load_config(config_path, params)) {
        std::cerr << "Failed to load configuration from " << config_path << std::endl;
        return 1; // Indicate failure
    }

    // --- Print Verification ---
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Configuration loaded successfully!" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    // Print a few key parameters to verify they were loaded correctly
    std::cout << "Verification:" << std::endl;
    std::cout << "  Dataset Type:         " << params.dataset << std::endl;
    std::cout << "  Buffer Delay:         " << params.buffer_delay << std::endl;
    std::cout << "  Hor Resolution Max:   " << params.hor_resolution_max << std::endl;
    std::cout << "  Ver Resolution Max:   " << params.ver_resolution_max << std::endl;
    std::cout << "  Case 1 Depth Thr:     " << params.depth_thr1 << std::endl;
    std::cout << "  Case 2 Vel Min Thr:   " << params.v_min_thr2 << std::endl;
    std::cout << "  Case 3 Occl Times Thr:" << params.occluding_times_thr3 << std::endl;
    std::cout << "  Cluster Voxel Res:    " << params.Cluster_Voxel_revolusion << std::endl;
    std::cout << "-------------------------------------" << std::endl;


    // --- Placeholder for future steps ---
    // Data loading would go here
    // Filter initialization would go here
    // Processing loop would go here

    return 0; // Indicate success
}