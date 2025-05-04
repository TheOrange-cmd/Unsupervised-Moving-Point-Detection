#include "gtest/gtest.h"
#include "config/config_loader.h"
#include <fstream> // To create temporary test files
#include <cstdio>  // For std::remove

// Helper to create a temporary YAML file
void create_yaml_file(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    file << content;
    file.close();
}

TEST(ConfigLoaderTest, LoadFullConfig) {
    // Arrange
    const std::string filename = "test_full_config.yaml";
    create_yaml_file(filename, R"(
dyn_obj:
  buffer_delay: 0.2
  dataset: 1
  hor_resolution_max: 0.01
  ver_resolution_max: 0.02
  interp_hor_thr: 0.015 # Should result in interp_hor_num = 2
  interp_ver_thr: 0.035 # Should result in interp_ver_num = 2
  fov_up: 10.0 # degrees
  fov_down: -20.0 # degrees
  frame_dur: 0.1
  max_depth_map_num: 5
  depth_map_dur: 0.2
  # ... add other necessary params ...
)");
    DynObjFilterParams params;

    // Act
    bool result = load_config(filename, params);

    // Assert
    ASSERT_TRUE(result);
    ASSERT_DOUBLE_EQ(params.buffer_delay, 0.2);
    ASSERT_EQ(params.dataset, 1);
    ASSERT_FLOAT_EQ(params.hor_resolution_max, 0.01f);
    ASSERT_FLOAT_EQ(params.ver_resolution_max, 0.02f);
    ASSERT_FLOAT_EQ(params.interp_hor_thr, 0.015f);
    ASSERT_FLOAT_EQ(params.interp_ver_thr, 0.035f);

    // Check derived parameters
    ASSERT_EQ(params.interp_hor_num, 2); // ceil(0.015 / 0.01) = ceil(1.5) = 2
    ASSERT_EQ(params.interp_ver_num, 2); // ceil(0.035 / 0.02) = ceil(1.75) = 2

    // Check FOV calculations (example, verify exact values)
    int expected_fov_up = static_cast<int>(std::floor((10.0 * PI_MATH / 180.0 + 0.5 * PI_MATH) / 0.02));
    int expected_fov_down = static_cast<int>(std::floor((-20.0 * PI_MATH / 180.0 + 0.5 * PI_MATH) / 0.02));
    ASSERT_EQ(params.pixel_fov_up, expected_fov_up);
    ASSERT_EQ(params.pixel_fov_down, expected_fov_down);

    int expected_max_pointers = static_cast<int>(std::round((5 * 0.2 + 0.2) / 0.1)) + 1; // round(1.2 / 0.1) + 1 = 12 + 1 = 13
    ASSERT_EQ(params.max_pointers_num, expected_max_pointers);


    // Clean up
    std::remove(filename.c_str());
}

TEST(ConfigLoaderTest, LoadMissingParams) {
    // Arrange
    const std::string filename = "test_missing_config.yaml";
     create_yaml_file(filename, R"(
dyn_obj:
  buffer_delay: 0.3 # Load this one
  # dataset is missing, should use default (0)
  hor_resolution_max: 0.01
  ver_resolution_max: 0.02
  interp_hor_thr: 0.015
  interp_ver_thr: 0.035
  fov_up: 10.0
  fov_down: -20.0
  frame_dur: 0.1
  max_depth_map_num: 5
  depth_map_dur: 0.2
)");
    DynObjFilterParams params; // Constructor sets defaults

    // Act
    bool result = load_config(filename, params);

    // Assert
    ASSERT_TRUE(result); // Should still succeed using defaults
    ASSERT_DOUBLE_EQ(params.buffer_delay, 0.3); // Loaded value
    ASSERT_EQ(params.dataset, 0); // Default value used

    // Check derived parameters are calculated using the mix
    ASSERT_EQ(params.interp_hor_num, 2);
    ASSERT_EQ(params.interp_ver_num, 2);
    // ... check others ...

    // Clean up
    std::remove(filename.c_str());
}

TEST(ConfigLoaderTest, InvalidFilePath) {
    DynObjFilterParams params;
    bool result = load_config("non_existent_file.yaml", params);
    ASSERT_FALSE(result);
}

// Add tests for invalid YAML and incorrect types if desired