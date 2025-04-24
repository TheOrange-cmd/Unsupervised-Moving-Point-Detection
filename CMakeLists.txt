cmake_minimum_required(VERSION 3.10) # 3.10 is okay

project(mdet_repurposed CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Good practice

# --- Find Required Packages ---
find_package(Eigen3 3.3 REQUIRED CONFIG)
find_package(PCL 1.8 REQUIRED COMPONENTS common io search kdtree filters segmentation)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED) 

message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "Found PCL: ${PCL_INCLUDE_DIRS}")
message(STATUS "Found yaml-cpp: ${yaml-cpp_INCLUDE_DIRS}")
message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found TBB: ${TBB_INCLUDE_DIRS}")
message(STATUS "Found OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Found GTest: ${GTEST_FOUND}") # Check if GTest was found

# ========================================================================
# Define Shared Library (NEW)
# ========================================================================
# List source files shared between the main app and tests
set(SHARED_SOURCES
    src/config_loader.cpp
    src/point_cloud_utils.cpp
    src/dyn_obj_filter.cpp      # Add others as needed
    src/data_loader.cpp         # Add others as needed
)

# Create a static library (or SHARED if you prefer)
add_library(m_detector_lib STATIC ${SHARED_SOURCES})

# Set properties for the library: Includes and Dependencies
# Use PUBLIC so that targets linking to this library inherit these
target_include_directories(m_detector_lib PUBLIC
    include # Project's include directory
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${yaml-cpp_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}      # Include if headers in shared sources need it
    ${TBB_INCLUDE_DIRS}         # Include if headers in shared sources need it
)

target_link_libraries(m_detector_lib PUBLIC
    Eigen3::Eigen
    ${PCL_LIBRARIES}
    yaml-cpp::yaml-cpp
    ${OpenCV_LIBS}              # Link if shared sources use OpenCV functions
    TBB::tbb                    # Link if shared sources use TBB functions
    ${OpenMP_CXX_LIBRARIES}     # Link if shared sources use OpenMP
)

# Apply compile options (like OpenMP) to the library if needed
target_compile_options(m_detector_lib PRIVATE ${OpenMP_CXX_FLAGS})

# ========================================================================
# Define Main Executable (MODIFIED)
# ========================================================================
add_executable(m_detector_cli
    src/main.cpp
)

# Link the main executable against the shared library
target_link_libraries(m_detector_cli PRIVATE
    m_detector_lib # Link our shared library
    # Dependencies are now inherited from the library via PUBLIC,
    # so you might not need to list Eigen, PCL, yaml-cpp etc. here again,
    # unless main.cpp uses them directly. Be explicit if unsure.
    Eigen3::Eigen          # Keep if main.cpp uses Eigen directly
    ${PCL_LIBRARIES}       # Keep if main.cpp uses PCL directly
    yaml-cpp::yaml-cpp     # Keep if main.cpp uses yaml-cpp directly
    ${OpenCV_LIBS}         # Keep if main.cpp uses OpenCV directly
    TBB::tbb               # Keep if main.cpp uses TBB directly
    ${OpenMP_CXX_LIBRARIES} # Keep if main.cpp uses OpenMP directly
)

# Include directories are also inherited, but you can keep this for clarity
# or if main.cpp needs includes not covered by the library's PUBLIC includes.
target_include_directories(m_detector_cli PRIVATE include) # Still need project include

# Compiler flags for the executable (warnings, optimizations)
target_compile_options(m_detector_cli PRIVATE ${OpenMP_CXX_FLAGS}) # Already on library, but needed if main.cpp uses OpenMP
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(m_detector_cli PRIVATE -Wall -Wextra -pedantic)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")


# ========================================================================
# Unit Testing Setup (MODIFIED)
# ========================================================================
enable_testing()

add_executable(m_detector_tests
    test/point_cloud_utils/test_spherical_projection.cpp
    test/point_cloud_utils/test_invalid_point_check.cpp
    test/point_cloud_utils/test_is_self_point.cpp
    test/point_cloud_utils/test_check_vertical_fov.cpp
    test/point_cloud_utils/test_neighbor_static_depth_range.cpp
    test/point_cloud_utils/test_interpolations.cpp
    # test/test_config_loader.cpp # Add this when you create it
)

# Link the test executable against the shared library and GTest
target_link_libraries(m_detector_tests PRIVATE
    m_detector_lib      # Link our shared library (provides utils, config, etc.)
    GTest::gtest        # Link Google Test
    GTest::gtest_main   # Link Google Test main
    TBB::tbb
    # Dependencies like Eigen, PCL, yaml-cpp are inherited from m_detector_lib
)

# Includes are also inherited from the library
# target_include_directories(m_detector_tests PRIVATE include) # Usually inherited

# --- Copy test data files next to the test executable ---
set(TEST_CONFIG_FILES
    test/config/test_full_config.yaml
    test/config/test_missing_config.yaml
)
foreach(TEST_FILE ${TEST_CONFIG_FILES})
    get_filename_component(TEST_FILE_NAME ${TEST_FILE} NAME)
    add_custom_command(
        TARGET m_detector_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}"
        "$<TARGET_FILE_DIR:m_detector_tests>/${TEST_FILE_NAME}"
        COMMENT "Copying test data file: ${TEST_FILE_NAME}"
    )
endforeach()

# --- Add Tests to CTest ---
include(GoogleTest)
gtest_add_tests(TARGET m_detector_tests)