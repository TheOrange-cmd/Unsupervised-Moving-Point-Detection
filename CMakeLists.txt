cmake_minimum_required(VERSION 3.12) # 3.11 is okay

project(mdet_repurposed CXX)

# # Explicitly tell the linker to search the Conda lib directory
# # Explicitly tell the linker to search the Conda lib directory
# link_directories(${CMAKE_PREFIX_PATH}/lib)

# # Set the RPATH for built executables/libraries
# # This helps them find dependencies within the Conda env at runtime *without* LD_LIBRARY_PATH
# # Use CMAKE_INSTALL_RPATH to set the RPATH added when installing AND building
# set(CMAKE_INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib")
# # Ensure that directories discovered by the linker during the build are added to the RPATH
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# # Tell CMake to actually use the CMAKE_INSTALL_RPATH setting during the build, not just install
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

option(MDET_ENABLE_INTERPOLATION_DEBUG "Enable interpolation debug prints in the library" OFF)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Good practice

# --- Find Required Packages ---
find_package(X11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED CONFIG)
find_package(PCL 1.8 REQUIRED COMPONENTS common io search kdtree filters segmentation)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED) 
find_package(pybind11 REQUIRED CONFIG)

message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "Found PCL: ${PCL_INCLUDE_DIRS}")
message(STATUS "Found yaml-cpp: ${yaml-cpp_INCLUDE_DIRS}")
message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found TBB: ${TBB_INCLUDE_DIRS}")
message(STATUS "Found OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Found GTest: ${GTEST_FOUND}") # Check if GTest was found
message(STATUS "Found pybind11: Version ${pybind11_VERSION} in ${pybind11_DIR}")

# ========================================================================
# Define Shared Library (NEW)
# ========================================================================
# List source files shared between the main app and tests
set(SHARED_SOURCES
    src/config/config_loader.cpp
    src/point_cloud_utils/point_cloud_utils.cpp
    src/filtering/dyn_obj_filter.cpp 
    src/filtering/consistency_checks.cpp
)

# Create a static library (or SHARED if you prefer)
add_library(m_detector_lib STATIC ${SHARED_SOURCES})
# set_target_properties(m_detector_lib PROPERTIES COMPILE_FLAGS "-O0") // set no optimization as check for debugging - don't use in release

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

# Add the compile definition CONDITIONALLY based on the option
if(MDET_ENABLE_INTERPOLATION_DEBUG)
    target_compile_definitions(m_detector_lib PRIVATE ENABLE_INTERPOLATION_DEBUG_PRINT)
    message(STATUS "Interpolation debugging prints ENABLED in m_detector_lib")
endif()

# ========================================================================
# Define Main Executable
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
# Unit Testing Setup
# ========================================================================
enable_testing()

# --- Utils Tests ---
add_executable(utils_tests
    # Source files for utils tests
    test_cpp/point_cloud_utils/test_spherical_projection.cpp
    test_cpp/point_cloud_utils/test_invalid_point_check.cpp
    test_cpp/point_cloud_utils/test_is_self_point.cpp
    test_cpp/point_cloud_utils/test_check_vertical_fov.cpp
    test_cpp/point_cloud_utils/test_neighbor_static_depth_range.cpp
    test_cpp/point_cloud_utils/test_interpolations.cpp
    test_cpp/point_cloud_utils/test_find_neighbor_cells.cpp
    test_cpp/point_cloud_utils/test_find_points_in_cells.cpp
    # Note: test_consistency_helpers.cpp is NOT included here, assuming it's only needed by filtering tests
)

# Apply compile definitions if needed by utils tests
# (Assuming interpolation debug might be used in test_interpolations.cpp)
target_compile_definitions(utils_tests PRIVATE ENABLE_INTERPOLATION_DEBUG_PRINT)

# Link utils tests
target_link_libraries(utils_tests PRIVATE
    m_detector_lib      # Link our shared library
    GTest::gtest        # Link Google Test
    GTest::gtest_main   # Link Google Test main
    TBB::tbb            # Link TBB if utils tests or m_detector_lib needs it
)

# Copy test data files needed by utils tests 
set(UTILS_TEST_CONFIG_FILES
    # Add any config files SPECIFICALLY needed ONLY by utils tests, if applicable
    # If they share the same config files, list them here too
    test_cpp/config/test_full_config.yaml
    test_cpp/config/test_missing_config.yaml
)
foreach(TEST_FILE ${UTILS_TEST_CONFIG_FILES})
    get_filename_component(TEST_FILE_NAME ${TEST_FILE} NAME)
    add_custom_command(
        TARGET utils_tests POST_BUILD # Apply to utils_tests target
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}"
        "$<TARGET_FILE_DIR:utils_tests>/${TEST_FILE_NAME}" # Use utils_tests directory
        COMMENT "Copying test data file for utils_tests: ${TEST_FILE_NAME}"
    )
endforeach()

# Discover tests in utils_tests and assign the "utils" label
gtest_discover_tests(utils_tests
    PROPERTIES LABELS "utils"
)

# --- Filtering Tests ---
add_executable(filtering_tests
    # Source files for filtering tests
    test_cpp/filtering/test_consistency_helpers.cpp # only contains helper functions, no tests
    test_cpp/filtering/test_map_consistency_checks.cpp
    test_cpp/filtering/test_depth_consistency_checks.cpp
    test_cpp/filtering/test_acceleration_consistency.cpp
    test_cpp/filtering/test_occlusion_check_relationship.cpp
    test_cpp/filtering/test_occlusion_find_relationship.cpp
    test_cpp/filtering/test_ring_buffer.cpp
)

# Apply compile definitions if needed by filtering tests
# (Assuming interpolation debug might be used indirectly via consistency checks)
target_compile_definitions(filtering_tests PRIVATE ENABLE_INTERPOLATION_DEBUG_PRINT)

# Link filtering tests
target_link_libraries(filtering_tests PRIVATE
    m_detector_lib      # Link our shared library
    GTest::gtest        # Link Google Test
    GTest::gtest_main   # Link Google Test main
    TBB::tbb            # Link TBB if filtering tests or m_detector_lib needs it
)

# # Copy test data files needed by filtering tests
# set(FILTERING_TEST_CONFIG_FILES
#     test_cpp/config/test_full_config.yaml
#     test_cpp/config/test_missing_config.yaml
# )
# foreach(TEST_FILE ${FILTERING_TEST_CONFIG_FILES})
#     get_filename_component(TEST_FILE_NAME ${TEST_FILE} NAME)
#     add_custom_command(
#         TARGET filtering_tests POST_BUILD # Apply to filtering_tests target
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE}"
#         "$<TARGET_FILE_DIR:filtering_tests>/${TEST_FILE_NAME}" # Use filtering_tests directory
#         COMMENT "Copying test data file for filtering_tests: ${TEST_FILE_NAME}"
#     )
# endforeach()

# Discover tests in filtering_tests and assign the "filtering" label
gtest_discover_tests(filtering_tests
    PROPERTIES LABELS "filtering"
)

# --- Add Tests to CTest ---
include(GoogleTest)

# ========================================================================
# Python Module
# ========================================================================

# Define the source file(s) for the bindings
set(PYBIND_SOURCES
    src/pybind_utils/bindings.cpp 
)

# Create the Python module using pybind11's helper function
# This creates a shared library with the correct suffix (.so/.pyd)
# The first argument is the name Python will use to import the module
pybind11_add_module(mpy_detector SHARED ${PYBIND_SOURCES})

# Link the Python module against C++ library
# This makes the functions and classes defined in m_detector_lib available
# to your binding code in bindings.cpp
target_link_libraries(mpy_detector PRIVATE
    m_detector_lib
    # Add other direct dependencies of bindings.cpp if any
    # For example, if bindings.cpp uses TBB directly: TBB::tbb
    # pybind11::module / pybind11::headers are usually handled automatically
    # by pybind11_add_module
)

# Include directories are inherited from m_detector_lib (because it uses PUBLIC)
# If bindings.cpp needs other specific includes, add them here:
# target_include_directories(mpy_detector PRIVATE /path/to/other/includes)

# Apply compile options if needed (e.g., OpenMP if bindings code uses it)
# target_compile_options(mpy_detector PRIVATE ${OpenMP_CXX_FLAGS})

# Optional: Set output directory for easier Python import dur

# ========================================================================
# Python Unit Testing Setup
# ========================================================================

# --- Find the Python Interpreter ---
set(PYTHON_TEST_EXECUTABLE "" CACHE STRING "Python executable to use for running pytest")
set(PYTHON_FOUND_METHOD "None") # For logging/debugging

# 1. Check for active Conda environment using CONDA_PREFIX environment variable
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PYTHON_EXE "$ENV{CONDA_PREFIX}/bin/python")
    if(EXISTS "${CONDA_PYTHON_EXE}")
        set(PYTHON_TEST_EXECUTABLE ${CONDA_PYTHON_EXE})
        set(PYTHON_FOUND_METHOD "Conda Environment (CONDA_PREFIX)")
        message(STATUS "Found Python executable via Conda environment: ${PYTHON_TEST_EXECUTABLE}")
    else()
        message(WARNING "CONDA_PREFIX is set ($ENV{CONDA_PREFIX}), but Python executable not found at '${CONDA_PYTHON_EXE}'. Will try other methods.")
    endif()
endif()

# 2. If Conda method failed or wasn't applicable, try the pybind11 variable
# (Ensure find_package(pybind11 REQUIRED CONFIG) was called earlier)
if(PYTHON_FOUND_METHOD STREQUAL "None")
    if(DEFINED pybind11_PYTHON_INTERPRETER AND EXISTS "${pybind11_PYTHON_INTERPRETER}")
        set(PYTHON_TEST_EXECUTABLE ${pybind11_PYTHON_INTERPRETER})
        set(PYTHON_FOUND_METHOD "pybind11 Variable (pybind11_PYTHON_INTERPRETER)")
        message(STATUS "Found Python executable via pybind11 variable: ${PYTHON_TEST_EXECUTABLE}")
    # else()
        # message(STATUS "pybind11_PYTHON_INTERPRETER variable not found or invalid.") # Optional debug message
    endif()
endif()

# 3. If still not found, fall back to find_package(Python3)
if(PYTHON_FOUND_METHOD STREQUAL "None")
    message(WARNING "Python not found via Conda or pybind11 variable. Falling back to find_package(Python3).")
    # Give FindPython3 a hint to check the PATH first, where Conda's bin should be.
    find_package(Python3 COMPONENTS Interpreter QUIET HINTS $ENV{PATH})
    if(Python3_FOUND AND EXISTS "${Python3_EXECUTABLE}")
        set(PYTHON_TEST_EXECUTABLE ${Python3_EXECUTABLE})
        set(PYTHON_FOUND_METHOD "FindPython3 (with PATH hint)")
        message(STATUS "Found Python executable via FindPython3: ${PYTHON_TEST_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Could not find a suitable Python interpreter for running tests using any method.")
    endif()
endif()

# Final Verification
if(NOT PYTHON_TEST_EXECUTABLE OR NOT EXISTS "${PYTHON_TEST_EXECUTABLE}")
     message(FATAL_ERROR "Selected Python executable for tests is empty or does not exist: '${PYTHON_TEST_EXECUTABLE}'")
endif()

message(STATUS "Using Python executable for tests: ${PYTHON_TEST_EXECUTABLE} (Found via: ${PYTHON_FOUND_METHOD})")

# --- Define the Pytest Command ---
add_test(
    NAME PythonBindingsTest
    # Add '-v' to the pytest command
    COMMAND ${PYTHON_TEST_EXECUTABLE} -m pytest -v -s ${CMAKE_CURRENT_SOURCE_DIR}/test_py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Optional: Add labels to Python tests
set_tests_properties(PythonBindingsTest PROPERTIES LABELS "python;bindings")