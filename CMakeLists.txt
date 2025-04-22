cmake_minimum_required(VERSION 3.10) # 3.10 is okay, but 3.13+ is often better for modern CMake features

project(mdet_repurposed CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Good practice

# --- Find Required Packages ---
find_package(Eigen3 3.3 REQUIRED CONFIG) # Found via config file, good.

# Find PCL - Specify components likely needed by DynObjFilter/Cluster
# common, io: Basic types and file I/O
# search, kdtree: Neighbor searches (likely used in clustering/filtering)
# filters: VoxelGrid, etc. (used in DynObjFilter/Cluster)
# segmentation: Often used with clustering (safer to include)
find_package(PCL 1.8 REQUIRED COMPONENTS common io search kdtree filters segmentation)

find_package(yaml-cpp REQUIRED) # Found via CMake config or find module
find_package(OpenCV REQUIRED)   # Find OpenCV
find_package(TBB REQUIRED)      # Find Intel TBB
find_package(OpenMP REQUIRED)   # Find OpenMP support

message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "Found PCL: ${PCL_INCLUDE_DIRS}")
message(STATUS "Found yaml-cpp: ${yaml-cpp_INCLUDE_DIRS}")
message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found TBB: ${TBB_INCLUDE_DIRS}")
message(STATUS "Found OpenMP: ${OpenMP_CXX_FOUND}")

# --- Include Directories ---
include_directories(
    include # Our project's include directory
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${yaml-cpp_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    # OpenMP typically doesn't add include dirs this way
)

# --- Define Executable ---
add_executable(m_detector_cli
    src/main.cpp
    src/config_loader.cpp
    src/data_loader.cpp
    # --- Add refactored source files below as you create/modify them ---
    # src/DynObjFilter.cpp
    # src/DynObjCluster.cpp
    # src/voxel_cluster.cpp # Assuming implementation is in .cpp
    # src/EA_disk.cpp       # Assuming implementation is in .cpp
    # ... other .cpp files corresponding to your include headers ...
)

# --- Link Libraries ---
target_link_libraries(m_detector_cli
    PRIVATE
    Eigen3::Eigen          # Modern Eigen target
    ${PCL_LIBRARIES}       # PCL libraries for the found components
    yaml-cpp::yaml-cpp     # Modern yaml-cpp target (if available) or ${yaml-cpp_LIBRARIES}
    ${OpenCV_LIBS}         # OpenCV libraries
    ${TBB_LIBRARIES}       # TBB libraries
    ${OpenMP_CXX_LIBRARIES} # OpenMP runtime libraries
)

# --- Compiler Flags ---
# Enable OpenMP flags
target_compile_options(m_detector_cli PRIVATE ${OpenMP_CXX_FLAGS})

# Optional: Enable compiler warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(m_detector_cli PRIVATE -Wall -Wextra -pedantic)
endif()

# Optional: Enable optimizations for release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")