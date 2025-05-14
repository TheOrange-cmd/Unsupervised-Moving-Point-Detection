// pybind_utils/conversions.h

#ifndef PYBIND_UTILS_CONVERSIONS_H
#define PYBIND_UTILS_CONVERSIONS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h> // For pcl::PointXYZI
#include <stdexcept>      // For runtime_error
#include <string>
#include <memory>         // For std::shared_ptr, std::make_shared

namespace py = pybind11;

/**
 * @brief Converts a NumPy array (Nx4: x,y,z,intensity) to a pcl::PointCloud<pcl::PointXYZI>::Ptr.
 *        Assumes the NumPy array is of type float32 and contiguous.
 * @param points_np The input NumPy array.
 * @return A shared pointer to a pcl::PointCloud<pcl::PointXYZI>.
 * @throws std::runtime_error if input is not a 2D array, doesn't have at least 4 columns,
 *         or is not C-contiguous float32.
 */
inline pcl::PointCloud<pcl::PointXYZI>::Ptr numpy_to_pcl(const py::array_t<float, py::array::c_style | py::array::forcecast>& points_np) {
    // 1. Request buffer information from NumPy array
    py::buffer_info buf_info = points_np.request();

    // 2. Validate dimensions
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input NumPy array must be 2-dimensional (NxC).");
    }
    if (buf_info.shape[1] < 4) {
        throw std::runtime_error("Input NumPy array must have at least 4 columns (x, y, z, intensity).");
    }
    // Note: py::array_t<float, py::array::c_style | py::array::forcecast> handles type and contiguity checks/conversions.

    // 3. Create an empty PCL PointCloud of the correct type
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    // 4. Reserve space for efficiency
    size_t num_points = buf_info.shape[0];
    cloud->reserve(num_points);

    // 5. Get pointer to the data
    const float* ptr = static_cast<const float*>(buf_info.ptr);
    size_t num_features = buf_info.shape[1]; // Number of columns (stride in floats)

    // 6. Loop through points and copy data
    for (size_t i = 0; i < num_points; ++i) {
        pcl::PointXYZI p; // Use the correct point type
        p.x = ptr[i * num_features + 0];
        p.y = ptr[i * num_features + 1];
        p.z = ptr[i * num_features + 2];
        p.intensity = ptr[i * num_features + 3];
        cloud->points.push_back(p);
    }

    // 7. Set PCL header info (optional but good practice)
    cloud->width = cloud->points.size();
    cloud->height = 1; // Unordered point cloud
    cloud->is_dense = true; // Assuming no NaN/Inf values in input x,y,z,i

    return cloud;
}

/**
 * @brief Converts a std::vector<DynObjLabel> to a NumPy array of integers.
 */
inline py::array_t<int> labels_to_numpy(const std::vector<DynObjLabel>& labels) {
    // Create a NumPy array of the same size, using int32 as dtype
    py::array_t<int> result(labels.size());
    // Get mutable buffer access
    py::buffer_info buf_info = result.request();
    int* ptr = static_cast<int*>(buf_info.ptr);

    // Copy data, casting enum to int
    for (size_t i = 0; i < labels.size(); ++i) {
        ptr[i] = static_cast<int>(labels[i]);
    }
    return result;
}

// TODO: Add pcl_to_numpy function later if needed

#endif // PYBIND_UTILS_CONVERSIONS_H