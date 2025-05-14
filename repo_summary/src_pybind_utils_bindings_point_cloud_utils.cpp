// // src/bindings/bindings_point_cloud_utils.cpp

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>       // For std::vector, std::array
// #include <pybind11/eigen.h>     // For Eigen types
// #include <pybind11/numpy.h>     // For py::array_t (though Eigen often handles it)
// #include <pybind11/operators.h> // For operators if needed

// #include "config/config_loader.h"
// #include "common/dyn_obj_datatypes.h" // Includes point_soph, DepthMap, enums
// #include "point_cloud_utils/point_cloud_utils.h" // The utils we want to bind

// namespace py = pybind11;
// // Make types from PointCloudUtils accessible without qualification within the module definition
// using namespace PointCloudUtils;
