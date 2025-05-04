// src/pybind_utils/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "filtering/dyn_obj_filter.h"
#include "config/config_loader.h" // Needed for DynObjFilter constructor binding
#include "filtering/dyn_obj_datatypes.h" // Include the header defining dyn_obj_flg
#include "pybind_utils/conversions.h" // For numpy_to_pcl

#include <memory> // For std::shared_ptr, std::make_shared
#include <stdexcept> // For std::runtime_error
#include <iostream> // For std::cout in lambda

namespace py = pybind11;

// --- Main Module Definition ---
PYBIND11_MODULE(mpy_detector, m) { // Name must match pybind11_add_module
    m.doc() = "Python bindings for the m_detector_lib";

    // --- Bind Enums ---
    // Keep enums that might be relevant for the filter's *output* or configuration modes
    // If Interpolation* enums are purely internal C++ details now, they can be removed too.
    py::enum_<dyn_obj_flg>(m, "DynObjFlg")
        .value("STATIC", dyn_obj_flg::STATIC)       // 0
        .value("CASE1", dyn_obj_flg::CASE1)         // 1
        .value("CASE2", dyn_obj_flg::CASE2)         // 2
        .value("CASE3", dyn_obj_flg::CASE3)         // 3
        .value("SELF", dyn_obj_flg::SELF)           // 4
        .value("UNCERTAIN", dyn_obj_flg::UNCERTAIN) // 5
        .value("INVALID", dyn_obj_flg::INVALID)     // 6
        .export_values();

    // Remove Interpolation enums if they are no longer part of the Python interface
    // py::enum_<PointCloudUtils::InterpolationNeighborType>(m, "InterpolationNeighborType") ...
    // py::enum_<PointCloudUtils::InterpolationStatus>(m, "InterpolationStatus") ...

    // --- Bind Structs ---
    // REMOVE DynObjFilterParams binding
    // REMOVE InterpolationResult binding

    // --- Bind Functions ---
    // REMOVE m.def("load_config", ...)

    // --- Bind DynObjFilter Class ---
    py::class_<DynObjFilter, std::shared_ptr<DynObjFilter>>(m, "DynObjFilter")
        .def(py::init([](const std::string& config_path) {
                 // ... (constructor binding as before) ...
                 std::cout << "Pybind: Constructing DynObjFilter with config: " << config_path << std::endl;
                 DynObjFilterParams params;
                 try {
                     bool success = load_config(config_path, params);
                     if (!success) {
                         py::print("Warning: Config loaded with issues.");
                     }
                     return std::make_shared<DynObjFilter>(params);
                 } catch (const std::exception &e) {
                     throw std::runtime_error(std::string("Error during DynObjFilter construction: ") + e.what());
                 }
             }),
             py::arg("config_path"),
             "Constructs and initializes the DynObjFilter using a configuration file.")

        // Bind the placeholder method (renamed for clarity in Python)
        .def("process_scan_placeholder", [](DynObjFilter& self,
                                             const py::array_t<float, py::array::c_style | py::array::forcecast>& points_np,
                                             const Eigen::Matrix3d& rotation, // Keep pose info for future
                                             const Eigen::Vector3d& position, // Keep pose info for future
                                             double timestamp) -> py::array_t<int> // Return NumPy array of ints
             {
                 std::cout << "Pybind: Calling process_scan_placeholder..." << std::endl;
                 ScanFrame::PointCloudPtr cloud_ptr;
                 try {
                     cloud_ptr = numpy_to_pcl(points_np);
                     std::cout << "  Pybind: NumPy -> PCL conversion successful. Cloud size: " << (cloud_ptr ? cloud_ptr->size() : 0) << std::endl;
                 } catch (const std::exception& e) {
                      throw std::runtime_error(std::string("Error during numpy_to_pcl conversion: ") + e.what());
                 }

                 // Construct Pose (might not be needed by placeholder, but keep structure)
                 Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
                 pose.linear() = rotation;
                 pose.translation() = position;

                 // Call the C++ placeholder method
                 std::vector<dyn_obj_flg> labels;
                 try {
                    // Pass only the cloud, as the placeholder doesn't need pose/time
                    labels = self.placeholder_labeling(cloud_ptr);
                    std::cout << "  Pybind: C++ placeholder_labeling call successful. Got " << labels.size() << " labels." << std::endl;
                 } catch (const std::exception& e) {
                     throw std::runtime_error(std::string("Error during C++ placeholder_labeling execution: ") + e.what());
                 }

                 // Convert the resulting label vector to a NumPy array
                 try {
                     py::array_t<int> labels_np = labels_to_numpy(labels);
                     std::cout << "  Pybind: C++ labels -> NumPy conversion successful." << std::endl;
                     return labels_np;
                 } catch (const std::exception& e) {
                     throw std::runtime_error(std::string("Error during labels_to_numpy conversion: ") + e.what());
                 }
             },
             "Processes scan using placeholder logic (labels = index % 7). Returns NumPy array of labels.",
             py::arg("points_np"), py::arg("rotation"), py::arg("position"), py::arg("timestamp"));
}