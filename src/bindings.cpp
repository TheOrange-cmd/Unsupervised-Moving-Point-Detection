// src/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
// #include <pybind11/smart_holder.h> // Keep commented out

#include "config/config_loader.h" // for DynObjFilterParams
#include "filtering/dyn_obj_datatypes.h" // Include the header defining dyn_obj_flg
#include "point_cloud_utils/point_cloud_utils.h"

namespace py = pybind11;

// --- Helper/Wrapper Functions 
DynObjFilterParams load_config_py(const std::string& filename) {
    DynObjFilterParams params; // Qualify if needed
    try {
        bool success = load_config(filename, params);
        if (!success) {
            py::print("Warning: Configuration loaded with some non-critical issues (using defaults).");
        }
        return params;
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Error loading config: ") + e.what());
    }
}

// --- Main Module Definition ---
PYBIND11_MODULE(mpy_detector, m) { // Name must match pybind11_add_module
    m.doc() = "Python bindings for the m_detector_lib";

    // --- Bind Enums ---

    // *** CORRECTED ENUM BINDING ***
    // Replace 'dyn_obj_flg' with 'm_detector::dyn_obj_flg' if it's in a namespace
    py::enum_<dyn_obj_flg>(m, "DynObjFlg") // Use the correct enum type and a suitable Python name
        .value("STATIC", dyn_obj_flg::STATIC)         // Bind actual values
        .value("CASE1", dyn_obj_flg::CASE1)
        .value("CASE2", dyn_obj_flg::CASE2)
        .value("CASE3", dyn_obj_flg::CASE3)
        .value("SELF", dyn_obj_flg::SELF)
        .value("UNCERTAIN", dyn_obj_flg::UNCERTAIN)
        .value("INVALID", dyn_obj_flg::INVALID)
        .export_values(); // Exposes enum members directly (e.g., mdet.DynObjFlg.STATIC)

    py::enum_<PointCloudUtils::InterpolationNeighborType>(m, "InterpolationNeighborType")
        .value("STATIC_ONLY", PointCloudUtils::InterpolationNeighborType::STATIC_ONLY)
        .value("ALL_VALID", PointCloudUtils::InterpolationNeighborType::ALL_VALID)
        .export_values();

    py::enum_<PointCloudUtils::InterpolationStatus>(m, "InterpolationStatus")
        .value("SUCCESS", PointCloudUtils::InterpolationStatus::SUCCESS)
        .value("NOT_ENOUGH_NEIGHBORS", PointCloudUtils::InterpolationStatus::NOT_ENOUGH_NEIGHBORS)
        .value("NO_VALID_TRIANGLE", PointCloudUtils::InterpolationStatus::NO_VALID_TRIANGLE)
        .export_values();

    // --- Bind Structs ---
    py::class_<DynObjFilterParams>(m, "DynObjFilterParams")
        .def(py::init<>())
        .def_readwrite("buffer_delay", &DynObjFilterParams::buffer_delay)
        .def_readwrite("buffer_size", &DynObjFilterParams::buffer_size)
        // ... Add ALL other members ...
        .def_readonly("interp_hor_num", &DynObjFilterParams::interp_hor_num)
        .def_readonly("interp_ver_num", &DynObjFilterParams::interp_ver_num)
        // ... etc. ...
        .def("__repr__",
             [](const DynObjFilterParams &p) {
                 // Add namespace qualifier if needed
                 return "<DynObjFilterParams: buffer_delay=" + std::to_string(p.buffer_delay) + "...>";
             });

    // Assuming PointCloudUtils namespace:
    py::class_<PointCloudUtils::InterpolationResult>(m, "InterpolationResult")
        .def(py::init<>())
        .def_readwrite("status", &PointCloudUtils::InterpolationResult::status)
        .def_readwrite("depth", &PointCloudUtils::InterpolationResult::depth)
        .def("__repr__",
             [](const PointCloudUtils::InterpolationResult &r) {
                 // Add namespace qualifier if needed
                 return "<InterpolationResult: status=" + std::to_string(static_cast<int>(r.status)) +
                        ", depth=" + std::to_string(r.depth) + ">";
             });

    // TODO: Bind point_soph
    // TODO: Bind DepthMap

    // --- Bind Functions ---
    m.def("load_config", &load_config_py, "Loads configuration from a YAML file.",
          py::arg("filename"));

    // TODO: Bind PointCloudUtils functions
    // TODO: Bind DynObjFilter class
}