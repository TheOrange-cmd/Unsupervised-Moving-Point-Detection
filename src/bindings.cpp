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
using namespace PointCloudUtils;

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

// Wrapper for SphericalProjection to handle output parameter
// point_soph spherical_projection_wrapper(
//     const V3D& global_point, // Input: Global point coordinates
//     int depth_index,
//     const M3D& rot,
//     const V3D& transl,
//     const DynObjFilterParams& params)
// {
//     // Create a temporary input point_soph (only 'glob' is needed by SphericalProjection's logic)
//     point_soph p_in;
//     p_in.glob = global_point;
//     // Create an output point_soph object
//     point_soph p_out;
//     // Call the actual function
//     PointCloudUtils::SphericalProjection(p_in, depth_index, rot, transl, params, p_out);
//     // Return the result
//     return p_out;
// }

// // Wrapper for findNeighborStaticDepthRange to return min/max
// std::pair<float, float> find_neighbor_static_depth_range_wrapper(
//     const point_soph& p,
//     const DepthMap& map_info,
//     const DynObjFilterParams& params)
// {
//     float min_d = std::numeric_limits<float>::max(); // Initialize appropriately
//     float max_d = 0.0f; // Initialize appropriately
//     PointCloudUtils::findNeighborStaticDepthRange(p, map_info, params, min_d, max_d);
//     return {min_d, max_d};
// }

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
        // Bind ALL members as readwrite unless they are derived
        // Example:
        .def_readwrite("buffer_delay", &DynObjFilterParams::buffer_delay)
        .def_readwrite("buffer_size", &DynObjFilterParams::buffer_size)
        .def_readwrite("points_num_perframe", &DynObjFilterParams::points_num_perframe)
        .def_readwrite("depth_map_dur", &DynObjFilterParams::depth_map_dur)
        .def_readwrite("max_depth_map_num", &DynObjFilterParams::max_depth_map_num)
        .def_readwrite("max_pixel_points", &DynObjFilterParams::max_pixel_points)
        .def_readwrite("frame_dur", &DynObjFilterParams::frame_dur)
        .def_readwrite("dataset", &DynObjFilterParams::dataset)
        .def_readwrite("buffer_dur", &DynObjFilterParams::buffer_dur)
        .def_readwrite("point_index", &DynObjFilterParams::point_index)
        .def_readwrite("self_x_f", &DynObjFilterParams::self_x_f)
        .def_readwrite("self_x_b", &DynObjFilterParams::self_x_b)
        .def_readwrite("self_y_l", &DynObjFilterParams::self_y_l)
        .def_readwrite("self_y_r", &DynObjFilterParams::self_y_r)
        .def_readwrite("blind_dis", &DynObjFilterParams::blind_dis)
        .def_readwrite("fov_up", &DynObjFilterParams::fov_up)
        .def_readwrite("fov_down", &DynObjFilterParams::fov_down)
        .def_readwrite("fov_cut", &DynObjFilterParams::fov_cut)
        .def_readwrite("fov_left", &DynObjFilterParams::fov_left)
        .def_readwrite("fov_right", &DynObjFilterParams::fov_right)
        .def_readwrite("checkneighbor_range", &DynObjFilterParams::checkneighbor_range)
        .def_readwrite("stop_object_detect", &DynObjFilterParams::stop_object_detect)
        // ... Bind ALL other loaded parameters ...
        .def_readwrite("depth_thr1", &DynObjFilterParams::depth_thr1)
        // ... etc for Case1, Case2, Case3, Interpolation, Other ...
        .def_readwrite("dyn_filter_en", &DynObjFilterParams::dyn_filter_en)
        .def_readwrite("debug_en", &DynObjFilterParams::debug_en)
        .def_readwrite("hor_resolution_max", &DynObjFilterParams::hor_resolution_max)
        .def_readwrite("ver_resolution_max", &DynObjFilterParams::ver_resolution_max)
        .def_readwrite("frame_id", &DynObjFilterParams::frame_id)
        .def_readwrite("time_file", &DynObjFilterParams::time_file)
        .def_readwrite("time_breakdown_file", &DynObjFilterParams::time_breakdown_file)

        // Bind Derived Parameters as readonly
        .def_readonly("interp_hor_num", &DynObjFilterParams::interp_hor_num)
        .def_readonly("interp_ver_num", &DynObjFilterParams::interp_ver_num)
        .def_readonly("pixel_fov_up", &DynObjFilterParams::pixel_fov_up)
        .def_readonly("pixel_fov_down", &DynObjFilterParams::pixel_fov_down)
        .def_readonly("pixel_fov_cut", &DynObjFilterParams::pixel_fov_cut)
        .def_readonly("pixel_fov_left", &DynObjFilterParams::pixel_fov_left)
        .def_readonly("pixel_fov_right", &DynObjFilterParams::pixel_fov_right)
        .def_readonly("max_pointers_num", &DynObjFilterParams::max_pointers_num)
        .def("__repr__", // Basic representation
             [](const DynObjFilterParams &p) {
                 return "<DynObjFilterParams: buffer_delay=" + std::to_string(p.buffer_delay) + ", buffer_size=" + std::to_string(p.buffer_size) + "...>";
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