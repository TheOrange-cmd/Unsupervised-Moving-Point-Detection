// src/pybind_utils/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "filtering/dyn_obj_filter.h"
#include "config/config_loader.h" // Needed for DynObjFilter constructor binding
#include "filtering/dyn_obj_datatypes.h" // Include the header defining DynObjLabel
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
    py::enum_<DynObjLabel>(m, "DynObjLabel")
        .value("STATIC", DynObjLabel::STATIC)               // 0
        .value("APPEARING", DynObjLabel::APPEARING)         // 1
        .value("OCCLUDING", DynObjLabel::OCCLUDING)         // 2
        .value("DISOCCLUDED", DynObjLabel::DISOCCLUDED)     // 3
        .value("SELF", DynObjLabel::SELF)                   // 4
        .value("UNCERTAIN", DynObjLabel::UNCERTAIN)         // 5
        .value("INVALID", DynObjLabel::INVALID)             // 6
        .export_values();

    // --- Bind ProcessedPointInfo Struct ---
    py::class_<ProcessedPointInfo>(m, "ProcessedPointInfo")
        .def(py::init<>()) // Default constructor
        .def_readonly("original_index", &ProcessedPointInfo::original_index)
        .def_readonly("label", &ProcessedPointInfo::label)
        .def_readonly("local_x", &ProcessedPointInfo::local_x)
        .def_readonly("local_y", &ProcessedPointInfo::local_y)
        .def_readonly("local_z", &ProcessedPointInfo::local_z)
        .def_readonly("global_x", &ProcessedPointInfo::global_x)
        .def_readonly("global_y", &ProcessedPointInfo::global_y)
        .def_readonly("global_z", &ProcessedPointInfo::global_z)
        .def_readonly("intensity", &ProcessedPointInfo::intensity)
        .def_readonly("grid_pos", &ProcessedPointInfo::grid_pos)
        .def_readonly("spherical_azimuth", &ProcessedPointInfo::spherical_azimuth)
        .def_readonly("spherical_elevation", &ProcessedPointInfo::spherical_elevation)
        .def_readonly("spherical_depth", &ProcessedPointInfo::spherical_depth)
        // Add a __repr__ for easier debugging in Python
        .def("__repr__",
             [](const ProcessedPointInfo &p) {
                 return "<ProcessedPointInfo Idx:" + std::to_string(p.original_index) +
                        " Label:" + std::to_string(static_cast<int>(p.label)) + // Cast label to int for printing
                        " Loc:(" + std::to_string(p.local_x) + "," + std::to_string(p.local_y) + "," + std::to_string(p.local_z) + ")" +
                        " GridPos:" + std::to_string(p.grid_pos) + ">";
             });

    // --- Bind DynObjFilter Class ---
    py::class_<DynObjFilter, std::shared_ptr<DynObjFilter>>(m, "DynObjFilter")
        .def(py::init([](const std::string& config_path) {
                 std::cout << "Pybind: Constructing DynObjFilter with config: " << config_path << std::endl;
                 DynObjFilterParams params;
                 try {
                     bool success = load_config(config_path, params);
                     if (!success) {
                         // Consider throwing an error or logging more verbosely
                         py::print("Pybind Warning: Config loaded with issues or failed to load.");
                         // Optionally throw: throw std::runtime_error("Failed to load configuration.");
                     }
                     // Check essential params?
                     if (params.history_length <= 0) {
                        throw std::runtime_error("Invalid config: history_length must be > 0.");
                     }
                     if (params.max_depth_map_num <= 0) {
                        throw std::runtime_error("Invalid config: max_depth_map_num must be > 0.");
                     }
                     return std::make_shared<DynObjFilter>(params);
                 } catch (const std::exception &e) {
                     throw std::runtime_error(std::string("Error during DynObjFilter construction: ") + e.what());
                 }
             }),
             py::arg("config_path"),
             "Constructs and initializes the DynObjFilter using a configuration file.")

        // Bind the primary addScan method
        .def("add_scan", [](DynObjFilter& self,
                             const py::array_t<float, py::array::c_style | py::array::forcecast>& points_np,
                             const Eigen::Matrix3d& rotation,
                             const Eigen::Vector3d& position,
                             double timestamp)
             {
                 //std::cout << "Pybind: Calling add_scan..." << std::endl; // Can be verbose
                 ScanFrame::PointCloudPtr cloud_ptr;
                 try {
                     cloud_ptr = numpy_to_pcl(points_np);
                 } catch (const std::exception& e) {
                      throw std::runtime_error(std::string("Pybind Error during numpy_to_pcl conversion: ") + e.what());
                 }

                 Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
                 pose.linear() = rotation;
                 pose.translation() = position;

                 try {
                    self.addScan(cloud_ptr, pose, timestamp);
                 } catch (const std::exception& e) {
                     throw std::runtime_error(std::string("Pybind Error during C++ addScan execution: ") + e.what());
                 }
                 // No return value needed for add_scan itself
             },
             "Adds a new lidar scan (points, pose, timestamp) to the filter for processing.",
             py::arg("points_np"), py::arg("rotation"), py::arg("position"), py::arg("timestamp"))

        // --- Bind Getters ---
        .def("get_depth_map_count", &DynObjFilter::get_depth_map_count,
             "Returns the current number of depth maps stored.")
        .def("get_last_processed_seq_id", &DynObjFilter::get_last_processed_seq_id,
             "Returns the sequence ID of the last scan frame successfully processed.")
        .def("get_scan_buffer_capacity", &DynObjFilter::get_scan_buffer_capacity,
             "Returns the capacity of the internal scan history buffer.")
        .def("get_scan_buffer_size", &DynObjFilter::get_scan_buffer_size,
             "Returns the current number of scans stored in the internal history buffer.")
        .def("get_map_total_point_count", &DynObjFilter::get_map_total_point_count,
            py::arg("map_absolute_index"), "Get total points in map with the given absolute index")
        .def("get_processed_points_info", &DynObjFilter::getProcessedPointsInfo, py::arg("seq_id"),
            py::return_value_policy::copy, // Return a copy of the vector
            "Gets detailed info for points processed in the specified frame.")

        // --- Keep Placeholder Binding (Optional) ---
        .def("placeholder_labeling", [](DynObjFilter& self,
                                          const py::array_t<float, py::array::c_style | py::array::forcecast>& points_np) -> py::array_t<int>
             {
                 // This bypasses the main addScan/processBufferedFrames logic
                 std::cout << "Pybind: Calling placeholder_labeling..." << std::endl;
                 ScanFrame::PointCloudPtr cloud_ptr = numpy_to_pcl(points_np);
                 std::vector<DynObjLabel> labels = self.placeholder_labeling(cloud_ptr);
                 return labels_to_numpy(labels);
             },
             "Processes scan using placeholder logic (labels = index % 7). Returns NumPy array of labels. Bypasses main filter state.",
             py::arg("points_np"));


}