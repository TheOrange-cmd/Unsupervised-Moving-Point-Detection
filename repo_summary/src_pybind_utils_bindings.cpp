// src/pybind_utils/bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "filtering/dyn_obj_filter.h"
#include "config/config_loader.h"
#include "common/dyn_obj_datatypes.h"
#include "pybind_utils/conversions.h"

#include <memory>
#include <stdexcept>
#include <iostream>

#include "common/logging_setup.h" // Make sure logging is initialized
#include <spdlog/spdlog.h>
#include <Eigen/Geometry> // For Eigen::Quaterniond

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
            const Eigen::Matrix3d& rotation, // Sensor->World R
            const Eigen::Vector3d& position, // Sensor->World T
            double timestamp)
            {
            // --- START LOGGING ---
            auto logger = spdlog::get("Filter_Core"); // Use an appropriate logger
            uint64_t seq_id_to_log = self.get_next_scan_seq_id(); // Get upcoming Seq ID

            if (logger) { // Check if logger exists
                logger->info("Pybind add_scan (Seq ID {}): Received Position (Sensor->World): ({:.3f}, {:.3f}, {:.3f})",
                            seq_id_to_log, position.x(), position.y(), position.z());

                try {
                    Eigen::Quaterniond q_eigen(rotation);
                    logger->info("Pybind add_scan (Seq ID {}): Received Rotation (Sensor->World Quat w,x,y,z): ({:.4f}, {:.4f}, {:.4f}, {:.4f})",
                                seq_id_to_log, q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
                } catch (const std::exception& e) {
                    logger->error("Pybind add_scan (Seq ID {}): Error converting rotation matrix to Eigen::Quaterniond: {}", seq_id_to_log, e.what());
                }

                // --- Calculate and log World->Sensor transform BEFORE calling internal addScan ---
                Eigen::Matrix3d project_R = rotation.transpose();
                Eigen::Vector3d project_T = -(project_R * position); // Use project_R here
                logger->info("Pybind add_scan (Seq ID {}): Calculated project_T (World->Sensor): ({:.3f}, {:.3f}, {:.3f})",
                            seq_id_to_log, project_T.x(), project_T.y(), project_T.z());
            } else {
                // Fallback if logger not found (shouldn't happen if setup_logging works)
                std::cerr << "Pybind add_scan: Logger 'Filter_Core' not found!" << std::endl;
            }
            // --- END LOGGING ---


            // --- Existing conversion and internal call ---
            ScanFrame::PointCloudPtr cloud_ptr;
            try {
                cloud_ptr = numpy_to_pcl(points_np);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Pybind Error during numpy_to_pcl conversion: ") + e.what());
            }

            // Assuming ScanFrame::PoseType is Eigen::Isometry3d based on your code
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.linear() = rotation;
            pose.translation() = position;

            try {
                // This call will use the rotation/position to eventually store
                // the calculated project_R/project_T internally.
                self.addScan(cloud_ptr, pose, timestamp);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Pybind Error during C++ addScan execution: ") + e.what());
            }
            // No return value needed for add_scan itself
            },
            "Adds a new lidar scan (points, pose, timestamp) to the filter for processing.",
            py::arg("points_np"), py::arg("rotation"), py::arg("position"), py::arg("timestamp")
        )

        // --- Bind Getters ---
        .def("get_next_scan_seq_id", &DynObjFilter::get_next_scan_seq_id,
            "Gets the sequence ID that will be assigned to the next scan added.")
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