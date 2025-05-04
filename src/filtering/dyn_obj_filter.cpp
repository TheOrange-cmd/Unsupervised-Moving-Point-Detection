#include <iostream> // For stub output
#include <cmath>    // For floor, round etc. in init
#include <vector>
#include <stdexcept> // For potential errors
#include <algorithm>  // For std::min, std::max
#include <iomanip>

#include "filtering/dyn_obj_filter.h"
#include "point_cloud_utils/point_cloud_utils.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h" // Include to get DynObjFilterParams definition
#include "filtering/consistency_checks.h"

// --- Constructor ---
DynObjFilter::DynObjFilter(const DynObjFilterParams& config_params) :
    params_(config_params),
    scan_history_buffer_(params_.history_length > 0 ? params_.history_length : 1),
    map_index_(-1), // Initialize map index to -1 (first map will be 0)
    next_scan_seq_id_(0),
    last_processed_seq_id_(-1) // Use -1 or similar sentinel for "none processed"
{
    std::cout << "DynObjFilter constructed. History length: " << params_.history_length << std::endl;
    std::cout << "  RingBuffer capacity: " << scan_history_buffer_.capacity() << std::endl;
    std::cout << "DynObjFilter initialization complete." << std::endl;
}

// --- Method to add a new scan ---
void DynObjFilter::addScan(const ScanFrame::PointCloudPtr& cloud, const ScanFrame::PoseType& pose, double timestamp) {
    if (!cloud) {
        std::cerr << "[DynObjFilter::addScan] Warning: Received null point cloud. Skipping." << std::endl;
        return;
    }

    uint64_t current_seq_id = next_scan_seq_id_++;
    // Note: ScanFrame constructor takes PointCloudPtr (cloud) directly
    ScanFrame current_frame(timestamp, cloud, pose, current_seq_id);

    std::cout << "[DynObjFilter::addScan] Adding scan Timestamp: " << timestamp
              << ", SeqID: " << current_seq_id
              << ", Points: " << cloud->size() << std::endl;

    scan_history_buffer_.add(std::move(current_frame));
    processBufferedFrames();
}



// --- Process Buffered Frames ---
void DynObjFilter::processBufferedFrames() {
    bool processed_a_frame = false;

    for (size_t buf_idx = 0; buf_idx < scan_history_buffer_.size(); ++buf_idx) {
        const ScanFrame& frame_to_check = scan_history_buffer_.get(buf_idx);

        if (frame_to_check.seq_id == last_processed_seq_id_ + 1) {
            std::cout << "Processing frame seq_id: " << frame_to_check.seq_id << std::endl;
            processed_a_frame = true;

            if (!frame_to_check.cloud) {
                std::cerr << "Warning: Frame " << frame_to_check.seq_id << " has null cloud pointer. Skipping." << std::endl;
                last_processed_seq_id_ = frame_to_check.seq_id;
                break;
            }

            std::vector<std::shared_ptr<point_soph>> processed_points; // For adding to map
            std::vector<ProcessedPointInfo> current_frame_info_vec; // For returning info
            processed_points.reserve(frame_to_check.cloud->size());
            current_frame_info_vec.reserve(frame_to_check.cloud->size()); // Reserve space

            // --- Loop through raw points ---
            for (size_t i = 0; i < frame_to_check.cloud->size(); ++i) {
                const auto& raw_point = frame_to_check.cloud->points[i];

                // 1. Basic Validity Check
                V3D local_point_coords(raw_point.x, raw_point.y, raw_point.z);
                if (PointCloudUtils::isPointInvalid(local_point_coords, params_)) {
                    continue;
                }

                // 2. Create and Populate point_soph
                auto p_soph = std::make_shared<point_soph>();
                p_soph->time = frame_to_check.timestamp;
                p_soph->intensity = raw_point.intensity;
                p_soph->local = local_point_coords;
                p_soph->glob = frame_to_check.sensor_pose * p_soph->local;
                p_soph->GetVec(p_soph->local, params_.hor_resolution_max, params_.ver_resolution_max);
                p_soph->reset();
                p_soph->dyn = DynObjLabel::STATIC; // Default label
                p_soph->raw_curvature = 0.0f;
                p_soph->is_distort = false;

                // 3. Self Point Check
                if (PointCloudUtils::isSelfPoint(p_soph->local, params_)) {
                    p_soph->dyn = DynObjLabel::SELF;
                    // processed_points.push_back(p_soph); // Add self points to map? Maybe not needed? Check original logic. Let's add them for now.
                    // continue; // Skip further checks for self points
                } else if (params_.dyn_filter_en) { // 4. Case 1 Check (only if not SELF)
                    if (checkAppearingPoint(*p_soph)) {
                        p_soph->dyn = DynObjLabel::APPEARING;
                        // processed_points.push_back(p_soph); // Add appearing points
                        // continue; // Skip Case 2/3 checks
                    }
                    // --- 5. Placeholder for Future Checks (Case 2, Case 3) ---
                    // else if (checkOccludingPoint(...)) { ... }
                    // else if (checkDisoccludedPoint(...)) { ... }
                }

                // --- Store Processed Info ---
                // Populate the info struct *after* all label checks for this point are done
                ProcessedPointInfo point_info;
                point_info.original_index = i;
                point_info.label = p_soph->dyn; // Capture the final label
                point_info.local_x = p_soph->local.x();
                point_info.local_y = p_soph->local.y();
                point_info.local_z = p_soph->local.z();
                point_info.global_x = p_soph->glob.x();
                point_info.global_y = p_soph->glob.y();
                point_info.global_z = p_soph->glob.z();
                point_info.intensity = p_soph->intensity;
                point_info.grid_pos = p_soph->position;
                point_info.spherical_azimuth = p_soph->vec(0);
                point_info.spherical_elevation = p_soph->vec(1);
                point_info.spherical_depth = p_soph->vec(2);
                current_frame_info_vec.push_back(point_info);

                // --- 6. Add point to be inserted into the map ---
                // Only add non-invalid points to the map list
                if (p_soph->dyn != DynObjLabel::INVALID) {
                    processed_points.push_back(p_soph);
                }


            } // End loop over raw points

            std::cout << "  - Processed " << current_frame_info_vec.size() << " total points (before map add) from frame " << frame_to_check.seq_id << std::endl;
            std::cout << "  - Adding " << processed_points.size() << " valid points to map processing." << std::endl;

            // --- Call updateDepthMaps ---
            updateDepthMaps(processed_points, frame_to_check);

            // Update the ID and info of the last successfully processed frame
            last_processed_seq_id_ = frame_to_check.seq_id;
            last_processed_frame_info_ = std::move(current_frame_info_vec); // Store the collected info

            break; // Found and processed the frame
        }
    }

    // Optional: Log if no frame was processed
    // if (!processed_a_frame && !scan_history_buffer_.empty()) { ... }
}

// --- Update Depth Maps ---
void DynObjFilter::updateDepthMaps(
    const std::vector<std::shared_ptr<point_soph>>& processed_points,
    const ScanFrame& current_frame_info)
{
    // --- Map List Management ---
    if (depth_map_list_.empty()) {
        // FIX 3: Use map_index_ and pre-increment for first map (makes it 0)
        int current_map_idx = ++map_index_;
        std::cout << "Creating first DepthMap (Index: " << current_map_idx << ") at time " << current_frame_info.timestamp << std::endl;
        auto new_map = std::make_shared<DepthMap>(
            // FIX 4: Use .rotation()
            current_frame_info.sensor_pose.rotation(),
            current_frame_info.sensor_pose.translation(),
            current_frame_info.timestamp,
            current_map_idx // Assign the calculated index
        );
        depth_map_list_.push_back(new_map);
    } else {
        double time_since_last_map = current_frame_info.timestamp - depth_map_list_.back()->time;
        if (time_since_last_map >= params_.depth_map_dur) {
            // FIX 3: Use map_index_ and pre-increment
            int next_map_idx = ++map_index_;
            std::cout << "Time threshold reached (" << time_since_last_map << " >= " << params_.depth_map_dur << "). ";
            if (depth_map_list_.size() >= params_.max_depth_map_num) {
                std::cout << "Rotating DepthMap (Reusing Index: " << depth_map_list_.front()->map_index
                          << ", New Index: " << next_map_idx << ") at time " << current_frame_info.timestamp << std::endl;
                DepthMap::Ptr oldest_map = depth_map_list_.front();
                depth_map_list_.pop_front();
                oldest_map->Reset(
                    // FIX 4: Use .rotation()
                    current_frame_info.sensor_pose.rotation(),
                    current_frame_info.sensor_pose.translation(),
                    current_frame_info.timestamp,
                    next_map_idx // Assign the new index
                );
                depth_map_list_.push_back(oldest_map);
            } else {
                 std::cout << "Creating new DepthMap (Index: " << next_map_idx << ") at time " << current_frame_info.timestamp << std::endl;
                 auto new_map = std::make_shared<DepthMap>(
                    // FIX 4: Use .rotation()
                    current_frame_info.sensor_pose.rotation(),
                    current_frame_info.sensor_pose.translation(),
                    current_frame_info.timestamp,
                    next_map_idx // Assign the new index
                );
                depth_map_list_.push_back(new_map);
            }
        }
    }

    if (depth_map_list_.empty()) {
         std::cerr << "Warning: No depth map available in updateDepthMaps!" << std::endl;
         return;
    }

    DepthMap& latest_map = *depth_map_list_.back();
    int points_added_to_map = 0;

    // --- Add Processed Points to Grid ---
    for (const auto& p_soph : processed_points) {
        int pos = p_soph->position;
        if (pos >= 0 && pos < MAX_2D_N) {
            if (latest_map.depth_map[pos].size() < params_.max_pixel_points) {
                latest_map.depth_map[pos].push_back(p_soph);
                points_added_to_map++;

                // --- Statistics Update (logic seems okay, minor tweak for clarity) ---
                float current_depth = p_soph->vec(2);
                bool is_first_in_cell = latest_map.depth_map[pos].size() == 1;

                // Update max_depth_all
                if (is_first_in_cell || current_depth > latest_map.max_depth_all[pos]) {
                    latest_map.max_depth_all[pos] = current_depth;
                    latest_map.max_depth_index_all[pos] = latest_map.depth_map[pos].size() - 1;
                }
                // Update min_depth_all
                if (is_first_in_cell || current_depth < latest_map.min_depth_all[pos]) {
                     latest_map.min_depth_all[pos] = current_depth;
                     latest_map.min_depth_index_all[pos] = latest_map.depth_map[pos].size() - 1;
                }
                // Update static depths
                if (p_soph->dyn == DynObjLabel::STATIC) {
                     bool first_static = true;
                     // Check if any *previous* point in the cell was static
                     for(size_t k=0; k < latest_map.depth_map[pos].size() - 1; ++k) {
                         if(latest_map.depth_map[pos][k]->dyn == DynObjLabel::STATIC) {
                             first_static = false;
                             break;
                         }
                     }
                     // Update min/max if this is the first static or sets a new bound
                     if (first_static || current_depth < latest_map.min_depth_static[pos]) {
                         latest_map.min_depth_static[pos] = current_depth;
                     }
                     if (first_static || current_depth > latest_map.max_depth_static[pos]) {
                         latest_map.max_depth_static[pos] = current_depth;
                     }
                }
                // --- End Statistics Update ---
            }
        } else {
             std::cerr << "Warning: Invalid position index " << pos << " for point." << std::endl;
        }
    }
     std::cout << "  - Added " << points_added_to_map << " points to DepthMap index " << latest_map.map_index << std::endl;
}

// --- Check Appearing Point (Case 1) ---
bool DynObjFilter::checkAppearingPoint(point_soph& p) {
    int depth_map_num = depth_map_list_.size();

    if (depth_map_num < params_.occluded_map_thr1) {
        return false;
    }

    int occluded_map_count = 0;

    for (int i = depth_map_num - 1; i >= 0; --i) {
        const DepthMap& map = *depth_map_list_[i];
        point_soph p_spherical_in_map;
        point_soph point_for_check;
        // --- Setup point_for_check (Projection, local coords, etc.) ---
        PointCloudUtils::SphericalProjection(p, map.map_index, map.project_R, map.project_T, params_, p_spherical_in_map);
        point_for_check.local = map.project_R * p.glob + map.project_T;
        point_for_check.vec = p_spherical_in_map.vec;
        point_for_check.hor_ind = p_spherical_in_map.hor_ind;
        point_for_check.ver_ind = p_spherical_in_map.ver_ind;
        point_for_check.position = p_spherical_in_map.position;
        point_for_check.time = p.time;
        point_for_check.is_distort = p.is_distort;
        // --- End Setup ---

        if (params_.debug_en) {
             std::cout << std::fixed << std::setprecision(5);
             std::cout << "[checkAppearingPoint] Checking MapIdx=" << map.map_index << " (List Idx=" << i << ")" << std::endl;
             std::cout << "[checkAppearingPoint]   Proj Coords : H=" << point_for_check.hor_ind << " V=" << point_for_check.ver_ind
                       << " Pos=" << point_for_check.position << " D=" << point_for_check.vec(2) << std::endl;
             std::cout << "[checkAppearingPoint]   Local Coords: X=" << point_for_check.local.x()
                       << " Y=" << point_for_check.local.y() << " Z=" << point_for_check.local.z() << std::endl;
        }

        bool map_shows_inconsistency = false; // Assume consistent

        // Condition 1: Invalid Projection
        if (point_for_check.position < 0 || point_for_check.position >= MAX_2D_N || point_for_check.vec(2) <= 0.0f) {
            map_shows_inconsistency = true;
            if (params_.debug_en) {
                std::cout << "[checkAppearingPoint] MapIdx=" << map.map_index << " -> Inconsistent (Invalid Projection)" << std::endl;
            }
        // Condition 2: Inside Self-Box (Treated as Consistent for this check)
        } else if (PointCloudUtils::isSelfPoint(point_for_check.local, params_)) {
            map_shows_inconsistency = false;
            if (params_.debug_en) {
                std::cout << "[checkAppearingPoint] MapIdx=" << map.map_index << " -> Consistent (Inside Self-Box)" << std::endl;
            }
        // Condition 3: Check Neighbors & Interpolation
        } else {
            std::vector<V3F> neighbors = PointCloudUtils::findInterpolationNeighbors(
                point_for_check, map, params_, PointCloudUtils::InterpolationNeighborType::STATIC_ONLY);

            if (neighbors.empty()) {
                // 3.b: Absence of Static Surface
                map_shows_inconsistency = true;
                if (params_.debug_en) {
                    std::cout << "[checkAppearingPoint] MapIdx=" << map.map_index << " -> Inconsistent (No Static Neighbors Found)" << std::endl;
                }
            } else if (neighbors.size() >= 3) {
                // Attempt interpolation only if enough neighbors exist
                V2F target_projection = point_for_check.vec.head<2>();
                PointCloudUtils::InterpolationResult interp_result = PointCloudUtils::computeBarycentricDepth(
                    target_projection, neighbors, params_);

                if (interp_result.status == PointCloudUtils::InterpolationStatus::SUCCESS) {
                    // 3.a: Geometric Contradiction (Significantly In Front)
                    float depth_diff = point_for_check.vec(2) - interp_result.depth;
                    // Calculate CASE1 threshold...
                    float threshold = params_.interp_thr1;
                    if (point_for_check.vec(2) > params_.interp_start_depth1) { /* ... */ }
                    if (params_.dataset == 0 && point_for_check.is_distort && params_.enlarge_distort > 1.0f) { /* ... */ }

                    if (depth_diff < -threshold) { // Significantly IN FRONT
                        map_shows_inconsistency = true;
                         if (params_.debug_en) {
                             std::cout << "[checkAppearingPoint] MapIdx=" << map.map_index << " -> Inconsistent (SUCCESS, In Front: Diff=" << depth_diff << " < Thr=" << -threshold << ")" << std::endl;
                         }
                    } else {
                        map_shows_inconsistency = false; // Consistent if SUCCESS and not in front
                         if (params_.debug_en) {
                             std::cout << "[checkAppearingPoint] MapIdx=" << map.map_index << " -> Consistent (SUCCESS, Not In Front: Diff=" << depth_diff << " >= Thr=" << -threshold << ")" << std::endl;
                        }
                    }
                } else { // interp_result.status != SUCCESS (e.g., NO_VALID_TRIANGLE)
                    // 3.c: Geometric Ambiguity
                    map_shows_inconsistency = true;
                    if (params_.debug_en) {
                        std::cout << "[checkAppearingPoint] MapIdx=" << map.map_index << " -> Inconsistent ("
                                  << PointCloudUtils::interpolationStatusToString(interp_result.status) << " with >=3 neighbors)" << std::endl;
                    }
                }
            } else { // 1 or 2 neighbors found
                // Sparsity: Not enough neighbors for interpolation. Treat as Consistent.
                map_shows_inconsistency = false;
                 if (params_.debug_en) {
                     std::cout << "[checkAppearingPoint] MapIdx=" << map.map_index << " -> Consistent (Sparse Static Neighbors: " << neighbors.size() << ")" << std::endl;
                 }
            }
        
            // --- End CASE1 Evaluation ---

            if (map_shows_inconsistency) {
                occluded_map_count++;
                 if (params_.debug_en) { // Added debug print for increment
                     std::cout << "[checkAppearingPoint]   -> Inconsistent with map. Treating as occluded. Count=" << occluded_map_count << std::endl;
                 }
            } else {
                 if (params_.debug_en) { // Added matching print for consistency
                     std::cout << "[checkAppearingPoint]   -> Consistent with map. Not occluded relative to this map." << std::endl;
                 }
            }
        } // End of valid projection/not-self-box block

        // Check success condition
        if (occluded_map_count >= params_.occluded_map_thr1) {
            if (params_.debug_en) {
                std::cout << "[checkAppearingPoint] -> Returning TRUE (Threshold reached: " << occluded_map_count << " >= " << params_.occluded_map_thr1 << ")" << std::endl;
            }
            return true;
        }

        // Check early exit condition
        if (occluded_map_count + i < params_.occluded_map_thr1) {
            if (params_.debug_en) {
                std::cout << "[checkAppearingPoint] -> Returning FALSE (Early exit: Cannot reach threshold. Count=" << occluded_map_count << ", Remaining=" << i << ", Threshold=" << params_.occluded_map_thr1 << ")" << std::endl;
            }
            return false;
        }
    } // End loop through maps

    // If loop finishes without reaching the threshold
    if (params_.debug_en) {
        std::cout << "[checkAppearingPoint] -> Returning FALSE (Loop finished, threshold not reached)" << std::endl;
    }
    return false;
}


// --- Getters ---

// --- Getter for Processed Point Info ---
std::vector<ProcessedPointInfo> DynObjFilter::getProcessedPointsInfo(uint64_t seq_id) const {
    if (seq_id == last_processed_seq_id_) {
        return last_processed_frame_info_; // Return a copy
    } else {
        std::cerr << "[getProcessedPointsInfo] Warning: Requested seq_id " << seq_id
                  // AND HERE:
                  << " does not match last processed seq_id " << last_processed_seq_id_
                  << ". Returning empty vector." << std::endl;
        return {};
    }
}

size_t DynObjFilter::get_depth_map_count() const {
    return depth_map_list_.size();
}

uint64_t DynObjFilter::get_last_processed_seq_id() const {
    // Return the actual ID, or the initial sentinel value if none processed
    return last_processed_seq_id_;
}

size_t DynObjFilter::get_scan_buffer_capacity() const {
    return scan_history_buffer_.capacity();
}

size_t DynObjFilter::get_scan_buffer_size() const {
    return scan_history_buffer_.size();
}

// --- Getter for map total point count ---
int DynObjFilter::get_map_total_point_count(int map_absolute_index) const {
    for (const auto& map_ptr : depth_map_list_) {
        if (map_ptr->map_index == map_absolute_index) {
            int total_points = 0;
            for (const auto& cell : map_ptr->depth_map) {
                total_points += cell.size();
            }
            return total_points;
        }
    }
    return -1; // Not found
}

// --- Placeholder Labeling to debug python-c++-python pipeline without actual algorithm ---
std::vector<DynObjLabel> DynObjFilter::placeholder_labeling(
    const ScanFrame::PointCloudPtr& cloud)
{
    std::vector<DynObjLabel> labels;
    if (!cloud) {
        std::cerr << "Warning: placeholder_labeling received a null point cloud." << std::endl;
        return labels;
    }
    size_t num_points = cloud->size();
    labels.reserve(num_points);
    std::cout << "  DynObjFilter::placeholder_labeling processing " << num_points << " points..." << std::endl;
    for (size_t i = 0; i < num_points; ++i) {
        labels.push_back(static_cast<DynObjLabel>(i % 7));
    }
    std::cout << "  DynObjFilter::placeholder_labeling finished." << std::endl;
    return labels;
}