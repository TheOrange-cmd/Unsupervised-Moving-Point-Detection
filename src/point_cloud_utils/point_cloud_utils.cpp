// file: src/point_cloud_utils.cpp

#include "point_cloud_utils/point_cloud_utils.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h" // Include to get DynObjFilterParams definition
#include <cmath>      // For fabs, pow, sqrt, atan2f
#include <vector>     // Used internally by interpolation functions
#include <algorithm>  // For std::min, std::max
#include <iostream>

// Define a small epsilon for floating-point comparisons in barycentric calculation
constexpr double CACHE_VALID_THRESHOLD = 10e-5; // Or 1e-5






namespace PointCloudUtils {

    std::string interpolationStatusToString(InterpolationStatus status) {
        switch (status) {
            case InterpolationStatus::SUCCESS: return "SUCCESS";
            case InterpolationStatus::NOT_ENOUGH_NEIGHBORS: return "NOT_ENOUGH_NEIGHBORS";
            case InterpolationStatus::NO_VALID_TRIANGLE: return "NO_VALID_TRIANGLE";
            // Add other cases like DEGENERACY if you define them
            default: return "UNKNOWN_STATUS";
        }
    }

    void SphericalProjection(point_soph &p, int depth_index, const M3D &rot, 
      const V3D &transl, const DynObjFilterParams& params, point_soph &p_spherical)
    {
        // Calculate cache index once
        const int cache_idx = depth_index % HASH_PRIM;

        // Check if cache is valid (using the range/depth component)
        if(std::fabs(p.last_vecs.at(cache_idx)[2]) > CACHE_VALID_THRESHOLD) 
        {
            // Cache Hit: Copy cached data to output
            p_spherical.vec = p.last_vecs.at(cache_idx);
            p_spherical.hor_ind = p.last_positions.at(cache_idx)[0];
            p_spherical.ver_ind = p.last_positions.at(cache_idx)[1];
            p_spherical.position = p.last_positions.at(cache_idx)[2];
        }
        else
        {
            // Cache Miss: Project point and calculate spherical coordinates
            V3D p_proj(rot * (p.glob - transl));

            // *** CORRECTION: Use resolution from params struct ***
            p_spherical.GetVec(p_proj, params.hor_resolution_max, params.ver_resolution_max);

            // Update the cache in the input point 'p'
            p.last_vecs.at(cache_idx) = p_spherical.vec;
            p.last_positions.at(cache_idx)[0] = p_spherical.hor_ind;
            p.last_positions.at(cache_idx)[1] = p_spherical.ver_ind;
            p.last_positions.at(cache_idx)[2] = p_spherical.position;
        }
    }

    bool isPointInvalid(const V3D& point, const DynObjFilterParams& params) {
        // Check 1: Too close to the origin (Blind Distance Sphere)
        // Using squaredNorm for efficiency (avoids sqrt)
        if (point.squaredNorm() < (params.blind_dis * params.blind_dis)) {
            return true;
        }

        // Check 2: Inside the optional invalid bounding box near the origin
        if (params.enable_invalid_box_check) {
            if (std::fabs(point.x()) < params.invalid_box_x_half_width &&
                std::fabs(point.y()) < params.invalid_box_y_half_width &&
                std::fabs(point.z()) < params.invalid_box_z_half_width) {
                return true; // Invalid if enabled and inside the box
            }
        }

        // If none of the invalid conditions are met
        return false;
    }

    // Define the boxes for dataset 0 (using the optional struct)
    // Note: Using V3D constructor assuming V3D(x, y, z)
    const std::vector<AABB> self_boxes_dataset0 = {
        { V3D(-1.2, -1.7, -0.65), V3D(-0.4, -1.0, -0.4) },
        { V3D(-1.75, 1.0, -0.75), V3D(-0.85, 1.6, -0.40) },
        { V3D(1.4, -1.3, -0.8),  V3D(1.7, -0.9, -0.6) },
        { V3D(2.45, -0.6, -1.0),  V3D(2.6, -0.45, -0.9) },
        { V3D(2.45, 0.45, -1.0),  V3D(2.6, 0.6, -0.9) }
    };


    bool isSelfPoint(const V3D& local_point, const DynObjFilterParams& params) {
        // Check if the point's X and Y coordinates fall within the rectangular region
        // defined by the parameters. Assuming Z is not used for self-filtering here.
        return (local_point.x() >= params.self_x_b && local_point.x() <= params.self_x_f &&
                local_point.y() >= params.self_y_r && local_point.y() <= params.self_y_l);
        // Note: Uses >= and <= matching the checkOcclusionRelationship implementation.
        // Adjust if strict inequality (>) was intended.
    }

    // Optional helper function implementation
    bool isInsideAABB(const V3D& point, const AABB& box) {
        // Assuming V3D has x(), y(), z() accessors
        return (point.x() > box.min_corner.x() && point.x() < box.max_corner.x() &&
                point.y() > box.min_corner.y() && point.y() < box.max_corner.y() &&
                point.z() > box.min_corner.z() && point.z() < box.max_corner.z());
        // Or using body(0) style if needed:
        // return (point(0) > box.min_corner(0) && point(0) < box.max_corner(0) &&
        //         point(1) > box.min_corner(1) && point(1) < box.max_corner(1) &&
        //         point(2) > box.min_corner(2) && point(2) < box.max_corner(2));
    }

    bool checkVerticalFov(const point_soph& p, const DepthMap& map_info, const DynObjFilterParams& params) {
        bool found_support_down = false;
        bool found_support_up = false;
    
        // --- Input Sanity Checks (Optional but Recommended) ---
        // Ensure hor_ind and ver_ind are within expected ranges based on how they are calculated
        if (p.hor_ind < 0 || p.hor_ind >= MAX_1D || p.ver_ind < 0 || p.ver_ind >= MAX_1D_HALF) {
            // Log an error or warning - point has invalid indices
            // Depending on desired behavior, could return true (treat as isolated) or false
            // Let's assume for now indices are valid if they reach here, but this is a potential improvement.
            // std::cerr << "Warning: checkVerticalFov called with invalid indices: hor="
            //           << p.hor_ind << ", ver=" << p.ver_ind << std::endl;
            return true; // Treat invalid index points as isolated? Or handle differently?
        }
        // Ensure map_info.depth_map has the expected size
        // if (map_info.depth_map.size() != MAX_2D_N) { /* Handle error */ }
    
    
        // --- Define Search Limits ---
        // Clamp the parameter limits to the valid index range [0, MAX_1D_HALF - 1]
        const int search_limit_down = std::max(0, params.pixel_fov_down);
        const int search_limit_up = std::min(MAX_1D_HALF - 1, params.pixel_fov_up);
    
        // --- Check Downwards ---
        // Start from the point's vertical index, go down to the clamped limit
        for (int i = p.ver_ind; i >= search_limit_down; --i) {
            // Calculate the 1D index
            int cur_pos = p.hor_ind * MAX_1D_HALF + i;
    
            // Bounds check for the calculated index (essential)
            if (cur_pos < 0 || cur_pos >= MAX_2D_N) {
                // This shouldn't happen if hor_ind/ver_ind and MAX_1D_HALF are correct,
                // but guards against potential issues.
                continue; // Skip invalid index
            }
    
            // Check for points in the map cell
            // Add check map_info.depth_map[cur_pos] != nullptr if it's a pointer type
            if (!map_info.depth_map[cur_pos].empty()) {
                found_support_down = true;
                break; // Found support, stop searching downwards
            }
        }
    
        // --- Check Upwards ---
        // Start from the point's vertical index, go up to the clamped limit
        for (int i = p.ver_ind; i <= search_limit_up; ++i) {
            // Calculate the 1D index
            int cur_pos = p.hor_ind * MAX_1D_HALF + i;
    
            // Bounds check for the calculated index (essential)
            if (cur_pos < 0 || cur_pos >= MAX_2D_N) {
                continue; // Skip invalid index
            }
    
            // Check for points in the map cell
            if (!map_info.depth_map[cur_pos].empty()) {
                found_support_up = true;
                break; // Found support, stop searching upwards
            }
        }
    
        // --- Return Result ---
        // Return false if support found both ways, true otherwise
        return !(found_support_up && found_support_down);
    }

    std::vector<int> findNeighborCells(
        const point_soph& center_point,
        int hor_range,
        int ver_range,
        bool include_center,
        bool wrap_horizontal)
    {
        std::vector<int> neighbor_indices;
    
        int effective_hor_range = std::max(0, hor_range);
        int effective_ver_range = std::max(0, ver_range);
    
        size_t max_neighbors = static_cast<size_t>(2 * effective_hor_range + 1) *
                               static_cast<size_t>(2 * effective_ver_range + 1);
    
        if (!include_center && max_neighbors > 0) {
            max_neighbors--;
        }
    
        if (max_neighbors > 0) {
             neighbor_indices.reserve(max_neighbors);
        }
    
        // Call the updated forEachNeighborCell
        forEachNeighborCell(center_point, hor_range, ver_range, include_center, wrap_horizontal,
            [&neighbor_indices](int pos) {
                neighbor_indices.push_back(pos);
            }
        );
    
        return neighbor_indices;
    }

    void findNeighborStaticDepthRange(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        float& min_depth,
        float& max_depth)
    {
        // --- Input Sanity Checks (Performed by forEachNeighborCell and within lambda) ---
        // Check if map_info vectors seem ready (basic check)
        if (map_info.depth_map.empty() || map_info.min_depth_static.empty() || map_info.max_depth_static.empty()) {
            std::cerr << "Warning in findNeighborStaticDepthRange: map_info appears uninitialized or empty." << std::endl;
            // Caller's min_depth/max_depth remain unchanged. Consider if an error/exception is better.
            return;
        }
         // Check if map vectors have the expected size
        if (map_info.depth_map.size() != MAX_2D_N ||
            map_info.min_depth_static.size() != MAX_2D_N ||
            map_info.max_depth_static.size() != MAX_2D_N) {
             std::cerr << "Warning in findNeighborStaticDepthRange: map_info vector sizes do not match MAX_2D_N." << std::endl;
             // Decide how critical this is. Maybe proceed cautiously or return.
             // For now, we proceed, but the lambda will implicitly handle out-of-bounds if pos >= size().
        }


        bool first_neighbor_found = false;
        const int search_range = std::max(0, params.checkneighbor_range); // Ensure non-negative range

        // If we use a range of 0, always include the center
        bool should_include_center = (search_range == 0);

        // Use the helper, enabling wrap-around, excluding center
        forEachNeighborCell(p, search_range, search_range, should_include_center, true, // include_center=false, wrap_horizontal=true
            // Lambda captures necessary variables by reference
            [&](int pos) {
                // Check if index is valid for the map vectors (safety check)
                 if (pos < 0 || pos >= map_info.depth_map.size()) {
                     // This check might be redundant if MAX_2D_N is correct and vectors are sized properly,
                     // but adds robustness if map_info could be inconsistent.
                     // std::cerr << "Warning: Neighbor index " << pos << " out of bounds for map_info vectors." << std::endl;
                     return; // Skip this index
                 }

                // Check if the neighbor cell contains any points
                if (!map_info.depth_map[pos].empty()) {
                    // Retrieve pre-computed static depths for this valid neighbor cell
                    // Add bounds check for safety if sizes might mismatch MAX_2D_N
                    float cur_min_depth = (pos < map_info.min_depth_static.size()) ? map_info.min_depth_static[pos] : std::numeric_limits<float>::max();
                    float cur_max_depth = (pos < map_info.max_depth_static.size()) ? map_info.max_depth_static[pos] : 0.0f;


                    if (!first_neighbor_found) {
                        // This is the first valid neighbor, initialize min/max
                        min_depth = cur_min_depth;
                        max_depth = cur_max_depth;
                        first_neighbor_found = true;
                    } else {
                        // Update min/max with subsequent neighbors
                        // Important: Only update if the current neighbor actually provided valid depths
                        // (e.g., min_depth_static might be 0 if no static points were in the cell)
                        // Assuming 0 is not a valid depth for comparison, or adjust logic as needed.
                        // If cur_min_depth > 0: // Example check if 0 means invalid/no static point
                        min_depth = std::min(min_depth, cur_min_depth);
                        max_depth = std::max(max_depth, cur_max_depth);
                    }
                }
            } // End of lambda
        ); // End of forEachNeighborCell call

        // If no neighbors were found, min_depth/max_depth remain unchanged from caller's initialization.
    }

    std::string interpolationNeighborTypeToString(InterpolationNeighborType type) {
        switch (type) {
            case InterpolationNeighborType::ALL_VALID: return "ALL_VALID";
            case InterpolationNeighborType::STATIC_ONLY: return "STATIC_ONLY";
            default: return "UNKNOWN";
        }
    }

    std::vector<V3F> findInterpolationNeighbors(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type)
    {
        std::vector<V3F> neighbors;
        // neighbors.reserve( estimate ); // Consider adding reservation
    
        // --- Input Sanity Checks ---
        if (map_info.depth_map.empty() || map_info.depth_map.size() != MAX_2D_N) {
             std::cerr << "Warning in findInterpolationNeighbors: map_info.depth_map invalid." << std::endl;
             return neighbors;
        }
    
        // *** ADD DEBUG PRINT (Outer) ***
        bool is_bg_point_debug = p.local.x() > 40.0; // Heuristic for this specific test
        if (params.debug_en && is_bg_point_debug && type == InterpolationNeighborType::STATIC_ONLY) {
            std::cout << "[findInterpNeighbors DEBUG BG Point] Center H=" << p.hor_ind << ", V=" << p.ver_ind
                      << " (Depth=" << p.vec(2) << ", LocX="<< p.local.x() <<") Checking MapIdx=" << map_info.map_index
                      << " for Type=" << interpolationNeighborTypeToString(type) << std::endl;
        }
        // *** END DEBUG PRINT ***
    
        // Use the helper, enabling wrap-around, including center
        forEachNeighborCell(p, params.interp_hor_num, params.interp_ver_num, true, true,
            // Lambda captures necessary variables
            [&](int pos) {
                 // Check if index is valid for the map_info.depth_map (safety check)
                 if (pos < 0 || pos >= map_info.depth_map.size()) {
                     return; // Skip invalid index
                 }
    
                // Get the points in the neighbor cell
                const auto& cell_points = map_info.depth_map[pos];
    
                // *** ADD DEBUG PRINT (Inner Setup) ***
                int found_in_cell = 0;
                int valid_for_interp_in_cell = 0;
                int passed_time_filter = 0;
                int passed_angular_filter = 0;
                int passed_label_filter = 0;
                bool is_center_cell = false; // Flag for more detailed printing of center cell
                 if (!cell_points.empty()) {
                    found_in_cell = cell_points.size();
                    int current_h_idx_dbg = pos % MAX_1D;
                    int current_v_idx_dbg = pos / MAX_1D;
                    is_center_cell = (current_h_idx_dbg == p.hor_ind && current_v_idx_dbg == p.ver_ind);
                }
                // *** END DEBUG PRINT (Inner Setup) ***
    
                if (cell_points.empty()) {
                     // *** ADD DEBUG PRINT (Inner - Empty Cell) ***
                     if (params.debug_en && is_bg_point_debug && type == InterpolationNeighborType::STATIC_ONLY) {
                         int current_h_idx = pos % MAX_1D;
                         int current_v_idx = pos / MAX_1D;
                         // Only print if it's a cell within the search range
                         if (std::abs(current_h_idx - p.hor_ind) <= params.interp_hor_num && std::abs(current_v_idx - p.ver_ind) <= params.interp_ver_num) {
                             // Adjust for wrap-around in horizontal check if necessary
                             int hor_diff_abs = std::abs(current_h_idx - p.hor_ind);
                             if (hor_diff_abs > MAX_1D / 2) hor_diff_abs = MAX_1D - hor_diff_abs; // Handle wrap
                             if (hor_diff_abs <= params.interp_hor_num) {
                                std::cout << "[findInterpNeighbors DEBUG BG Point]  Cell H=" << current_h_idx << ", V=" << current_v_idx
                                          << ": Found=0" << std::endl;
                             }
                         }
                     }
                     // *** END DEBUG PRINT ***
                    return; // Skip empty cells
                }
    
                // Iterate through points within that cell
                for (const auto& neighbor_ptr : cell_points) {
                    if (!neighbor_ptr) continue; // Skip null pointers
    
                    // --- Apply original filtering logic ---
                    bool time_ok = false, angular_ok = false, label_ok = false, use_neighbor = false;
    
                    // Filter 1: Time difference check
                    float time_diff = std::fabs(neighbor_ptr->time - p.time);
                    time_ok = (time_diff >= params.frame_dur);
                    if (time_ok) passed_time_filter++; else continue; // Continue if time fails
    
                    // Filter 2: Projection distance check (angular distance)
                    float hor_diff_raw = neighbor_ptr->vec.x() - p.vec.x();
                    float ver_diff = neighbor_ptr->vec.y() - p.vec.y();
                    float hor_diff = hor_diff_raw;
                    if (hor_diff > M_PI) hor_diff -= 2.0f * M_PI;
                    else if (hor_diff < -M_PI) hor_diff += 2.0f * M_PI;
                    angular_ok = (std::fabs(hor_diff) < params.interp_hor_thr && std::fabs(ver_diff) < params.interp_ver_thr);
                     if (angular_ok) passed_angular_filter++; else continue; // Continue if angular fails
    
                    // Filter 3: Neighbor Type check
                    if (type == InterpolationNeighborType::ALL_VALID) {
                        label_ok = true; // Always true for ALL_VALID
                    } else if (type == InterpolationNeighborType::STATIC_ONLY) {
                        label_ok = (neighbor_ptr->dyn == DynObjLabel::STATIC);
                        // *** ADD DEBUG PRINT ON LABEL FAILURE ***
                        if (!label_ok && params.debug_en && is_bg_point_debug) { // is_bg_point_debug is the outer heuristic check (p.local.x() > 40.0)
                            // This block executes ONLY if the STATIC check fails for a background point neighbor search
                            std::cout << "[findInterpNeighbors DEBUG BG Point]    #### LABEL CHECK FAILED ####" << std::endl;
                            std::cout << "[findInterpNeighbors DEBUG BG Point]    Checking against MapIdx=" << map_info.map_index << std::endl;
                            std::cout << "[findInterpNeighbors DEBUG BG Point]    Neighbor Point in Cell H=" << (pos % MAX_1D) << ", V=" << (pos / MAX_1D) << std::endl;
                            std::cout << "[findInterpNeighbors DEBUG BG Point]    Neighbor's Actual Label = " << static_cast<int>(neighbor_ptr->dyn)
                                    << " (Expected STATIC = " << static_cast<int>(DynObjLabel::STATIC) << ")" << std::endl;
                            std::cout << "[findInterpNeighbors DEBUG BG Point]    Neighbor's Time = " << neighbor_ptr->time
                                    << ", Depth = " << neighbor_ptr->vec(2) << std::endl;
                            std::cout << "[findInterpNeighbors DEBUG BG Point]    (Center Point Time = " << p.time
                                    << ", Depth = " << p.vec(2) << ")" << std::endl;

                        }
                        // *** END DEBUG PRINT ***
                        }
                    if (label_ok) passed_label_filter++; else continue; // Continue if label fails
    
                    // If all filters passed
                    use_neighbor = true; // Redundant now, but follows structure
                    neighbors.push_back(neighbor_ptr->vec);
                    valid_for_interp_in_cell++;
    
                } // End loop through points in cell
    
                // *** ADD DEBUG PRINT (Inner Summary) ***
                if (params.debug_en && is_bg_point_debug && type == InterpolationNeighborType::STATIC_ONLY) {
                     int current_h_idx = pos % MAX_1D;
                     int current_v_idx = pos / MAX_1D;
                     std::cout << "[findInterpNeighbors DEBUG BG Point]  Cell H=" << current_h_idx << ", V=" << current_v_idx
                               << ": Found=" << found_in_cell << ", PassTime=" << passed_time_filter
                               << ", PassAng=" << passed_angular_filter << ", PassLabel(STATIC)=" << passed_label_filter
                               << " -> ValidForInterp=" << valid_for_interp_in_cell << std::endl;
                     // Optional: More detail for the center cell if interpolation fails
                     if (is_center_cell && valid_for_interp_in_cell == 0 && found_in_cell > 0) {
                         std::cout << "    Center Cell Detail: Points found but none valid. Check filters." << std::endl;
                         // Could add printout of the first few points in cell_points here if needed
                     }
                }
                // *** END DEBUG PRINT ***
    
            } // End of lambda
        ); // End of forEachNeighborCell call
    
        // *** ADD DEBUG PRINT (Final) ***
         if (params.debug_en && is_bg_point_debug && type == InterpolationNeighborType::STATIC_ONLY) {
            std::cout << "[findInterpNeighbors DEBUG BG Point] Total Valid Neighbors Found For Interpolation: " << neighbors.size() << std::endl;
         }
        // *** END DEBUG PRINT ***
    
        return neighbors;
    }
    

    InterpolationResult computeBarycentricDepth(
        const V2F& target_point_projection,
        const std::vector<V3F>& neighbors,
        const DynObjFilterParams& params)
    {
        // --- DEBUG PRINT: Input ---
        // std::cout << "[computeBarycentricDepth] Target Proj: (" << target_point_projection.x() << ", " << target_point_projection.y() << ")" << std::endl;
        // std::cout << "[computeBarycentricDepth] Received " << neighbors.size() << " neighbors:" << std::endl;
        // for(size_t i = 0; i < neighbors.size(); ++i) {
        //     std::cout << "  Neighbor " << i << ": Proj=(" << neighbors[i].x() << ", " << neighbors[i].y() << "), Depth=" << neighbors[i].z() << std::endl;
        // }
        // --- END DEBUG ---

        if (neighbors.size() < 3) {
            // std::cout << "[computeBarycentricDepth] Failing early: Not enough neighbors." << std::endl;
            return {InterpolationStatus::NOT_ENOUGH_NEIGHBORS, 0.0f};
        }
    
        InterpolationResult result;
        result.status = InterpolationStatus::NO_VALID_TRIANGLE; // Assume failure until success
    
        constexpr float BARY_DEGENERACY_EPSILON = 1e-12f;
        constexpr float BARY_INSIDE_CHECK_EPSILON = 1e-6f;
        constexpr float TWO_PI = 2.0f * M_PI;
    
        for (size_t i = 0; i < neighbors.size(); ++i) {
            for (size_t j = i + 1; j < neighbors.size(); ++j) {
                for (size_t k = j + 1; k < neighbors.size(); ++k) {
                    // Get original 2D projections
                    V2F v0_orig = neighbors[i].head<2>();
                    V2F v1_orig = neighbors[j].head<2>();
                    V2F v2_orig = neighbors[k].head<2>();
    
                    // --- Unwrap Azimuths relative to target ---
                    // Adjust neighbor azimuths to be numerically close to target azimuth
                    auto unwrap_azimuth = [&](float neighbor_az, float target_az) {
                        float diff = neighbor_az - target_az;
                        // If diff is more than PI away, adjust by 2*PI
                        if (diff > M_PI) {
                            neighbor_az -= TWO_PI;
                        } else if (diff <= -M_PI) {
                            neighbor_az += TWO_PI;
                        }
                        return neighbor_az;
                    };
    
                    V2F v0 = v0_orig;
                    v0.x() = unwrap_azimuth(v0.x(), target_point_projection.x());
                    V2F v1 = v1_orig;
                    v1.x() = unwrap_azimuth(v1.x(), target_point_projection.x());
                    V2F v2 = v2_orig;
                    v2.x() = unwrap_azimuth(v2.x(), target_point_projection.x());
                    // Target azimuth does not need unwrapping relative to itself
                    V2F target_unwrapped = target_point_projection;
                    // --- End Unwrap ---
    
                    // --- DEBUG PRINT: Triangle (Unwrapped) ---
                    // std::cout << "[computeBarycentricDepth] Checking triangle (i=" << i << ", j=" << j << ", k=" << k << ")" << std::endl;
                    // std::cout << "  v0 (orig): (" << v0_orig.x() << ", " << v0_orig.y() << ")" << std::endl; // Optional
                    // std::cout << "  v1 (orig): (" << v1_orig.x() << ", " << v1_orig.y() << ")" << std::endl; // Optional
                    // std::cout << "  v2 (orig): (" << v2_orig.x() << ", " << v2_orig.y() << ")" << std::endl; // Optional
                    // std::cout << "  v0 (unwr): (" << v0.x() << ", " << v0.y() << ")" << std::endl;
                    // std::cout << "  v1 (unwr): (" << v1.x() << ", " << v1.y() << ")" << std::endl;
                    // std::cout << "  v2 (unwr): (" << v2.x() << ", " << v2.y() << ")" << std::endl;
                    // std::cout << "  Target   : (" << target_unwrapped.x() << ", " << target_unwrapped.y() << ")" << std::endl;
                    // --- END DEBUG ---
    
                    // Calculate barycentric coordinates using UNWRAPPED v0, v1, v2 and target
                    V2F vec_v0v1 = v1 - v0;
                    V2F vec_v0v2 = v2 - v0;
                    V2F vec_v0p = target_unwrapped - v0;
    
                    float d00 = vec_v0v1.dot(vec_v0v1);
                    float d01 = vec_v0v1.dot(vec_v0v2);
                    float d11 = vec_v0v2.dot(vec_v0v2);
                    float d20 = vec_v0p.dot(vec_v0v1);
                    float d21 = vec_v0p.dot(vec_v0v2);
                    float denom = d00 * d11 - d01 * d01;
    
                    // --- DEBUG PRINT: Calculations (Unwrapped) ---
                    // std::cout << "  (Unwrapped) d00=" << d00 << ", d01=" << d01 << ", d11=" << d11 << ", d20=" << d20 << ", d21=" << d21 << std::endl;
                    // std::cout << "  (Unwrapped) denom=" << denom << " (Degeneracy Epsilon: " << std::scientific << BARY_DEGENERACY_EPSILON << std::fixed << ")" << std::endl;
                    // --- END DEBUG ---
    
                    if (std::fabs(denom) < BARY_DEGENERACY_EPSILON) {
                        // std::cout << "  -> Skipping: Degenerate triangle (denom < " << std::scientific << BARY_DEGENERACY_EPSILON << std::fixed << ")." << std::endl;
                        continue;
                    }
    
                    float v = (d11 * d20 - d01 * d21) / denom;
                    float w = (d00 * d21 - d01 * d20) / denom;
                    float u = 1.0f - v - w;
    
                    // --- DEBUG PRINT: Barycentric Coords (Unwrapped) ---
                    // std::cout << "  (Unwrapped) Barycentric coords: u=" << u << ", v=" << v << ", w=" << w << std::endl;
                    // --- END DEBUG ---
    
                    bool is_inside = (u >= -BARY_INSIDE_CHECK_EPSILON &&
                                      v >= -BARY_INSIDE_CHECK_EPSILON &&
                                      w >= -BARY_INSIDE_CHECK_EPSILON);
    
                    // std::cout << "  (Unwrapped) Inside Check (Tolerance: " << -BARY_INSIDE_CHECK_EPSILON << "): " << (is_inside ? "PASS" : "FAIL") << std::endl;
    
                    if (is_inside) {
                        // Use original neighbor indices (i, j, k) to get correct depths
                        result.depth = u * neighbors[i].z() + v * neighbors[j].z() + w * neighbors[k].z();
                        result.status = InterpolationStatus::SUCCESS;
                        std::cout << "[computeBarycentricDepth] -> SUCCESS! Interpolated depth: " << result.depth << std::endl;
                        return result;
                    }
                } // end k loop
            } // end j loop
        } // end i loop
    
        std::cout << "[computeBarycentricDepth] -> FAIL: No valid triangle found containing the target after checking all combinations." << std::endl;
        return result; // result.status is still NO_VALID_TRIANGLE
    }
    

    InterpolationResult interpolateDepth(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type)
    {
        // 1. Find neighbors using the refactored function
        std::vector<V3F> neighbors = findInterpolationNeighbors(p, map_info, params, type);

        // 2. Compute depth using barycentric coordinates
        //    Need the 2D projection of the target point 'p'
        V2F target_projection = p.vec.head<2>(); // Assuming p.vec holds azimuth, elevation

        InterpolationResult result = computeBarycentricDepth(target_projection, neighbors, params);

        // Optional: Add logic here based on the original code if specific actions
        // need to be taken depending on the InterpolationStatus (e.g., fallback methods).

        return result;
    }  

    std::vector<std::shared_ptr<point_soph>> findPointsInCells( // <-- Return shared_ptr vector
        const std::vector<int>& cell_indices,
        const DepthMap& map_info)
    {
        std::vector<std::shared_ptr<point_soph>> found_points; // <-- Store shared_ptrs
        const size_t map_total_size = map_info.depth_map.size(); // Cache the total size
    
        // --- Optional: Pre-calculate total points for reservation ---
        size_t estimated_total_points = 0;
        for (int cell_index : cell_indices) {
            // Check index validity before accessing
            if (cell_index >= 0 && static_cast<size_t>(cell_index) < map_total_size) {
                // Access the vector of points for the cell and add its size
                estimated_total_points += map_info.depth_map[cell_index].size();
            }
        }
        found_points.reserve(estimated_total_points); // Reserve space
        // --- End Optional Reservation ---
    
    
        // --- Main loop to collect points ---
        for (int cell_index : cell_indices) {
            // Check index validity again (important!)
            if (cell_index >= 0 && static_cast<size_t>(cell_index) < map_total_size) {
                // Get a const reference to the vector of points in the cell
                const auto& points_ptrs_in_cell = map_info.depth_map[cell_index];

                // --- FIX: Iterate through pointers and push_back the POINTER itself ---
                for (const auto& point_ptr : points_ptrs_in_cell) {
                    // Check if the shared_ptr is valid (points to something)
                    if (point_ptr) {
                        // Add the shared_ptr itself to the result vector.
                        // This increments the reference count.
                        found_points.push_back(point_ptr);
                    }
                    // Else: Handle null pointers if necessary
                }
            } else {
                // Optional: Log a warning for invalid indices if desired during debugging
                // std::cerr << "[findPointsInCells] Warning: Skipping invalid cell index " << cell_index
                //           << " (Map size: " << map_total_size << ")" << std::endl;
            }
        }
    
        return found_points;
    }

}