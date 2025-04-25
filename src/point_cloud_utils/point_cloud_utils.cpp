// file: src/point_cloud_utils.cpp

#include "point_cloud_utils/point_cloud_utils.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h" // Include to get DynObjFilterParams definition
#include <cmath>      // For fabs, pow, sqrt, atan2f
#include <vector>     // Used internally by interpolation functions
#include <algorithm>  // For std::min, std::max
#include <iostream>

constexpr double CACHE_VALID_THRESHOLD = 10e-5; // Or 1e-5
// Define a small epsilon for floating-point comparisons in barycentric calculation


namespace PointCloudUtils {

    void SphericalProjection(point_soph &p, int depth_index, const M3D &rot, const V3D &transl, const DynObjFilterParams& params, point_soph &p_spherical)
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

    bool isPointInvalid(const V3D& point, float blind_distance, int dataset_id) {
        // Check 1: Too close to the origin
        // Using squaredNorm for efficiency (avoids sqrt)
        if (point.squaredNorm() < (blind_distance * blind_distance)) {
            return true;
        }
    
        // Check 2: Specific bounding box for dataset 1 (Combined condition)
        if (dataset_id == 1 &&
            std::fabs(point.x()) < 0.1f &&
            std::fabs(point.y()) < 1.0f &&
            std::fabs(point.z()) < 0.1f) { // Z check is part of the main condition
            return true;
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


    bool isSelfPoint(const V3D& point, int dataset_id) { // Mark dyn as unused if keeping it
        if (dataset_id != 0) {
            return false; // This check only applies to dataset 0
        }

        // Check against predefined boxes for dataset 0
        for (const auto& box : self_boxes_dataset0) {
            if (isInsideAABB(point, box)) {
                return true; // Point is inside one of the self-filter boxes
            }
        }

        return false; // Point is not inside any self-filter box for dataset 0
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

    void findNeighborStaticDepthRange(const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        float& min_depth,
        float& max_depth){
        // Ensure range is sensible
        const int n = std::max(0, params.checkneighbor_range);

        // Flag to handle initialization cleanly on the first valid neighbor found
        bool first_neighbor_found = false;

        // --- Input Sanity Checks (Optional but Recommended) ---
        // Check if p has valid indices
        if (p.hor_ind < 0 || p.hor_ind >= MAX_1D || p.ver_ind < 0 || p.ver_ind >= MAX_1D_HALF) {
            // Log error or warning - point has invalid indices
            // The caller's min_depth/max_depth will remain unchanged.
            return;
        }
        // Check if map_info arrays are allocated (though constructor should handle this)
        if (!map_info.depth_map.data() || !map_info.min_depth_static || !map_info.max_depth_static) {
            // Log error or warning - map_info seems invalid
            return;
        }

        for (int i = -n; i <= n; ++i) {
            int neighbor_hor_ind = p.hor_ind + i;
            // --- Skip if neighbor_hor_ind is out of non-wrapped bounds ---
            // (This replicates original behavior; add modulo for wrap-around)
            if (neighbor_hor_ind < 0 || neighbor_hor_ind >= MAX_1D) {
                continue;
            }

            for (int j = -n; j <= n; ++j) {
                int neighbor_ver_ind = p.ver_ind + j;
                // --- Skip if neighbor_ver_ind is out of bounds ---
                if (neighbor_ver_ind < 0 || neighbor_ver_ind >= MAX_1D_HALF) {
                    continue;
                }

                // Calculate the 1D index for the neighbor
                int cur_pos = neighbor_hor_ind * MAX_1D_HALF + neighbor_ver_ind;

                // --- Bounds check for the 1D index (redundant if above checks are done, but safe) ---
                // if (cur_pos < 0 || cur_pos >= MAX_2D_N) {
                //     continue; // Should not happen with the hor/ver checks above
                // }

                // Check if the neighbor cell has points
                // Add check map_info.depth_map[cur_pos] != nullptr if it's a pointer type
                if (!map_info.depth_map[cur_pos].empty()) {
                    // Retrieve pre-computed static depths for this valid neighbor cell
                    float cur_min_depth = map_info.min_depth_static[cur_pos];
                    float cur_max_depth = map_info.max_depth_static[cur_pos];

                    if (!first_neighbor_found) {
                        // This is the first valid neighbor, initialize min/max
                        min_depth = cur_min_depth;
                        max_depth = cur_max_depth;
                        first_neighbor_found = true;
                    } else {
                        // Update min/max with subsequent neighbors
                        min_depth = std::min(min_depth, cur_min_depth);
                        max_depth = std::max(max_depth, cur_max_depth);
                    }
                }
            }
        }
        // If no neighbors were found, min_depth and max_depth retain the values
        // they had when passed into the function by the caller.
    }

    // Reimplementation of the interpolation functions, DepthInterpolationStatic and DepthInterpolationAll

    std::vector<V3F> findInterpolationNeighbors(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type)
    {
        #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
        std::cout << "[FIN] Target p: hor_ind=" << p.hor_ind << ", ver_ind=" << p.ver_ind
                  << ", az=" << p.vec.x() << ", el=" << p.vec.y() << ", time=" << p.time
                  << ", type_req=" << (type == InterpolationNeighborType::STATIC_ONLY ? "STATIC" : "ALL")
                  << std::endl;
        std::cout << "[FIN] Params: interp_h_num=" << params.interp_hor_num << ", interp_v_num=" << params.interp_ver_num
                  << ", interp_h_thr=" << params.interp_hor_thr << ", interp_v_thr=" << params.interp_ver_thr
                  << ", frame_dur=" << params.frame_dur << std::endl;
        #endif
    
        std::vector<V3F> neighbors;
        // Reserve some space heuristically, e.g., based on search area * typical density
        // neighbors.reserve((2 * params.interp_hor_num + 1) * (2 * params.interp_ver_num + 1) * 5);
    
        // Ensure indices of p are valid before proceeding
        if (p.hor_ind < 0 || p.hor_ind >= MAX_1D || p.ver_ind < 0 || p.ver_ind >= MAX_1D_HALF) {
            #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                    std::cout << "[FIN] ERROR: Target point p has invalid indices!" << std::endl;
            #endif
                    return neighbors; // Return empty vector
                }
            
            // Loop through the grid neighborhood defined by interp_hor_num and interp_ver_num
            for (int offset_h = -params.interp_hor_num; offset_h <= params.interp_hor_num; ++offset_h) {
                int neighbor_hor_idx = getWrappedIndex(p.hor_ind, offset_h, MAX_1D);
        
                for (int offset_v = -params.interp_ver_num; offset_v <= params.interp_ver_num; ++offset_v) {
                    // Vertical index does not wrap in the original logic's calculation style
                    int neighbor_ver_idx = p.ver_ind + offset_v;
        
                    // Skip if vertical index is out of bounds
                    if (neighbor_ver_idx < 0 || neighbor_ver_idx >= MAX_1D_HALF) {
                        continue;
                    }
        
                    // Calculate the 1D index for the neighbor cell
                    int neighbor_1d_index = neighbor_hor_idx * MAX_1D_HALF + neighbor_ver_idx;
            
                    #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                // Optional: Print every cell being checked
                                // std::cout << "[FIN] Checking cell: hor_idx=" << neighbor_hor_idx << ", ver_idx=" << neighbor_ver_idx << ", 1d_idx=" << neighbor_1d_index << std::endl;
                    #endif
                    
                                // Bounds check for the 1D index (safety check)
                                if (neighbor_1d_index < 0 || neighbor_1d_index >= MAX_2D_N) {
                    #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                    std::cout << "[FIN] WARNING: Calculated invalid neighbor_1d_index: " << neighbor_1d_index << std::endl;
                    #endif
                                    continue;
                                }
                    
                            // Get the points in the neighbor cell
                            const auto& cell_points = map_info.depth_map[neighbor_1d_index];
                
                            if (cell_points.empty()) { // Check if cell is empty before iterating
                                continue;
                            }
                
                    #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                if (!cell_points.empty()) {
                                    std::cout << "[FIN] Cell [" << neighbor_hor_idx << "," << neighbor_ver_idx << "] (1d=" << neighbor_1d_index << ") has " << cell_points.size() << " points." << std::endl;
                                }
                    #endif
                    
                                // Iterate through points within that cell
                                for (const point_soph* neighbor_ptr : cell_points) {
                                    if (!neighbor_ptr) continue;
                    
                    #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                    std::cout << "[FIN]   Considering neighbor: az=" << neighbor_ptr->vec.x() << ", el=" << neighbor_ptr->vec.y()
                                            << ", depth=" << neighbor_ptr->vec.z() << ", time=" << neighbor_ptr->time << ", dyn=" << neighbor_ptr->dyn << std::endl;
                    #endif
                    
                                    // Filter 1: Time difference check
                                    float time_diff = std::fabs(neighbor_ptr->time - p.time);
                                    if (time_diff < params.frame_dur) {
                    #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                        std::cout << "[FIN]     SKIP: Time diff " << time_diff << " < " << params.frame_dur << std::endl;
                    #endif
                                        continue; // Skip points from the same frame/scan
                                }
                
                                // Filter 2: Projection distance check (using azimuth/elevation directly)
                                float hor_diff_raw = neighbor_ptr->vec.x() - p.vec.x();
                                float ver_diff = neighbor_ptr->vec.y() - p.vec.y();
                
                                // Handle azimuth wrap-around for difference calculation
                                float hor_diff = hor_diff_raw;
                                if (hor_diff > PI_MATH) hor_diff -= 2.0f * PI_MATH;
                                else if (hor_diff < -PI_MATH) hor_diff += 2.0f * PI_MATH;
                
                                float abs_hor_diff = std::fabs(hor_diff);
                                float abs_ver_diff = std::fabs(ver_diff);
                
                                if (abs_hor_diff >= params.interp_hor_thr || abs_ver_diff >= params.interp_ver_thr) {
                #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                    std::cout << "[FIN]     SKIP: Proj dist |hor_diff|=" << abs_hor_diff << " (thr=" << params.interp_hor_thr
                                            << ") or |ver_diff|=" << abs_ver_diff << " (thr=" << params.interp_ver_thr << ")" << std::endl;
                #endif
                                    continue; // Skip points too far in projection space
                                }
                
                                // Filter 3: Neighbor Type check
                                bool use_neighbor = false;
                                if (type == InterpolationNeighborType::ALL_VALID) {
                                    use_neighbor = true;
                                } else if (type == InterpolationNeighborType::STATIC_ONLY) {
                                    if (neighbor_ptr->dyn == STATIC) {
                                        use_neighbor = true;
                                    } else {
                #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                        std::cout << "[FIN]     SKIP: STATIC_ONLY requested, neighbor is dyn=" << neighbor_ptr->dyn << std::endl;
                #endif
                                    }
                                }
                
                                // Add the neighbor's vector (azimuth, elevation, depth) if it passed all filters
                                if (use_neighbor) {
                #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                                    std::cout << "[FIN]     KEEPING neighbor." << std::endl;
                #endif
                                    neighbors.push_back(neighbor_ptr->vec);
                                }
                            } // End loop through points in cell
                        } // End loop vertical offset
                    } // End loop horizontal offset
                
                #ifdef ENABLE_INTERPOLATION_DEBUG_PRINT
                    std::cout << "[FIN] Returning " << neighbors.size() << " neighbors." << std::endl;
                #endif
                    return neighbors;
            }
    
    
    InterpolationResult computeBarycentricDepth(
        const V2F& target_point_projection, // (azimuth, elevation) of point p
        const std::vector<V3F>& neighbors,  // (azimuth, elevation, depth) of neighbors
        const DynObjFilterParams& params)   // Unused currently, but kept for potential future use
    {
        // return InterpolationResult{InterpolationStatus::NOT_ENOUGH_NEIGHBORS, 0.0f};
        // Need at least 3 points to form a triangle
        if (neighbors.size() < 3) {
            return { InterpolationStatus::NOT_ENOUGH_NEIGHBORS, 0.0f };
        }
    
        // Variables to store the best triangle found so far
        const V3F* best_v1 = nullptr;
        const V3F* best_v2 = nullptr;
        const V3F* best_v3 = nullptr;
        float best_bary_u = 0.0f;
        float best_bary_v = 0.0f;
        float best_bary_w = 0.0f;
        float overall_min_dist_sum = std::numeric_limits<float>::max();
        bool overall_found_triangle = false;
    
        // Iterate through all possible combinations of 3 neighbors
        for (size_t i = 0; i < neighbors.size(); ++i) {
            const V3F& v1 = neighbors[i];
            // Calculate offset from target point to v1 in projection space
            float target_offset_x = target_point_projection.x() - v1.x();
            float target_offset_y = target_point_projection.y() - v1.y();
            // Handle azimuth wrap-around for offset
            if (target_offset_x > PI_MATH) target_offset_x -= 2.0f * PI_MATH;
            else if (target_offset_x < -PI_MATH) target_offset_x += 2.0f * PI_MATH;
    
    
            for (size_t j = i + 1; j < neighbors.size(); ++j) {
                const V3F& v2 = neighbors[j];
                // Calculate Manhattan distance from target to v2 projection
                float dx2 = v2.x() - target_point_projection.x();
                float dy2 = v2.y() - target_point_projection.y();
                if (dx2 > PI_MATH) dx2 -= 2.0f * PI_MATH; else if (dx2 < -PI_MATH) dx2 += 2.0f * PI_MATH;
                float dist_to_v2 = std::fabs(dx2) + std::fabs(dy2);
    
                // Optimization: If dist_to_v2 alone is already worse than the best sum found,
                // no combination involving v2 can be better.
                if (dist_to_v2 >= overall_min_dist_sum && overall_found_triangle) {
                     continue;
                }
    
                for (size_t k = j + 1; k < neighbors.size(); ++k) {
                    const V3F& v3 = neighbors[k];
                    // Calculate Manhattan distance from target to v3 projection
                    float dx3 = v3.x() - target_point_projection.x();
                    float dy3 = v3.y() - target_point_projection.y();
                    if (dx3 > PI_MATH) dx3 -= 2.0f * PI_MATH; else if (dx3 < -PI_MATH) dx3 += 2.0f * PI_MATH;
                    float dist_to_v3 = std::fabs(dx3) + std::fabs(dy3);
    
                    float current_dist_sum = dist_to_v2 + dist_to_v3;
    
                    // Check if this combination is potentially better than the best found so far
                    if (current_dist_sum < overall_min_dist_sum) {
                        // Calculate vectors relative to v1 in projection space
                        float v21_x = v2.x() - v1.x();
                        float v21_y = v2.y() - v1.y();
                        float v31_x = v3.x() - v1.x();
                        float v31_y = v3.y() - v1.y();
                        // Handle azimuth wrap-around for relative vectors
                        if (v21_x > PI_MATH) v21_x -= 2.0f * PI_MATH; else if (v21_x < -PI_MATH) v21_x += 2.0f * PI_MATH;
                        if (v31_x > PI_MATH) v31_x -= 2.0f * PI_MATH; else if (v31_x < -PI_MATH) v31_x += 2.0f * PI_MATH;
    
    
                        // Calculate denominator for barycentric coordinates (related to triangle area)
                        float denom = v21_x * v31_y - v31_x * v21_y;
    
                        // Check for collinear points (denominator close to zero)
                        if (std::fabs(denom) > BARYCENTRIC_EPSILON) {
                            // Calculate barycentric coordinates (u for v2, v for v3, w for v1)
                            float bary_u = (target_offset_x * v31_y - target_offset_y * v31_x) / denom;
                            float bary_v = (target_offset_y * v21_x - target_offset_x * v21_y) / denom;
                            float bary_w = 1.0f - bary_u - bary_v;
    
                            // Check if the target point is inside the triangle (or on edge)
                            // Allow slight tolerance due to float precision? Or stick to strict >=0?
                            constexpr float BARY_CHECK_EPSILON = -1e-5f; // Allow slightly negative due to precision
                            if (bary_u >= BARY_CHECK_EPSILON && bary_u <= 1.0f - BARY_CHECK_EPSILON &&
                                bary_v >= BARY_CHECK_EPSILON && bary_v <= 1.0f - BARY_CHECK_EPSILON &&
                                bary_w >= BARY_CHECK_EPSILON && bary_w <= 1.0f - BARY_CHECK_EPSILON)
                            {
                                // Valid triangle enclosing the point found, and it's the best so far
                                overall_min_dist_sum = current_dist_sum;
                                best_v1 = &v1;
                                best_v2 = &v2;
                                best_v3 = &v3;
                                // Clamp coordinates just in case epsilon allowed slight over/undershoot
                                best_bary_u = std::max(0.0f, std::min(1.0f, bary_u));
                                best_bary_v = std::max(0.0f, std::min(1.0f, bary_v));
                                best_bary_w = 1.0f - best_bary_u - best_bary_v; // Recalculate w for consistency
                                overall_found_triangle = true;
                            }
                        } // end if !collinear
                    } // end if potential best
                } // end loop k (v3)
            } // end loop j (v2)
        } // end loop i (v1)
    
        // After checking all combinations, calculate depth if a valid triangle was found
        if (overall_found_triangle && best_v1 && best_v2 && best_v3) {
            // Interpolate the depth (z-coordinate) using barycentric coordinates
            // Assuming vec.z() holds the depth/range
            float interpolated_depth = best_bary_w * best_v1->z() +
                                       best_bary_u * best_v2->z() +
                                       best_bary_v * best_v3->z();
    
            // Basic sanity check on interpolated depth (optional)
            if (interpolated_depth < 0.0f) {
                 // This might indicate an issue if depths should always be positive
                 // std::cerr << "Warning: Negative interpolated depth calculated: " << interpolated_depth << std::endl;
                 // Decide how to handle: return error, clamp to 0, or allow negative?
                 // Let's return failure for now if negative depth is unexpected.
                 return { InterpolationStatus::NO_VALID_TRIANGLE, 0.0f }; // Or a new status?
            }
    
            return { InterpolationStatus::SUCCESS, interpolated_depth };
        } else {
            // No valid triangle enclosing the point was found among the neighbors
            return { InterpolationStatus::NO_VALID_TRIANGLE, 0.0f };
        }
    }
    
    
    InterpolationResult interpolateDepth(
        const point_soph& p,
        const DepthMap& map_info,
        const DynObjFilterParams& params,
        InterpolationNeighborType type)
    {
        // return InterpolationResult{InterpolationStatus::NOT_ENOUGH_NEIGHBORS, 0.0f};
        // Step 1: Find potential neighbors based on type and filters
        std::vector<V3F> neighbors = findInterpolationNeighbors(p, map_info, params, type);
    
        // Step 2: Extract the 2D projection coordinates of the target point p
        // Assuming vec.x() = azimuth, vec.y() = elevation
        V2F target_proj(p.vec.x(), p.vec.y());
    
        // Step 3: Compute depth using barycentric interpolation on the neighbors
        InterpolationResult result = computeBarycentricDepth(target_proj, neighbors, params);
    
        return result;
    }
    


}