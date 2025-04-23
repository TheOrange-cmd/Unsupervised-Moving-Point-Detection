#include "point_cloud_utils.h"
#include "dyn_obj_datatypes.h"
#include "config_loader.h" // Include to get DynObjFilterParams definition
#include <cmath>      // For fabs, pow, sqrt, atan2f
#include <vector>     // Used internally by interpolation functions
#include <algorithm>  // For std::min, std::max

constexpr double CACHE_VALID_THRESHOLD = 10e-5; // Or 1e-5

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



}