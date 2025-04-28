#include "detection_logic/detection_logic.h" 
#include <vector>
#include <utility> // For std::pair

namespace DetectionLogic {

    // Helper for modular arithmetic (like in PointCloudUtils)
    inline int getWrappedIndex(int base_idx, int offset, int max_dim) {
        int idx = base_idx + offset;
        // Ensure result is always non-negative and less than max_dim
        return (idx % max_dim + max_dim) % max_dim;
    }
    
    
    std::vector<std::pair<int, int>> findNeighborCells(
        const point_soph& p,
        const DynObjFilterParams& params)
    {
        std::vector<std::pair<int, int>> neighbor_indices;
        // Reserve space roughly based on the neighborhood size (2*range+1)^2
        neighbor_indices.reserve((2 * params.checkneighbor_range + 1) * (2 * params.checkneighbor_range + 1));
    
        int center_ver = p.ver_ind;
        int center_hor = p.hor_ind;
    
        // Check if center indices are valid (basic sanity check)
        if (center_ver < 0 || center_ver >= params.map_height ||
            center_hor < 0 || center_hor >= params.map_width) {
            // Log an error or return empty vector if indices are invalid
            // std::cerr << "Warning: Invalid center indices in findNeighborCells: ("
            //           << center_ver << ", " << center_hor << ")" << std::endl;
            return neighbor_indices; // Return empty list
        }
    
    
        for (int dv = -params.checkneighbor_range; dv <= params.checkneighbor_range; ++dv) {
            int current_ver = center_ver + dv;
    
            // Check vertical bounds
            if (current_ver < 0 || current_ver >= params.map_height) {
                continue; // Skip cells outside the vertical map boundaries
            }
    
            for (int dh = -params.checkneighbor_range; dh <= params.checkneighbor_range; ++dh) {
                // Calculate horizontal index with wrap-around
                int current_hor = getWrappedIndex(center_hor, dh, params.map_width);
    
                neighbor_indices.emplace_back(current_ver, current_hor);
            }
        }
    
        return neighbor_indices;
    }

// ... Implementations of other DetectionLogic functions ...

} // namespace DetectionLogic