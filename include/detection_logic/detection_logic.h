#include "common/types.h"
#include "filtering/dyn_obj_datatypes.h" // Includes point_soph, DepthMap, Eigen types etc.
#include "config/config_loader.h"     // Include to get DynObjFilterParams definition

namespace DetectionLogic {

    /**
     * @brief Finds the 2D grid indices (vertical, horizontal) of neighboring cells around a point.
     *
     * Calculates the indices of cells within a square neighborhood defined by params.checkneighbor_range
     * centered on the input point's spherical projection indices (p.ver_ind, p.hor_ind).
     * Handles horizontal wrap-around for the azimuth index. Includes the center cell itself.
     *
     * @param p The point containing valid ver_ind and hor_ind.
     * @param params Parameters struct containing map_width, map_height, and checkneighbor_range.
     * @return std::vector<std::pair<int, int>> A list of (ver_ind, hor_ind) pairs for neighbor cells.
     */
    std::vector<std::pair<int, int>> findNeighborCells(
        const point_soph& p,
        const DynObjFilterParams& params);
    
    // ... other DetectionLogic functions ...
    
    } // namespace DetectionLogic