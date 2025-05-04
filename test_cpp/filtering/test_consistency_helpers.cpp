// file: test/filtering/test_consistency_helpers.cpp

#include "test/test_consistency_helpers.hpp"
#include "filtering/consistency_checks.h"
#include "filtering/dyn_obj_datatypes.h"
#include "config/config_loader.h"
#include "point_cloud_utils/point_cloud_utils.h" // For types used internally by consistency_checks

// Define SetUp here
    void ConsistencyChecksTest::SetUp() {
    // Load parameters from the full config file
    try {
        bool loaded = load_config(config_path, params);
        ASSERT_TRUE(loaded) << "Failed to load config file: " << config_path;
        ASSERT_GT(params.hor_resolution_max, 0) << "Horizontal resolution invalid.";
        ASSERT_GT(params.ver_resolution_max, 0) << "Vertical resolution invalid.";
    } catch (const std::exception& e) {
        FAIL() << "Exception during config loading: " << e.what();
    }

    // Initialize/Clear the depth map
    test_map.depth_map.assign(MAX_2D_N, std::vector<std::shared_ptr<point_soph>>());
    test_map.time = 0.0; // Reset timestamp if needed
    // Initialize static min/max if they are used directly or indirectly (optional)
    // test_map.min_depth_static.assign(MAX_2D_N, std::numeric_limits<float>::max());
    // test_map.max_depth_static.assign(MAX_2D_N, 0.0f);
    // ... same for min/max_depth_all ...

    // Initialize a default center point (can be overridden in tests)
    // Place it somewhere away from edges and the self-box by default
    center_point = createTestPointWithIndices(20.0, 5.0, 0.0, params, 0.1); // time=0.1
    center_point.vec(2) = 20.0;
    ASSERT_GE(center_point.position, 0) << "Default center_point has invalid position.";
    ASSERT_LT(center_point.position, MAX_2D_N) << "Default center_point has invalid position.";
}

point_soph ConsistencyChecksTest::addNeighborInRelativeCell(
    const point_soph& target,
    DepthMap& map, // Pass map by reference
    const DynObjFilterParams& params, // Pass params
    int delta_hor_ind, // e.g., -1, 0, 1
    int delta_ver_ind, // e.g., -1, 0, 1
    float depth,
    double time_offset_sec, // Time offset in SECONDS
    DynObjLabel status)
{
    int neighbor_hor_ind = (target.hor_ind + delta_hor_ind + MAX_1D) % MAX_1D;
    int neighbor_ver_ind = target.ver_ind + delta_ver_ind;

    // Clamp vertical index
    neighbor_ver_ind = std::max(0, std::min(MAX_1D_HALF - 1, neighbor_ver_ind));

    float neighbor_az, neighbor_el;
    indicesToSpherical(neighbor_hor_ind, neighbor_ver_ind, params.hor_resolution_max, params.ver_resolution_max, neighbor_az, neighbor_el);

    V3D neighbor_local = sphericalToCartesian(neighbor_az, neighbor_el, depth);

    // Use target time + offset
    double neighbor_time = target.time + time_offset_sec;

    point_soph neighbor_point = createTestPointWithIndices(
        neighbor_local.x(), neighbor_local.y(), neighbor_local.z(),
        params, neighbor_time, status);

    // Debug print in helper
    std::cout << "[addNeighborInRelativeCell] Target H=" << target.hor_ind << ", V=" << target.ver_ind
              << ". Adding neighbor with delta H=" << delta_hor_ind << ", V=" << delta_ver_ind
              << " (Target Cell " << neighbor_hor_ind << "," << neighbor_ver_ind << ")"
              << " at Depth=" << depth << " Time=" << neighbor_time
              << ". Resulting Point H=" << neighbor_point.hor_ind << ", V=" << neighbor_point.ver_ind
              << ", Az=" << neighbor_point.vec(0) << ", El=" << neighbor_point.vec(1) << std::endl;

    // Add the point to the map passed by reference
    addPointToMap(neighbor_point); 

    return neighbor_point;
}


// Helper to create neighbors for interpolation tests
// Creates 3 neighbors guaranteed to be in different cells adjacent to the target
void ConsistencyChecksTest::addInterpolationTriangle(
    const point_soph& target, 
    float depth1, 
    float depth2, 
    float depth3,
    DynObjLabel status, 
    double time_offset) {

    int target_hor_ind = target.hor_ind;
    int target_ver_ind = target.ver_ind;

    // Define neighbor indices in adjacent cells (handle wrapping/clamping)
    // Neighbor 1: Lower-left cell
    int n1_hor_ind = (target_hor_ind - 1 + MAX_1D) % MAX_1D; // Wrap horizontally
    int n1_ver_ind = std::max(0, target_ver_ind - 1);       // Clamp vertically

    // Neighbor 2: Lower-right cell
    int n2_hor_ind = (target_hor_ind + 1) % MAX_1D;         // Wrap horizontally
    int n2_ver_ind = std::max(0, target_ver_ind - 1);       // Clamp vertically (same row as n1)

    // Neighbor 3: Cell directly above
    int n3_hor_ind = target_hor_ind;                        // Same column as target
    int n3_ver_ind = std::min(MAX_1D_HALF - 1, target_ver_ind + 1); // Clamp vertically

    // --- Convert target indices back to approximate spherical angles ---
    float az1, el1, az2, el2, az3, el3;
    indicesToSpherical(n1_hor_ind, n1_ver_ind, params.hor_resolution_max, params.ver_resolution_max, az1, el1);
    indicesToSpherical(n2_hor_ind, n2_ver_ind, params.hor_resolution_max, params.ver_resolution_max, az2, el2);
    indicesToSpherical(n3_hor_ind, n3_ver_ind, params.hor_resolution_max, params.ver_resolution_max, az3, el3);

    // --- Convert spherical back to Cartesian ---
    V3D n1_local = sphericalToCartesian(az1, el1, depth1);
    V3D n2_local = sphericalToCartesian(az2, el2, depth2);
    V3D n3_local = sphericalToCartesian(az3, el3, depth3);

    // --- Create point_soph objects using the calculated local Cartesian coords ---
    // These points *should* now map back to the intended nX_hor_ind, nX_ver_ind
    point_soph n1 = createTestPointWithIndices(n1_local.x(), n1_local.y(), n1_local.z(), params, target.time + time_offset, status);
    point_soph n2 = createTestPointWithIndices(n2_local.x(), n2_local.y(), n2_local.z(), params, target.time + time_offset, status);
    point_soph n3 = createTestPointWithIndices(n3_local.x(), n3_local.y(), n3_local.z(), params, target.time + time_offset, status);

    // --- Add to map ---
    addPointToMap(n1);
    addPointToMap(n2);
    addPointToMap(n3);

    // --- Debug Print (Crucial for this attempt) ---
    std::cout << "Target Pos=" << target.position << " (H:" << target.hor_ind << ", V:" << target.ver_ind << ")" << std::endl;
    std::cout << "Added Neighbors (Cell-based):" << std::endl;
    std::cout << " N1 Target (H:" << n1_hor_ind << ", V:" << n1_ver_ind << ") -> Actual Pos=" << n1.position << " (H:" << n1.hor_ind << ", V:" << n1.ver_ind << ") Proj=(" << n1.vec(0) << "," << n1.vec(1) << ")" << std::endl;
    std::cout << " N2 Target (H:" << n2_hor_ind << ", V:" << n2_ver_ind << ") -> Actual Pos=" << n2.position << " (H:" << n2.hor_ind << ", V:" << n2.ver_ind << ") Proj=(" << n2.vec(0) << "," << n2.vec(1) << ")" << std::endl;
    std::cout << " N3 Target (H:" << n3_hor_ind << ", V:" << n3_ver_ind << ") -> Actual Pos=" << n3.position << " (H:" << n3.hor_ind << ", V:" << n3.ver_ind << ") Proj=(" << n3.vec(0) << "," << n3.vec(1) << ")" << std::endl;
}

// Define any other complex helpers here