// file: common/dyn_obj_datatypes.h

#ifndef DYN_OBJ_DATATYPES_H
#define DYN_OBJ_DATATYPES_H

/**
 * @file dyn_obj_datatypes.h
 * @brief Defines core data structures, constants, and enums used for dynamic object filtering.
 *
 * This includes the sophisticated point representation `point_soph` and the `DepthMap`
 * structure used for organizing points in a spherical projection grid.
 */

#include <common/types.h>     // For V3D, V3F, M3D, etc. 

// Standard Library
#include <vector>      // For std::vector (used extensively in DepthMap and DepthMap2D)
#include <array>       // For std::array (used in point_soph)
#include <cmath>       // For math functions (e.g., atan2f, sqrt, pow, floor)
#include <string>      // Potentially needed if any string members exist or for future use
#include <memory>      // For std::shared_ptr (RECOMMENDED replacement for std::shared_ptr)
#include <algorithm>   // For std::fill_n, std::for_each
#include <execution>   // For std::execution::par (if keeping parallel Reset) - Requires C++17
#include <cstddef>     // For nullptr_t (used with nullptr)
#include <iostream>    // For std::cout
#include <limits>      // For numeric_limits
#include <deque>
#include <spdlog/fmt/fmt.h> 

// External Libraries
#include <Eigen/Dense> // Includes Core, Geometry, LU, etc. Or include specific modules if preferred.
#include <omp.h>       // For omp_get_wtime (if keeping timed Reset) - Requires OpenMP flag during compilation
#include <ostream>

#include <spdlog/spdlog.h>
#include "common/logging_setup.h" // Ensure this is included for logger access

// --- Constants for Dynamic Object Filtering, copied as is from original code ---

/** @brief Mathematical constant PI (float precision). */
// #define M_PI (3.141593f)
/** @brief Prime number used potentially for hashing or indexing. */
#define HASH_P 116101
/** @brief Maximum number of points, related to buffer sizing. */
#define MAX_N 100000
/** @brief Total number of cells in the 2D spherical projection grid (MAX_1D * MAX_1D_HALF). */
#define MAX_2D_N       (564393) //1317755(1317755) //(50086) 39489912
/** @brief Maximum number of horizontal bins in the spherical projection (approx. 2*pi / hor_resolution). */
#define MAX_1D         (1257) //(2095) //(317)
/** @brief Maximum number of vertical bins in the spherical projection (approx. pi / ver_resolution). */
#define MAX_1D_HALF    (449) //3142() //158 31416
/** @brief Constant related to depth width, potentially for visualization or indexing. */
#define DEPTH_WIDTH    (80 * 10)
/** @brief Small coefficient, possibly used for scaling or comparisons. */
#define COE_SMALL   1000
/** @brief Number of historical depth maps or frames to keep (related to point_soph::last_depth_interps). */
#define MAP_NUM     17 //30
/** @brief Prime number used as the size for caching arrays in point_soph (last_vecs, last_positions). */
#define HASH_PRIM   19 //37


/**
 * @brief Enumeration defining the classification status of a point regarding dynamic objects.
 */
enum class DynObjLabel { // Renamed from dyn_obj_flg
    STATIC,         ///< Point is considered static.
    APPEARING,      ///< Point likely appeared in free space (formerly CASE1).
    OCCLUDING,      ///< Point is likely occluding background (formerly CASE2).
    DISOCCLUDED,    ///< Point was likely previously occluded (formerly CASE3).
    SELF,           ///< Point belongs to the ego-vehicle or sensor platform.
    UNCERTAIN,      ///< Point's dynamic status is currently uncertain.
    INVALID         ///< Point is considered invalid (e.g., too close, outside FOV).
};

// --- BEGIN {fmt} Formatter Specialization for DynObjLabel ---
template <>
struct fmt::formatter<DynObjLabel> {
    // Presentation format: 's' - short string (e.g., "STA"), default - full string
    char presentation = ' ';

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it != end && (*it == 's')) presentation = *it++;
        if (it != end && *it != '}') throw format_error("invalid format");
        return it;
    }

    template <typename FormatContext>
    auto format(DynObjLabel label, FormatContext& ctx) const -> decltype(ctx.out()) {
        std::string_view name = "UNKNOWN"; // Use string_view for efficiency
        switch (label) {
            // Match YOUR enum definition
            case DynObjLabel::STATIC:     name = (presentation == 's' ? "STA" : "STATIC"); break;
            case DynObjLabel::APPEARING:  name = (presentation == 's' ? "APP" : "APPEARING"); break;
            case DynObjLabel::OCCLUDING:  name = (presentation == 's' ? "OCCG": "OCCLUDING"); break;
            case DynObjLabel::DISOCCLUDED:name = (presentation == 's' ? "DIS" : "DISOCCLUDED"); break;
            case DynObjLabel::SELF:       name = (presentation == 's' ? "SLF" : "SELF"); break;
            case DynObjLabel::UNCERTAIN:  name = (presentation == 's' ? "UNC" : "UNCERTAIN"); break; // Added UNCERTAIN
            case DynObjLabel::INVALID:    name = (presentation == 's' ? "INV" : "INVALID"); break;
            // Removed DYNAMIC, OCCLUDED as they weren't in your final enum definition
            default: {
                // Handle potential unknown values by formatting their underlying int value
                // Note: Requires C++17 for std::to_chars, or use sprintf/stringstream alternatively
                 // Using format_to directly is simpler here
                 return fmt::format_to(ctx.out(), "UNKNOWN({})", static_cast<int>(label));
            }
        }
        return fmt::format_to(ctx.out(), "{}", name);
    }
};
// --- END {fmt} Formatter Specialization ---


// --- REMOVE after refactoring code to use logger everywhere ---
/**
 * @brief Overload the stream insertion operator for DynObjLabel for easy printing.
 * @param os The output stream.
 * @param label The DynObjLabel value to print.
 * @return The output stream.
 */
inline std::ostream& operator<<(std::ostream& os, DynObjLabel label) {
    // IMPORTANT: Keep this logic IDENTICAL to the fmt::formatter logic
    switch (label) {
        case DynObjLabel::STATIC:     os << "STATIC"; break;
        case DynObjLabel::APPEARING:  os << "APPEARING"; break;
        case DynObjLabel::OCCLUDING:  os << "OCCLUDING"; break;
        case DynObjLabel::DISOCCLUDED: os << "DISOCCLUDED"; break;
        case DynObjLabel::SELF:       os << "SELF"; break;
        case DynObjLabel::UNCERTAIN:  os << "UNCERTAIN"; break;
        case DynObjLabel::INVALID:    os << "INVALID"; break;
        default:
            os << "UNKNOWN(" << static_cast<int>(label) << ")"; break;
    }
    return os;
}
// --- End of optional operator<< ---

// --- Forward Declaration ---
struct point_soph;

/**
 * @brief A sophisticated point structure holding comprehensive information for dynamic object filtering.
 *
 * This structure stores not only the point's coordinates (global, local, spherical)
 * but also temporal information, occlusion counts, dynamic status, transformation data,
 * interpolation results, and caching structures for efficient recalculation.
 */
struct point_soph
{
    int          hor_ind;                                   ///< Horizontal index in the spherical projection grid.
    V3F          vec;                                       ///< Spherical coordinates (azimuth, elevation, range/depth) as a float vector.
    int          ver_ind;                                   ///< Vertical index in the spherical projection grid.
    int          position;                                  ///< Linearized 1D index in the spherical grid (hor_ind * MAX_1D_HALF + ver_ind).
    int          occu_times;                                ///< Counter for how many times this point has occluded others.
    int          is_occu_times;                             ///< Counter for how many times this point has been occluded.
    Eigen::Vector3i occu_index;                             ///< Index related to the last point this point occluded (-1 if none).
    Eigen::Vector3i is_occu_index;                          ///< Index related to the last point that occluded this point (-1 if none).
    double       time;                                      ///< Timestamp of the point measurement.
    V3F          occ_vec;                                   ///< Spherical coordinates of the last point this point occluded.
    V3F          is_occ_vec;                                ///< Spherical coordinates of the last point that occluded this point.
    M3D          rot;                                       ///< Rotation matrix associated with the point's frame/pose.
    V3D          transl;                                    ///< Translation vector associated with the point's frame/pose.
    DynObjLabel  dyn;                                       ///< Dynamic object classification flag (@see DynObjLabel).
    V3D          glob;                                      ///< Global 3D coordinates of the point (double precision).
    V3D          local;                                     ///< Local 3D coordinates (relative to sensor frame at measurement time).
    V3F          cur_vec;                                   ///< Current spherical coordinates (potentially updated).
    float        intensity;                                 ///< Intensity value of the point (often from LiDAR).
    bool         is_distort;                                ///< Flag indicating if motion distortion correction was applied.
    V3D          last_closest;                              ///< Coordinates of the last closest point found (purpose context-dependent).
    float        raw_curvature;                             ///< Raw value read from input cloud's curvature field (potentially used as distortion marker).
    std::array<float, MAP_NUM> last_depth_interps;          ///< Cached depth interpolation results from previous maps.
    std::array<V3F, HASH_PRIM> last_vecs;                   ///< Cache for spherical coordinates from recent projections. Index using `depth_index % HASH_PRIM`.
    std::array<Eigen::Vector3i, HASH_PRIM> last_positions;  ///< Cache for spherical indices from recent projections. Index using `depth_index % HASH_PRIM`. Stores (hor_ind, ver_ind, position).
    size_t       original_index;                            ///< Index of this point in its original input scan frame.    

    /** @brief Shared pointer type definition for point_soph. */
    typedef std::shared_ptr<point_soph> Ptr; // Consider std::shared_ptr<point_soph>

    /**
     * @brief Constructor initializing from a 3D point and resolution parameters.
     * Calculates initial spherical coordinates and indices.
     * @param point Input 3D point coordinates (V3D).
     * @param hor_resolution_max Maximum horizontal angular resolution (radians).
     * @param ver_resolution_max Maximum vertical angular resolution (radians).
     */
    point_soph(V3D & point, float & hor_resolution_max, float & ver_resolution_max)
        : // Use member initializer list for better practice
          hor_ind(0), ver_ind(0), position(0), occu_times(0), is_occu_times(0),
          occu_index(-1 * Eigen::Vector3i::Ones()), is_occu_index(-1 * Eigen::Vector3i::Ones()),
          time(-1), occ_vec(V3F::Zero()), is_occ_vec(V3F::Zero()),
          rot(M3D::Identity()), transl(V3D::Zero()), dyn(DynObjLabel::UNCERTAIN),
          glob(point), local(V3D::Zero()), // Initialize local properly later if needed
          cur_vec(V3F::Zero()), intensity(0.0f), is_distort(false),
          last_closest(V3D::Zero()), raw_curvature(0.0f),
          original_index(static_cast<size_t>(-1)) // Initialize to invalid index
    {
        GetVec(point, hor_resolution_max, ver_resolution_max); // Calculate vec, hor_ind, ver_ind, position
        last_depth_interps.fill(0.0f);
        last_vecs.fill(V3F::Zero());
        last_positions.fill(Eigen::Vector3i::Zero());
    };

    point_soph()
        : // Use member initializer list
          hor_ind(0), ver_ind(0), position(0), occu_times(0), is_occu_times(0),
          occu_index(-1 * Eigen::Vector3i::Ones()), is_occu_index(-1 * Eigen::Vector3i::Ones()),
          time(-1), vec(V3F::Zero()), occ_vec(V3F::Zero()), is_occ_vec(V3F::Zero()),
          rot(M3D::Identity()), transl(V3D::Zero()), glob(V3D::Zero()), local(V3D::Zero()),
          dyn(DynObjLabel::UNCERTAIN), cur_vec(V3F::Zero()), intensity(0.0f),
          is_distort(false), last_closest(V3D::Zero()), raw_curvature(0.0f),
          original_index(static_cast<size_t>(-1)) // Initialize to invalid index
    {
         last_depth_interps.fill(0.0f);
         last_vecs.fill(V3F::Zero());
         last_positions.fill(Eigen::Vector3i::Zero());
    };

    point_soph(V3F s, int ind1, int ind2, int pos)
        : // Use member initializer list
          hor_ind(ind1), ver_ind(ind2), position(pos), occu_times(0), is_occu_times(0),
          occu_index(-1 * Eigen::Vector3i::Ones()), is_occu_index(-1 * Eigen::Vector3i::Ones()),
          time(-1), vec(s), occ_vec(V3F::Zero()), is_occ_vec(V3F::Zero()),
          rot(M3D::Identity()), transl(V3D::Zero()), glob(V3D::Zero()), local(V3D::Zero()),
          dyn(DynObjLabel::UNCERTAIN), cur_vec(V3F::Zero()), intensity(0.0f),
          is_distort(false), last_closest(V3D::Zero()), raw_curvature(0.0f),
          original_index(static_cast<size_t>(-1)) // Initialize to invalid index
    {
         last_depth_interps.fill(0.0f);
         last_vecs.fill(V3F::Zero());
         last_positions.fill(Eigen::Vector3i::Zero());
    };

    // Default copy constructor and destructor are likely fine now
    point_soph(const point_soph & cur) = default;
    ~point_soph() = default;


    /**
     * @brief Calculates spherical coordinates (vec) and indices (hor_ind, ver_ind, position) from 3D Cartesian coordinates.
     * Handles wrap-around for the horizontal index correctly.
     * @param point Input 3D point coordinates (V3D, assumed to be in the sensor's local frame for this calculation).
     * @param hor_resolution_max Maximum horizontal angular resolution (radians).
     * @param ver_resolution_max Maximum vertical angular resolution (radians).
     */
    void GetVec(const V3D & point, const float & hor_resolution_max, const float & ver_resolution_max)
    {
        // Calculate spherical coordinates
        vec(2)    = float(point.norm()); // Range
        vec(0)    = atan2f(float(point(1)), float(point(0))); // Azimuth [-pi, pi]
        vec(1)    = atan2f(float(point(2)), std::sqrt(std::pow(float(point(0)), 2) + std::pow(float(point(1)), 2))); // Elevation [-pi/2, pi/2]

        // Calculate raw horizontal index based on azimuth [-pi, pi] mapped to [0, 2*pi]
        // Add M_PI to shift range to [0, 2*pi], then divide by resolution.
        int raw_hor_ind = static_cast<int>(std::floor((vec(0) + M_PI) / hor_resolution_max));

        // Calculate the theoretical number of horizontal bins for wrap-around check
        // Use round to get the nearest integer number of bins.
        int num_hor_bins = static_cast<int>(std::round(2.0f * M_PI / hor_resolution_max));

        // Handle wrap-around: If index is exactly at or beyond the expected number of bins
        // (can happen at azimuth near +pi due to floating point precision), wrap it to 0.
        // Also handle potential negative index if azimuth is exactly -pi (floor might give -1).
        // Using modulo is a robust way: (index % N + N) % N
        hor_ind = (raw_hor_ind % num_hor_bins + num_hor_bins) % num_hor_bins;
        // Original logic check:
        // if (raw_hor_ind >= num_hor_bins) {
        //     raw_hor_ind = 0; // Wrap the index back to 0
        // }
        // hor_ind = raw_hor_ind; // Assign potentially corrected index

        // Calculate vertical index based on elevation [-pi/2, pi/2] mapped to [0, pi]
        // Add 0.5 * M_PI to shift range to [0, pi], then divide by resolution.
        ver_ind = static_cast<int>(std::floor((vec(1) + 0.5f * M_PI) / ver_resolution_max));

        // Clamp vertical index to be within valid range [0, MAX_1D_HALF - 1]
        // This prevents out-of-bounds access if elevation is slightly outside [-pi/2, pi/2]
        // or if MAX_1D_HALF is slightly smaller than pi/ver_resolution_max.
        ver_ind = std::max(0, std::min(ver_ind, MAX_1D_HALF - 1));

        // Calculate the final 1D position index using the CORRECTED hor_ind and clamped ver_ind
        position = hor_ind * MAX_1D_HALF + ver_ind;

        // Bounds check for position (optional sanity check)
        // if (position < 0 || position >= MAX_2D_N) {
        //     std::cerr << "Warning: Calculated position out of bounds: " << position << std::endl;
        //     // Handle error? Clamp position?
        //     position = std::max(0, std::min(position, MAX_2D_N - 1));
        // }
    };

    /**
     * @brief Resets occlusion-related counters, indices, vectors, caches, and distortion flag.
     * Typically called when reusing a point_soph object for a new point in a subsequent frame.
     */
    void reset()
    {
        occu_times = is_occu_times = 0;
        occu_index = -1 * Eigen::Vector3i::Ones();
        is_occu_index = -1 * Eigen::Vector3i::Ones();
        occ_vec.setZero();
        is_occ_vec.setZero();
        last_closest.setZero();
        last_depth_interps.fill(0.0f);
        last_vecs.fill(V3F::Zero());
        last_positions.fill(Eigen::Vector3i::Zero());
        is_distort = false;
        raw_curvature = 0.0f;
        // Note: Does not reset time, glob, local, rot, transl, dyn, intensity, vec, indices
    };
};

/**
 * @brief Type definition for the 2D depth map grid.
 * A vector where each element represents a cell in the linearized 1D grid (size MAX_2D_N).
 * Each cell contains a vector of pointers to point_soph objects that fall into that cell.
 */
typedef std::vector<std::vector<point_soph::Ptr>> DepthMap2D;

/**
 * @brief Class representing a single depth map (snapshot) for a specific time or frame.
 *
 * Contains the 2D grid (`depth_map`) storing pointers to points, the associated pose
 * (`project_R`, `project_T`), timestamp (`time`), frame index (`map_index`),
 * pre-computed min/max depth information for static and all points per cell,
 * and potentially a collection of point pointers for memory management.
 */
// class DepthMap
// {
// public:
//     DepthMap2D       depth_map;         ///< The 2D grid storing pointers to points in each cell.
//     double           time;              ///< Timestamp associated with this depth map.
//     int              map_index;         ///< Frame index or identifier for this depth map.
//     M3D              project_R;         ///< Rotation matrix (world to sensor frame) for this map's pose.
//     V3D              project_T;         ///< Translation vector (world to sensor frame) for this map's pose.
//     std::vector<point_soph::Ptr> point_sopth_pointer; ///< Vector holding shared pointers to points (potentially for ownership).
//     int              point_sopth_pointer_count; ///< Counter associated with point_sopth_pointer (usage unclear).
//     std::vector<float> min_depth_static;  ///< Array (size MAX_2D_N) storing minimum depth of STATIC points per cell.
//     std::vector<float> min_depth_all;     ///< Array (size MAX_2D_N) storing minimum depth of ALL points per cell.
//     std::vector<float> max_depth_all;     ///< Array (size MAX_2D_N) storing maximum depth of ALL points per cell.
//     std::vector<float> max_depth_static;  ///< Array (size MAX_2D_N) storing maximum depth of STATIC points per cell.
//     std::vector<int> max_depth_index_all; ///< Array (size MAX_2D_N) storing index within cell vector of the point with max depth.
//     std::vector<int> min_depth_index_all; ///< Array (size MAX_2D_N) storing index within cell vector of the point with min depth.
//     std::vector<int> index_vector;      ///< Helper vector containing indices 0 to MAX_2D_N-1, used for parallel iteration.

//     /** @brief Shared pointer type definition for DepthMap. */
//     typedef std::shared_ptr<DepthMap> Ptr; // Consider std::shared_ptr<DepthMap>

//     /**
//      * @brief Default constructor. Initializes the depth map grid and allocates/initializes depth arrays.
//      * @note Prints "build depth map2" to stdout.
//      */
//     DepthMap() :
//         time(0.0),
//         map_index(-1),
//         project_R(M3D::Identity()),
//         project_T(V3D::Zero()),
//         point_sopth_pointer_count(0)
//         // min_depth_static(nullptr),
//         // min_depth_all(nullptr),
//         // max_depth_all(nullptr),
//         // max_depth_static(nullptr),
//         // max_depth_index_all(nullptr),
//         // min_depth_index_all(nullptr)
//     {
//         // std::cout << "build depth map2\n"; // Consider using a proper logger
//         try {
//             depth_map.assign(MAX_2D_N, std::vector<point_soph::Ptr>());
//             index_vector.resize(MAX_2D_N); // Pre-allocate index vector

//             // min_depth_static = new float[MAX_2D_N];
//             // min_depth_all = new float[MAX_2D_N];
//             // max_depth_all = new float[MAX_2D_N];
//             // max_depth_static = new float[MAX_2D_N];
//             // max_depth_index_all = new int[MAX_2D_N];
//             // min_depth_index_all = new int[MAX_2D_N];

//             min_depth_static.resize(MAX_2D_N);
//             min_depth_all.resize(MAX_2D_N);
//             max_depth_all.resize(MAX_2D_N);
//             max_depth_static.resize(MAX_2D_N);
//             max_depth_index_all.resize(MAX_2D_N);
//             min_depth_index_all.resize(MAX_2D_N);

//             // Initialize allocated arrays
//             std::fill(min_depth_static.begin(), min_depth_static.end(), 0.0f);
//             std::fill(min_depth_all.begin(), min_depth_all.end(), 0.0f); 
//             std::fill(max_depth_all.begin(), max_depth_all.end(), 0.0f);
//             std::fill(max_depth_static.begin(), max_depth_static.end(), 0.0f);
//             std::fill(min_depth_index_all.begin(), min_depth_index_all.end(), -1);
//             std::fill(max_depth_index_all.begin(), max_depth_index_all.end(), -1);

//             // Fill index vector
//             for (int i = 0; i < MAX_2D_N; ++i) {
//                 index_vector[i] = i;
//             }
//         } catch (const std::bad_alloc& e) {
//             std::cerr << "Error: Failed to allocate memory for DepthMap members. " << e.what() << std::endl;
//             // Clean up any partially allocated memory if possible, though RAII helps here
//             // delete[] min_depth_static; min_depth_static = nullptr;
//             // delete[] min_depth_all; min_depth_all = nullptr;
//             // ... etc for other arrays
//             throw; // Re-throw the exception
//         }
//     }

//     /**
//      * @brief Constructor initializing with pose, time, and frame index.
//      * Allocates and initializes the depth map grid and depth arrays.
//      * @param rot Rotation matrix (M3D) for the map's pose.
//      * @param transl Translation vector (V3D) for the map's pose.
//      * @param cur_time Timestamp (double) for the map.
//      * @param frame Frame index (int) for the map.
//      */
//     DepthMap(M3D rot, V3D transl, double cur_time, int frame) :
//         time(cur_time),
//         map_index(frame),
//         project_R(rot),
//         project_T(transl),
//         point_sopth_pointer_count(0)
//         // min_depth_static(nullptr),
//         // min_depth_all(nullptr),
//         // max_depth_all(nullptr),
//         // max_depth_static(nullptr),
//         // max_depth_index_all(nullptr),
//         // min_depth_index_all(nullptr)
//     {
//          try {
//             depth_map.assign(MAX_2D_N, std::vector<point_soph::Ptr>());
//             index_vector.resize(MAX_2D_N);

//             // min_depth_static = new float[MAX_2D_N];
//             // min_depth_all = new float[MAX_2D_N];
//             // max_depth_all = new float[MAX_2D_N];
//             // max_depth_static = new float[MAX_2D_N];
//             // max_depth_index_all = new int[MAX_2D_N];
//             // min_depth_index_all = new int[MAX_2D_N];

//             min_depth_static.resize(MAX_2D_N);
//             min_depth_all.resize(MAX_2D_N);
//             max_depth_all.resize(MAX_2D_N);
//             max_depth_static.resize(MAX_2D_N);
//             max_depth_index_all.resize(MAX_2D_N);
//             min_depth_index_all.resize(MAX_2D_N);

//             // Initialize allocated arrays
//             std::fill(min_depth_static.begin(), min_depth_static.end(), 0.0f);
//             std::fill(min_depth_all.begin(), min_depth_all.end(), 0.0f); 
//             std::fill(max_depth_all.begin(), max_depth_all.end(), 0.0f);
//             std::fill(max_depth_static.begin(), max_depth_static.end(), 0.0f);
//             std::fill(min_depth_index_all.begin(), min_depth_index_all.end(), -1);
//             std::fill(max_depth_index_all.begin(), max_depth_index_all.end(), -1);

//             for (int i = 0; i < MAX_2D_N; ++i) {
//                 index_vector[i] = i;
//             }
//         } catch (const std::bad_alloc& e) {
//             std::cerr << "Error: Failed to allocate memory for DepthMap members. " << e.what() << std::endl;
//             // delete[] min_depth_static; min_depth_static = nullptr;
//             // delete[] min_depth_all; min_depth_all = nullptr;
//             // ... etc
//             throw;
//         }
//     }

//     /**
//      * @brief Copy constructor. Performs a deep copy of the depth arrays.
//      * @param cur The DepthMap object to copy from.
//      * @note The `depth_map` and `point_sopth_pointer` members are copied by value,
//      *       meaning the new DepthMap will contain copies of the vectors, but these
//      *       vectors will still point to the *same* `point_soph` objects as the original.
//      *       This implies shared ownership or careful lifetime management of the `point_soph` objects.
//      */
//     DepthMap(const DepthMap & cur) :
//         depth_map(cur.depth_map), // Copies the vector of vectors (shallow copy of pointers)
//         time(cur.time),
//         map_index(cur.map_index),
//         project_R(cur.project_R),
//         project_T(cur.project_T),
//         point_sopth_pointer(cur.point_sopth_pointer), // Copies the vector of shared_ptrs (increases ref count)
//         point_sopth_pointer_count(cur.point_sopth_pointer_count),
//         min_depth_static(cur.min_depth_static), // Allocate new arrays
//         min_depth_all(cur.min_depth_all),
//         max_depth_all(cur.max_depth_all),
//         max_depth_static(cur.max_depth_static),
//         max_depth_index_all(cur.max_depth_index_all),
//         min_depth_index_all(cur.min_depth_index_all),
//         index_vector(cur.index_vector) // Copy the index vector
//     {
//         // try {
//         //     // min_depth_static = new float[MAX_2D_N];
//         //     // min_depth_all = new float[MAX_2D_N];
//         //     // max_depth_all = new float[MAX_2D_N];
//         //     // max_depth_static = new float[MAX_2D_N];
//         //     // max_depth_index_all = new int[MAX_2D_N];
//         //     // min_depth_index_all = new int[MAX_2D_N];

//         //     // // Deep copy the contents of the depth arrays
//         //     // std::copy_n(cur.min_depth_static, MAX_2D_N, min_depth_static);
//         //     // std::copy_n(cur.max_depth_static, MAX_2D_N, max_depth_static);
//         //     // std::copy_n(cur.min_depth_all, MAX_2D_N, min_depth_all);
//         //     // std::copy_n(cur.max_depth_all, MAX_2D_N, max_depth_all);
//         //     // std::copy_n(cur.max_depth_index_all, MAX_2D_N, max_depth_index_all);
//         //     // std::copy_n(cur.min_depth_index_all, MAX_2D_N, min_depth_index_all);

//         // } catch (const std::bad_alloc& e) {
//         //      std::cerr << "Error: Failed to allocate memory during DepthMap copy construction. " << e.what() << std::endl;
//         //     // delete[] min_depth_static; min_depth_static = nullptr;
//         //     // delete[] min_depth_all; min_depth_all = nullptr;
//         //      // ... etc
//         //     throw;
//         // }
//         // Note: index_vector is already copied by vector's copy constructor
//     }

//     // Rule of 5: If providing copy constructor, also consider copy assignment, move constructor, move assignment.
//     // For simplicity here, we assume default move operations might be okay if needed,
//     // but a proper implementation would handle resource transfer correctly.
//     // DepthMap& operator=(const DepthMap& other) { /* ... implement deep copy ... */ }
//     // DepthMap(DepthMap&& other) noexcept { /* ... implement move ... */ }
//     // DepthMap& operator=(DepthMap&& other) noexcept { /* ... implement move assignment ... */ }


//     /**
//      * @brief Destructor. Deallocates the dynamically allocated depth arrays.
//      */
//     ~DepthMap()
//     {
//         // delete [] min_depth_static;
//         // delete [] min_depth_all;
//         // delete [] max_depth_all;
//         // delete [] max_depth_static;
//         // delete [] max_depth_index_all;
//         // delete [] min_depth_index_all;
//         // Note: point_soph objects pointed to by depth_map and point_sopth_pointer
//         // are NOT deleted here. Their lifetime must be managed elsewhere (e.g., by the owner
//         // who populates point_sopth_pointer or another mechanism).
//     }

//     /**
//      * @brief Resets the DepthMap for reuse, clearing the grid and depth arrays, and updating pose/time.
//      * Uses parallel execution policy for clearing the depth_map grid.
//      * @param rot New rotation matrix (M3D).
//      * @param transl New translation vector (V3D).
//      * @param cur_time New timestamp (double).
//      * @param frame New frame index (int).
//      * @note Requires C++17 for std::execution::par and linking with TBB or another backend.
//      *       Requires OpenMP for omp_get_wtime if the timing code were active.
//      */
//     void Reset(M3D rot, V3D transl, double cur_time, int frame)
//     {
//         time = cur_time;
//         project_R = rot;
//         project_T = transl;
//         map_index = frame;

//         // Clear the vectors within the main depth_map vector in parallel
//         // Ensure index_vector is correctly sized if it wasn't already
//         if (index_vector.size() != MAX_2D_N) {
//             index_vector.resize(MAX_2D_N);
//             for(int i=0; i<MAX_2D_N; ++i) index_vector[i] = i;
//         }
//         // Check if depth_map itself needs resizing (shouldn't if constructed correctly)
//         if (depth_map.size() != MAX_2D_N) {
//             depth_map.assign(MAX_2D_N, std::vector<point_soph::Ptr>());
//         }

//         // double t = omp_get_wtime(); // Timing code (currently commented out)
//         std::for_each(std::execution::par, index_vector.begin(), index_vector.end(), [&](const int &i)
//         {
//             // Ensure index is valid before accessing depth_map
//             // This check is technically redundant if index_vector is always correct [0, MAX_2D_N-1]
//             // and depth_map has size MAX_2D_N, but adds safety.
//             if (i >= 0 && i < depth_map.size()) {
//                  depth_map[i].clear(); // Clear the vector of pointers for this cell
//             }
//         });
//         // double t2 = omp_get_wtime(); // Timing code
//         // printf("clear time %f\n", t2-t); // Consider using logger

//         // Reset depth arrays (check for nullptrs before filling)
//         std::fill(min_depth_static.begin(), min_depth_static.end(), 0.0f);
//         std::fill(min_depth_all.begin(), min_depth_all.end(), 0.0f); 
//         std::fill(max_depth_all.begin(), max_depth_all.end(), 0.0f);
//         std::fill(max_depth_static.begin(), max_depth_static.end(), 0.0f);
//         std::fill(min_depth_index_all.begin(), min_depth_index_all.end(), -1);
//         std::fill(max_depth_index_all.begin(), max_depth_index_all.end(), -1);

//         // Reset point pointer vector and count (optional, depends on usage)
//         // point_sopth_pointer.clear();
//         // point_sopth_pointer_count = 0;
//     }
// };

class DepthMap
{
public:
    // --- Pose Information ---
    // Store both Sensor->World and World->Sensor transforms explicitly
    M3D sensor_R;             ///< Rotation matrix (Sensor -> World) for this map's pose.
    V3D sensor_T;             ///< Translation vector (Sensor -> World) for this map's pose.
    M3D project_R;            ///< Rotation matrix (World -> Sensor) calculated from sensor_R.
    V3D project_T;            ///< Translation vector (World -> Sensor) calculated from sensor_R/T.

    // --- Other Members ---
    DepthMap2D       depth_map;
    double           time;
    int              map_index;
    std::vector<point_soph::Ptr> point_sopth_pointer;
    int              point_sopth_pointer_count;
    std::vector<float> min_depth_static;
    std::vector<float> min_depth_all;
    std::vector<float> max_depth_all;
    std::vector<float> max_depth_static;
    std::vector<int> max_depth_index_all;
    std::vector<int> min_depth_index_all;
    std::vector<int> index_vector;

    typedef std::shared_ptr<DepthMap> Ptr;

    // --- Default Constructor ---
    DepthMap() :
        sensor_R(M3D::Identity()), // Initialize new members
        sensor_T(V3D::Zero()),     // Initialize new members
        project_R(M3D::Identity()),
        project_T(V3D::Zero()),
        time(0.0),
        map_index(-1),
        point_sopth_pointer_count(0)
    {
        // std::cout << "build depth map2\n"; // Use logger
        auto logger = spdlog::get("Filter_Map");
        if(logger) logger->trace("DepthMap default constructor called.");

        try {
            depth_map.assign(MAX_2D_N, std::vector<point_soph::Ptr>());
            index_vector.resize(MAX_2D_N);
            min_depth_static.resize(MAX_2D_N);
            min_depth_all.resize(MAX_2D_N);
            max_depth_all.resize(MAX_2D_N);
            max_depth_static.resize(MAX_2D_N);
            max_depth_index_all.resize(MAX_2D_N);
            min_depth_index_all.resize(MAX_2D_N);

            // Initialize allocated arrays
            std::fill(min_depth_static.begin(), min_depth_static.end(), std::numeric_limits<float>::infinity()); // Use infinity
            std::fill(min_depth_all.begin(), min_depth_all.end(), std::numeric_limits<float>::infinity());    // Use infinity
            std::fill(max_depth_all.begin(), max_depth_all.end(), -std::numeric_limits<float>::infinity());   // Use -infinity
            std::fill(max_depth_static.begin(), max_depth_static.end(), -std::numeric_limits<float>::infinity()); // Use -infinity
            std::fill(min_depth_index_all.begin(), min_depth_index_all.end(), -1);
            std::fill(max_depth_index_all.begin(), max_depth_index_all.end(), -1);

            for (int i = 0; i < MAX_2D_N; ++i) { index_vector[i] = i; }
        } catch (const std::bad_alloc& e) {
            if(logger) logger->critical("DepthMap default constructor: Failed memory allocation: {}", e.what());
            else std::cerr << "CRITICAL Error: Failed to allocate memory for DepthMap members. " << e.what() << std::endl;
            throw;
        }
    }

    // --- Constructor with Pose ---
    // Takes Sensor->World pose (R_sw, T_sw) as input
    DepthMap(const M3D& R_sw, const V3D& T_sw, double cur_time, int frame) :
        sensor_R(R_sw),             // Store Sensor->World R
        sensor_T(T_sw),             // Store Sensor->World T
        time(cur_time),
        map_index(frame),
        point_sopth_pointer_count(0)
    {
        auto logger = spdlog::get("Filter_Map");
        if(logger) logger->trace("DepthMap constructor called for Index {}.", frame);

        // --- Calculate and store World->Sensor transform ---
        project_R = sensor_R.transpose();
        project_T = -(project_R * sensor_T);
        // --- END Calculation ---

        // --- ADD LOGGING HERE ---
        if (logger) {
            logger->info("DepthMap Constructor (Index {}): Received sensor_T (Sensor->World): ({:.3f}, {:.3f}, {:.3f})",
                         map_index, sensor_T.x(), sensor_T.y(), sensor_T.z());
            logger->info("DepthMap Constructor (Index {}): Storing project_T (World->Sensor): ({:.3f}, {:.3f}, {:.3f})",
                         map_index, project_T.x(), project_T.y(), project_T.z());
        } else {
             std::cerr << "WARN: Logger 'Filter_Map' not available in DepthMap constructor for Index " << map_index << std::endl;
        }
        // --- END LOGGING ---


        try {
            // --- Initialize vectors (same as default constructor) ---
            depth_map.assign(MAX_2D_N, std::vector<point_soph::Ptr>());
            index_vector.resize(MAX_2D_N);
            min_depth_static.resize(MAX_2D_N);
            min_depth_all.resize(MAX_2D_N);
            max_depth_all.resize(MAX_2D_N);
            max_depth_static.resize(MAX_2D_N);
            max_depth_index_all.resize(MAX_2D_N);
            min_depth_index_all.resize(MAX_2D_N);

            std::fill(min_depth_static.begin(), min_depth_static.end(), std::numeric_limits<float>::infinity());
            std::fill(min_depth_all.begin(), min_depth_all.end(), std::numeric_limits<float>::infinity());
            std::fill(max_depth_all.begin(), max_depth_all.end(), -std::numeric_limits<float>::infinity());
            std::fill(max_depth_static.begin(), max_depth_static.end(), -std::numeric_limits<float>::infinity());
            std::fill(min_depth_index_all.begin(), min_depth_index_all.end(), -1);
            std::fill(max_depth_index_all.begin(), max_depth_index_all.end(), -1);

            for (int i = 0; i < MAX_2D_N; ++i) { index_vector[i] = i; }
            // --- End Initialize vectors ---
        } catch (const std::bad_alloc& e) {
             if(logger) logger->critical("DepthMap constructor (Index {}): Failed memory allocation: {}", map_index, e.what());
             else std::cerr << "CRITICAL Error: Failed to allocate memory for DepthMap members for Index " << map_index << ". " << e.what() << std::endl;
            throw;
        }
    }

    // --- Copy Constructor (Update for new members) ---
    DepthMap(const DepthMap & cur) :
        sensor_R(cur.sensor_R), // Copy new members
        sensor_T(cur.sensor_T), // Copy new members
        project_R(cur.project_R),
        project_T(cur.project_T),
        depth_map(cur.depth_map),
        time(cur.time),
        map_index(cur.map_index),
        point_sopth_pointer(cur.point_sopth_pointer),
        point_sopth_pointer_count(cur.point_sopth_pointer_count),
        min_depth_static(cur.min_depth_static), // Vector copy is fine
        min_depth_all(cur.min_depth_all),
        max_depth_all(cur.max_depth_all),
        max_depth_static(cur.max_depth_static),
        max_depth_index_all(cur.max_depth_index_all),
        min_depth_index_all(cur.min_depth_index_all),
        index_vector(cur.index_vector)
    {
        auto logger = spdlog::get("Filter_Map");
        if(logger) logger->trace("DepthMap copy constructor called for Index {}.", cur.map_index);
        // No deep copy of arrays needed as we switched to std::vector
    }

    // --- Destructor ---
    ~DepthMap() {
        // No manual deletion needed for std::vector members
        auto logger = spdlog::get("Filter_Map");
        // Optional: Log destruction if needed for debugging complex lifetime issues
        // if(logger) logger->trace("DepthMap destructor called for Index {}.", map_index);
    }

    // --- Reset Method ---
    // Takes Sensor->World pose (R_sw, T_sw) as input
    void Reset(const M3D& R_sw, const V3D& T_sw, double cur_time, int frame)
    {
        auto logger = spdlog::get("Filter_Map");
        if(logger) logger->trace("DepthMap Reset called for New Index {}.", frame);

        // --- Update Pose Info ---
        sensor_R = R_sw; // Store Sensor->World R
        sensor_T = T_sw; // Store Sensor->World T
        time = cur_time;
        map_index = frame;

        // --- Calculate and store World->Sensor transform ---
        project_R = sensor_R.transpose();
        project_T = -(project_R * sensor_T);
        // --- END Calculation ---

        // --- ADD LOGGING HERE ---
        if (logger) {
             logger->info("DepthMap Reset (New Index {}): Received sensor_T (Sensor->World): ({:.3f}, {:.3f}, {:.3f})",
                          map_index, sensor_T.x(), sensor_T.y(), sensor_T.z());
             logger->info("DepthMap Reset (New Index {}): Storing project_T (World->Sensor): ({:.3f}, {:.3f}, {:.3f})",
                          map_index, project_T.x(), project_T.y(), project_T.z());
        } else {
             std::cerr << "WARN: Logger 'Filter_Map' not available in DepthMap Reset for New Index " << map_index << std::endl;
        }
        // --- END LOGGING ---


        // --- Clear/Reset Vectors (Parallel part) ---
        if (index_vector.size() != MAX_2D_N) { /* resize if needed */ }
        if (depth_map.size() != MAX_2D_N) { /* assign if needed */ }

        // double t = omp_get_wtime(); // Timing code
        try { // Add try-catch around parallel execution as exceptions might not propagate well
            std::for_each(std::execution::par, index_vector.begin(), index_vector.end(), [&](const int &i)
            {
                if (i >= 0 && i < depth_map.size()) {
                     depth_map[i].clear();
                }
            });
        } catch (const std::exception& e) {
             if(logger) logger->error("DepthMap Reset (Index {}): Exception during parallel clear: {}", map_index, e.what());
             // Consider falling back to sequential clear or rethrowing
        }
        // double t2 = omp_get_wtime(); // Timing code
        // if(logger) logger->trace("DepthMap Reset (Index {}): Parallel clear time: {:.6f}s", map_index, t2-t);


        // --- Reset depth arrays (Sequential part) ---
        std::fill(min_depth_static.begin(), min_depth_static.end(), std::numeric_limits<float>::infinity());
        std::fill(min_depth_all.begin(), min_depth_all.end(), std::numeric_limits<float>::infinity());
        std::fill(max_depth_all.begin(), max_depth_all.end(), -std::numeric_limits<float>::infinity());
        std::fill(max_depth_static.begin(), max_depth_static.end(), -std::numeric_limits<float>::infinity());
        std::fill(min_depth_index_all.begin(), min_depth_index_all.end(), -1);
        std::fill(max_depth_index_all.begin(), max_depth_index_all.end(), -1);

        // Reset point pointer vector and count (optional)
        // point_sopth_pointer.clear();
        // point_sopth_pointer_count = 0;
    }

    // Consider adding Rule of 5: copy assignment, move constructor, move assignment
    // DepthMap& operator=(const DepthMap& other) { /* ... */ return *this; } // Needs careful implementation
    // DepthMap(DepthMap&& other) noexcept { /* ... */ }
    // DepthMap& operator=(DepthMap&& other) noexcept { /* ... */ return *this; }

}; // End class DepthMap


/**
 * @brief Struct to hold information about a processed point for Python bindings.
 */
struct ProcessedPointInfo {
    uint64_t original_index; // Index in the original ScanFrame cloud
    DynObjLabel label;
    float local_x, local_y, local_z;
    float global_x, global_y, global_z;
    float intensity;
    int grid_pos; // The calculated 1D grid position index
    float spherical_azimuth; // vec(0)
    float spherical_elevation; // vec(1)
    float spherical_depth; // vec(2)

    // Default constructor (optional but good practice)
    ProcessedPointInfo() :
        original_index(0), label(DynObjLabel::INVALID),
        local_x(0.0f), local_y(0.0f), local_z(0.0f),
        global_x(0.0f), global_y(0.0f), global_z(0.0f),
        intensity(0.0f), grid_pos(-1),
        spherical_azimuth(0.0f), spherical_elevation(0.0f), spherical_depth(0.0f) {}
};

#endif