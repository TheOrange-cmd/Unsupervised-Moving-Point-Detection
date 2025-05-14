/**
 * @file point_validity.cpp
 * @brief Implements utility functions for checking point validity based on proximity, bounding boxes, etc.
*/

#include "point_cloud_utils/point_validity.h"
#include "common/dyn_obj_datatypes.h" // For V3D, AABB
#include "config/config_loader.h"     // For DynObjFilterParams
#include <cmath>                      // For std::fabs
#include <vector>
#include "point_cloud_utils/logging_context.h"
// No spdlog needed here currently, as these are simple checks. Add if debugging is required.

namespace PointCloudUtils {


    bool isPointInvalid(const V3D& point, const DynObjFilterParams& params) {
        auto logger = spdlog::get("Utils_Validity"); // Or "Utils_Validity"
        // Assuming no specific point context needed here, but frame ID might be useful
        uint64_t frame_id = g_current_logging_seq_id; 

        bool invalid_by_blind_dist = false;
        bool invalid_by_box = false;

        if (point.squaredNorm() < (params.blind_dis * params.blind_dis)) {
            invalid_by_blind_dist = true;
        }

        if (!invalid_by_blind_dist && params.enable_invalid_box_check) {
            if (std::fabs(point.x()) < params.invalid_box_x_half_width &&
                std::fabs(point.y()) < params.invalid_box_y_half_width &&
                std::fabs(point.z()) < params.invalid_box_z_half_width) {
                invalid_by_box = true;
            }
        }
        
        bool is_invalid = invalid_by_blind_dist || invalid_by_box;

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[isPointInvalid] FrameID={}: Point({:.3f},{:.3f},{:.3f}). Params: BlindDis={:.2f}, EnableBox={}, BoxXHW={:.2f}, BoxYHW={:.2f}, BoxZHW={:.2f}. Result: Invalid={}, byBlindDist={}, byBox={}",
                          frame_id, point.x(), point.y(), point.z(),
                          params.blind_dis, params.enable_invalid_box_check, params.invalid_box_x_half_width, params.invalid_box_y_half_width, params.invalid_box_z_half_width,
                          is_invalid, invalid_by_blind_dist, invalid_by_box);
        }
        return is_invalid;
    }
 
    // Define the boxes for dataset 0 (KITTI example) - Keep for potential future use
    const std::vector<AABB> self_boxes_dataset0 = {
        { V3D(-1.2, -1.7, -0.65), V3D(-0.4, -1.0, -0.4) },
        { V3D(-1.75, 1.0, -0.75), V3D(-0.85, 1.6, -0.40) },
        { V3D(1.4, -1.3, -0.8),  V3D(1.7, -0.9, -0.6) },
        { V3D(2.45, -0.6, -1.0),  V3D(2.6, -0.45, -0.9) },
        { V3D(2.45, 0.45, -1.0),  V3D(2.6, 0.6, -0.9) }
        // Add boxes for other datasets if needed
    };

    bool isSelfPoint(const V3D& local_point, const DynObjFilterParams& params) {
        auto logger = spdlog::get("Utils_Validity");
        uint64_t frame_id = g_current_logging_seq_id;
        bool is_self = (local_point.x() >= params.self_x_b && local_point.x() <= params.self_x_f &&
                        local_point.y() >= params.self_y_r && local_point.y() <= params.self_y_l);

        if (logger && logger->should_log(spdlog::level::trace)) {
            logger->trace("[isSelfPoint] FrameID={}: LocalPoint({:.3f},{:.3f},{:.3f}). Params: SelfX_B={:.2f}, SelfX_F={:.2f}, SelfY_R={:.2f}, SelfY_L={:.2f}. Result: IsSelf={}",
                          frame_id, local_point.x(), local_point.y(), local_point.z(),
                          params.self_x_b, params.self_x_f, params.self_y_r, params.self_y_l,
                          is_self);
        }
        return is_self;
    }

    // Implementation of the AABB check helper - Keep for potential future use
    bool isInsideAABB(const V3D& point, const AABB& box) {
        // Using strict inequality (>) based on original implementation comment.
        // Change to >= if inclusive boundaries are desired for this helper.
        return (point.x() > box.min_corner.x() && point.x() < box.max_corner.x() &&
                point.y() > box.min_corner.y() && point.y() < box.max_corner.y() &&
                point.z() > box.min_corner.z() && point.z() < box.max_corner.z());
    }
 
 } // namespace PointCloudUtils