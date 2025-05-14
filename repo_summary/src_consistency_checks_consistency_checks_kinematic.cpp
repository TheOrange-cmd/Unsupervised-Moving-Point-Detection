/**
 * @file consistency_checks_kinematic.cpp
 * @brief Implements kinematic consistency checks (acceleration).
 */

 #include "consistency_checks/consistency_checks.h" // Includes function declarations, enums, DynObjFilterParams
 #include "consistency_checks/consistency_checks_utils.h" // For getCaseStringUtil

 #include <cmath>     // For std::fabs
 #include <iomanip>   // For logging output formatting
 #include <stdexcept> // For std::invalid_argument

 // Logging
 #include <spdlog/spdlog.h>

 namespace ConsistencyChecks {

     // --- Acceleration Limit Check ---
     bool checkAccelerationLimit(
         float velocity1,
         float velocity2,
         double time_delta_between_velocity_centers,
         const DynObjFilterParams& params,
         ConsistencyCheckType check_type)
     {
         auto logger = spdlog::get("Consistency");

         // --- Select parameters and case string ---
         float acceleration_threshold;
         const char* case_str = getCaseStringUtil(check_type); // <<--- USE UTILITY

         switch (check_type) {
             case ConsistencyCheckType::CASE2_OCCLUDER_SEARCH:
                 acceleration_threshold = params.acc_thr2;
                 // case_str = "CASE2"; // Set by utility
                 break;
             case ConsistencyCheckType::CASE3_OCCLUDED_SEARCH:
                 acceleration_threshold = params.acc_thr3;
                 // case_str = "CASE3"; // Set by utility
                 break;
             case ConsistencyCheckType::CASE1_FALSE_REJECTION:
             default: // Handle invalid case
                  // Keep error level as is
                  if (logger) logger->error("[AccelCheck {}] ERROR: Invalid check_type ({}) received.", case_str, static_cast<int>(check_type));
                 throw std::invalid_argument("checkAccelerationLimit received an invalid check_type.");
         }

         // Apply should_log for trace
         if (logger && logger->should_log(spdlog::level::trace)) {
             logger->trace("[AccelCheck {}] Inputs: V1={:.4f}, V2={:.4f}, DeltaT={:.4f}",
                          case_str, velocity1, velocity2, time_delta_between_velocity_centers);
             logger->trace("[AccelCheck {}] Param: AccelThr={:.3f}", case_str, acceleration_threshold);
         }

         // Handle negligible time difference
         constexpr double epsilon_time = 1e-6;
         if (time_delta_between_velocity_centers <= epsilon_time) {
             constexpr float epsilon_vel = 1e-4;
             float delta_v_abs = std::fabs(velocity1 - velocity2);
             bool is_consistent = delta_v_abs < epsilon_vel;

             // Apply should_log for trace and debug
             if (logger && logger->should_log(spdlog::level::trace)) {
                 logger->trace("[AccelCheck {}] Near-Zero DeltaT detected (<= {:.1e}). Comparing |V1-V2|={:.4f} vs VelEpsilon={:.1e}. Consistent={}",
                              case_str, epsilon_time, delta_v_abs, epsilon_vel, is_consistent);
             }
             if (logger && logger->should_log(spdlog::level::debug)) {
                 logger->debug("[AccelCheck {}] -> Returning {} (Near-Zero DeltaT)", case_str, is_consistent);
             }
             return is_consistent;
         }

         // Main Check
         float delta_v_abs = std::fabs(velocity1 - velocity2);
         double velocity_change_limit = time_delta_between_velocity_centers * static_cast<double>(acceleration_threshold);
         bool is_consistent = delta_v_abs < velocity_change_limit;

         // Apply should_log for trace and debug
         if (logger && logger->should_log(spdlog::level::trace)) {
             logger->trace("[AccelCheck {}] Main Check: Comparing |V1-V2|={:.4f} vs Limit (DeltaT*AccelThr)={:.4f}. Consistent={}",
                          case_str, delta_v_abs, velocity_change_limit, is_consistent);
         }
         if (logger && logger->should_log(spdlog::level::debug)) {
             logger->debug("[AccelCheck {}] -> Returning {}", case_str, is_consistent);
         }
         return is_consistent;
     }

 } // namespace ConsistencyChecks