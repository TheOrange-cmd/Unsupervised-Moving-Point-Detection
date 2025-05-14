// include/point_cloud_utils/logging_context.h
#ifndef POINT_CLOUD_UTILS_LOGGING_CONTEXT_H
#define POINT_CLOUD_UTILS_LOGGING_CONTEXT_H

#include "common/dyn_obj_datatypes.h" // For point_soph
#include <cstdint>                    // For uint64_t
#include <limits>                     // For numeric_limits

namespace PointCloudUtils {

// --- Thread-Local Storage Variables (Declaration) ---
constexpr uint64_t INVALID_SEQ_ID = std::numeric_limits<uint64_t>::max();
extern thread_local uint64_t g_current_logging_seq_id;
extern thread_local const point_soph* g_current_logging_point_ptr;


// --- RAII Helper Class ---
class LoggingContextSetter {
public:
    LoggingContextSetter(uint64_t seq_id, const point_soph& p);
    ~LoggingContextSetter();

    // Disable copy/move
    LoggingContextSetter(const LoggingContextSetter&) = delete;
    LoggingContextSetter& operator=(const LoggingContextSetter&) = delete;
    LoggingContextSetter(LoggingContextSetter&&) = delete;
    LoggingContextSetter& operator=(LoggingContextSetter&&) = delete;

private:
    uint64_t previous_seq_id_;
    const point_soph* previous_point_ptr_;
    bool context_was_set_;
};

} // namespace PointCloudUtils

#endif // POINT_CLOUD_UTILS_LOGGING_CONTEXT_H