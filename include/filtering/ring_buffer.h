// file: filtering/ring_buffer.h

#pragma once // Prevent multiple inclusions

// Standard Library Includes needed by RingBuffer and/or ScanFrame
#include <vector>
#include <stdexcept>    // For std::out_of_range
#include <cstddef>      // For size_t
#include <utility>      // For std::move
#include <memory>       // For std::shared_ptr
#include <cstdint>      // For uint64_t

// External Library Includes needed by ScanFrame
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>    // Defines pcl::PointXYZI
#include <Eigen/Core>
#include <Eigen/Geometry>       // For Eigen::Isometry3d


/**
 * @brief Structure to hold data for a single lidar scan, including the point cloud,
 *        timestamp, and sensor pose.
 */
struct ScanFrame {
    // ... Type Aliases ...
    using PointCloudType = pcl::PointCloud<pcl::PointXYZI>;
    using PointCloudPtr = PointCloudType::Ptr;
    using PoseType = Eigen::Isometry3d;

    // --- Member Variables ---
    double timestamp = 0.0;
    PointCloudPtr cloud = nullptr;
    PoseType sensor_pose = PoseType::Identity();
    uint64_t seq_id = 0; // Unique sequence ID for this scan
    // bool processed = false; // Alternative state tracking

    // --- Constructors ---
    ScanFrame() = default;

    // Constructor now includes seq_id
    ScanFrame(double ts, PointCloudPtr pc, const PoseType& pose, uint64_t id) :
        timestamp(ts), cloud(std::move(pc)), sensor_pose(pose), seq_id(id) {}

    // --- Special Member Functions (Move Semantics & Copy Control) ---
    // Allow moving ScanFrames (efficient transfer of ownership, especially for the cloud pointer)
    ScanFrame(ScanFrame&& other) noexcept = default;
    ScanFrame& operator=(ScanFrame&& other) noexcept = default;

    // Prevent copying by default: Copying a ScanFrame would involve deciding
    // whether to do a shallow copy (both ScanFrames point to the *same* cloud)
    // or a deep copy (create a new copy of the potentially large point cloud).
    // Deleting the copy operations forces the user to be explicit if copying is needed.
    ScanFrame(const ScanFrame&) = delete;
    ScanFrame& operator=(const ScanFrame&) = delete;
};


/**
 * @brief A circular buffer (ring buffer) implementation using std::vector as storage.
 *
 * Stores a fixed number of elements of type T. When the buffer is full,
 * adding a new element overwrites the oldest element. Access is provided
 * relative to the age of elements (oldest at index 0).
 *
 * @tparam T The type of elements to store. Should be default constructible,
 *           and ideally movable (MoveConstructible/MoveAssignable) for
 *           efficiency when adding elements.
 */
template <typename T>
class RingBuffer {
public:
    /**
     * @brief Constructs a RingBuffer with a specified capacity.
     * @param capacity The maximum number of elements the buffer can hold. Must be > 0.
     * @throws std::invalid_argument if capacity is 0.
     */
    explicit RingBuffer(size_t capacity) : capacity_(0), head_(0), count_(0) // Initialize members before check
    {
        if (capacity == 0) {
            throw std::invalid_argument("RingBuffer capacity must be greater than 0.");
        }
        capacity_ = capacity;
        // Pre-allocate the vector to the required capacity.
        // Elements will be default-constructed initially.
        buffer_.resize(capacity_); // Use resize to create default-constructed elements
    }

    /**
     * @brief Resizes the buffer.
     * If the new capacity is smaller, elements are discarded.
     * If larger, new elements are default-constructed.
     * Resets the buffer state (head, count).
     * @param new_capacity The new capacity. Must be > 0.
     * @throws std::invalid_argument if new_capacity is 0.
     */
    void resize(size_t new_capacity) {
         if (new_capacity == 0) {
            throw std::invalid_argument("RingBuffer new capacity must be greater than 0.");
        }
        buffer_.resize(new_capacity); // Resize underlying vector
        capacity_ = new_capacity;
        head_ = 0; // Reset state
        count_ = 0;
    }


    /**
     * @brief Adds an item to the buffer using move semantics, potentially overwriting
     *        the oldest item if the buffer is full.
     * @param item The item to add (will be moved from).
     */
    void add(T&& item) {
        buffer_[head_] = std::move(item); // Move item into the buffer slot
        head_ = (head_ + 1) % capacity_;  // Advance head index, wrap around if needed
        if (count_ < capacity_) {
            count_++; // Increase count only if not yet full
        }
        // If count_ == capacity_, we just overwrote the oldest element.
    }

    /**
     * @brief Adds an item to the buffer by copying, potentially overwriting
     *        the oldest item if the buffer is full.
     * @param item The item to add (will be copied).
     */
    void add(const T& item) {
        buffer_[head_] = item; // Copy item into the buffer slot
        head_ = (head_ + 1) % capacity_; // Advance head index, wrap around if needed
        if (count_ < capacity_) {
            count_++; // Increase count only if not yet full
        }
    }

    /**
     * @brief Gets a constant reference to the element at the specified index relative to age.
     * Index 0 refers to the oldest element currently in the buffer.
     * Index size()-1 refers to the newest element currently in the buffer.
     * @param index The relative index (0 <= index < size()).
     * @return Constant reference to the element.
     * @throws std::out_of_range if the index is invalid (index >= size()) or the buffer is empty.
     */
    const T& get(size_t index) const {
        if (empty() || index >= count_) {
            throw std::out_of_range("RingBuffer::get() index out of range or buffer empty.");
        }
        // Calculate the actual index in the underlying std::vector storage.
        size_t internal_index;
        if (full()) {
            // If full, the oldest element is at the current 'head_' position
            // (since head points to the *next* slot to write, which was the oldest).
            // The element at relative index 'i' is 'i' steps *after* head (with wrap-around).
            internal_index = (head_ + index) % capacity_;
        } else {
            // If not full, the oldest element is always at vector index 0.
            // The element at relative index 'i' is simply at vector index 'i'.
            internal_index = index;
        }
        return buffer_[internal_index];
    }

     /**
     * @brief Gets a non-constant reference to the element at the specified index relative to age.
     * Index 0 refers to the oldest element, index size()-1 refers to the newest.
     * Use with caution - modifying historical data might have unintended consequences.
     * @param index The relative index (0 <= index < size()).
     * @return Non-constant reference to the element.
     * @throws std::out_of_range if the index is invalid (index >= size()) or the buffer is empty.
     */
    T& get(size_t index) {
        if (empty() || index >= count_) {
            throw std::out_of_range("RingBuffer::get() index out of range or buffer empty.");
        }
        // Logic is the same as the const version for calculating the index.
        size_t internal_index;
        if (full()) {
            internal_index = (head_ + index) % capacity_;
        } else {
            internal_index = index;
        }
        return buffer_[internal_index];
    }

    /**
     * @brief Gets a constant reference to the newest element added to the buffer.
     * @return Constant reference to the newest element.
     * @throws std::out_of_range if the buffer is empty.
     */
    const T& newest() const {
        if (empty()) {
            throw std::out_of_range("RingBuffer::newest() called on empty buffer.");
        }
        // The newest element is the one *before* the head index (wrapping around if head is 0).
        size_t newest_internal_index = (head_ == 0) ? (capacity_ - 1) : (head_ - 1);
        return buffer_[newest_internal_index];
    }

    /**
     * @brief Gets a non-constant reference to the newest element added to the buffer.
     * @return Non-constant reference to the newest element.
     * @throws std::out_of_range if the buffer is empty.
     */
    T& newest() {
        if (empty()) {
            throw std::out_of_range("RingBuffer::newest() called on empty buffer.");
        }
        size_t newest_internal_index = (head_ == 0) ? (capacity_ - 1) : (head_ - 1);
        return buffer_[newest_internal_index];
    }


    /**
     * @brief Gets a constant reference to the oldest element currently in the buffer.
     * @return Constant reference to the oldest element.
     * @throws std::out_of_range if the buffer is empty.
     */
    const T& oldest() const {
         if (empty()) {
            throw std::out_of_range("RingBuffer::oldest() called on empty buffer.");
        }
        // If full, oldest is at head_. If not full, oldest is at 0.
        size_t oldest_internal_index = full() ? head_ : 0;
        return buffer_[oldest_internal_index];
    }

    /**
     * @brief Gets a non-constant reference to the oldest element currently in the buffer.
     * @return Non-constant reference to the oldest element.
     * @throws std::out_of_range if the buffer is empty.
     */
    T& oldest() {
         if (empty()) {
            throw std::out_of_range("RingBuffer::oldest() called on empty buffer.");
        }
        size_t oldest_internal_index = full() ? head_ : 0;
        return buffer_[oldest_internal_index];
    }


    /**
     * @brief Returns the number of elements currently stored in the buffer.
     * @return The current size (number of elements), between 0 and capacity().
     */
    size_t size() const noexcept {
        return count_;
    }

    /**
     * @brief Returns the maximum number of elements the buffer can hold.
     * @return The capacity.
     */
    size_t capacity() const noexcept {
        return capacity_;
    }

    /**
     * @brief Checks if the buffer is empty (contains no elements).
     * @return true if size() == 0, false otherwise.
     */
    bool empty() const noexcept {
        return count_ == 0;
    }

    /**
     * @brief Checks if the buffer is full (contains capacity() elements).
     * @return true if size() == capacity(), false otherwise.
     */
    bool full() const noexcept {
        return count_ == capacity_;
    }

    /**
     * @brief Removes all elements from the buffer, resetting its state.
     * Resets size to 0 but retains the allocated capacity. Does not explicitly
     * destroy elements (relies on overwriting or vector destruction).
     */
    void clear() noexcept {
        head_ = 0; // Reset head pointer
        count_ = 0; // Reset element count
        // Note: The underlying std::vector `buffer_` still holds its allocated memory
        // and potentially old objects. These objects will be overwritten by future `add`
        // calls or their destructors called when the RingBuffer itself is destroyed.
    }

private:
    std::vector<T> buffer_;    // Underlying storage (vector of T)
    size_t capacity_;          // Maximum number of items buffer can hold
    size_t head_;              // Index within buffer_ where the *next* element will be inserted
    size_t count_;             // Number of valid elements currently in the buffer (<= capacity_)
};