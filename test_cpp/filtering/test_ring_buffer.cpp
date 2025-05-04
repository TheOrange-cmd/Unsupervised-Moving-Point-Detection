// file: test/test_ring_buffer.cpp

#include <gtest/gtest.h>
#include "filtering/ring_buffer.h" // Include the header we are testing
#include <vector>
#include <string>   // For testing with strings
#include <memory>   // For std::make_unique, std::unique_ptr
#include <stdexcept> // For std::invalid_argument, std::out_of_range

// --- Test Fixture ---
class RingBufferTest : public ::testing::Test {
protected:
    // Helper to create a dummy ScanFrame with specific ts and seq_id
    ScanFrame create_dummy_frame(double ts, uint64_t id) {
        // No need for actual cloud or complex pose for most buffer tests
        return ScanFrame(ts, nullptr, ScanFrame::PoseType::Identity(), id);
    }
};

// --- Test Cases ---

// Test Construction with Valid and Invalid Capacity (Unchanged)
TEST_F(RingBufferTest, Construction) {
    // Valid capacity
    ASSERT_NO_THROW(RingBuffer<int> buffer(5));
    RingBuffer<int> buffer(5);
    ASSERT_EQ(buffer.capacity(), 5);
    ASSERT_EQ(buffer.size(), 0);
    ASSERT_TRUE(buffer.empty());
    ASSERT_FALSE(buffer.full());

    // Invalid capacity (0) should throw
    ASSERT_THROW(RingBuffer<int> buffer_zero(0), std::invalid_argument);
}

// Test Adding Elements (int) until Full (Unchanged)
TEST_F(RingBufferTest, AddAndGetUntilFull_Int) {
    RingBuffer<int> buffer(3);
    ASSERT_TRUE(buffer.empty());

    buffer.add(10); // Uses add(const T&)
    ASSERT_EQ(buffer.size(), 1);
    ASSERT_FALSE(buffer.empty());
    ASSERT_FALSE(buffer.full());
    ASSERT_EQ(buffer.get(0), 10);
    ASSERT_EQ(buffer.oldest(), 10);
    ASSERT_EQ(buffer.newest(), 10);

    buffer.add(20);
    ASSERT_EQ(buffer.size(), 2);
    ASSERT_FALSE(buffer.full());
    ASSERT_EQ(buffer.get(0), 10); // Oldest
    ASSERT_EQ(buffer.get(1), 20); // Newest
    ASSERT_EQ(buffer.oldest(), 10);
    ASSERT_EQ(buffer.newest(), 20);

    buffer.add(30);
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 10); // Oldest
    ASSERT_EQ(buffer.get(1), 20);
    ASSERT_EQ(buffer.get(2), 30); // Newest
    ASSERT_EQ(buffer.oldest(), 10);
    ASSERT_EQ(buffer.newest(), 30);
}

// Test Wrap-Around Behavior (int) (Unchanged)
TEST_F(RingBufferTest, WrapAround_Int) {
    RingBuffer<int> buffer(3);
    buffer.add(10);
    buffer.add(20);
    buffer.add(30); // Buffer is now [10, 20, 30] (oldest to newest)

    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.oldest(), 10);
    ASSERT_EQ(buffer.newest(), 30);

    // Add 40, should overwrite 10
    buffer.add(40); // Buffer should be [20, 30, 40]
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 20); // Oldest is now 20
    ASSERT_EQ(buffer.get(1), 30);
    ASSERT_EQ(buffer.get(2), 40); // Newest is 40
    ASSERT_EQ(buffer.oldest(), 20);
    ASSERT_EQ(buffer.newest(), 40);

    // Add 50, should overwrite 20
    buffer.add(50); // Buffer should be [30, 40, 50]
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 30); // Oldest is now 30
    ASSERT_EQ(buffer.get(1), 40);
    ASSERT_EQ(buffer.get(2), 50); // Newest is 50
    ASSERT_EQ(buffer.oldest(), 30);
    ASSERT_EQ(buffer.newest(), 50);

    // Add 60, should overwrite 30
    buffer.add(60); // Buffer should be [40, 50, 60]
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 40); // Oldest is now 40
    ASSERT_EQ(buffer.get(1), 50);
    ASSERT_EQ(buffer.get(2), 60); // Newest is 60
    ASSERT_EQ(buffer.oldest(), 40);
    ASSERT_EQ(buffer.newest(), 60);
}

// Test Getting Elements with Invalid Indices (Unchanged)
TEST_F(RingBufferTest, GetOutOfBounds) {
    RingBuffer<int> buffer(3);

    // Test on empty buffer
    ASSERT_THROW(buffer.get(0), std::out_of_range);
    ASSERT_THROW(buffer.oldest(), std::out_of_range);
    ASSERT_THROW(buffer.newest(), std::out_of_range);

    buffer.add(1);
    buffer.add(2); // size is 2

    ASSERT_NO_THROW(buffer.get(0));
    ASSERT_NO_THROW(buffer.get(1));
    ASSERT_THROW(buffer.get(2), std::out_of_range); // Index == size() is invalid
    ASSERT_THROW(buffer.get(3), std::out_of_range);

    buffer.add(3); // size is 3 (full)
    ASSERT_NO_THROW(buffer.get(0));
    ASSERT_NO_THROW(buffer.get(1));
    ASSERT_NO_THROW(buffer.get(2));
    ASSERT_THROW(buffer.get(3), std::out_of_range); // Index == size() is invalid

    buffer.add(4); // wrap around, size still 3
    ASSERT_NO_THROW(buffer.get(0));
    ASSERT_NO_THROW(buffer.get(1));
    ASSERT_NO_THROW(buffer.get(2));
    ASSERT_THROW(buffer.get(3), std::out_of_range);
}

// Test Clear Method (Unchanged)
TEST_F(RingBufferTest, Clear) {
    RingBuffer<int> buffer(4);
    buffer.add(1);
    buffer.add(2);

    ASSERT_EQ(buffer.size(), 2);
    ASSERT_EQ(buffer.capacity(), 4);
    ASSERT_FALSE(buffer.empty());

    buffer.clear();

    ASSERT_EQ(buffer.size(), 0);
    ASSERT_EQ(buffer.capacity(), 4); // Capacity remains unchanged
    ASSERT_TRUE(buffer.empty());
    ASSERT_FALSE(buffer.full());

    // Check if adding works after clearing
    buffer.add(100);
    ASSERT_EQ(buffer.size(), 1);
    ASSERT_EQ(buffer.get(0), 100);
    ASSERT_EQ(buffer.oldest(), 100);
    ASSERT_EQ(buffer.newest(), 100);

    // Clear a full buffer
    buffer.add(200);
    buffer.add(300);
    buffer.add(400);
    ASSERT_TRUE(buffer.full());
    buffer.clear();
    ASSERT_EQ(buffer.size(), 0);
    ASSERT_TRUE(buffer.empty());
}

// Test with a copyable type (std::string) to check add(const T&) (Unchanged)
TEST_F(RingBufferTest, StringType_Copy) {
    RingBuffer<std::string> buffer(2);
    std::string h = "hello";
    std::string w = "world";
    std::string e = "!";

    buffer.add(h); // Copy
    buffer.add(w); // Copy

    ASSERT_EQ(buffer.size(), 2);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), "hello");
    ASSERT_EQ(buffer.get(1), "world");

    buffer.add(e); // Copy, overwrites "hello"

    ASSERT_EQ(buffer.size(), 2);
    ASSERT_EQ(buffer.get(0), "world");
    ASSERT_EQ(buffer.get(1), "!");
    ASSERT_EQ(buffer.oldest(), "world");
    ASSERT_EQ(buffer.newest(), "!");
}

// Test Move Semantics using std::unique_ptr (Unchanged)
TEST_F(RingBufferTest, MoveSemantics_UniquePtr) {
    RingBuffer<std::unique_ptr<int>> buffer(2);

    auto ptr1 = std::make_unique<int>(10);
    auto ptr2 = std::make_unique<int>(20);
    int* p1_addr = ptr1.get(); // Store address for checking later
    int* p2_addr = ptr2.get();

    // Check adding via move
    buffer.add(std::move(ptr1)); // Uses add(T&&)
    ASSERT_EQ(buffer.size(), 1);
    ASSERT_NE(buffer.get(0), nullptr);
    ASSERT_EQ(*buffer.get(0), 10);
    ASSERT_EQ(buffer.get(0).get(), p1_addr); // Ensure it's the same object
    ASSERT_EQ(ptr1, nullptr); // Original pointer should be null after move

    buffer.add(std::move(ptr2));
    ASSERT_EQ(buffer.size(), 2);
    ASSERT_NE(buffer.get(1), nullptr);
    ASSERT_EQ(*buffer.get(1), 20);
    ASSERT_EQ(buffer.get(1).get(), p2_addr);
    ASSERT_EQ(ptr2, nullptr);

    // Check wrap-around with move
    auto ptr3 = std::make_unique<int>(30);
    int* p3_addr = ptr3.get();
    buffer.add(std::move(ptr3)); // Overwrites the first element (originally ptr1)
    ASSERT_EQ(buffer.size(), 2);
    ASSERT_NE(buffer.get(0), nullptr);
    ASSERT_EQ(*buffer.get(0), 20); // Oldest is now the one originally from ptr2
    ASSERT_EQ(buffer.get(0).get(), p2_addr);
    ASSERT_NE(buffer.get(1), nullptr);
    ASSERT_EQ(*buffer.get(1), 30); // Newest is the one from ptr3
    ASSERT_EQ(buffer.get(1).get(), p3_addr);
    ASSERT_EQ(ptr3, nullptr);
}

// Test Resize Method (Unchanged)
TEST_F(RingBufferTest, Resize) {
    RingBuffer<int> buffer(5);
    buffer.add(1);
    buffer.add(2);
    buffer.add(3); // size = 3

    // Resize smaller (data loss expected, state reset)
    ASSERT_NO_THROW(buffer.resize(2));
    ASSERT_EQ(buffer.capacity(), 2);
    ASSERT_EQ(buffer.size(), 0); // Size resets
    ASSERT_TRUE(buffer.empty());
    ASSERT_THROW(buffer.get(0), std::out_of_range); // Old data gone

    // Add to resized buffer
    buffer.add(100);
    buffer.add(200);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 100);
    ASSERT_EQ(buffer.get(1), 200);

    // Resize larger
    ASSERT_NO_THROW(buffer.resize(4));
    ASSERT_EQ(buffer.capacity(), 4);
    ASSERT_EQ(buffer.size(), 0); // Size resets
    ASSERT_TRUE(buffer.empty());

    // Add to larger buffer
    buffer.add(5);
    buffer.add(6);
    ASSERT_EQ(buffer.size(), 2);
    ASSERT_FALSE(buffer.full());
    ASSERT_EQ(buffer.get(0), 5);
    ASSERT_EQ(buffer.get(1), 6);

    // Resize to 0 should throw
    ASSERT_THROW(buffer.resize(0), std::invalid_argument);
}


// Enhanced Test using ScanFrame focusing on move and seq_id
TEST_F(RingBufferTest, ScanFrameSpecifics) {
    RingBuffer<ScanFrame> buffer(3); // Capacity 3

    // Create frames to add
    ScanFrame frame1 = create_dummy_frame(1.0, 101);
    ScanFrame frame2 = create_dummy_frame(2.0, 102);
    ScanFrame frame3 = create_dummy_frame(3.0, 103);
    ScanFrame frame4 = create_dummy_frame(4.0, 104);
    ScanFrame frame5 = create_dummy_frame(5.0, 105);

    // --- Add until full using move ---
    ASSERT_TRUE(buffer.empty());

    buffer.add(std::move(frame1)); // Move frame1 in
    ASSERT_EQ(buffer.size(), 1);
    ASSERT_FALSE(buffer.empty());
    ASSERT_FALSE(buffer.full());
    ASSERT_EQ(buffer.get(0).timestamp, 1.0);
    ASSERT_EQ(buffer.get(0).seq_id, 101);
    ASSERT_EQ(buffer.oldest().seq_id, 101);
    ASSERT_EQ(buffer.newest().seq_id, 101);
    // frame1 is now in a valid but unspecified state after move

    buffer.add(std::move(frame2)); // Move frame2 in
    ASSERT_EQ(buffer.size(), 2);
    ASSERT_FALSE(buffer.full());
    ASSERT_EQ(buffer.get(0).seq_id, 101); // Oldest
    ASSERT_EQ(buffer.get(1).seq_id, 102); // Newest
    ASSERT_EQ(buffer.oldest().seq_id, 101);
    ASSERT_EQ(buffer.newest().seq_id, 102);

    buffer.add(std::move(frame3)); // Move frame3 in
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0).seq_id, 101); // Oldest
    ASSERT_EQ(buffer.get(1).seq_id, 102);
    ASSERT_EQ(buffer.get(2).seq_id, 103); // Newest
    ASSERT_EQ(buffer.oldest().seq_id, 101);
    ASSERT_EQ(buffer.newest().seq_id, 103);

    // --- Test Wrap-around with ScanFrame ---
    buffer.add(std::move(frame4)); // Move frame4 in, should overwrite frame1 (seq_id 101)
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0).seq_id, 102); // Oldest is now frame2
    ASSERT_EQ(buffer.get(1).seq_id, 103);
    ASSERT_EQ(buffer.get(2).seq_id, 104); // Newest is frame4
    ASSERT_EQ(buffer.oldest().seq_id, 102);
    ASSERT_EQ(buffer.newest().seq_id, 104);

    buffer.add(std::move(frame5)); // Move frame5 in, should overwrite frame2 (seq_id 102)
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0).seq_id, 103); // Oldest is now frame3
    ASSERT_EQ(buffer.get(1).seq_id, 104);
    ASSERT_EQ(buffer.get(2).seq_id, 105); // Newest is frame5
    ASSERT_EQ(buffer.oldest().seq_id, 103);
    ASSERT_EQ(buffer.newest().seq_id, 105);

    // --- Test non-const access ---
    // Modify the newest frame's timestamp via non-const get()
    ASSERT_NO_THROW(buffer.get(2).timestamp = 5.5);
    ASSERT_EQ(buffer.get(2).timestamp, 5.5);
    // Modify the oldest frame's timestamp via non-const oldest()
    ASSERT_NO_THROW(buffer.oldest().timestamp = 3.3);
    ASSERT_EQ(buffer.get(0).timestamp, 3.3); // Verify change through get()
    ASSERT_EQ(buffer.oldest().timestamp, 3.3);

    // --- Test clear ---
    buffer.clear();
    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(buffer.size(), 0);
    ASSERT_THROW(buffer.oldest(), std::out_of_range); // Verify empty after clear
}

// Test Non-Const Access Methods
TEST_F(RingBufferTest, NonConstAccess) {
    RingBuffer<int> buffer(3);
    buffer.add(10);
    buffer.add(20);
    buffer.add(30); // [10, 20, 30]

    // Modify via non-const get()
    ASSERT_NO_THROW(buffer.get(1) = 25); // Modify middle element
    ASSERT_EQ(buffer.get(1), 25);
    ASSERT_EQ(buffer.get(0), 10);
    ASSERT_EQ(buffer.get(2), 30);

    // Modify via non-const oldest()
    ASSERT_NO_THROW(buffer.oldest() = 15); // Modify oldest (index 0)
    ASSERT_EQ(buffer.oldest(), 15);
    ASSERT_EQ(buffer.get(0), 15);

    // Modify via non-const newest()
    ASSERT_NO_THROW(buffer.newest() = 35); // Modify newest (index 2)
    ASSERT_EQ(buffer.newest(), 35);
    ASSERT_EQ(buffer.get(2), 35);

    // Wrap around and modify
    buffer.add(40); // [25, 35, 40], oldest is 25
    ASSERT_EQ(buffer.oldest(), 25);
    ASSERT_NO_THROW(buffer.oldest() = 26);
    ASSERT_EQ(buffer.get(0), 26); // Verify oldest was modified

    ASSERT_EQ(buffer.newest(), 40);
    ASSERT_NO_THROW(buffer.newest() = 41);
    ASSERT_EQ(buffer.get(2), 41); // Verify newest was modified
}