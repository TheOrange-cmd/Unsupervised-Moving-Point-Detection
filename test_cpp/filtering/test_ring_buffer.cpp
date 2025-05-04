#include <gtest/gtest.h>
#include "filtering/ring_buffer.h" // Include the header we are testing
#include <vector>
#include <string> // For testing with strings
#include <memory> // For std::make_unique

// --- Test Fixture (Optional but can be useful) ---
// We don't strictly need one here, but shown for structure
class RingBufferTest : public ::testing::Test {
protected:
    // Per-test-suite set-up.
    static void SetUpTestSuite() {}

    // Per-test-suite tear-down.
    static void TearDownTestSuite() {}

    // Per-test set-up.
    void SetUp() override {
        // Code here will be called immediately before each test
        // (right after constructor)
    }

    // Per-test tear-down.
    void TearDown() override {
        // Code here will be called immediately after each test
        // (right before destructor)
    }
};

// --- Test Cases ---

// Test Construction with Valid and Invalid Capacity
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

// Test Adding Elements until Full
TEST_F(RingBufferTest, AddAndGetUntilFull) {
    RingBuffer<int> buffer(3);

    ASSERT_TRUE(buffer.empty());

    buffer.add(10);
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

// Test Wrap-Around Behavior (Overwriting Oldest)
TEST_F(RingBufferTest, WrapAround) {
    RingBuffer<int> buffer(3);
    buffer.add(10);
    buffer.add(20);
    buffer.add(30); // Buffer is now [10, 20, 30], head points to index 0 (next write)

    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.oldest(), 10);
    ASSERT_EQ(buffer.newest(), 30);

    // Add 40, should overwrite 10
    buffer.add(40); // Buffer should be [40, 20, 30], head points to index 1
    ASSERT_EQ(buffer.size(), 3); // Size remains capacity
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 20); // Oldest is now 20
    ASSERT_EQ(buffer.get(1), 30);
    ASSERT_EQ(buffer.get(2), 40); // Newest is 40
    ASSERT_EQ(buffer.oldest(), 20);
    ASSERT_EQ(buffer.newest(), 40);


    // Add 50, should overwrite 20
    buffer.add(50); // Buffer should be [40, 50, 30], head points to index 2
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 30); // Oldest is now 30
    ASSERT_EQ(buffer.get(1), 40);
    ASSERT_EQ(buffer.get(2), 50); // Newest is 50
    ASSERT_EQ(buffer.oldest(), 30);
    ASSERT_EQ(buffer.newest(), 50);

    // Add 60, should overwrite 30
    buffer.add(60); // Buffer should be [40, 50, 60], head points to index 0
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), 40); // Oldest is now 40
    ASSERT_EQ(buffer.get(1), 50);
    ASSERT_EQ(buffer.get(2), 60); // Newest is 60
    ASSERT_EQ(buffer.oldest(), 40);
    ASSERT_EQ(buffer.newest(), 60);
}

// Test Getting Elements with Invalid Indices
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

// Test Clear Method
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

// Test with a different type (std::string)
TEST_F(RingBufferTest, StringType) {
    RingBuffer<std::string> buffer(2);
    buffer.add("hello");
    buffer.add("world");

    ASSERT_EQ(buffer.size(), 2);
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.get(0), "hello");
    ASSERT_EQ(buffer.get(1), "world");

    buffer.add("!"); // Overwrites "hello"

    ASSERT_EQ(buffer.size(), 2);
    ASSERT_EQ(buffer.get(0), "world");
    ASSERT_EQ(buffer.get(1), "!");
    ASSERT_EQ(buffer.oldest(), "world");
    ASSERT_EQ(buffer.newest(), "!");
}

// Test Move Semantics (using std::unique_ptr as a move-only type proxy)
TEST_F(RingBufferTest, MoveSemantics) {
    RingBuffer<std::unique_ptr<int>> buffer(2);

    auto ptr1 = std::make_unique<int>(10);
    auto ptr2 = std::make_unique<int>(20);

    // Check adding via move
    buffer.add(std::move(ptr1));
    ASSERT_EQ(buffer.size(), 1);
    ASSERT_NE(buffer.get(0), nullptr);
    ASSERT_EQ(*buffer.get(0), 10);
    ASSERT_EQ(ptr1, nullptr); // Original pointer should be null after move

    buffer.add(std::move(ptr2));
    ASSERT_EQ(buffer.size(), 2);
    ASSERT_NE(buffer.get(1), nullptr);
    ASSERT_EQ(*buffer.get(1), 20);
    ASSERT_EQ(ptr2, nullptr);

    // Check wrap-around with move
    auto ptr3 = std::make_unique<int>(30);
    buffer.add(std::move(ptr3)); // Overwrites the first element (originally ptr1)
    ASSERT_EQ(buffer.size(), 2);
    ASSERT_NE(buffer.get(0), nullptr);
    ASSERT_EQ(*buffer.get(0), 20); // Oldest is now the one originally from ptr2
    ASSERT_NE(buffer.get(1), nullptr);
    ASSERT_EQ(*buffer.get(1), 30); // Newest is the one from ptr3
    ASSERT_EQ(ptr3, nullptr);
}

// Test Resize Method
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


// Basic Test using ScanFrame to ensure compilation and basic operation
TEST_F(RingBufferTest, ScanFrameType) {
    RingBuffer<ScanFrame> buffer(3);

    // Create a dummy ScanFrame (using default constructor)
    ScanFrame frame1;
    frame1.timestamp = 1.0;
    // frame1.cloud remains nullptr
    // frame1.sensor_pose remains Identity

    // Add using copy (if ScanFrame had a copy constructor)
    // buffer.add(frame1); // This would fail as copy constructor is deleted

    // Add using move
    buffer.add(std::move(frame1));
    ASSERT_EQ(buffer.size(), 1);
    ASSERT_FALSE(buffer.empty());
    ASSERT_EQ(buffer.get(0).timestamp, 1.0);
    // ASSERT_EQ(frame1.timestamp, 1.0); // State of moved-from object is valid but unspecified (don't rely on it)

    // Add another frame constructed in place (implicitly moved)
    ScanFrame::PoseType pose = ScanFrame::PoseType::Identity(); // Example pose
    ScanFrame::PointCloudPtr cloud_ptr = nullptr; // No cloud data needed for this test
    buffer.add(ScanFrame(2.0, cloud_ptr, pose));

    ASSERT_EQ(buffer.size(), 2);
    ASSERT_EQ(buffer.get(0).timestamp, 1.0); // Oldest
    ASSERT_EQ(buffer.get(1).timestamp, 2.0); // Newest
    ASSERT_EQ(buffer.newest().timestamp, 2.0);
    ASSERT_EQ(buffer.oldest().timestamp, 1.0);

    // Add one more to fill
     buffer.add(ScanFrame(3.0, nullptr, pose));
     ASSERT_TRUE(buffer.full());
     ASSERT_EQ(buffer.get(2).timestamp, 3.0);

    // Add one more to wrap
    buffer.add(ScanFrame(4.0, nullptr, pose));
    ASSERT_TRUE(buffer.full());
    ASSERT_EQ(buffer.size(), 3);
    ASSERT_EQ(buffer.oldest().timestamp, 2.0); // frame1 (ts=1.0) was overwritten
    ASSERT_EQ(buffer.get(0).timestamp, 2.0);
    ASSERT_EQ(buffer.get(1).timestamp, 3.0);
    ASSERT_EQ(buffer.get(2).timestamp, 4.0);
    ASSERT_EQ(buffer.newest().timestamp, 4.0);

    // Test clear
    buffer.clear();
    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(buffer.size(), 0);
}