// test_cpp/common/test_environment.h
#ifndef TEST_ENVIRONMENT_H
#define TEST_ENVIRONMENT_H

#include <gtest/gtest.h>

class GlobalTestEnvironment : public ::testing::Environment {
public:
    ~GlobalTestEnvironment() override {}

    // Override this to define how to set up the environment.
    void SetUp() override;

    // Override this to define how to tear down the environment.
    // Optional: Can be used for cleanup like spdlog::shutdown() if needed.
    void TearDown() override;
};

#endif // TEST_ENVIRONMENT_H