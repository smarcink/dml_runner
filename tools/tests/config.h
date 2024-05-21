#pragma once
#include <cstdint>

struct TestConfig
{
    bool run_dml = false;
    std::uint32_t iterations = 1;
};

inline TestConfig g_test_config;