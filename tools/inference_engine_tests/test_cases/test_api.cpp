#include "test_d3d12_context.h"
#include <gtest/gtest.h>



TEST(ApiTest, GPU_create_context_0)
{
    test_ctx::TestGpuContext ctx;
    EXPECT_TRUE(nullptr != ctx.get());
}


