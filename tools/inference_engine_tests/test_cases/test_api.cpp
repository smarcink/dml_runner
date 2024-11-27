#include "test_gpu_context.h"
#include <gtest/gtest.h>



TEST(ApiTest, GPU_create_context_0)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());
    EXPECT_TRUE(nullptr != ctx);
    inferenceEngineDestroyContext(ctx);
}


