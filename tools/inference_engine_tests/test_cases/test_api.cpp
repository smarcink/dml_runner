#include <inference_engine.h>
#include <inference_engine_operators.h>

#include <gtest/gtest.h>

static inference_engine_kernel_t create_kernel(const char* kernel_name, const void* kernel_code,
    size_t kernel_code_size, const char* build_options)
{
    std::cout << "Dummy callback example" << std::endl;
    return nullptr;
}

TEST(ApiTest, Basic_0)
{
    inference_engine_context_callbacks_t callbacks{};
    callbacks.fn_gpu_create_kernel = &create_kernel;
    auto h_ctx = inferenceEngineCreateContext(INFERENCE_ENGINE_ACCELERATOR_TYPE_GPU, callbacks);
    EXPECT_TRUE(h_ctx != nullptr);
    inferenceEngineDestroyContext(h_ctx);
}
