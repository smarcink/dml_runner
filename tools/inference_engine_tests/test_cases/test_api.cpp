#include <inference_engine.h>

#include <gtest/gtest.h>


TEST(ApiTest, Basic_0)
{
    inference_engine_context_handle_t h_ctx = nullptr;
    auto result = inferenceEngineCreateContext(h_ctx);

    EXPECT_EQ(result, inference_engine_result_t::XESS_RESULT_SUCCESS);
}
