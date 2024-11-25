#include "inference_engine.h"

namespace inference_engine
{
    struct DummyGPUContext
    {

    };
}

INFERENCE_ENGINE_API inference_engine_context_handle_t inferenceEngineCreateContext(inference_engine_accelerator_type_t type, inference_engine_context_callbacks_t callbacks)
{
    auto ctx = new inference_engine::DummyGPUContext{};
    return reinterpret_cast<inference_engine_context_handle_t>(ctx);
}

INFERENCE_ENGINE_API void inferenceEngineDestroyContext(inference_engine_context_handle_t ctx)
{
    auto typed_ctx = reinterpret_cast<inference_engine::DummyGPUContext*>(ctx);
    delete typed_ctx;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineGetLastError()
{
    return INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN;
}

