#include "inference_engine.h"
#include "impl/gpu_context.h"
#include <cassert>


INFERENCE_ENGINE_API inference_engine_context_handle_t inferenceEngineCreateContext(inference_engine_device_t device, inference_engine_context_callbacks_t callbacks)
{
    auto ctx = new inference_engine::GpuContext(device, callbacks);
    return reinterpret_cast<inference_engine_context_handle_t>(ctx);
}

INFERENCE_ENGINE_API void inferenceEngineDestroyContext(inference_engine_context_handle_t ctx)
{
    auto typed_ctx = reinterpret_cast<inference_engine::GpuContext*>(ctx);
    delete typed_ctx;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineGetLastError()
{
    return INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN;
}
