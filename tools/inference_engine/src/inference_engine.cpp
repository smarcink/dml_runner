#include "inference_engine.h"
#include "impl/gpu_context.h"
#include "impl/error.h"


INFERENCE_ENGINE_API inference_engine_context_handle_t inferenceEngineCreateContext(inference_engine_device_t device, inference_engine_context_callbacks_t callbacks)
{
    try {
        auto ctx = new inference_engine::GpuContext(device, callbacks);
        return reinterpret_cast<inference_engine_context_handle_t>(ctx);
    }
    catch (const std::bad_alloc&)
    {
        inference_engine::set_last_error(INFERENCE_ENGINE_RESULT_BAD_ALLOC);
    }
    catch (const inference_engine::inference_engine_exception& ex) {
        inference_engine::set_last_error(ex.val_);
    }
    catch (...) {
        inference_engine::set_last_error(INFERENCE_ENGINE_RESULT_OTHER);
    }
    return nullptr;
}

INFERENCE_ENGINE_API void inferenceEngineDestroyContext(inference_engine_context_handle_t ctx)
{
    auto typed_ctx = reinterpret_cast<inference_engine::GpuContext*>(ctx);
    delete typed_ctx;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineGetLastError()
{
    return inference_engine::get_last_error();
}

