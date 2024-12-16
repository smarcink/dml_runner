#include "ai_driver.h"
#include "impl/gpu_context.h"


AI_DRIVER_API ai_driver_context_handle_t aiDriverCreateContext(ai_driver_device_t device, ai_driver_context_callbacks_t callbacks)
{
    try {
        auto ctx = new ai_driver::GpuContext(device, callbacks);
        return reinterpret_cast<ai_driver_context_handle_t>(ctx);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "exception: " << ex.what() << '\n';
    }
    catch (...) {
        std::cerr << "unknown exception!\n";
    }
    return nullptr;
}

AI_DRIVER_API void aiDriverDestroyContext(ai_driver_context_handle_t ctx)
{
    auto typed_ctx = reinterpret_cast<ai_driver::GpuContext*>(ctx);
    delete typed_ctx;
}

