#include "gpu_context.h"

void ai_driver::GpuStream::dispatch_resource_barrier(GpuResource& resource)
{
    assert(G_GPU_CBCS.fn_gpu_stream_resource_barrier != nullptr);

    std::cout << "GpuStream::dispatch_resource_barrier" <<std::endl;
    std::vector<ai_driver_resource_t> rscs(1);
    rscs[0] = resource.get();
    G_GPU_CBCS.fn_gpu_stream_resource_barrier(handle_, rscs.data(), rscs.size());
}

void ai_driver::GpuStream::dispatch_kernel(const GpuKernel& kernel, uint32_t gws[3], uint32_t lws[3])
{
    assert(G_GPU_CBCS.fn_gpu_stream_execute_kernel != nullptr);
    std::cout << "GpuStream::dispatch_kernel" << std::endl;
    G_GPU_CBCS.fn_gpu_stream_execute_kernel(handle_, kernel.get(), gws, lws);
}

