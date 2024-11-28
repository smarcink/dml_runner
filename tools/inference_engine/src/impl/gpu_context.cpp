#include "gpu_context.h"

void inference_engine::GpuStream::dispatch_resource_barrier(GpuResource& resource)
{
    assert(G_GPU_CBCS.fn_gpu_stream_resource_barrier != nullptr);

    std::cout << "GpuStream::dispatch_resource_barrier" <<std::endl;
    std::vector<inference_engine_resource_t> rscs(1);
    rscs[0] = resource.get();
    G_GPU_CBCS.fn_gpu_stream_resource_barrier(handle_, rscs.data(), rscs.size());
}

