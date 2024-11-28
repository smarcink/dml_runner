#include "gpu_context.h"

void inference_engine::GpuStream::dispatch_resource_barrier(std::span<GpuResource::Ptr> resources)
{
    std::vector<inference_engine_resource_t> rscs;
    rscs.reserve(resources.size());
    for (auto& r : resources)
    {
        rscs.push_back(r->get());
    }
    assert(G_GPU_CBCS.fn_gpu_stream_resource_barrier);
    G_GPU_CBCS.fn_gpu_stream_resource_barrier(handle_, rscs.data(), rscs.size());
}

