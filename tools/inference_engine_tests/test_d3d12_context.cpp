#include "test_d3d12_context.h"

#include <iostream>

static inference_engine_kernel_t gpu_device_create_kernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options)
{
    std::cout << "Dummy callback gpu_device_create_kernel" << std::endl;
    return nullptr;
}

static inference_engine_resource_t gpu_device_allocate_resource(inference_engine_device_t device, size_t size)
{
    std::cout << "Dummy callback gpu_device_allocate_resource" << std::endl;
    return nullptr;
}

static void gpu_kernel_set_arg_resource(inference_engine_kernel_t kernel, uint32_t index, inference_engine_resource_t resource)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_resource" << std::endl;
}

static void gpu_kernel_set_arg_uint32(inference_engine_kernel_t kernel, uint32_t index, uint32_t value)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_uint32" << std::endl;
}

static void gpu_kernel_set_arg_float(inference_engine_kernel_t kernel, uint32_t index, float value)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_float" << std::endl;
}

static void gpu_stream_execute_kernel(inference_engine_stream_t stream, inference_engine_kernel_t kernel, uint32_t gws[3], uint32_t lws[3])
{
    std::cout << "Dummy callback gpu_stream_execute_kernel" << std::endl;
}

static void gpu_stream_fill_memory(inference_engine_stream_t stream, inference_engine_resource_t dst_resource, size_t size, inference_engine_event_t* out_event, inference_engine_event_t* dep_events, size_t dep_events_count)
{
    std::cout << "Dummy callback gpu_stream_fill_memory" << std::endl;
}

test_ctx::TestGpuContext::TestGpuContext()
{
    inference_engine_context_callbacks_t callbacks{};

    callbacks.fn_gpu_device_create_kernel = &gpu_device_create_kernel;
    callbacks.fn_gpu_device_allocate_resource = &gpu_device_allocate_resource;

    callbacks.fn_gpu_kernel_set_arg_resource = &gpu_kernel_set_arg_resource;
    callbacks.fn_gpu_kernel_set_arg_uint32 = &gpu_kernel_set_arg_uint32;
    callbacks.fn_gpu_kernel_set_arg_float = &gpu_kernel_set_arg_float;

    callbacks.fn_gpu_stream_execute_kernel = &gpu_stream_execute_kernel;
    callbacks.fn_gpu_stream_fill_memory = &gpu_stream_fill_memory;

    ctx_ = inferenceEngineCreateContext(INFERENCE_ENGINE_ACCELERATOR_TYPE_GPU, device_, callbacks);
}
