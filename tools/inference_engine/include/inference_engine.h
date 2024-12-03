#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include "inference_engine_export.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _inference_engine_context_handle_t* inference_engine_context_handle_t;
typedef struct _inference_engine_device_t* inference_engine_device_t;
typedef struct _inference_engine_stream_t* inference_engine_stream_t;
typedef struct _inference_engine_kernel_t* inference_engine_kernel_t;
typedef struct _inference_engine_resource_t* inference_engine_resource_t;

typedef enum _inference_engine_kernel_language_t
{
    INFERENCE_ENGINE_KERNEL_LANGUAGE_CM = 0,


    INFERENCE_ENGINE_KERNEL_LANGUAGE_UNKNOWN = -1000,
} inference_engine_kernel_language_t;

typedef inference_engine_kernel_t(*FN_GPU_DEVICE_CREATE_KERNEL)(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language);
typedef inference_engine_resource_t(*FN_GPU_DEVICE_ALLOCATE_RESOURCE)(inference_engine_device_t device, size_t size);
typedef void(*FN_GPU_KERNEL_SET_ARG_RESOURCE)(inference_engine_kernel_t kernel, uint32_t index, inference_engine_resource_t resource);
typedef void(*FN_GPU_KERNEL_SET_ARG_UINT32)(inference_engine_kernel_t kernel, uint32_t index, uint32_t value);
typedef void(*FN_GPU_KERNEL_SET_ARG_FLOAT)(inference_engine_kernel_t kernel, uint32_t index, float value);
typedef void(*FN_GPU_KERNEL_DESTROY)(inference_engine_kernel_t kernel);


typedef void(*FN_GPU_STREAM_EXECUTE_KERNEL)(inference_engine_stream_t stream, inference_engine_kernel_t kernel, uint32_t gws[3], uint32_t lws[3]);
typedef void(*FN_GPU_STREAM_FILL_MEMORY)(inference_engine_stream_t stream, inference_engine_resource_t dst_resource, size_t size);
typedef void(*FN_GPU_STREAM_RESOURCE_BARRIER)(inference_engine_stream_t stream, inference_engine_resource_t* rsc, size_t rsc_count);

typedef struct _inference_engine_callbacks_t
{
    // device
    FN_GPU_DEVICE_CREATE_KERNEL fn_gpu_device_create_kernel;
    FN_GPU_DEVICE_ALLOCATE_RESOURCE fn_gpu_device_allocate_resource;

    // kernel 
    FN_GPU_KERNEL_SET_ARG_RESOURCE fn_gpu_kernel_set_arg_resource;
    FN_GPU_KERNEL_SET_ARG_UINT32   fn_gpu_kernel_set_arg_uint32;
    FN_GPU_KERNEL_SET_ARG_FLOAT    fn_gpu_kernel_set_arg_float;
    FN_GPU_KERNEL_DESTROY          fn_gpu_kernel_destroy;

    // stream
    FN_GPU_STREAM_EXECUTE_KERNEL   fn_gpu_stream_execute_kernel;
    FN_GPU_STREAM_FILL_MEMORY      fn_gpu_stream_fill_memory;
    FN_GPU_STREAM_RESOURCE_BARRIER fn_gpu_stream_resource_barrier;

} inference_engine_context_callbacks_t;

INFERENCE_ENGINE_API inference_engine_context_handle_t inferenceEngineCreateContext(inference_engine_device_t device, inference_engine_context_callbacks_t callbacks);
INFERENCE_ENGINE_API void inferenceEngineDestroyContext(inference_engine_context_handle_t ctx);

#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_ENGINE_H