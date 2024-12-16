#ifndef AI_DRIVER_H
#define AI_DRIVER_H

#include "ai_driver_export.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ai_driver_context_handle_t* ai_driver_context_handle_t;
typedef struct _ai_driver_device_t* ai_driver_device_t;
typedef struct _ai_driver_stream_t* ai_driver_stream_t;
typedef struct _ai_driver_kernel_t* ai_driver_kernel_t;
typedef struct _ai_driver_resource_t* ai_driver_resource_t;

typedef enum _ai_driver_kernel_language_t
{
    AI_DRIVER_KERNEL_LANGUAGE_OCL = 0,


    AI_DRIVER_KERNEL_LANGUAGE_UNKNOWN = -1000,
} ai_driver_kernel_language_t;

typedef ai_driver_kernel_t(*FN_GPU_DEVICE_CREATE_KERNEL)(ai_driver_device_t device, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, ai_driver_kernel_language_t language);
typedef ai_driver_resource_t(*FN_GPU_DEVICE_ALLOCATE_RESOURCE)(ai_driver_device_t device, size_t size);
typedef void(*FN_GPU_KERNEL_SET_ARG_RESOURCE)(ai_driver_kernel_t kernel, uint32_t index, ai_driver_resource_t resource, size_t offset);
typedef void(*FN_GPU_KERNEL_SET_ARG_UINT32)(ai_driver_kernel_t kernel, uint32_t index, uint32_t value);
typedef void(*FN_GPU_KERNEL_SET_ARG_FLOAT)(ai_driver_kernel_t kernel, uint32_t index, float value);
typedef void(*FN_GPU_KERNEL_DESTROY)(ai_driver_kernel_t kernel);


typedef void(*FN_GPU_STREAM_EXECUTE_KERNEL)(ai_driver_stream_t stream, ai_driver_kernel_t kernel, uint32_t gws[3], uint32_t lws[3]);
typedef void(*FN_GPU_STREAM_FILL_MEMORY)(ai_driver_stream_t stream, ai_driver_resource_t dst_resource, size_t size);
typedef void(*FN_GPU_STREAM_RESOURCE_BARRIER)(ai_driver_stream_t stream, ai_driver_resource_t* rsc, size_t rsc_count);

typedef struct _ai_driver_callbacks_t
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
    //FN_GPU_STREAM_FILL_MEMORY      fn_gpu_stream_fill_memory;
    FN_GPU_STREAM_RESOURCE_BARRIER fn_gpu_stream_resource_barrier;

} ai_driver_context_callbacks_t;

AI_DRIVER_API ai_driver_context_handle_t aiDriverCreateContext(ai_driver_device_t device, ai_driver_context_callbacks_t callbacks);
AI_DRIVER_API void aiDriverDestroyContext(ai_driver_context_handle_t ctx);

#ifdef __cplusplus
}
#endif

#endif  // AI_DRIVER_H