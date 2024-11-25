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

typedef inference_engine_kernel_t(*gpu_create_kernel)(const char* kernel_name, const void* kernel_code,
    size_t kernel_code_size, const char* build_options);

typedef struct _inference_engine_callbacks_t
{
    // gpu callbacks..
    gpu_create_kernel fn_gpu_create_kernel;

    // cpu callbacks

    // npu callbacks

} inference_engine_context_callbacks_t;

typedef enum _inference_engine_result_t
{
    INFERENCE_ENGINE_RESULT_SUCCESS = 0,


    INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN = -1000,
} inference_engine_result_t;

typedef enum _inference_engine_accelerator_type_t
{
    INFERENCE_ENGINE_ACCELERATOR_TYPE_GPU = 0,
    INFERENCE_ENGINE_ACCELERATOR_TYPE_CPU,
    INFERENCE_ENGINE_ACCELERATOR_TYPE_NPU,
    
    INFERENCE_ENGINE_ACCELERATOR_TYPE_UNKNOWN = -1000,
} inference_engine_accelerator_type_t;

INFERENCE_ENGINE_API inference_engine_context_handle_t inferenceEngineCreateContext(inference_engine_accelerator_type_t type, inference_engine_context_callbacks_t callbacks);
INFERENCE_ENGINE_API void inferenceEngineDestroyContext(inference_engine_context_handle_t ctx);
INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineGetLastError();




#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_ENGINE_H