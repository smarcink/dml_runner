#ifndef INFERENCE_ENGINE_TENSOR_H
#define INFERENCE_ENGINE_TENSOR_H

#define INFERENCE_ENGINE_MAX_TENSOR_DIMS 8
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef enum _inference_engine_data_type_t
    {
        XESS_DATA_TYPE_FP32 = 0,
        XESS_DATA_TYPE_FP16,


        XESS_DATA_TYPE_UNKNOWN = -1000,
    } inference_engine_data_type_t;

typedef struct _inference_engine_tensor_t
{
    inference_engine_data_type_t data_type;
    uint64_t dims[INFERENCE_ENGINE_MAX_TENSOR_DIMS];
    uint64_t strides[INFERENCE_ENGINE_MAX_TENSOR_DIMS];
} inference_engine_tensor_t;

#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_ENGINE_TENSOR_H