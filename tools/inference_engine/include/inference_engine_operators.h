#ifndef INFERENCE_ENGINE_OPERATORS_H
#define INFERENCE_ENGINE_OPERATORS_H

#include "inference_engine.h"
#include "inference_engine_tensor.h"
#include "inference_engine_export.h"
#include <stdint.h>

#define INFERENCE_ENGINE_INVALID_NODE_ID -1

#ifdef __cplusplus
extern "C" {
#endif
typedef uint64_t inference_engine_node_id_t;

typedef struct _inference_engine_port_desc_t
{
    inference_engine_data_type_t data_type;
} inference_engine_port_desc_t;

typedef struct _inference_engine_matmul_desc_t
{
    inference_engine_node_id_t input_a;
    inference_engine_node_id_t input_b;
    // params..
} inference_engine_matmul_desc_t;


typedef enum _inference_engine_activation_type_t
{
    INFERENCE_ENGINE_ACTIVATION_TYPE_RELU = 0,
    INFERENCE_ENGINE_ACTIVATION_TYPE_LINEAR,

    INFERENCE_ENGINE_ACTIVATION_TYPE_UNKNOWN = -1000,
} inference_engine_activation_type_t;

typedef struct _inference_engine_activation_linear_params_t
{
    float a;
    float b;
} inference_engine_activation_linear_params_t;

typedef struct _inference_engine_activation_desc_t
{
    inference_engine_node_id_t input;
    inference_engine_activation_type_t type;
    
    union
    {
        inference_engine_activation_linear_params_t linear;
    } params;

} inference_engine_activation_desc_t;

#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_ENGINE_OPERATORS_H