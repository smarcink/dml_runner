#ifndef INFERENCE_ENGINE_OPERATORS_H
#define INFERENCE_ENGINE_OPERATORS_H

#include "inference_engine.h"
#include "inference_engine_tensor.h"
#include "inference_engine_export.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _inference_engine_node_t* inference_engine_node_t;
INFERENCE_ENGINE_API void inferenceEngineDestroyNode(inference_engine_node_t node);
INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineSetResource(inference_engine_node_t node, inference_engine_resource_t resource);

typedef struct _inference_engine_port_t* inference_engine_port_t;
typedef struct _inference_engine_port_desc_t
{
    inference_engine_tensor_t tensor;
} inference_engine_port_desc_t;
INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreatePort(inference_engine_port_desc_t desc);

typedef struct _inference_engine_matmul_desc_t
{
    inference_engine_node_t input_a;
    inference_engine_node_t input_b;
    // params..
} inference_engine_matmul_desc_t;
INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreateMatMul(inference_engine_matmul_desc_t desc);

typedef enum _inference_engine_activation_type_t
{
    INFERENCE_ENGINE_ACTIVATION_TYPE_RELU = 0,


    INFERENCE_ENGINE_ACTIVATION_TYPE_UNKNOWN = -1000,
} inference_engine_activation_type_t;

typedef struct _inference_engine_activation_desc_t
{
    inference_engine_node_t input;
    inference_engine_activation_type_t type;
    // params...
} inference_engine_activation_desc_t;
INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreateActivation(inference_engine_activation_desc_t desc);

#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_ENGINE_OPERATORS_H