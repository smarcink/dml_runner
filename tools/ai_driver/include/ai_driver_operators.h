#ifndef AI_DRIVER_OPERATORS_H
#define AI_DRIVER_OPERATORS_H

#include "ai_driver.h"
#include "ai_driver_tensor.h"
#include "ai_driver_export.h"
#include <stdint.h>

#define AI_DRIVER_INVALID_NODE_ID -1

#ifdef __cplusplus
extern "C" {
#endif
typedef uint64_t ai_driver_node_id_t;

typedef struct _ai_driver_port_desc_t
{
    ai_driver_data_type_t data_type;
} ai_driver_port_desc_t;

typedef struct _ai_driver_matmul_desc_t
{
    ai_driver_node_id_t input_a;
    ai_driver_node_id_t input_b;
    // params..

    ai_driver_data_type_t out_data_type;
} ai_driver_matmul_desc_t;


typedef enum _ai_driver_activation_type_t
{
    AI_DRIVER_ACTIVATION_TYPE_RELU = 0,
    AI_DRIVER_ACTIVATION_TYPE_LINEAR,

    AI_DRIVER_ACTIVATION_TYPE_UNKNOWN = -1000,
} ai_driver_activation_type_t;

typedef struct _ai_driver_activation_linear_params_t
{
    float a;
    float b;
} ai_driver_activation_linear_params_t;

typedef struct _ai_driver_activation_desc_t
{
    ai_driver_node_id_t input;
    ai_driver_activation_type_t type;
    ai_driver_data_type_t out_data_type;

    union
    {
        ai_driver_activation_linear_params_t linear;
    } params;

} ai_driver_activation_desc_t;

typedef enum _ai_driver_elementwise_type_t
{
    AI_DRIVER_ELEMENTWISE_TYPE_ADD = 0,

    AI_DRIVER_ELEMENTWISE_TYPE_UNKNOWN = -1000,
} ai_driver_elementwise_type_t;

typedef struct _ai_driver_elementwise_desc_t
{
    ai_driver_node_id_t input_a;
    ai_driver_node_id_t input_b;
    ai_driver_elementwise_type_t type;
    // params..

    ai_driver_data_type_t out_data_type;
} ai_driver_elementwise_desc_t;

typedef struct _ai_driver_convolution_desc_t
{
    ai_driver_node_id_t input;
    // filter params...
    // params...

    ai_driver_data_type_t out_data_type;
} ai_driver_convolution_desc_t;

#ifdef __cplusplus
}
#endif

#endif  // AI_DRIVER_OPERATORS_H