#ifndef AI_DRIVER_MODEL_DESCRIPTOR_H
#define AI_DRIVER_MODEL_DESCRIPTOR_H

#include "ai_driver.h"
#include "ai_driver_operators.h"
#include "ai_driver_export.h"
#include "ai_driver_tensor.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct _ai_driver_model_descriptor_t* ai_driver_model_descriptor_t;
typedef struct _ai_driver_model_t* ai_driver_model_t;

typedef struct _ai_driver_tensor_mapping_t
{
    ai_driver_node_id_t id;
    ai_driver_tensor_t tensor;
} ai_driver_tensor_mapping_t;

AI_DRIVER_API ai_driver_model_descriptor_t aiDriverCreateModelDescriptor();
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddPort(ai_driver_model_descriptor_t model_desc, ai_driver_port_desc_t desc);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddPortNamed(ai_driver_model_descriptor_t model_desc, ai_driver_port_desc_t desc, const char* name);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddMatMul(ai_driver_model_descriptor_t model_desc, ai_driver_matmul_desc_t desc);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddMatMulNamed(ai_driver_model_descriptor_t model_desc, ai_driver_matmul_desc_t desc, const char* name);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddActivation(ai_driver_model_descriptor_t model_desc, ai_driver_activation_desc_t desc);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddActivationNamed(ai_driver_model_descriptor_t model_desc, ai_driver_activation_desc_t desc, const char* name);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddElementwise(ai_driver_model_descriptor_t model_desc, ai_driver_elementwise_desc_t desc);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddElementwiseNamed(ai_driver_model_descriptor_t model_desc, ai_driver_elementwise_desc_t desc, const char* name);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddConvolution(ai_driver_model_descriptor_t model_desc, ai_driver_convolution_desc_t desc);
AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddConvolutionNamed(ai_driver_model_descriptor_t model_desc, ai_driver_convolution_desc_t desc, const char* name);
//AI_DRIVER_API ai_driver_model_descriptor_t aiDriverCreateModelDescriptor(ai_driver_node_t* out_nodes, uint32_t out_nodes_count);
AI_DRIVER_API void aiDriverDestroyModelDescriptor(ai_driver_model_descriptor_t md);
AI_DRIVER_API void aiDriverDestroyModel(ai_driver_model_t model);
AI_DRIVER_API ai_driver_model_t aiDriverCompileModelDescriptor(ai_driver_context_handle_t context, ai_driver_stream_t stream, ai_driver_model_descriptor_t model_desc, ai_driver_tensor_mapping_t* input_mapping_list, size_t input_mapping_size);
//set resource for inputs and outputs
AI_DRIVER_API bool aiDriverModelSetResource(ai_driver_model_t model, ai_driver_node_id_t id, ai_driver_resource_t resource);
// call with empty list to get size
AI_DRIVER_API bool aiDriverModelGetOutputs(ai_driver_model_t model, ai_driver_tensor_mapping_t* list, size_t* size);
AI_DRIVER_API bool aiDriverExecuteModel(ai_driver_model_t model, ai_driver_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // AI_DRIVER_MODEL_DESCRIPTOR_H