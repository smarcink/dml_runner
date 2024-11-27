#include "inference_engine_operators.h"
#include "impl/model.h"

#include <iostream>
#include <cassert>

INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreatePort(inference_engine_port_desc_t desc)
{
	std::cout << "Created Port" << std::endl;

	void* data_ptr = nullptr; // todo... pass from outside...
	auto ret = new inference_engine::Port(desc, data_ptr);
	return reinterpret_cast<inference_engine_node_t>(ret);
}

INFERENCE_ENGINE_API void inferenceEngineDestroyNode(inference_engine_node_t node)
{
    std::cout << "Destroyed Node" << std::endl;
    auto typed_node = reinterpret_cast<inference_engine::INode*>(node);
    delete typed_node;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineSetResource(inference_engine_node_t node, inference_engine_resource_t resource)
{
    std::cout << "inferenceEngineSetResource" << std::endl;
    auto typed_node = reinterpret_cast<inference_engine::INode*>(node);
    typed_node->set_resource(resource);
    return INFERENCE_ENGINE_RESULT_SUCCESS;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineCheckNodeInputs(inference_engine_node_t node)
{
	std::cout << "inferenceEngineCheckNodeInputs" << std::endl;
	auto typed_node = reinterpret_cast<inference_engine::INode*>(node);
	if (typed_node->check_inputs())
		return INFERENCE_ENGINE_RESULT_SUCCESS;
	
	return INFERENCE_ENGINE_RESULT_WRONG_INPUTS;
}

INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreateMatMul(inference_engine_matmul_desc_t desc)
{
	std::cout << "Created MatMul" << std::endl;

	auto ret = new inference_engine::MatMul(desc);
	return reinterpret_cast<inference_engine_node_t>(ret);
}

INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreateActivation(inference_engine_activation_desc_t desc)
{
    std::cout << "Created Activation" << std::endl;

    auto ret = new inference_engine::Activation(desc);
    return reinterpret_cast<inference_engine_node_t>(ret);
}

