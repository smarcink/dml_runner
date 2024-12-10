#include "inference_engine_operators.h"
#include "inference_engine_model.h"
#include "impl/model.h"

#include "impl/nodes/port.h"
#include "impl/nodes/activation.h"
#include "impl/nodes/matmul.h"
#include "impl/nodes/elementwise.h"
#include "impl/nodes/convolution.h"

#include <iostream>
#include <cassert>

template <typename Func>
requires (!std::is_void_v<decltype(std::declval<Func>()())>)
inference_engine_node_id_t handle_exceptions(Func func) {
	using ReturnType = decltype(func());
	try {
		return func();
	}
    catch (const std::exception& ex)
    {
        std::cerr << "exception: " << ex.what() << '\n';
    }
	catch (...) {
		std::cerr << "unknown exception!\n";
	}
	return ReturnType(-1);
}


INFERENCE_ENGINE_API inference_engine_node_id_t inferenceEngineModelDescriptorAddPort(inference_engine_model_descriptor_t model_desc, inference_engine_port_desc_t desc)
{
    return handle_exceptions([&]() {
        std::cout << "Created Port" << std::endl;
		auto typed_md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
		return typed_md->add_node<inference_engine::Port>(desc);
		});
}

INFERENCE_ENGINE_API inference_engine_node_id_t inferenceEngineModelDescriptorAddMatMul(inference_engine_model_descriptor_t model_desc, inference_engine_matmul_desc_t desc)
{
    return handle_exceptions([&]() {
        std::cout << "Created MatMul" << std::endl;
        auto typed_md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
		return typed_md->add_node<inference_engine::MatMul>(desc);
        });
}

INFERENCE_ENGINE_API inference_engine_node_id_t inferenceEngineModelDescriptorAddActivation(inference_engine_model_descriptor_t model_desc, inference_engine_activation_desc_t desc)
{
    return handle_exceptions([&]() {
        std::cout << "Created Activation" << std::endl;
        auto typed_md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
        return typed_md->add_node<inference_engine::Activation>(desc);
        });
}

INFERENCE_ENGINE_API inference_engine_node_id_t inferenceEngineModelDescriptorAddElementwise(inference_engine_model_descriptor_t model_desc, inference_engine_elementwise_desc_t desc)
{
    return handle_exceptions([&]() {
        std::cout << "Created Elementwise" << std::endl;
        auto typed_md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
        return typed_md->add_node<inference_engine::Elementwise>(desc);
        });
}

INFERENCE_ENGINE_API inference_engine_node_id_t inferenceEngineModelDescriptorAddConvolution(inference_engine_model_descriptor_t model_desc, inference_engine_convolution_desc_t desc)
{
    return handle_exceptions([&]() {
        std::cout << "Created Convolution" << std::endl;
        auto typed_md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
        return typed_md->add_node<inference_engine::Convolution>(desc);
        });
}
