#include "ai_driver_operators.h"
#include "ai_driver_model.h"
#include "impl/model.h"

#include "impl/nodes/port.h"
#include "impl/nodes/constant_port.h"
#include "impl/nodes/activation.h"
#include "impl/nodes/matmul.h"
#include "impl/nodes/elementwise.h"
#include "impl/nodes/convolution.h"

#include <iostream>
#include <cassert>

template <typename Func>
requires (!std::is_void_v<decltype(std::declval<Func>()())>)
ai_driver_node_id_t handle_exceptions(Func func) {
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


AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddPort(ai_driver_model_descriptor_t model_desc, ai_driver_port_desc_t desc)
{
    return aiDriverModelDescriptorAddPortNamed(model_desc, desc, "");
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddPortNamed(ai_driver_model_descriptor_t model_desc, ai_driver_port_desc_t desc, const char* name)
{
    return handle_exceptions([&]() {
        std::cout << "Created Port " << name << std::endl;
        auto typed_md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(model_desc);
        return typed_md->add_node<ai_driver::Port>(desc, name);
        });
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddConstantPort(ai_driver_model_descriptor_t model_desc, ai_driver_constant_port_desc_t desc)
{
    return aiDriverModelDescriptorAddConstantPortNamed(model_desc, desc, "");
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddConstantPortNamed(ai_driver_model_descriptor_t model_desc, ai_driver_constant_port_desc_t desc, const char* name)
{
    return handle_exceptions([&]() {
        std::cout << "Created Constant Port " << name << std::endl;
        auto typed_md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(model_desc);
        return typed_md->add_node<ai_driver::ConstantPort>(desc, name);
        });
}


AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddMatMul(ai_driver_model_descriptor_t model_desc, ai_driver_matmul_desc_t desc)
{
    return aiDriverModelDescriptorAddMatMulNamed(model_desc, desc, "");
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddMatMulNamed(ai_driver_model_descriptor_t model_desc, ai_driver_matmul_desc_t desc, const char* name)
{
    return handle_exceptions([&]() {
        std::cout << "Created MatMul " << name << std::endl;
        auto typed_md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(model_desc);
        return typed_md->add_node<ai_driver::MatMul>(desc, name);
        });
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddActivation(ai_driver_model_descriptor_t model_desc, ai_driver_activation_desc_t desc)
{
    return aiDriverModelDescriptorAddActivationNamed(model_desc, desc, "");
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddActivationNamed(ai_driver_model_descriptor_t model_desc, ai_driver_activation_desc_t desc, const char* name)
{
    return handle_exceptions([&]() {
        std::cout << "Created Activation " << name << std::endl;
        auto typed_md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(model_desc);
        return typed_md->add_node<ai_driver::Activation>(desc, name);
        });
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddElementwise(ai_driver_model_descriptor_t model_desc, ai_driver_elementwise_desc_t desc)
{
    return aiDriverModelDescriptorAddElementwiseNamed(model_desc, desc, "");
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddElementwiseNamed(ai_driver_model_descriptor_t model_desc, ai_driver_elementwise_desc_t desc, const char* name)
{
    return handle_exceptions([&]() {
        std::cout << "Created Elementwise " << name << std::endl;
        auto typed_md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(model_desc);
        return typed_md->add_node<ai_driver::Elementwise>(desc, name);
        });
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddConvolution(ai_driver_model_descriptor_t model_desc, ai_driver_convolution_desc_t desc)
{
    return aiDriverModelDescriptorAddConvolutionNamed(model_desc, desc, "");
}

AI_DRIVER_API ai_driver_node_id_t aiDriverModelDescriptorAddConvolutionNamed(ai_driver_model_descriptor_t model_desc, ai_driver_convolution_desc_t desc, const char* name)
{
    return handle_exceptions([&]() {
        std::cout << "Created Convolution " << name << std::endl;
        auto typed_md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(model_desc);
        return typed_md->add_node<ai_driver::Convolution>(desc, name);
        });
}
