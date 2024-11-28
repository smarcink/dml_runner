#include "inference_engine_operators.h"
#include "impl/model.h"
#include "impl/error.h"

#include <iostream>
#include <cassert>

template <typename Func>
requires (!std::is_void_v<decltype(std::declval<Func>()())>)
auto handle_exceptions(Func func) {
	using ReturnType = decltype(func());
	try {
		return func();
	}
	catch (const std::bad_alloc&) {
		inference_engine::set_last_error(INFERENCE_ENGINE_RESULT_BAD_ALLOC);
	}
	catch (const inference_engine::inference_engine_exception& ex) {
		inference_engine::set_last_error(ex.val_);
	}
	catch (...) {
		inference_engine::set_last_error(INFERENCE_ENGINE_RESULT_OTHER);
	}
	return ReturnType{};
}

INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreatePort(inference_engine_port_desc_t desc)
{
	return handle_exceptions([&]() {
		std::cout << "Created Port" << std::endl;
		auto ret = new inference_engine::Port(desc);
		return reinterpret_cast<inference_engine_node_t>(ret);
	});
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
    typed_node->set_resource(std::make_shared<inference_engine::GpuResource>(resource));
    return INFERENCE_ENGINE_RESULT_SUCCESS;
}

INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreateMatMul(inference_engine_matmul_desc_t desc)
{
	std::cout << "Created MatMul" << std::endl;
	return handle_exceptions([&]() {
		auto ret = new inference_engine::MatMul(desc);
		return reinterpret_cast<inference_engine_node_t>(ret);
	});
}

INFERENCE_ENGINE_API inference_engine_node_t inferenceEngineCreateActivation(inference_engine_activation_desc_t desc)
{
	std::cout << "Created Activation" << std::endl;
	return handle_exceptions([&]() {
		auto ret = new inference_engine::Activation(desc);
		return reinterpret_cast<inference_engine_node_t>(ret);
	});
}

