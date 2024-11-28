#include "inference_engine_model.h"
#include "impl/gpu_context.h"
#include "impl/model.h"

#include <iostream>
#include <cassert>

inline void add_node_to_model_descriptor(inference_engine::INode* node, std::vector<inference_engine::INode*>& node_list)
{
    if (!node)
    {
        return;
    }
    node_list.push_back(node);
    for (const auto& i : node->inputs())
    {
        add_node_to_model_descriptor(i, node_list);
    }
}

INFERENCE_ENGINE_API inference_engine_model_descriptor_t inferenceEngineCreateModelDescriptor(inference_engine_node_t* nodes, uint32_t out_nodes_count)
{
    if (!nodes || out_nodes_count == 0)
    {
        return nullptr;
    }

    std::cout << "Created Model Descriptor" << std::endl;
    
    std::vector<inference_engine::INode*> nodes_list{};
    for(std::uint32_t i = 0; i < out_nodes_count;i ++)
    {
        const auto typed_node = reinterpret_cast<inference_engine::INode*>(nodes[i]);
        add_node_to_model_descriptor(typed_node, nodes_list);
    }
    auto md = new inference_engine::ModelDescriptor(std::move(nodes_list));
    return reinterpret_cast<inference_engine_model_descriptor_t>(md);
}

INFERENCE_ENGINE_API void inferenceEngineDestroyModelDescriptor(inference_engine_model_descriptor_t md)
{
    std::cout << "Destroyed Model Descriptor" << std::endl;
    auto typed_md = reinterpret_cast<inference_engine::ModelDescriptor*>(md);
    delete typed_md;
}

INFERENCE_ENGINE_API inference_engine_model_t inferenceEngineCompileModelDescriptor(inference_engine_context_handle_t context, inference_engine_stream_t stream, inference_engine_model_descriptor_t model_desc)
{
    std::cout << "inferenceEngineCompileModelDescriptor" << std::endl;
    auto ctx = reinterpret_cast<inference_engine::GpuContext*>(context);
    auto md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
    auto str = reinterpret_cast<inference_engine::GpuStream*>(stream);
    auto* exec_model = new inference_engine::ExecutableModel(md->compile(*ctx, *str));
    return reinterpret_cast<inference_engine_model_t>(exec_model);
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineExecuteModel(inference_engine_model_t model, inference_engine_stream_t stream)
{
    std::cout << "inferenceEngineExecuteModel" << std::endl;
    try
    {
        auto md = reinterpret_cast<inference_engine::ExecutableModel*>(model);
        auto str = inference_engine::GpuStream(stream);  // temporary object, handle passed by user, library do not own this.
        md->execute(str);
    }
    catch (...)
    {
        return INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN;
    }
    return INFERENCE_ENGINE_RESULT_SUCCESS;
}

