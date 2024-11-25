#include "inference_engine_model.h"
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

INFERENCE_ENGINE_API inference_engine_model_descriptor_t inferenceEngineCreateModelDescriptor(inference_engine_model_descriptor_config_t config, inference_engine_node_t* nodes, uint32_t out_nodes_count)
{
    if (!nodes || out_nodes_count == 0)
    {
        return nullptr;
    }
    
    if (!config.preffered_accelerator_list)
    {
        std::cout << "preffered_accelerator_list is empty, model descriptor will pick default one: GPU" << std::endl;
    }
    else
    {
        assert(!"not implemeneted");
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

INFERENCE_ENGINE_API inference_engine_model_partition_t* inferenceEngineGetPartitions(inference_engine_model_descriptor_t model_desc)
{
    auto md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
    std::cout << "Model descriptor get partitions" << std::endl;
    const auto partitions = md->get_partitions();
    return nullptr;
}

