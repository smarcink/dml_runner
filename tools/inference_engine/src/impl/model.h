#pragma once
#include "gpu_context.h"
#include "inference_engine_tensor.h"
#include "inference_engine_operators.h"
#include "tensor.h"
#include "node.h"

#include <vector>
#include <variant>
#include <string>
#include <span>
#include <iostream>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <stack>

namespace inference_engine
{

class DAG 
{
public:
    DAG();

    template<typename T, typename TDesc>
    std::size_t add_node(const TDesc& desc, std::string_view name)
    {  
        // add node
        auto id = nodes_.size();
        nodes_.push_back(std::make_unique<T>(desc, id, name));
        
        // Early validation: valid it's inputs
        const auto& inputs = nodes_.back()->inputs();
        for (const auto& in : inputs)
        {
            // Dont allow for loops/cycles.
            if (in == id)
            {
                throw std::runtime_error("Nodes input can't point to itself.");
            }
            // We know that inputs have to be already added to graph, so their ID has to be lower.
            if (in >= nodes_.size())
            {
                throw std::runtime_error("Invalid input for node. Can't add node to the model.");
            }
        }
        return id;
    }

public:
    std::vector<std::unique_ptr<GpuNode>> compile(std::span<TensorMapping> input_mappings);


private:
    void create_adjacency_list();

    std::vector<INode*> topological_sort();
    void topological_sort_util(INode* node, std::unordered_set<INode*>& visited, std::stack<INode*>& stack);

private:
    std::vector<std::unique_ptr<INode>> nodes_;
    std::unordered_map<INode*, std::vector<INode*>> adjacency_list_;
};


struct ExecutableModel
{
public:
    ExecutableModel(std::vector<std::unique_ptr<GpuNode>>&& nodes);
    ~ExecutableModel();

    void execute(GpuStream& stream);
    void set_resource(inference_engine_node_id_t id, GpuResource::Ptr rsc);

    const std::vector<TensorMapping>& get_outputs() const;

private:
    const std::vector<std::unique_ptr<GpuNode>> nodes_;
    const std::vector<TensorMapping> output_mappings_;
};

class ModelDescriptor
{
public:
    ModelDescriptor();
    ~ModelDescriptor();

    template<typename T, typename TDesc>
    std::size_t add_node(const TDesc& desc, std::string_view name)
    {
        return dag_.add_node<T>(desc, name);
    }

    ExecutableModel compile(GpuContext& ctx, GpuStream& stream, std::span<TensorMapping> input_mappings);

private:
    DAG dag_;
};


} // namespace inference_engine