#pragma once
#include "impl/gpu_context.h"

#include <vector>
#include <variant>
#include <string>
#include <span>
#include <iostream>
#include <map>

namespace inference_engine
{
enum class ModelNodeType
{
    ePort,
    eMatmul,
    eActivation,

    eUnknown
};
inline const char* model_node_type_to_string(ModelNodeType t)
{
    switch (t)
    {
    case ModelNodeType::ePort: return "Port";
    case ModelNodeType::eMatmul: return "MatMul";
    case ModelNodeType::eActivation: return "Activation";
    }
    return "Unknown";
}

struct Tensor
{
    inference_engine_data_type_t data_type;
    std::vector<std::uint64_t> dims;
    std::vector<std::uint64_t> strides;

    Tensor() = default;
    Tensor(const inference_engine_tensor_t& tensor_desc)
        : data_type(tensor_desc.data_type)
    {
        for (int i = 0; i < INFERENCE_ENGINE_MAX_TENSOR_DIMS && tensor_desc.dims[i] != 0; ++i)
        {
            dims.push_back(tensor_desc.dims[i]);
            strides.push_back(tensor_desc.strides[i]);
        }
    }
};



class INode
{
public:
    INode(ModelNodeType type, const std::vector<INode*>& inputs)
        : type_(type), inputs_(inputs)
    {
        for (auto& i : inputs_)
        {
            i->add_output(this);
        }
        compute_output_tensors();
    }

    virtual ~INode() = default;

    virtual const std::vector<INode*>& inputs() const {
        return inputs_;
    }

    virtual const std::vector<INode*>& output() const {
        return outputs_;
    }

    virtual void add_output(INode* n)
    {
        outputs_.push_back(n);
    }

    virtual const ModelNodeType type() const {
        return type_;
    }

    virtual void set_resource(std::shared_ptr<GpuResource> r) {
        resources_.push_back(r);
    }

    virtual const std::vector<Tensor>& output_tensors() const
    {
        return output_tensors_;
    }

    virtual std::vector<GpuResource::Ptr> execute() = 0;

protected:
    virtual std::vector<Tensor> compute_output_tensors() { return {}; }  // Should be pure virtual =0?

protected:
    std::vector<INode*> inputs_;
    std::vector<INode*> outputs_;
    ModelNodeType type_ = ModelNodeType::eUnknown;
    std::vector<GpuResource::Ptr> resources_;
    std::vector<Tensor> output_tensors_;
};

inline INode* to_node(inference_engine_node_t n)
{
    return reinterpret_cast<INode*>(n);
}

class Port : public INode
{
public:
    Port(const inference_engine_port_desc_t& desc)
        : INode(ModelNodeType::ePort, {})
        , desc_(desc)
    {
    }

    std::vector<GpuResource::Ptr> execute() override
    {
        std::cout << "[Port] Execute." << std::endl;
        return resources_;
    }

protected:
    std::vector<Tensor> compute_output_tensors() override
    {
        // just an example
        const auto input = inputs()[0]->output_tensors()[0];
        return { input };
    }

private:
    inference_engine_port_desc_t desc_;
};

class MatMul : public INode
{
public:
    MatMul(const inference_engine_matmul_desc_t& desc) 
        : INode(ModelNodeType::eMatmul, { to_node(desc.input_a), to_node(desc.input_b) })
        , desc_(desc)
    {
    }

    std::vector<GpuResource::Ptr> execute() override
    {
        std::cout << "[MatMul] Execute." << std::endl;
        return resources_;
    }

protected:
    std::vector<Tensor> compute_output_tensors() override
    {
        // just an example
        const auto input_a = inputs()[0]->output_tensors()[0];
        const auto input_b = inputs()[1]->output_tensors()[0];
        
        Tensor ret{};
        ret.dims[0] = input_a.dims[0];
        ret.dims[1] = input_a.dims[1];
        ret.dims[2] = input_a.dims[2];
        ret.dims[3] = input_b.dims[3];
        return { ret };
    }

private:
    inference_engine_matmul_desc_t desc_{};
};

class Activation : public INode
{
public:
    Activation(const inference_engine_activation_desc_t& desc)
        : INode(ModelNodeType::eActivation, { to_node(desc.input) })
        , desc_(desc)
    {
    }

    std::vector<GpuResource::Ptr> execute() override
    {
        std::cout << "[Activation] Execute." << std::endl;
        return resources_;
    }

protected:
    std::vector<Tensor> compute_output_tensors() override
    {
        // just an example
        const auto input = inputs()[0]->output_tensors()[0];
        return { input };
    }
private:
    inference_engine_activation_desc_t desc_;
};


struct ExecutableModel
{
public:
    ExecutableModel(const std::vector<INode*>& nodes)
        : nodes_(nodes)
    {
        std::cout << "ExecutableModel:" << std::endl;
    }

    ~ExecutableModel()
    {
        std::cout << "~ExecutableModel:" << std::endl;
    }

    void execute(GpuStream& stream)
    {
        std::cout << "ExecutableModel execute()" << std::endl;
        // We should have topological order of nodes here, we know that model desc just reversed list. For now it should work.
        std::vector<GpuResource::Ptr> resources_for_barrier{};
        resources_for_barrier.reserve(100);
        for (auto& n : nodes_)
        {
            std::cout << "\t[Executing] " << model_node_type_to_string(n->type()) << std::endl;
            auto out_resources = n->execute();
            // aggregate resources
            resources_for_barrier.assign(out_resources.begin(), out_resources.end());
            // we should know dependency graph and when to put resource barriers, but for now always put barrier, after every primitive
            stream.dispatch_resource_barrier(resources_for_barrier);
            resources_for_barrier = {}; // clear list
        }
    }

private:
    std::vector<INode*> nodes_;
};

class ModelDescriptor
{
public:
    ModelDescriptor(std::vector<INode*>&& nodes)
        : nodes_(std::move(nodes))
    {
        std::cout << "ModelDescriptor:" << std::endl;
    }

    ~ModelDescriptor()
    {
        std::cout << "~ModelDescriptor:" << std::endl;
    }

    ExecutableModel compile(GpuContext& ctx, GpuStream& stream) const
    {
        //ToDo: we need some data structure to represent graph (random order of example features below)
        // 1) Sorting graph
        // 2) Traversing graph (i.e. layout prorogation)
        // 2b) Memory allocations and ping-pong (memory reuse)
        // 3) Graph optimization passes (layers fusions) - we should be able to register and unregister passes etc.
        // 4) Uploading constant data to gpu
        // 5) compiling shaders (picking optimized implementations)

        std::cout << "[Compile][Info] -- Nodes added to model desc:" << std::endl;
        for (const auto& n : nodes_)
        {
            std::cout <<"\t" << model_node_type_to_string(n->type()) << std::endl;
        }
        // pass example with current dummy code
        std::cout << "[Compile][Pass-0] -- Topological sort" << std::endl;
        // hacky way so we can pass tests (reverse traversal of node, this will not work in more complex scenarios).
        std::vector<INode*> sorted_nodes{};
        sorted_nodes.reserve(nodes_.size());
        for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it)
        {
            sorted_nodes.push_back((*it));
        }
        // more  passes...
        return ExecutableModel(sorted_nodes);
    }

private:
    std::vector<INode*> nodes_;
};


}