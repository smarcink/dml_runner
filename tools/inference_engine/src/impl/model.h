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

    std::size_t bytes_width() const
    {
        std::size_t size = 1;
        for (const auto& d : dims)
        {
            size *= d;
        }
        switch (data_type)
        {
        case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32:
            return size * sizeof(float);
        case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16:
            return size * sizeof(std::uint16_t);
        default:
            assert(!"unsupported");
        }
        return 1;
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
    }

    virtual ~INode() = default;

    virtual const std::vector<INode*>& inputs() const {
        return inputs_;
    }

    virtual const std::vector<INode*>& outputs() const {
        return outputs_;
    }

    virtual const ModelNodeType type() const {
        return type_;
    }

    virtual void set_resource(GpuResource::Ptr r) {
        resource_ = r;
    }

    virtual GpuResource::Ptr get_resource() {
        return resource_;
    }

    virtual const Tensor& output_tensor() const
    {
        return output_tensor_;
    }

    virtual void compute_output_tensor() = 0;
    virtual void compile(GpuContext& ctx) = 0;
    virtual void initalize(GpuStream& stream) = 0;
    virtual GpuResource::Ptr execute(GpuStream& stream) = 0;

private:
    void add_output(INode* n)
    {
        outputs_.push_back(n);
    }

protected:
    std::vector<INode*> inputs_;
    std::vector<INode*> outputs_;
    ModelNodeType type_ = ModelNodeType::eUnknown;
    GpuResource::Ptr resource_;
    Tensor output_tensor_;
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

    void compile(GpuContext& ctx) override
    {
        std::cout << "[Port] Compile." << std::endl;
    }

    void initalize(GpuStream& stream) override
    {
        std::cout << "[Port] Initialize." << std::endl;
    }

    GpuResource::Ptr execute(GpuStream& stream) override
    {
        std::cout << "[Port] Execute." << std::endl;
        return resource_;
    }

    void compute_output_tensor() override
    {
        // just an example
        output_tensor_ = Tensor(desc_.tensor);
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

    void compile(GpuContext& ctx) override
    {
        std::cout << "[MatMul] Compile." << std::endl;
    }

    void initalize(GpuStream& stream) override
    {
        std::cout << "[MatMul] Initialize." << std::endl;
    }

    GpuResource::Ptr execute(GpuStream& stream) override
    {
        std::cout << "[MatMul] Execute." << std::endl;
        return resource_;
    }

    void compute_output_tensor() override
    {
        // just an example
        const auto input_a = inputs()[0]->output_tensor();
        const auto input_b = inputs()[1]->output_tensor();
        
        output_tensor_.data_type = input_a.data_type;
        output_tensor_.dims.push_back(input_a.dims[0]);
        output_tensor_.dims.push_back(input_a.dims[1]);
        output_tensor_.dims.push_back(input_a.dims[2]);
        output_tensor_.dims.push_back(input_b.dims[3]);
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

    void compile(GpuContext& ctx) override
    {
        std::cout << "[Activation] Compile." << std::endl;
        assert(kernel_ == nullptr); // compile can happen only once
        return;
        const char* kernel_str =
            ""
            ""
            "T"
            ""
            "";

        const char* build_options = "";
        kernel_ = ctx.create_kernel("activation_relu_ref", kernel_str, std::strlen(kernel_str), build_options, INFERENCE_ENGINE_KERNEL_LANGUAGE_CM);
    }

    void initalize(GpuStream& stream) override
    {
        std::cout << "[Activation] Initialize." << std::endl;
    }

    GpuResource::Ptr execute(GpuStream& stream) override
    {
        std::cout << "[Activation] Execute." << std::endl;
        //assert(kernel_);

        //kernel_->set_arg(0, inputs()[0]->)

        return resource_;
    }

    void compute_output_tensor() override
    {
        // just an example
        output_tensor_ = inputs()[0]->output_tensor();
    }
private:
    inference_engine_activation_desc_t desc_;
    GpuKernel::Ptr kernel_;
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
        for (auto& n : nodes_)
        {
            std::cout << "\t[Executing] " << model_node_type_to_string(n->type()) << std::endl;
            auto out_resource = n->execute(stream);

            // aggregate resources and dispatch barrier (sync point) - this is naive, as it will add sync point after each node
            if (out_resource)
            {
                // we should know dependency graph and when to put resource barriers, but for now always put barrier, after every primitive
                stream.dispatch_resource_barrier(*out_resource);
            }

        }
    }

private:
    std::vector<INode*> nodes_;
};

class ModelDescriptor
{
public:
    ModelDescriptor(const std::vector<INode*>& nodes)
        : graph_()
    {
        std::cout << "ModelDescriptor:" << std::endl;
        std::cout << "[Compile][Info] -- Nodes added to model desc:" << std::endl;
        for (const auto& n : nodes)
        {
            std::cout << "\t" << model_node_type_to_string(n->type()) << std::endl;
        }
        std::cout << "[Compile][Pre-process] -- Topological sort" << std::endl;
        // hacky way so we can pass tests (reverse traversal of node, this will not work in more complex scenarios).
        graph_.reserve(nodes.size());
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)
        {
            graph_.push_back((*it));
        }

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


        // pass example with current dummy code
        std::cout << "[Compile][Pass-X] -- Shape prorogations" << std::endl;
        for (auto& n : graph_)
        {
            n->compute_output_tensor();
        }
        std::cout << "[Compile][Pass-Q] -- Memory allocations" << std::endl;
        for (auto& n : graph_)
        {
            const auto has_resource = n->get_resource();
            const auto is_intermidate_node = !n->inputs().empty() && !n->outputs().empty();
            if (!has_resource && is_intermidate_node)
            {
                n->set_resource(std::make_shared<GpuResource>(ctx.allocate_resource(n->output_tensor().bytes_width())));
            }
        }
        std::cout << "[Compile][Pass-Y] -- Compile" << std::endl;
        for (auto& n : graph_)
        {
            n->compile(ctx);
        }
        std::cout << "[Compile][Pass-Z] -- Initialize" << std::endl;
        for (auto& n : graph_)
        {
            n->initalize(stream);
        }
        // more  passes...
        return ExecutableModel(graph_);
    }

private:
    std::vector<INode*> graph_;
};


}