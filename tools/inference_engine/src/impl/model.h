#pragma once
#include "gpu_context.h"
#include "inference_engine_tensor.h"
#include "inference_engine_operators.h"

#include <vector>
#include <variant>
#include <string>
#include <span>
#include <iostream>

namespace inference_engine
{
enum class ModelNodeType
{
    ePort,
    eMatmul,
    eActivation,
    eConvolution,
    eUnknown
};
const char* to_string(ModelNodeType t);
const char* to_string(inference_engine_tensor_layout_t t);


struct Tensor
{
    inference_engine_data_type_t data_type = XESS_DATA_TYPE_UNKNOWN;
    inference_engine_tensor_layout_t layout = TENSOR_LAYOUT_UNDEFINED;
    void* data = nullptr;   // non-owning pointer to data
    std::vector<std::uint64_t> dims;
	std::vector<std::uint64_t> strides;

	Tensor(const inference_engine_tensor_t& tensor_desc, void* data_ptr);

	std::size_t size() const;
};

struct INode
{
    INode()
    {
    }

    virtual ~INode() = default;

    virtual const std::vector<INode*>& inputs() const {
        return inputs_;
    }

    virtual const std::vector<INode*>& output() const {
        return outputs_;
    }

    virtual const ModelNodeType type() const {
        return type_;
    }

    virtual void set_resource(inference_engine_resource_t r) {
        resource_ = r;
    }

    virtual bool check_inputs() const {
        return true;
    }

protected:
    std::vector<INode*> inputs_;
    std::vector<INode*> outputs_;
    ModelNodeType type_ = ModelNodeType::eUnknown;
    inference_engine_resource_t resource_ = nullptr;
};

struct Port : public INode
{
	Port(const inference_engine_port_desc_t& desc, void* data_ptr)
		: tensor_(desc.tensor, data_ptr)
	{
		type_ = ModelNodeType::ePort;
		outputs_.push_back(this);
	}

	const Tensor& tensor() const
	{
		return tensor_;
	}

private:
	Tensor tensor_;
};

struct MatMul : public INode
{
	MatMul(const inference_engine_matmul_desc_t& desc)
	{
		type_ = ModelNodeType::eMatmul;
		inputs_.push_back(reinterpret_cast<Port*>(desc.tensor_a));
		inputs_.push_back(reinterpret_cast<Port*>(desc.tensor_b));
		outputs_.push_back(this);
	}

	const Tensor& tensor_a() const
	{
		return reinterpret_cast<Port*>(inputs_[0])->tensor();
	}

	const Tensor& tensor_b() const
	{
		return reinterpret_cast<Port*>(inputs_[1])->tensor();
	}

    bool check_inputs() const override;
};

struct Activation : public INode
{
    Activation(const inference_engine_activation_desc_t& desc)
    {
        type_ = ModelNodeType::eActivation;
        inputs_.push_back(reinterpret_cast<Port*>(desc.tensor));
        outputs_.push_back(this);
    }
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

    void execute()
    {
        std::cout << "ExecutableModel execute()" << std::endl;
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

    ExecutableModel compile_for_gpu(GpuContext& gpu_ctx) const
    {
        std::cout << "compile_for_gpu -- Nodes added to model desc:" << std::endl;
        for (const auto& n : nodes_)
        {
            std::cout <<"\t" << to_string(n->type()) << std::endl;
        }
        return ExecutableModel(nodes_);
    }

private:
    std::vector<INode*> nodes_;
};


}