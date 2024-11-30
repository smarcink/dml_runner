#pragma once
#include "inference_engine_operators.h"

#include "tensor.h"
#include "gpu_context.h"
#include <string>

namespace inference_engine
{
    class GpuNode;
    /*
    Wrapper over user created node. Lightweight object.
    */
    class INode
    {
    public:
        INode(std::size_t id, const std::vector<inference_engine_node_id_t>& inputs)
            : id_(id)
            , inputs_(inputs)
        {
            assert(id != INFERENCE_ENGINE_INVALID_NODE_ID);
        }

        INode(INode&& rhs) noexcept
        {
            std::swap(inputs_, rhs.inputs_);
        }

        INode& operator=(INode&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(inputs_, rhs.inputs_);
            }
            return *this;
        }

        virtual ~INode() = default;

        virtual const std::vector<inference_engine_node_id_t>& inputs() const {
            return inputs_;
        }

        virtual std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) = 0;
    protected:
        std::size_t id_ = INFERENCE_ENGINE_INVALID_NODE_ID;
        std::vector<inference_engine_node_id_t> inputs_{};
    };

    /*
    Concrete class which knows it's predecessors, successors, tensor and GPU objects.
    */
    class GpuNode
    {
    public:
        GpuNode(std::size_t user_id)
            : GpuNode(user_id, {}, {})
        {}
        GpuNode(std::size_t user_id, const Tensor& output_tensor, const std::vector<GpuNode*>& inputs)
            : id_(user_id)
            , inputs_(inputs)
            , output_tensor_(output_tensor)
        {}
        virtual ~GpuNode() = default;

        virtual void add_input(GpuNode* node)
        {
            inputs_.push_back(node);
        }

        virtual void add_output(GpuNode* node)
        {
            outputs_.push_back(node);
        }

        virtual const std::vector<GpuNode*>& get_inputs() const
        {
            return inputs_;
        }

        virtual const std::vector<GpuNode*>& get_outputs() const
        {
            return outputs_;
        }

        virtual void set_id(std::size_t id)
        {
            id_ = id;
        }

        virtual std::size_t get_id() const
        {
            return id_;
        }

        virtual std::string to_str() const = 0;
        virtual void compile(GpuContext& ctx) = 0;
        virtual void initalize(GpuStream& stream) = 0;
        virtual GpuResource::Ptr execute(GpuStream& stream) = 0;

        virtual void set_resource(GpuResource::Ptr r)
        {
            if (resource_)
            {
                std::cout << "Override resource, was it intended?" << std::endl;
            }
            resource_ = r;
        }

        virtual GpuResource::Ptr get_resource() {
            return resource_;
        }

        virtual const Tensor& get_output_tensor() const
        {
            return output_tensor_;
        }

    protected:
        std::size_t id_ = INFERENCE_ENGINE_INVALID_NODE_ID; // init with invalid id
        std::vector<GpuNode*> inputs_;
        std::vector<GpuNode*> outputs_;
        GpuResource::Ptr resource_;
        Tensor output_tensor_;
    };
}  // namespace inference_engine