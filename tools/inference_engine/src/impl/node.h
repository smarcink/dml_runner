#pragma once
#include "inference_engine_operators.h"

#include "tensor.h"
#include "gpu_context.h"
#include <string>
#include <variant>
#include <format>

namespace inference_engine
{
    class GpuNode;
    /*
    Wrapper over user created node. Lightweight object.
    */
    class INode
    {
    public:
        INode(std::size_t id, const std::vector<inference_engine_node_id_t>& inputs, std::string_view name)
            : id_(id)
            , inputs_(inputs)
            , name_(name)
        {
            assert(id != INFERENCE_ENGINE_INVALID_NODE_ID);
        }

        INode(INode&& rhs) noexcept
        {
            swap(*this, rhs);
        }

        INode& operator=(INode&& rhs) noexcept
        {
            swap(*this, rhs);
            return *this;
        }

        virtual ~INode() = default;

        friend void swap(INode& first, INode& second) noexcept
        {
            using std::swap;
            swap(first.id_, second.id_);
            swap(first.inputs_, second.inputs_);
            swap(first.name_, second.name_);
        }

        virtual const std::vector<inference_engine_node_id_t>& inputs() const {
            return inputs_;
        }

        virtual std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) = 0;
    protected:
        std::size_t id_ = INFERENCE_ENGINE_INVALID_NODE_ID;
        std::vector<inference_engine_node_id_t> inputs_{};
        std::string name_;
    };

    struct PostOp
    {
        struct ElemWisePosOp {
            inference_engine_elementwise_desc_t desc;
            GpuNode* additional_input{ nullptr };   // for elementwise_add, we have to bring second input if fused with unary operation...
        };

        // probably this structure will evolve, depending on the needs
        std::variant<inference_engine_activation_desc_t, ElemWisePosOp> params_;
    };

    /*
    Concrete class which knows it's predecessors, successors, tensor and GPU objects.
    */
    class GpuNode
    {
    public:
        GpuNode(std::size_t user_id)
            : GpuNode(user_id, {}, {}, {})
        {}
        GpuNode(std::size_t user_id, const std::string& name)
            : GpuNode(user_id, {}, {}, name)
        {
        }
        GpuNode(std::size_t user_id, const Tensor& output_tensor, const std::vector<GpuNode*>& inputs, const std::string& name)
            : id_(user_id)
            , inputs_(inputs)
            , output_tensor_(output_tensor)
            , name_(name)
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

        static void replace_input(GpuNode* node, const GpuNode* old_node, GpuNode* new_node)
        {
            for (auto&& input : node->inputs_)
            {
                if (input == old_node)
                {
                    std::cout << std::format("  Replace input for node: {}, old {}, new {}\n", node->to_str(), old_node->to_str(), new_node->to_str());
                    input = new_node;
                }
            }
        }

        static void replace_output(GpuNode* node, const GpuNode* old_node, GpuNode* new_node)
        {
            for (auto&& output_node : node->outputs_)
            {
                if (output_node == old_node)
                {
                    std::cout << std::format("  Replace output for node: {}, old {}, new {}\n", node->to_str(), old_node->to_str(), new_node->to_str());
                    output_node = new_node;
                }
            }
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

        virtual bool fuse_with(const class GpuActivation*) { return false; }     // false if no fusion happened
        virtual bool fuse_with(const class GpuElementwise*) { return false; } // false if no fusion happened

    protected:
        std::size_t id_ = INFERENCE_ENGINE_INVALID_NODE_ID; // init with invalid id
        std::vector<GpuNode*> inputs_;
        std::vector<GpuNode*> outputs_;
        GpuResource::Ptr resource_;
        Tensor output_tensor_;
        std::vector<PostOp> post_ops_;
        std::string name_;
    };

    namespace node_utils
    {
        // if name is empty, return node_type_name, otherwise return node_type_name + "name"
        std::string create_name(std::string_view node_type_name, const std::string& name);
    }
}  // namespace inference_engine