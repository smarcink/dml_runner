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
    std::vector<std::uint64_t> dims;
};



struct INode
{
    INode() = default;
    virtual ~INode() = default;

    const std::vector<INode*>& inputs() const {
        return inputs_;
    }

    const std::vector<INode*>& output() const {
        return outputs_;
    }

    const ModelNodeType type() const {
        return type_;
    }

protected:
    std::vector<INode*> inputs_;
    std::vector<INode*> outputs_;
    ModelNodeType type_ = ModelNodeType::eUnknown;
};

struct Port : public INode
{
    Port(const inference_engine_port_desc_t& desc)
    {
        type_ = ModelNodeType::ePort;
        outputs_.push_back(this);
    }
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

class ModelDescriptor
{
public:
    ModelDescriptor(std::vector<INode*>&& nodes)
        : nodes_(std::move(nodes))
    {
    }

    struct Partition
    {

    };

    std::vector<Partition> get_partitions() const
    {
        std::cout << "Nodes added to model desc:" << std::endl;
        for (const auto& n : nodes_)
        {
            std::cout <<"\t" << model_node_type_to_string(n->type()) << std::endl;
        }
        return {};
    }

private:
    std::vector<INode*> nodes_;
};
}