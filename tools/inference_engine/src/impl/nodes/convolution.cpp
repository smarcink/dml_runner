#include "convolution.h"
#include "activation.h"
#include "elementwise.h"
#include <format>

namespace inference_engine
{

    void GpuConvolution::compile(GpuContext& ctx)
    {
        std::cout << std::format("[{}] Compile.\n", to_str());
    }

    void GpuConvolution::initalize(GpuStream& stream)
    {
        std::cout << std::format("[{}] Initialize.\n", to_str());
    }

    inference_engine::GpuResource::Ptr GpuConvolution::execute(GpuStream& stream)
    {
        std::cout << std::format("[{}] Execute.\n", to_str());

        if (!post_ops_.empty())
        {
            std::cout << "  Post ops:\n";
            for (auto& op : post_ops_)
            {
                if (const auto activation_params = std::get_if<inference_engine_activation_desc_t>(&op.params_))
                {
                    std::cout << std::format("    Activation: \n");
                }
                else if (const auto elemwise_params = std::get_if<PostOp::ElemWisePosOp>(&op.params_))
                {
                    std::cout << std::format("    Elementwise add: additional input {}\n", elemwise_params->additional_input->to_str());
                }
            }
        }

        return resource_;
    }

    std::string GpuConvolution::to_str() const
    {
        // more details about the node here
        return node_utils::create_name("GpuConvolution", name_);
    }

    // todo: move this code to the base class...
    bool GpuConvolution::fuse_with(const GpuActivation* activation)
    {
        std::cout << "convolution fuse with... activation\n";
        outputs_ = activation->get_outputs();
        for (auto& out : outputs_)
            GpuNode::replace_input(out, activation, this);
        return true;
    }

    bool GpuConvolution::fuse_with(const GpuElementwise* elemwise)
    {
        assert(elemwise);
        std::cout << "convolution fuse with... elementwise_add\n";
        outputs_ = elemwise->get_outputs();
        for (auto& out : outputs_)
            GpuNode::replace_input(out, elemwise, this);

        // we have to update our inputs, convolution contains only one input, bur for elementwise_add we have bring its second input
        auto& elem_inputs = elemwise->get_inputs();
        assert(elem_inputs.size() == 2 && (elem_inputs[0] == this || elem_inputs[1] == this));
        auto other_input = elem_inputs[0] == this ? elem_inputs[1] : elem_inputs[0];
        GpuNode::replace_output(other_input, elemwise, this);
        inputs_.push_back(other_input);

        post_ops_.push_back(elemwise->create_post_op(other_input));
        return true;
    }

} // namespace inference_engine