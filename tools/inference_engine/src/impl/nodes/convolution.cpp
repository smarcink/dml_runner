#include "convolution.h"
#include "activation.h"
#include "elementwise_add.h"
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

    bool GpuConvolution::fuse_with(const GpuElementwiseAdd* elemwise)
    {
        assert(elemwise);
        std::cout << "activation fuse with... elementwise_add\n";
        outputs_ = elemwise->get_outputs();
        for (auto& out : outputs_)
            GpuNode::replace_input(out, elemwise, this);

        // we have to update out inputs, activation contains only one input, bur for elementwise_add we have bring its second input
        auto& elem_inputs = elemwise->get_inputs();
        assert(elem_inputs.size() == 2 && (elem_inputs[0] == this || elem_inputs[1] == this));
        inputs_.push_back(elem_inputs[0] == this ? elem_inputs[1] : elem_inputs[0]);

        post_ops_.push_back(elemwise->create_post_op());
        return true;
    }

} // namespace inference_engine