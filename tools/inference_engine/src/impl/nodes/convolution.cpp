#include "convolution.h"
#include "activation.h"
#include "..\gpu_visitor.h"
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
        return {};
    }

    std::string GpuConvolution::to_str() const
    {
        // more details about the node here
        return node_utils::create_name("GpuConvolution", name_);
    }

    // todo: move this code to the base class...
    void GpuConvolution::fuse_with(const GpuActivation* activation)
    {
        std::cout << "convolution fuse with... activation\n";
        outputs_ = activation->get_outputs();
        for (auto& out : outputs_)
            out->replace_input(activation, this);
    }

    void GpuConvolution::accept(GpuVisitor* visitor)
    {
        visitor->visit(this);
    }
} // namespace inference_engine