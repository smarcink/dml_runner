#include "convolution.h"
#include "activation.h"
#include "..\gpu_visitor.h"

namespace inference_engine
{
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