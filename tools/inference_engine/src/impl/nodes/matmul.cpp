#include "matmul.h"
#include "..\gpu_visitor.h"
#include "activation.h"
#include <iostream>

namespace inference_engine
{
    void GpuMatMul::fuse_with(const GpuActivation* activation)
    {
        std::cout << "matmul fuse with... activation\n";
        outputs_ = activation->get_outputs();
        for (auto& out : outputs_)
            out->replace_input(activation, this);
    }

    void GpuMatMul::accept(GpuVisitor* visitor)
    {
        visitor->visit(this);
    }
} // namespace inference_engine