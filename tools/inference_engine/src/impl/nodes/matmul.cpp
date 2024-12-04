#include "matmul.h"
#include "..\gpu_visitor.h"
#include "activation.h"
#include <iostream>


void inference_engine::GpuMatMul::fuse_with(const std::vector<GpuActivation*>& activations)
{
    assert(!activations.empty());
    std::cout << "matmul fuse with... " << activations.size() << " activations\n";
    outputs_ = activations[0]->get_outputs();
}

void inference_engine::GpuMatMul::accept(GpuVisitor* visitor)
{
    visitor->visit(this);
}
