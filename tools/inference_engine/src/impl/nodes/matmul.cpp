#include "matmul.h"
#include "..\gpu_visitor.h"
#include "activation.h"
#include <iostream>


void inference_engine::GpuMatMul::fuse_with(const GpuActivation* activation)
{
    std::cout << "matmul fuse with...\n";
    outputs_ = activation->get_outputs();
}

void inference_engine::GpuMatMul::accept(GpuVisitor* visitor)
{
    visitor->visit(this);
}
