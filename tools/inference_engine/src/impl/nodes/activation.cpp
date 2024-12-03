#include "activation.h"
#include "..\gpu_visitor.h"


void inference_engine::GpuActivation::accept(GpuVisitor* visitor)
{
    visitor->visit(this);
}
