#include "elementwise_add.h"
#include "..\gpu_visitor.h"


void inference_engine::GpuElementwiseAdd::accept(GpuVisitor* visitor)
{
    visitor->visit(this);
}
