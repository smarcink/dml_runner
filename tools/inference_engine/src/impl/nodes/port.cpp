#include "port.h"
#include "..\gpu_visitor.h"


void inference_engine::GpuPort::accept(GpuVisitor* visitor)
{
    visitor->visit(this);
}
