#pragma once
#include <inference_engine.h>

namespace inference_engine
{
    class GpuVisitor
    {
    public:
        virtual void process_sorted_nodes(std::vector<std::unique_ptr<class GpuNode>>& sorted_nodes) = 0;
        virtual void visit(class GpuPort*) = 0;
        virtual void visit(class GpuActivation*) = 0;
        virtual void visit(class GpuMatMul*) = 0;
        virtual void visit(class GpuElementwiseAdd*) = 0;
        virtual void visit(class GpuConvolution*) = 0;
    };

} // namespace inference_engine