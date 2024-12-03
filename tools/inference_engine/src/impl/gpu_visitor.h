#pragma once
#include <inference_engine.h>

namespace inference_engine
{
    class GpuVisitor
    {
    public:
        virtual void processSortedNodes(std::vector<std::unique_ptr<class GpuNode>>& sortedNodes) = 0;
        virtual void visit(class GpuPort*) = 0;
        virtual void visit(class GpuActivation*) = 0;
        virtual void visit(class GpuMatMul*) = 0;
    };

} // namespace inference_engine