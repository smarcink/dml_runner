#pragma once
#include <inference_engine.h>

namespace inference_engine
{
    class GpuVisitor
    {
    public:
        virtual void process_sorted_nodes(std::vector<std::unique_ptr<class GpuNode>>& sorted_nodes) = 0;
        // todo, consider using full version of the visitor pattern (if needed)
    };

} // namespace inference_engine