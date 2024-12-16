#pragma once
#include <ai_driver.h>

namespace ai_driver
{
    class GpuVisitor
    {
    public:
        virtual void process_sorted_nodes(std::vector<std::unique_ptr<class GpuNode>>& sorted_nodes) = 0;
        // todo, consider using full version of the visitor pattern (if needed)
    };

} // namespace ai_driver