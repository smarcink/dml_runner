#include "elementwise.h"
#include <format>

void ai_driver::GpuElementwise::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
}

void ai_driver::GpuElementwise::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

ai_driver::GpuResource::Ptr ai_driver::GpuElementwise::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
    return resource_;
}

std::string ai_driver::GpuElementwise::to_str() const
{
    // more details about the node here
    return node_utils::create_name("GpuElementwise", name_);
}
