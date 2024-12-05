#include "elementwise_add.h"
#include <format>

void inference_engine::GpuElementwise::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
}

void inference_engine::GpuElementwise::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

inference_engine::GpuResource::Ptr inference_engine::GpuElementwise::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
    return resource_;
}

std::string inference_engine::GpuElementwise::to_str() const
{
    // more details about the node here
    return node_utils::create_name("GpuElementwise", name_);
}
