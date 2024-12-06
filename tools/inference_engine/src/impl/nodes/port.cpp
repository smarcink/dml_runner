#include "port.h"
#include <format>

void inference_engine::GpuPort::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
}

void inference_engine::GpuPort::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

inference_engine::GpuResource::Ptr inference_engine::GpuPort::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
    return resource_;
}

std::string inference_engine::GpuPort::to_str() const
{
    return node_utils::create_name("GpuPort", name_);
}
