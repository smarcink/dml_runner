#include "port.h"
#include <format>

void ai_driver::GpuPort::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
}

void ai_driver::GpuPort::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

ai_driver::GpuResource::Ptr ai_driver::GpuPort::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
    return resource_;
}

std::string ai_driver::GpuPort::to_str() const
{
    return node_utils::create_name("GpuPort", name_);
}
