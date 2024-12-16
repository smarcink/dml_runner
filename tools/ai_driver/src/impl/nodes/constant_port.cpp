#include "constant_port.h"
#include <format>

ai_driver::GpuConstantPort::GpuConstantPort(std::size_t user_id, const ai_driver_constant_port_desc_t& desc, const std::string& name)
    : GpuNode(user_id, name)
    , desc_(desc)
{
    resource_ = std::make_unique<ai_driver::GpuResource>(desc_.resource);
}

void ai_driver::GpuConstantPort::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
}

void ai_driver::GpuConstantPort::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

ai_driver::GpuResource::Ptr ai_driver::GpuConstantPort::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
    return resource_;
}

std::string ai_driver::GpuConstantPort::to_str() const
{
    return node_utils::create_name("GpuPort", name_);
}

ai_driver::ConstantPort::ConstantPort(const ai_driver_constant_port_desc_t& desc, std::size_t id, std::string_view name)
    : INode(id, {/*no inputs*/ }, name)
    , desc_(desc)
{
    if (!desc_.resource)
    {
        throw std::runtime_error("Constant port requires to have resources connected to it during operator creation!");
    }
}

std::unique_ptr<ai_driver::GpuNode> ai_driver::ConstantPort::create_gpu_node(const std::vector<GpuNode*>& inputs)
{
    assert(inputs.empty());
    return std::make_unique<GpuConstantPort>(id_, desc_, name_);
}
