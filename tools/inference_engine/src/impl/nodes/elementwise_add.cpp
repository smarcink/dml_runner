#include "elementwise_add.h"
#include "..\gpu_visitor.h"
#include <format>

void inference_engine::GpuElementwiseAdd::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
}

void inference_engine::GpuElementwiseAdd::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

inference_engine::GpuResource::Ptr inference_engine::GpuElementwiseAdd::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
    return {};
}

std::string inference_engine::GpuElementwiseAdd::to_str() const
{
    // more details about the node here
    return node_utils::create_name("GpuElementwiseAdd", name_);
}

void inference_engine::GpuElementwiseAdd::accept(GpuVisitor* visitor)
{
    visitor->visit(this);
}
