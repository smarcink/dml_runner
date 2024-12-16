#pragma once
#include "../node.h"

namespace ai_driver
{

    class GpuConstantPort : public GpuNode
    {
    public:
    public:
        GpuConstantPort(std::size_t user_id, const ai_driver_constant_port_desc_t& desc, const std::string& name);

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;
        GpuResource::Ptr execute(GpuStream& stream) override;
        std::string to_str() const override;

    private:
        ai_driver_constant_port_desc_t desc_{};
    };

    class ConstantPort : public INode
    {
    public:
        ConstantPort(const ai_driver_constant_port_desc_t& desc, std::size_t id, std::string_view name);
        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override;

    private:
        ai_driver_constant_port_desc_t desc_;
    };
}  // namespace ai_driver