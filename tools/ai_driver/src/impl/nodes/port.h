#pragma once
#include "../node.h"

namespace ai_driver
{
    class GpuPort : public GpuNode
    {
    public:
        GpuPort(std::size_t user_id, const ai_driver_port_desc_t& desc, const std::string& name)
            : GpuNode(user_id, name)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;

        GpuResource::Ptr execute(GpuStream& stream) override;

        void set_tensor(const Tensor& tensor)
        {
            if (desc_.data_type != tensor.data_type)
            {
                throw std::runtime_error("Cant set tensor for port. It was created with different data type.");
            }
            output_tensor_ = tensor;
        }

        std::string to_str() const override;

    private:
        ai_driver_port_desc_t desc_{};
    };

    class Port : public INode
    {
    public:
        Port(const ai_driver_port_desc_t& desc, std::size_t id, std::string_view name)
            : INode(id, {/*no inputs*/ }, name)
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            assert(inputs.empty());
            return std::make_unique<GpuPort>(id_, desc_, name_);
        }

    private:
        ai_driver_port_desc_t desc_;
    };
}  // namespace ai_driver