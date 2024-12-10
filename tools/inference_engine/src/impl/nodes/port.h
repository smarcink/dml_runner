#pragma once
#include "../node.h"

namespace inference_engine
{
    class GpuPort : public GpuNode
    {
    public:
        GpuPort(std::size_t user_id, const inference_engine_port_desc_t& desc, const std::string& name)
            : GpuNode(user_id, name)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;

        GpuResource::Ptr execute(GpuStream& stream) override;

        void set_tensor(const Tensor& tensor)
        {
            assert(tensor.data_type == desc_.data_type);
            output_tensor_ = tensor;
        }

        std::string to_str() const override;

    private:
        inference_engine_port_desc_t desc_{};
    };

    class Port : public INode
    {
    public:
        Port(const inference_engine_port_desc_t& desc, std::size_t id, std::string_view name)
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
        inference_engine_port_desc_t desc_;
    };
}  // namespace inference_engine