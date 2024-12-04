#pragma once
#include "../node.h"

namespace inference_engine
{
    class GpuPort : public GpuNode
    {
    public:
        GpuPort(std::size_t user_id, const inference_engine_port_desc_t& desc)
            : GpuNode(user_id)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override
        {
            std::cout << "[Port] Compile." << std::endl;
        }

        void initalize(GpuStream& stream) override
        {
            std::cout << "[Port] Initialize." << std::endl;
        }

        GpuResource::Ptr execute(GpuStream& stream) override
        {
            std::cout << "[Port] Execute." << std::endl;
            return resource_;
        }

        void set_tensor(const Tensor& tensor)
        {
            assert(tensor.data_type == desc_.data_type);
            output_tensor_ = tensor;
        }

        std::string to_str() const override
        {
            // more details about the node here
            return "GpuPort";
        }
    private:
        inference_engine_port_desc_t desc_{};
    };

    class Port : public INode
    {
    public:
        Port(const inference_engine_port_desc_t& desc, std::size_t id)
            : INode(id, {/*no inputs*/ })
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            assert(inputs.empty());
            return std::make_unique<GpuPort>(id_, desc_);
        }

    private:
        inference_engine_port_desc_t desc_;
    };
}  // namespace inference_engine