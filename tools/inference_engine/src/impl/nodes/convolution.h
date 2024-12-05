#pragma once
#include "../node.h"

namespace inference_engine
{
    class GpuActivation;

    class GpuConvolution : public GpuNode
    {
    public:
        GpuConvolution(std::size_t user_id, GpuNode* input, const inference_engine_convolution_desc_t& desc)
            : GpuNode(user_id, input->get_output_tensor(), { input })
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override
        {
            std::cout << "[GpuConvolution] Compile." << std::endl;
        }

        void initalize(GpuStream& stream) override
        {
            std::cout << "[GpuConvolution] Initialize." << std::endl;
        }

        GpuResource::Ptr execute(GpuStream& stream) override
        {
            std::cout << "[GpuConvolution] Execute." << std::endl;
            return {};
        }

        std::string to_str() const override
        {
            // more details about the node here
            return "GpuConvolution";
        }

        void fuse_with(const GpuActivation*);
        void accept(GpuVisitor* visitor) override;        
    private:
        inference_engine_convolution_desc_t desc_{};
    };

    class Convolution : public INode
    {
    public:
        Convolution(const inference_engine_convolution_desc_t& desc, std::size_t id)
            : INode(id, { desc.input })
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            assert(inputs.size() == 1);
            return std::make_unique<GpuConvolution>(id_, inputs[0], desc_);
        }
    private:
        inference_engine_convolution_desc_t desc_;
    };

}  // namespace inference_engine