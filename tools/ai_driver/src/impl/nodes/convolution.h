#pragma once
#include "../node.h"

namespace ai_driver
{
    class GpuActivation;
    class GpuElementwise;

    class GpuConvolution : public GpuNode
    {
    public:
        GpuConvolution(std::size_t user_id, GpuNode* input, const ai_driver_convolution_desc_t& desc, const std::string& name)
            : GpuNode(user_id, input->get_output_tensor(), { input }, name)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;

        GpuResource::Ptr execute(GpuStream& stream) override;

        std::string to_str() const override;

        bool fuse_with(const GpuActivation*) override;      
        bool fuse_with(const GpuElementwise*) override;      

    private:
        ai_driver_convolution_desc_t desc_{};
    };

    class Convolution : public INode
    {
    public:
        Convolution(const ai_driver_convolution_desc_t& desc, std::size_t id, std::string_view name)
            : INode(id, { desc.input }, name)
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            assert(inputs.size() == 1);
            return std::make_unique<GpuConvolution>(id_, inputs[0], desc_, name_);
        }
    private:
        ai_driver_convolution_desc_t desc_;
    };

}  // namespace ai_driver