#pragma once
#include "../node.h"

namespace ai_driver
{
    class GpuActivation : public GpuNode
    {
    public:
        GpuActivation(std::size_t user_id, GpuNode* input, const ai_driver_activation_desc_t& desc, const std::string& name)
            : GpuNode(user_id, input->get_output_tensor(), { input }, name)
            , desc_(desc)
        {
            output_tensor_.data_type = desc_.out_data_type;
        }

        GpuActivation(GpuActivation&& rhs) noexcept
            : GpuNode(std::move(rhs))
        {
            std::swap(desc_, rhs.desc_);
            std::swap(kernel_, rhs.kernel_);
        }

        GpuActivation& operator=(GpuActivation&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(this->desc_, rhs.desc_);
                std::swap(this->kernel_, rhs.kernel_);
            }
            return *this;
        }

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;

        GpuResource::Ptr execute(GpuStream& stream) override;

        std::string to_str() const override;

        PostOp create_post_op() const { return PostOp{desc_}; }

    private:
        ai_driver_activation_desc_t desc_{};
        GpuKernel::Ptr kernel_;
    };

    class Activation : public INode
    {
    public:
        Activation(const ai_driver_activation_desc_t& desc, std::size_t id, std::string_view name)
            : INode(id, { desc.input }, name)
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            assert(inputs.size() == 1);
            return std::make_unique<GpuActivation>(id_, inputs[0], desc_, name_);
        }
    private:
        ai_driver_activation_desc_t desc_;
    };

}  // namespace ai_driver