#pragma once
#include "../node.h"

namespace inference_engine
{
    class GpuActivation : public GpuNode
    {
    public:
        GpuActivation(std::size_t user_id, GpuNode* input, const inference_engine_activation_desc_t& desc)
            : GpuNode(user_id, input->get_output_tensor(), { input })
            , desc_(desc)
        {
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

        void compile(GpuContext& ctx) override
        {
            std::cout << "[Activation] Compile." << std::endl;
            assert(kernel_ == nullptr); // compile can happen only once
            return;
            const char* kernel_str =
                ""
                ""
                "T"
                ""
                "";

            const char* build_options = "";
            kernel_ = ctx.create_kernel("activation_relu_ref", kernel_str, std::strlen(kernel_str), build_options, INFERENCE_ENGINE_KERNEL_LANGUAGE_CM);
        }

        void initalize(GpuStream& stream) override
        {
            std::cout << "[Activation] Initialize." << std::endl;
        }

        GpuResource::Ptr execute(GpuStream& stream) override
        {
            std::cout << "[Activation] Execute." << std::endl;
            //assert(kernel_);

            //kernel_->set_arg(0, inputs()[0]->)

            //return resource_;
            return {};
        }

        std::string to_str() const override
        {
            // more details about the node here
            return "GpuActivation";
        }

    private:
        inference_engine_activation_desc_t desc_{};
        GpuKernel::Ptr kernel_;
    };

    class Activation : public INode
    {
    public:
        Activation(const inference_engine_activation_desc_t& desc, std::size_t id)
            : INode(id, { desc.input })
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            assert(inputs.size() == 1);
            return std::make_unique<GpuActivation>(id_, inputs[0], desc_);
        }
    private:
        inference_engine_activation_desc_t desc_;
    };

}  // namespace inference_engine