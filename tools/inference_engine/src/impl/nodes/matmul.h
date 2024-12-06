#pragma once
#include "../node.h"

namespace inference_engine
{
    class GpuActivation;

    class GpuMatMul : public GpuNode
    {
    public:
        GpuMatMul(std::size_t user_id, Tensor output_tensor, const std::vector<GpuNode*>& inputs, const inference_engine_matmul_desc_t& desc, const std::string& name)
            : GpuNode(user_id, output_tensor, inputs, name)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;

        GpuResource::Ptr execute(GpuStream& stream) override;

        std::string to_str() const override;

        bool fuse_with(const GpuActivation*) override;

    private:
        std::uint32_t get_M() const;
        std::uint32_t get_N() const;
        std::uint32_t get_K() const;

    private:
        inference_engine_matmul_desc_t desc_{};
        GpuKernel::Ptr kernel_ = nullptr;
    };

    class MatMul : public INode
    {
    public:
        MatMul(const inference_engine_matmul_desc_t& desc, std::size_t id)
            : INode(id, { desc.input_a, desc.input_b })
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override;

    private:
        Tensor compute_output_tensor(const Tensor& input_a, const Tensor& input_b);

    private:
        inference_engine_matmul_desc_t desc_{};
    };

}  // namespace inference_engine