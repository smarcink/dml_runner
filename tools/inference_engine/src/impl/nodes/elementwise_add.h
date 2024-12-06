#pragma once
#include "../node.h"

namespace inference_engine
{
    class GpuElementwiseAdd : public GpuNode
    {
    public:
        GpuElementwiseAdd(std::size_t user_id, Tensor output_tensor, const std::vector<GpuNode*>& inputs, const inference_engine_elementwise_add_desc_t& desc, const std::string& name)
            : GpuNode(user_id, output_tensor, inputs, name)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;

        GpuResource::Ptr execute(GpuStream& stream) override;

        std::string to_str() const override;

        void accept(GpuVisitor* visitor) override;

    private:
        inference_engine_elementwise_add_desc_t desc_{};
    };

    class ElementwiseAdd : public INode
    {
    public:
        ElementwiseAdd(const inference_engine_elementwise_add_desc_t& desc, std::size_t id)
            : INode(id, { desc.input_a, desc.input_b })
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            assert(inputs.size() == 2);

            const auto& tensor_a = inputs[0]->get_output_tensor();
            const auto& tensor_b = inputs[1]->get_output_tensor();
            if (tensor_a.dims.size() == 0 || tensor_a != tensor_b)
            {
                throw std::invalid_argument("tensors don't match!");
            }

            return std::make_unique<GpuElementwiseAdd>(id_, compute_output_tensor(tensor_a, tensor_b), inputs, desc_, name_);
        }
    private:
        Tensor compute_output_tensor(const Tensor& input_a, const Tensor& input_b)
        {
            // just an example
            assert(input_a.data_type == input_b.data_type);
            assert(input_a.dims.size() == input_b.dims.size());
            assert(input_a.dims.size() == 4);
            Tensor ret{};
            ret.data_type = input_a.data_type;
            ret.dims = input_a.dims;
            ret.strides.assign({ 0,0,0,0 });
            return ret;
        }
    private:
        inference_engine_elementwise_add_desc_t desc_;
    };

}  // namespace inference_engine