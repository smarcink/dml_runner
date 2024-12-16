#pragma once
#include "../node.h"

namespace ai_driver
{
    class GpuElementwise : public GpuNode
    {
    public:
        GpuElementwise(std::size_t user_id, Tensor output_tensor, const std::vector<GpuNode*>& inputs, const ai_driver_elementwise_desc_t& desc, const std::string& name)
            : GpuNode(user_id, output_tensor, inputs, name)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override;

        void initalize(GpuStream& stream) override;

        GpuResource::Ptr execute(GpuStream& stream) override;

        std::string to_str() const override;

        PostOp create_post_op(GpuNode* new_input) const { return { PostOp::ElemWisePosOp{desc_, new_input } }; }

    private:
        ai_driver_elementwise_desc_t desc_{};
    };

    class Elementwise : public INode
    {
    public:
        Elementwise(const ai_driver_elementwise_desc_t& desc, std::size_t id, std::string_view name)
            : INode(id, { desc.input_a, desc.input_b }, name)
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

            return std::make_unique<GpuElementwise>(id_, compute_output_tensor(tensor_a, tensor_b), inputs, desc_, name_);
        }
    private:
        Tensor compute_output_tensor(const Tensor& input_a, const Tensor& input_b)
        {
            // just an example
            assert(input_a.data_type == input_b.data_type);
            assert(input_a.dims.size() == input_b.dims.size());
            assert(input_a.dims.size() == 4);
            Tensor ret{};
            ret.data_type = static_cast<DataType>(desc_.out_data_type);
            ret.dims = input_a.dims;
            ret.strides.assign({ 0,0,0,0 });
            return ret;
        }
    private:
        ai_driver_elementwise_desc_t desc_;
    };

}  // namespace ai_driver