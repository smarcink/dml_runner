#pragma once
#include "../node.h"

namespace inference_engine
{
    class GpuMatMul : public GpuNode
    {
    public:
        GpuMatMul(std::size_t user_id, Tensor output_tensor, const std::vector<GpuNode*>& inputs, const inference_engine_matmul_desc_t& desc)
            : GpuNode(user_id, output_tensor, inputs)
            , desc_(desc)
        {
        }

        void compile(GpuContext& ctx) override
        {
            std::cout << "[MatMul] Compile." << std::endl;
        }

        void initalize(GpuStream& stream) override
        {
            std::cout << "[MatMul] Initialize." << std::endl;
        }

        GpuResource::Ptr execute(GpuStream& stream) override
        {
            std::cout << "[MatMul] Execute." << std::endl;
            //return resource_;
            return {};
        }

        std::string to_str() const override
        {
            // more details about the node here
            return "GpuMatMul";
        }
    private:
        inference_engine_matmul_desc_t desc_{};
    };

    class MatMul : public INode
    {
    public:
        MatMul(const inference_engine_matmul_desc_t& desc, std::size_t id)
            : INode(id, { desc.input_a, desc.input_b })
            , desc_(desc)
        {
        }

        std::unique_ptr<GpuNode> create_gpu_node(const std::vector<GpuNode*>& inputs) override
        {
            auto are_tensors_compatible_for_matmul = [](const Tensor& tensor_a, const Tensor& tensor_b) {
                // Check if both tensors have at least 2 dimensions
                if (tensor_a.dims.size() < 2 || tensor_b.dims.size() < 2) {
                    return false;
                }

                // For 4D tensors, ensure the batch size and channels match, and the inner dimensions are compatible
                if (tensor_a.dims.size() == 4 && tensor_b.dims.size() == 4) {
                    std::size_t cols_a = tensor_a.dims[tensor_a.dims.size() - 1];
                    std::size_t rows_b = tensor_b.dims[tensor_b.dims.size() - 2];
                    return cols_a == rows_b;
                }

                // For 2D tensors, check if the number of columns in tensor_a matches the number of rows in tensor_b
                if (tensor_a.dims.size() == 2 && tensor_b.dims.size() == 2)
                {
                    std::size_t cols_a = tensor_a.dims[tensor_a.dims.size() - 1];
                    std::size_t rows_b = tensor_b.dims[tensor_b.dims.size() - 2];
                    return cols_a == rows_b;
                }
                return false; // unknown format?
                };
            if (inputs.size() != 2)
            {
                throw std::invalid_argument("there must be exactly two inputs for this operation!");
            }

            const auto tensor_a = inputs[0]->get_output_tensor();
            const auto tensor_b = inputs[1]->get_output_tensor();
            if (!are_tensors_compatible_for_matmul(tensor_a, tensor_b))
            {
                throw std::invalid_argument("tensors don't match!");
            }
            return std::make_unique<GpuMatMul>(id_, compute_output_tensor(tensor_a, tensor_b), inputs, desc_);
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
            ret.dims.push_back(input_a.dims[0]);
            ret.dims.push_back(input_a.dims[1]);
            ret.dims.push_back(input_a.dims[2]);
            ret.dims.push_back(input_b.dims[3]);
            ret.strides.assign({ 0,0,0,0 });
            return ret;
        }
    private:
        inference_engine_matmul_desc_t desc_{};
    };

}  // namespace inference_engine