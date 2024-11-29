#include "model.h"
#include "inference_engine_model.h"
#include "inference_engine_tensor.h"
#include "error.h"

namespace inference_engine {

    const char* to_string(ModelNodeType t)
    {
        switch (t)
        {
        case ModelNodeType::ePort: return "Port";
        case ModelNodeType::eMatmul: return "MatMul";
        case ModelNodeType::eActivation: return "Activation";
        case ModelNodeType::eConvolution: return "Convolution";
        }
        return "Unknown";
    }

    bool are_tensors_compatible_for_matmul(const Tensor& tensor_a, const Tensor& tensor_b) {
        // Check if both tensors have at least 2 dimensions
        if (tensor_a.dims.size() < 2 || tensor_b.dims.size() < 2) {
            return false;
        }

        // For 4D tensors, ensure the batch size and channels match, and the inner dimensions are compatible
        if (tensor_a.dims.size() == 4 && tensor_b.dims.size() == 4) {
            // todo...
            return true;
        }

        // For 2D tensors, check if the number of columns in tensor_a matches the number of rows in tensor_b
        if (tensor_a.dims.size() == 2 && tensor_b.dims.size() == 2)
        {
            std::size_t cols_a = tensor_a.dims[tensor_a.dims.size() - 1];
            std::size_t rows_b = tensor_b.dims[tensor_b.dims.size() - 2];
            return cols_a == rows_b;
        }

        return false; // unknown format?
    }

    MatMul::MatMul(const inference_engine_matmul_desc_t& desc)
        : INode(ModelNodeType::eMatmul, { to_node(desc.input_a), to_node(desc.input_b) })
        , desc_(desc)
    {
        /*if (inputs_.size() != 2 || 
            inputs_[0]->output_tensors().empty() || 
            inputs_[1]->output_tensors().empty() ||
            !are_tensors_compatible_for_matmul(inputs_[0]->output_tensors()[0], inputs_[1]->output_tensors()[0]))
            throw inference_engine_exception(INFERENCE_ENGINE_RESULT_INVALID_ARGUMENT);*/

        // what if our inputs have more outputs... ?
    }

} // namespace inference_engine