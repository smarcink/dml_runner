#include "model.h"
#include "inference_engine_model.h"
#include "inference_engine_tensor.h"

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

	const char* to_string(inference_engine_tensor_layout_t t)
	{
		switch (t)
		{
		case TENSOR_LAYOUT_NCHW: return "NCHW";
		case TENSOR_LAYOUT_NHWC: return "NHWC";
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
			if (tensor_a.layout == TENSOR_LAYOUT_NCHW && tensor_b.layout == TENSOR_LAYOUT_NCHW) {
				return tensor_a.dims[0] == tensor_b.dims[0] && // Batch size
					tensor_a.dims[1] == tensor_b.dims[1] && // Channels
					tensor_a.dims[3] == tensor_b.dims[2];   // Inner dimensions
			}
			
			if (tensor_a.layout == TENSOR_LAYOUT_NHWC && tensor_b.layout == TENSOR_LAYOUT_NHWC) {
				return tensor_a.dims[0] == tensor_b.dims[0] && // Batch size
					tensor_a.dims[3] == tensor_b.dims[3] && // Channels
					tensor_a.dims[2] == tensor_b.dims[1];   // Inner dimensions
			}

			// Handle other layouts or mismatched layouts if necessary
			return false;
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

	Tensor::Tensor(const inference_engine_tensor_t& tensor_desc, void* data_ptr) 
		: data_type(tensor_desc.data_type)
		, layout(tensor_desc.layout)
		, data(data_ptr)
	{
		for (int i = 0; i < INFERENCE_ENGINE_MAX_TENSOR_DIMS && tensor_desc.dims[i] != 0; ++i)
		{
			dims.push_back(tensor_desc.dims[i]);
			strides.push_back(tensor_desc.strides[i]);
		}
	}

	std::size_t Tensor::size() const
	{
		std::size_t total_size = 1;
		for (const auto& dim : dims)
		{
			total_size *= dim;
		}
		return total_size;
	}

	bool MatMul::check_inputs() const
	{
		return are_tensors_compatible_for_matmul(tensor_a(), tensor_b());
	}

} // namespace inference_engine