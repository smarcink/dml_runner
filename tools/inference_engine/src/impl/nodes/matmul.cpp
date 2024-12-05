#include "matmul.h"
#include "..\gpu_visitor.h"
#include "activation.h"
#include <iostream>

namespace inference_engine
{
    void GpuMatMul::fuse_with(const GpuActivation* activation)
    {
        std::cout << "matmul fuse with... activation\n";
        outputs_ = activation->get_outputs();
        for (auto& out : outputs_)
            out->replace_input(activation, this);
    }

    void GpuMatMul::accept(GpuVisitor* visitor)
    {
        visitor->visit(this);
    }
} // namespace inference_engine

void inference_engine::GpuMatMul::compile(GpuContext& ctx)
{
    std::cout << "[MatMul] Compile." << std::endl;
    assert(kernel_ == nullptr); // compile can happen only once

    const char* code_string
        =
        "#if defined(cl_khr_fp16)\n"
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable)\n "
        "#endif \n"
        "__attribute__((reqd_work_group_size(1,1, 1))) \n"
        "__kernel void matmul_ref(__global DT* input_a, __global DT* input_b, __global DT* output)\n "
        "{\n "
        "const uint id_m = get_global_id(0); \n"
        "const uint id_n = get_global_id(1); \n"
        "float accu = 0.0f; \n"
        "for(int id_k = 0; id_k < K; id_k++) \n"
        "{ \n"
        "    accu += input_a[id_m * K + id_k] * input_b[id_k * N + id_n];\n"
        "} \n"
        "output[id_m * N + id_n] = (DT)accu; \n"
        "} \n";

    const std::string dt = [](const auto dt)
        {
            switch (dt)
            {
            case INFERENCE_ENGINE_DATA_TYPE_FP16:
                return "half";
            case INFERENCE_ENGINE_DATA_TYPE_FP32:
                return "float";
            default:
                assert(!"unsupported");
            }
            return "";
        }(get_output_tensor().data_type);

    std::string build_options = " -DDT=" + dt;
    build_options += " -DM=" + std::to_string(get_M());
    build_options += " -DK=" + std::to_string(get_K());
    build_options += " -DN=" + std::to_string(get_N());
    kernel_ = ctx.create_kernel("matmul_ref", code_string, std::strlen(code_string), build_options.c_str(), INFERENCE_ENGINE_KERNEL_LANGUAGE_OCL);
}

inference_engine::GpuResource::Ptr inference_engine::GpuMatMul::execute(GpuStream& stream)
{
    std::cout << "[MatMul] Execute." << std::endl;
    auto input_rsc_a = get_inputs().at(0)->get_resource().get();
    auto input_rsc_b = get_inputs().at(1)->get_resource().get();
    assert(input_rsc_a);
    assert(input_rsc_b);
    auto output_rsc = resource_.get();
    assert(output_rsc);

    kernel_->set_arg(0, *input_rsc_a);
    kernel_->set_arg(1, *input_rsc_b);
    kernel_->set_arg(2, *output_rsc);

    std::uint32_t gws[3] = { get_M(), get_N(), 1 };
    std::uint32_t lws[3] = { 1, 1, 1 };
    stream.dispatch_kernel(*kernel_.get(), gws, lws);

    return resource_;
}

std::uint32_t inference_engine::GpuMatMul::get_M() const
{
    assert(!get_inputs().empty());
    const auto tensor_a = get_inputs()[0]->get_output_tensor();
    return static_cast<std::uint32_t>(tensor_a.dims[tensor_a.dims.size() - 2]);
}

std::uint32_t inference_engine::GpuMatMul::get_N() const
{
    assert(get_inputs().size() >= 2);
    const auto tensor_b = get_inputs()[1]->get_output_tensor();
    return static_cast<std::uint32_t>(tensor_b.dims[tensor_b.dims.size() - 1]);
}

std::uint32_t inference_engine::GpuMatMul::get_K() const
{
    assert(!get_inputs().empty());
    const auto tensor_a = get_inputs()[0]->get_output_tensor();
    return static_cast<std::uint32_t>(tensor_a.dims[tensor_a.dims.size() - 1]);
}

std::unique_ptr<inference_engine::GpuNode> inference_engine::MatMul::create_gpu_node(const std::vector<GpuNode*>& inputs)
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

    if (tensor_a.dims[0] != 1 || tensor_a.dims[1] != 1)
    {
        throw std::invalid_argument("Not supported path in MatMul yet. ToDo: add batch support.");
    }

    return std::make_unique<GpuMatMul>(id_, compute_output_tensor(tensor_a, tensor_b), inputs, desc_);
}

inference_engine::Tensor inference_engine::MatMul::compute_output_tensor(const Tensor& input_a, const Tensor& input_b)
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
