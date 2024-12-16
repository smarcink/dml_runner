#include "matmul.h"
#include "activation.h"
#include <iostream>
#include <format>

namespace ai_driver
{
    bool GpuMatMul::fuse_with(const GpuActivation* activation)
    {
        assert(activation);
        std::cout << "matmul fuse with... activation\n";
        outputs_ = activation->get_outputs();
        for (auto& out : outputs_)
            GpuNode::replace_input(out, activation, this);

        post_ops_.push_back(activation->create_post_op());
        return true;
    }


void GpuMatMul::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
    assert(kernel_ == nullptr); // compile can happen only once

    std::string code_string
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
            case DataType::fp16:
                return "half";
            case DataType::fp32:
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

    if (!post_ops_.empty())
    {
        // insert post ops just before "\noutput[id_m" in the code_string
        std::size_t pos = code_string.find("\noutput[id_m");
        assert(pos != std::string::npos);
        for (auto& op : post_ops_)
        {
            if (const auto activation_params = std::get_if<ai_driver_activation_desc_t>(&op.params_))
            {
                const std::string op_code = [](ai_driver_activation_desc_t* params) {
                    if (params->type == AI_DRIVER_ACTIVATION_TYPE_RELU)
                        return std::string("\naccu = fmax(accu, (DT)0.0f);");
                    if (params->type == AI_DRIVER_ACTIVATION_TYPE_LINEAR)
                        return std::format("\naccu = ({} * accu + {});", params->params.linear.a, params->params.linear.b);
                    assert(!"Unknown activation type. Cant create activation post ops..");
                    return std::string();
                    }(activation_params);
                code_string.insert(pos, op_code);
                pos += op_code.length();
            }
        }        
    }
    kernel_ = ctx.create_kernel("matmul_ref", code_string.c_str(), code_string.length(), build_options.c_str(), AI_DRIVER_KERNEL_LANGUAGE_OCL);
}

void GpuMatMul::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

ai_driver::GpuResource::Ptr GpuMatMul::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
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

std::string GpuMatMul::to_str() const
{
    return node_utils::create_name("GpuMatMul", name_);
}

std::uint32_t GpuMatMul::get_M() const
{
    assert(!get_inputs().empty());
    const auto tensor_a = get_inputs()[0]->get_output_tensor();
    return static_cast<std::uint32_t>(tensor_a.dims[tensor_a.dims.size() - 2]);
}

std::uint32_t GpuMatMul::get_N() const
{
    assert(get_inputs().size() >= 2);
    const auto tensor_b = get_inputs()[1]->get_output_tensor();
    return static_cast<std::uint32_t>(tensor_b.dims[tensor_b.dims.size() - 1]);
}

std::uint32_t GpuMatMul::get_K() const
{
    assert(!get_inputs().empty());
    const auto tensor_a = get_inputs()[0]->get_output_tensor();
    return static_cast<std::uint32_t>(tensor_a.dims[tensor_a.dims.size() - 1]);
}

std::unique_ptr<GpuNode> MatMul::create_gpu_node(const std::vector<GpuNode*>& inputs)
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

    return std::make_unique<GpuMatMul>(id_, compute_output_tensor(tensor_a, tensor_b), inputs, desc_, name_);
}

Tensor MatMul::compute_output_tensor(const Tensor& input_a, const Tensor& input_b)
{
    // just an example
    assert(input_a.data_type == input_b.data_type);
    assert(input_a.dims.size() == input_b.dims.size());
    assert(input_a.dims.size() == 4);
    Tensor ret{};
    ret.data_type = static_cast<DataType>(desc_.out_data_type);
    ret.dims.push_back(input_a.dims[0]);
    ret.dims.push_back(input_a.dims[1]);
    ret.dims.push_back(input_a.dims[2]);
    ret.dims.push_back(input_b.dims[3]);
    ret.strides.assign({ 0,0,0,0 });
    return ret;
}

} // namespace ai_driver