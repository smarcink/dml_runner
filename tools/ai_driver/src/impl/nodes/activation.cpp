#include "activation.h"
#include <sstream>
#include <iomanip>
#include <format>

void ai_driver::GpuActivation::compile(GpuContext& ctx)
{
    std::cout << std::format("[{}] Compile.\n", to_str());
    assert(kernel_ == nullptr); // compile can happen only once

    const char* code_string
        = 
        "#if defined(cl_khr_fp16)\n"
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable)\n "
        "#endif \n"

        "__attribute__((reqd_work_group_size(1,1, 1))) \n"
        "__kernel void activation_ref(__global DT* input, __global DT* output)\n "
        "{\n "
        "const uint id = get_global_id(0); \n"
        "output[id] = ACTIVATION_OP(input[id]); \n"
        "} \n";

    const std::string dt = [](const auto dt) {
        switch (dt)
        {
        case AI_DRIVER_DATA_TYPE_FP16:
            return "half";
        case AI_DRIVER_DATA_TYPE_FP32:
            return "float";
        default:
            assert(!"unsupported");
        }
        return "";
    }(get_output_tensor().data_type);

    std::string build_options = " -DDT=" + dt;

    switch (desc_.type)
    {
    case AI_DRIVER_ACTIVATION_TYPE_RELU:
    {
        build_options += " -DACTIVATION_OP(input)=\"fmax(input, (DT)0.0f)\"";
    }
    break;
    case AI_DRIVER_ACTIVATION_TYPE_LINEAR:
    {
        build_options += " -DLINEAR_A=" + std::to_string(desc_.params.linear.a);
        build_options += " -DLINEAR_B=" + std::to_string(desc_.params.linear.b);
        build_options += " -DACTIVATION_OP(input)=\"(LINEAR_A * input + LINEAR_B)\"";
    }
    break;
    default:
        assert(!"Unknown activation type. Cant create reference activation kernel.");
    }


    kernel_ = ctx.create_kernel("activation_ref", code_string, std::strlen(code_string), build_options.c_str(), AI_DRIVER_KERNEL_LANGUAGE_OCL);
}

void ai_driver::GpuActivation::initalize(GpuStream& stream)
{
    std::cout << std::format("[{}] Initialize.\n", to_str());
}

ai_driver::GpuResource::Ptr ai_driver::GpuActivation::execute(GpuStream& stream)
{
    std::cout << std::format("[{}] Execute.\n", to_str());
    assert(kernel_);

    auto input_rsc = get_inputs().at(0)->get_resource().get();
    assert(input_rsc);
    auto output_rsc = resource_.get();
    assert(output_rsc);

    const auto gws_x = [](const auto& dims)
        {
            std::uint32_t ret = 1;
            for (const auto& d : dims)
            {
                ret *= static_cast<std::uint32_t>(d);
            }
            return ret;
        }(output_tensor_.dims);

    kernel_->set_arg(0, *input_rsc);
    kernel_->set_arg(1, *output_rsc);

    std::uint32_t gws[3] = { gws_x, 1, 1 };
    std::uint32_t lws[3] = { 1, 1, 1 };
    stream.dispatch_kernel(*kernel_.get(), gws, lws);

    return resource_;
}

std::string ai_driver::GpuActivation::to_str() const
{
    // more details about the node here
    return node_utils::create_name("GpuActivation", name_);
}