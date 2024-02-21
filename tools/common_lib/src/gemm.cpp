#include "gemm.h"
#include "dnnl_utils.h"

std::vector<std::byte> dnnl_gemm_op::gemm(const bindings_t& bindings, opts_t opts)
{
    using namespace dnnl_utils;
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    static dnnl::stream stream(engine);
    const auto engine_kind = engine.get_kind();
    stream.wait();  // just to be sure we can freely upload the input data   


    dnnl::memory input_a_memory = [&](const auto& binding)
    {
        dnnl::memory ret;
        if (binding.transposed)
        {
            dnnl::memory::desc transposed_desc = matrix_transpose(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), binding.dt);
            ret = dnnl::memory(transposed_desc, engine);
        }
        else
        {
            ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        }
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_a);

    dnnl::memory input_b_memory = [&](const auto& binding)
    {
        dnnl::memory ret;
        if (binding.transposed)
        {
            dnnl::memory::desc transposed_desc = matrix_transpose(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), binding.dt);
            ret = dnnl::memory(transposed_desc, engine);
        }
        else
        {
            auto ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        }
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_b);
   
    dnnl::memory input_c_memory = [&](const auto& binding)
    {
        if (!binding.data)
        {
            return dnnl::memory{};
        }
        auto ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_c);

    dnnl::memory output_memory = [&]()
    {
        return dnnl::memory(to_dnnl_mem_desc(opts.output_shape, opts.out_layout, opts.out_dt), engine);
    }();

    dnnl::memory alpha_scale_memory = [&]()
    {
        const float alpha = opts.alpha / (opts.beta == 0.0f ? 1.0f : opts.beta);
        return create_dnnl_memory(binding_t{ reinterpret_cast<const std::byte*>(&alpha), DataType::eFp32, DataLayout::eW, TensorShape{1, 0, 0, 0} }, engine);
    }();

    dnnl::memory beta_scale_memory = [&]()
    {
        return create_dnnl_memory(binding_t{ reinterpret_cast<const std::byte*>(&opts.beta), DataType::eFp32, DataLayout::eW, TensorShape{1, 0, 0, 0} }, engine);
    }();

    dnnl::post_ops po{};
    dnnl::primitive_attr attrs{};
    if (opts.force_fp32_accumulator)
    {
        attrs.set_accumulation_mode(dnnl::accumulation_mode::strict);
    }

    auto has_scaling_factors = [&]()
    {
        return opts.alpha != 1.0f || opts.beta != 1.0f;
    };

    if (has_scaling_factors())
    {
        attrs.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
    }

    if (input_c_memory)
    {
        po.append_binary(dnnl::algorithm::binary_add, input_c_memory.get_desc());
    }

    if (has_scaling_factors())
    {
        po.append_binary(dnnl::algorithm::binary_mul, beta_scale_memory.get_desc());
    }

    if (opts.activation.type != ActivationType::eUnknown)
    {
        po.append_eltwise(to_dnnl_activation_type(opts.activation.type), opts.activation.alpha, opts.activation.beta);
    }

    attrs.set_post_ops(po);

    dnnl::matmul::primitive_desc matmul_desc(engine,
        input_a_memory.get_desc(),
        input_b_memory.get_desc(),
        {}, // we dont use bias for c_tensir
        output_memory.get_desc(),
        attrs
    );
    const auto guery_impl_str = matmul_desc.impl_info_str();
    std::cout << "ref query impl: " << guery_impl_str << std::endl;

    auto matmul = dnnl::matmul(matmul_desc);
    std::unordered_map<int, dnnl::memory> args;
    args.insert({ DNNL_ARG_SRC, input_a_memory });
    args.insert({ DNNL_ARG_WEIGHTS, input_b_memory });
    args.insert({ DNNL_ARG_DST, output_memory });

    std::size_t post_ops_idx = 0ull;
    if (input_c_memory)
    {
        args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_ops_idx) | DNNL_ARG_SRC_1, input_c_memory});
        post_ops_idx++;
    }

    if (has_scaling_factors())
    {
        args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, alpha_scale_memory });
        args.insert({ DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_ops_idx) | DNNL_ARG_SRC_1, beta_scale_memory });
        post_ops_idx++;
    }

    matmul.execute(stream, args);
    stream.wait();

    auto* out_dnnl_data = output_memory.map_data<uint8_t>();
    assert(out_dnnl_data != nullptr && "[dnnl][gemm] Couldnt map output memory!");

    const auto om_desc = output_memory.get_desc();
    const auto copy_size = om_desc.get_size();
    std::vector<std::byte> ret(copy_size);
    std::memcpy(ret.data(), out_dnnl_data, copy_size);
    output_memory.unmap_data(out_dnnl_data);
    return ret;
}