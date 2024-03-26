#include "conv.h"
#include "dnnl_utils.h"

inline dnnl::primitive_attr CreateEltwisePostOps(const ActivationSettings& activation, bool use_fp32_accu)
{
    // create a post-op with relu
    dnnl::post_ops ops;
    dnnl::primitive_attr attr;

    // sanity check
    assert(attr.get_scratchpad_mode() == dnnl::scratchpad_mode::library);
    // set scratchpad mode to user provided
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (use_fp32_accu)
    {
        attr.set_accumulation_mode(dnnl::accumulation_mode::strict);
    }

    if (activation.type != ActivationType::eUnknown)
    {
        ops.append_eltwise(dnnl_utils::to_dnnl_activation_type(activation.type), activation.alpha, activation.beta);
        // create an attribute and set the corresponding post op
        attr.set_post_ops(ops);
    }
    return attr;
}

template<typename T>
inline std::vector<std::byte> run_inference(const dnnl_conv_op::bindings_t& bindings, dnnl_conv_op::opts_t opts, dnnl::algorithm algo)
{
    using namespace dnnl_utils;
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);

    const auto enable_profiling = opts.execution_iterations > 1;
    dnnl::stream stream = [&]()
    {
        auto stream_flags = dnnl::stream::flags::default_flags;
        if (enable_profiling)
        {
            stream_flags |= dnnl::stream::flags::profiling;
        }
        return dnnl::stream(engine, stream_flags);
    }();
    const auto engine_kind = engine.get_kind();

    dnnl::set_jit_dump(false);

    stream.wait();  // just to be sure we can freely upload the input data    

    dnnl::memory input_memory = [&](const auto& binding)
    {
        auto ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input);

    dnnl::memory bias_memory = [&](const auto& binding)
    {
        if (!binding.data)  // no bias
        {
            return dnnl::memory{};

        }
        auto ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.bias);


    dnnl::memory output_memory = [&]()
    {
        return dnnl::memory(to_dnnl_mem_desc(opts.output_shape, opts.out_layout, opts.out_dt), engine);
    }();


    const dnnl::memory::dims pad{ opts.inp_pad, opts.inp_pad };
    const dnnl::memory::dims stride{ opts.stride.h, opts.stride.w };
    const dnnl::memory::dims dilates{ opts.dilates.h, opts.dilates.w };
    const dnnl::primitive_attr attr = CreateEltwisePostOps(opts.activation, opts.use_fp32_accu);

    const dnnl::memory::desc filter_any_fmt_desc = [](const auto& bind, const auto& group_count)
    {
        const auto fmt = dnnl::memory::format_tag::any;
        const auto dt = to_dnnl_data_type(bind.dt);  
        dnnl::memory::dims dims = to_dnnl_dims(bind.shape);
        if constexpr (std::is_same_v<T, dnnl::deconvolution_forward>)
        {
            std::swap(dims[0], dims[1]);
        }
        if (group_count != 1)
        {
            dims[0] /= group_count;
            dims[1] /= group_count;

            dnnl::memory::dims dims_temp{ group_count };
            dims_temp.insert(dims_temp.end(), dims.begin(), dims.end());
            dims = dims_temp;
        }
        return dnnl::memory::desc(dims, dt, fmt);
    }(bindings.filter, opts.groups);

    const auto conv_desc = T::primitive_desc(engine,
        dnnl::prop_kind::forward_inference,
        algo,
        input_memory.get_desc(),
        filter_any_fmt_desc,
        bindings.bias.data ? bias_memory.get_desc() : dnnl::memory::desc{},
        output_memory.get_desc(),
        stride, dilates, pad, pad, attr);


    const auto filter_output_desc_mem = conv_desc.query_md(dnnl::query::weights_md);
    dnnl::memory filter_memory(filter_output_desc_mem, engine);

    // weights reorder
    {
        dnnl::memory filter_input_memory = [&](const auto& binding)
        {
            const auto dims = filter_any_fmt_desc.get_dims();
            const auto dt = filter_any_fmt_desc.get_data_type();
            // use typed format for input resource for reorder primitive
            auto ft = dnnl::memory::format_tag::undef;
            if constexpr (std::is_same_v<T, dnnl::convolution_forward>)
            {
                if (binding.layout == DataLayout::eNCHW)
                {
                    ft = opts.groups > 1 ? dnnl::memory::format_tag::goihw : dnnl::memory::format_tag::oihw;
                }
                else if (binding.layout == DataLayout::eNHWC)
                {
                    ft = opts.groups > 1 ? dnnl::memory::format_tag::gohwi : dnnl::memory::format_tag::ohwi;
                }
            }
            else if constexpr (std::is_same_v<T, dnnl::deconvolution_forward>)
            {
                // for tranposed convolutions (deconvolutions) we need to flip input and output channels!
                if (binding.layout == DataLayout::eNCHW)
                {
                    ft = opts.groups > 1 ? dnnl::memory::format_tag::giohw : dnnl::memory::format_tag::iohw;
                }
                else if (binding.layout == DataLayout::eNHWC)
                {
                    ft = opts.groups > 1 ? dnnl::memory::format_tag::acdeb : dnnl::memory::format_tag::ihwo;
                }
            }
            assert(ft != dnnl::memory::format_tag::undef);
            auto ret = dnnl::memory({ dims, dt, ft }, engine);
            copy_to_dnnl_memory(ret, binding.data);
            return ret;
        }(bindings.filter);

        dnnl::reorder::primitive_desc reorder_desc(filter_input_memory, filter_memory);
        auto reorder = dnnl::reorder(reorder_desc);
        reorder.execute(stream, { { DNNL_ARG_SRC, filter_input_memory }, { DNNL_ARG_DST, filter_memory } });
        stream.wait();

        if (opts.dump_weights)
        {
            dump_buffer_to_file(filter_memory, "dnnl_weights_data.dat");
        }
    }

    const auto scratchpad_desc_mem = conv_desc.query_md(dnnl::query::scratchpad_md);
    dnnl::memory scratchpad_memory(scratchpad_desc_mem, engine);

    const auto guery_impl_str = conv_desc.impl_info_str();
    std::cout << "ref query impl: " << guery_impl_str << std::endl;

    const auto start = std::chrono::steady_clock::now();

    if (enable_profiling)
    {
        dnnl::reset_profiling(stream);
    }

    T convolution(conv_desc.get());
    for (int i = 0; i < opts.execution_iterations; i++)
    {
        convolution.execute(stream, { { DNNL_ARG_SRC, input_memory }, {DNNL_ARG_WEIGHTS, filter_memory}, {DNNL_ARG_BIAS, bias_memory}, {DNNL_ARG_DST, output_memory},
            {DNNL_ARG_SCRATCHPAD, scratchpad_memory} });
    }
    stream.wait();

    if (enable_profiling)
    {
        const auto profiling_usecs_data = dnnl::get_profiling_data(stream, dnnl::profiling_data_kind::time);
        const auto avg_perf = std::accumulate(profiling_usecs_data.begin(), profiling_usecs_data.end(), 0.0) / profiling_usecs_data.size();
        std::cout << "OneDNN avg performance time: " << (float)avg_perf / 1000.0f << " ms." << std::endl;
    }

    if (opts.dump_scratchpad)
    {
        dump_buffer_to_file(scratchpad_memory, "dnnl_scratchpad_data.dat");
    }

    auto* out_dnnl_data = output_memory.map_data<uint8_t>();
    assert(out_dnnl_data != nullptr && "[dnnl][conv] Couldnt map output memory!");

    const auto om_desc = output_memory.get_desc();
    const auto copy_size = om_desc.get_size();
    std::vector<std::byte> ret(copy_size);
    std::memcpy(ret.data(), out_dnnl_data, copy_size);
    output_memory.unmap_data(out_dnnl_data);
    return ret;
}


std::vector<std::byte> dnnl_conv_op::convolution(const bindings_t& bindings, opts_t opts)
{
    return run_inference<dnnl::convolution_forward>(bindings, opts, opts.force_winograd ? dnnl::algorithm::convolution_winograd : dnnl::algorithm::convolution_direct);
}

std::vector<std::byte> dnnl_conv_op::deconvolution(const bindings_t& bindings, opts_t opts)
{
    return run_inference<dnnl::deconvolution_forward>(bindings, opts, opts.force_winograd ? dnnl::algorithm::deconvolution_winograd : dnnl::algorithm::deconvolution_direct);
}