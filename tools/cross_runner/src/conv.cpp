#include "conv.h"
#include "dnnl_utils.h"

inline dnnl::primitive_attr CreateEltwisePostOps(dnnl::algorithm Activation, float alpha, float beta)
{
    // create a post-op with relu
    dnnl::post_ops ops;
    dnnl::primitive_attr attr;

    // sanity check
    assert(attr.get_scratchpad_mode() == dnnl::scratchpad_mode::library);
    // set scratchpad mode to user provided
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (Activation != dnnl::algorithm::undef)
    {
        ops.append_eltwise(Activation
            , alpha, beta);
        // create an attribute and set the corresponding post op
        attr.set_post_ops(ops);
    }
    return attr;
}


std::vector<std::byte> dnnl_conv_op::convolution(const bindings_t& bindings, opts_t opts)
{
    using namespace dnnl_utils;
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    static dnnl::stream stream(engine);
    const auto engine_kind = engine.get_kind();

    dnnl::set_jit_dump(false);

    stream.wait();  // just to be sure we can freely upload the input data    

    dnnl::memory input_memory = [&](const auto& binding)
    {
        const auto dims = to_dnnl_dims(binding.shape);
        const auto dt = to_dnnl_data_type(binding.dt);
        const auto ft = to_dnnl_format(binding.layout);
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input);

    dnnl::memory bias_memory = [&](const auto& binding)
    {
        if (!binding.data)  // no bias
        {
            return dnnl::memory{};
        }
        const auto dims = dnnl::memory::dims{ binding.shape.n };
        const auto dt = to_dnnl_data_type(binding.dt);
        const auto ft = dnnl::memory::format_tag::a;
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.bias);


    dnnl::memory output_memory = [&]()
    {
        return create_dnnl_memory(binding_t{ nullptr, opts.out_dt, opts.out_layout, opts.output_shape }, engine);
    }();


    const dnnl::memory::dims pad{ opts.inp_pad, opts.inp_pad };
    const dnnl::memory::dims stride{ opts.stride.h, opts.stride.w };
    const dnnl::primitive_attr attr = CreateEltwisePostOps(static_cast<dnnl::algorithm>(opts.activation_type), opts.activation_alpha, opts.activation_beta);
    const dnnl::convolution_forward::primitive_desc conv_desc(engine,
        dnnl::prop_kind::forward_inference, 
        opts.force_winograd ? dnnl::algorithm::convolution_winograd : dnnl::algorithm::convolution_direct,
        input_memory.get_desc(),
        dnnl::memory::desc{to_dnnl_dims(bindings.filter.shape), to_dnnl_data_type(bindings.filter.dt), dnnl::memory::format_tag::any },
        bindings.bias.data ? bias_memory.get_desc() : dnnl::memory::desc{},
        output_memory.get_desc(),
        stride, pad, pad, attr);

    const auto filter_output_desc_mem = conv_desc.query_md(dnnl::query::weights_md);
    dnnl::memory filter_memory(filter_output_desc_mem, engine);

    // weights reorder
    {
        dnnl::memory filter_input_memory = [&](const auto& binding)
        {
            const auto dims = to_dnnl_dims(binding.shape);
            const auto dt = to_dnnl_data_type(binding.dt);
            auto ft = dnnl::memory::format_tag::undef;
            if (binding.layout == DataLayout::eNCHW)
            {
                ft = dnnl::memory::format_tag::oihw;
            }
            else if (binding.layout == DataLayout::eNHWC)
            {
                ft = dnnl::memory::format_tag::ohwi;
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

    dnnl::convolution_forward convolution(conv_desc);
    convolution.execute(stream, { { DNNL_ARG_SRC, input_memory }, {DNNL_ARG_WEIGHTS, filter_memory}, {DNNL_ARG_BIAS, bias_memory}, {DNNL_ARG_DST, output_memory}, 
        {DNNL_ARG_SCRATCHPAD, scratchpad_memory} });
    stream.wait();

    if (opts.dump_scratchpad)
    {
        dump_buffer_to_file(scratchpad_memory, "dnnl_scratchpad_data.dat");
    }

    auto* out_dnnl_data = output_memory.map_data<uint8_t>();
    assert(out_dnnl_data != nullptr && "[dnnl][conv] Couldnt map output memory!");

    const auto om_desc = output_memory.get_desc();
    const auto om_dims = om_desc.get_dims();
    const auto copy_size = dimensions_product(om_dims) * dnnl::memory::data_type_size(om_desc.get_data_type());
    std::vector<std::byte> ret(copy_size);
    std::memcpy(ret.data(), out_dnnl_data, copy_size);
    output_memory.unmap_data(out_dnnl_data);
    return ret;
}