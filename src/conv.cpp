#include "conv.h"
#include "dnnl_utils.h"

inline dnnl::memory create_dnnl_memory(const cpu_op::binding_t binding, dnnl::engine& engine)
{
    const auto dims = to_dnnl_dims(binding.dims);
    const auto dt = to_dnnl_data_type(binding.dt);
    const auto ft = to_dnnl_format(binding.layout);
    return dnnl::memory({ dims, dt, ft }, engine);
}

std::vector<std::byte> cpu_op::convolution(const bindings_t& bindings, opts_t opts)
{
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    static dnnl::stream stream(engine);
    const auto engine_kind = engine.get_kind();

    stream.wait();  // just to be sure we can freely upload the input data    

    dnnl::memory input_memory = [&](const auto& binding)
    {
        const auto dims = to_dnnl_dims(binding.dims);
        const auto dt = to_dnnl_data_type(binding.dt);
        const auto ft = to_dnnl_format(binding.layout);
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input);

    dnnl::memory filter_memory = [&](const auto& binding)
    {
        const auto dims = to_dnnl_dims(binding.dims);
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


    dnnl::memory bias_memory = [&](const auto& binding)
    {
        if (!binding.data)  // no bias
        {
            return dnnl::memory{};
        }
        const auto dims = dnnl::memory::dims{ binding.dims[0] };
        const auto dt = to_dnnl_data_type(binding.dt);
        const auto ft = dnnl::memory::format_tag::a;
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.bias);


    dnnl::memory output_memory = [&]()
    {
        std::vector<std::uint32_t> output_dims;
        output_dims.push_back(bindings.input.dims[0]);
        output_dims.push_back(bindings.filter.dims[0]);
        output_dims.push_back((bindings.input.dims[2] - bindings.filter.dims[2] + opts.inp_pad + opts.inp_pad) / opts.stride + 1);
        output_dims.push_back((bindings.input.dims[3] - bindings.filter.dims[3] + opts.inp_pad + opts.inp_pad) / opts.stride + 1);
        return create_dnnl_memory(binding_t{ nullptr, opts.out_dt, opts.out_layout, output_dims }, engine);
    }();


    const dnnl::memory::dims pad{ opts.inp_pad, opts.inp_pad };
    const dnnl::memory::dims stride{ opts.stride, opts.stride };
    const dnnl::convolution_forward::primitive_desc conv_desc(engine,
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        input_memory.get_desc(), filter_memory.get_desc(), bindings.bias.data ? bias_memory.get_desc() : dnnl::memory::desc{}, output_memory.get_desc(), stride, pad, pad);

    dnnl::convolution_forward convolution(conv_desc);
    convolution.execute(stream, { { DNNL_ARG_SRC, input_memory }, {DNNL_ARG_WEIGHTS, filter_memory}, {DNNL_ARG_BIAS, bias_memory}, {DNNL_ARG_DST, output_memory} });
    stream.wait();

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