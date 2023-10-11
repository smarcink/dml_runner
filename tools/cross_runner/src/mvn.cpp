#include "mvn.h"
#include "dnnl_utils.h"

std::vector<std::byte> dnnl_mvn_op::mvn(const TensorShape& in_out_shape, DataLayout in_out_layout, DataType in_out_datatype, const std::byte* input_data, const std::byte* scale_data, const std::byte* bias_data, const float epsilon)
{
    using namespace dnnl_utils;

    /*
    *   This code seems to be broken. ToDo: fix it.
    */
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    static dnnl::stream stream(engine);
    const auto engine_kind = engine.get_kind();

    dnnl::set_jit_dump(true);

    stream.wait();  // just to be sure we can freely upload the input data    

    dnnl::memory input_memory = [&]()
    {
        const auto dims = to_dnnl_dims(in_out_shape);
        const auto dt = to_dnnl_data_type(in_out_datatype);
        const auto ft = to_dnnl_format(in_out_layout);
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, input_data);
        return ret;
    }();

    dnnl::memory output_memory = [&]()
    {
        const auto dims = to_dnnl_dims(in_out_shape);
        const auto dt = to_dnnl_data_type(in_out_datatype);
        const auto ft = to_dnnl_format(in_out_layout);
        return dnnl::memory({ dims, dt, ft }, engine);
    }();

    dnnl::memory scale_memory = [&]()
    {
        if (!scale_data)
        {
            return dnnl::memory{};
        }
        const auto dims = dnnl::memory::dims{ in_out_shape.c };
        const auto dt = to_dnnl_data_type(in_out_datatype);
        const auto ft = dnnl::memory::format_tag::a;
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, scale_data);
        return ret;
    }();

    dnnl::memory bias_memory = [&]()
    {
        if (!bias_data)
        {
            return dnnl::memory{};
        }
        const auto dims = dnnl::memory::dims{ in_out_shape.c };
        const auto dt = to_dnnl_data_type(in_out_datatype);
        const auto ft = dnnl::memory::format_tag::a;
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, bias_data);
        return ret;
    }();

    dnnl::normalization_flags flags = dnnl::normalization_flags::none;
    if (scale_data)
    {
        flags |= dnnl::normalization_flags::use_scale;
    }
    if (bias_data)
    {
        flags |= dnnl::normalization_flags::use_shift;
    }

    const dnnl::layer_normalization_forward::primitive_desc mvn_desc(engine, dnnl::prop_kind::forward_inference,
        input_memory.get_desc(), output_memory.get_desc(), epsilon, flags);

    dnnl::layer_normalization_forward mvn(mvn_desc);
    std::unordered_map<int, dnnl::memory> memory_map{ { DNNL_ARG_SRC, input_memory }, {DNNL_ARG_DST, output_memory} };
    if (bias_data)
    {
        memory_map.insert({ DNNL_ARG_SHIFT, bias_memory });
    }
    if (scale_data)
    {
        memory_map.insert({ DNNL_ARG_SCALE, scale_memory });
    }
    mvn.execute(stream, memory_map);
    stream.wait();

    auto* out_dnnl_data = output_memory.map_data<uint8_t>();
    assert(out_dnnl_data != nullptr && "[dnnl][mvn] Couldnt map output memory!");

    const auto om_desc = output_memory.get_desc();
    const auto om_dims = om_desc.get_dims();
    const auto copy_size = dimensions_product(om_dims) * dnnl::memory::data_type_size(om_desc.get_data_type());
    std::vector<std::byte> ret(copy_size);
    std::memcpy(ret.data(), out_dnnl_data, copy_size);
    output_memory.unmap_data(out_dnnl_data);
    return ret;
}
