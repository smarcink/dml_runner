#include "softmax.h"
#include "dnnl_utils.h"

std::vector<std::byte> cpu_op::softmax(std::uint32_t axis, const std::byte* in_data, const TensorShape& in_out_shape, DataType in_out_datatype, DataLayout in_out_layout)
{
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    static dnnl::stream stream(engine);
    const auto engine_kind = engine.get_kind();

    stream.wait();  // just to be sure we can freely upload the input data    

    dnnl::memory input_memory = [&]()
    {
        const auto dims = to_dnnl_dims(in_out_shape);
        const auto dt = to_dnnl_data_type(in_out_datatype);
        const auto ft = to_dnnl_format(in_out_layout);
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, in_data);
        return ret;
    }();

    dnnl::memory output_memory = [&]()
    {
        const auto dims = to_dnnl_dims(in_out_shape);
        const auto dt = to_dnnl_data_type(in_out_datatype);
        const auto ft = to_dnnl_format(in_out_layout);
        return dnnl::memory({ dims, dt, ft }, engine);
    }();

    const dnnl::softmax_forward::primitive_desc softmax_desc(engine, dnnl::prop_kind::forward_inference,
        dnnl::algorithm::softmax_accurate, input_memory.get_desc(), output_memory.get_desc(), static_cast<int32_t>(axis));

    dnnl::softmax_forward softmax(softmax_desc);
    softmax.execute(stream, { { DNNL_ARG_SRC, input_memory }, {DNNL_ARG_DST, output_memory} });
    stream.wait();

    auto* out_dnnl_data = output_memory.map_data<uint8_t>();
    assert(out_dnnl_data != nullptr && "[dnnl][softmax] Couldnt map output memory!");

    const auto om_desc = output_memory.get_desc();
    const auto om_dims = om_desc.get_dims();
    const auto copy_size = dimensions_product(om_dims) * dnnl::memory::data_type_size(om_desc.get_data_type());
    std::vector<std::byte> ret(copy_size);
    std::memcpy(ret.data(), out_dnnl_data, copy_size);
    output_memory.unmap_data(out_dnnl_data);
    return ret;

}