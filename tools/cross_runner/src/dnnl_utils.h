#include "conv.h"
#include <oneapi/dnnl/dnnl.hpp>

#include <numeric>
#include <span>
#include <cassert>

inline dnnl::memory::dim dimensions_product(const dnnl::memory::dims& dims)
{
    return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1, std::multiplies<dnnl::memory::dim>());
}

inline dnnl::memory::dims to_dnnl_dims(const TensorShape& shape)
{
    // nchw
    const dnnl::memory::dims dims{ shape.n, shape.c, shape.h, shape.w };
    return dims;
}

inline dnnl::memory::format_tag to_dnnl_format(const DataLayout l)
{
    switch (l)
    {
    case DataLayout::eNCHW: return dnnl::memory::format_tag::nchw;
    case DataLayout::eNHWC: return dnnl::memory::format_tag::nhwc;
    default:
        return dnnl::memory::format_tag::undef;
    }
    return dnnl::memory::format_tag::undef;
}

inline dnnl::memory::data_type to_dnnl_data_type(const DataType l)
{
    switch (l)
    {
    case DataType::eFp32: return dnnl::memory::data_type::f32;
    case DataType::eFp16: return dnnl::memory::data_type::f16;
    default:
        return dnnl::memory::data_type::undef;
    }
    return dnnl::memory::data_type::undef;
}

inline void copy_to_dnnl_memory(dnnl::memory& dst_memory, const std::byte* input_data)
{
    const auto desc = dst_memory.get_desc();
    auto dest_ptr = dst_memory.map_data<uint8_t>();
    const auto copy_size = dimensions_product(desc.get_dims()) * dnnl::memory::data_type_size(desc.get_data_type());
    std::memcpy(dest_ptr, input_data, copy_size);
    dst_memory.unmap_data(dest_ptr);
}