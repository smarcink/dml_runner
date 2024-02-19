#pragma once
#include <oneapi/dnnl/dnnl.hpp>

#include <numeric>
#include <span>
#include <cassert>

#include "layers_utils.h"

#include <DirectML.h>

namespace dnnl_utils
{
struct binding_t
{
    const std::byte* data = nullptr;
    DataType dt = DataType::eCount;
    DataLayout layout = DataLayout::eCount;
    TensorShape shape;
    bool transposed = false;
};

inline dnnl::memory::dim dimensions_product(const dnnl::memory::dims& dims)
{
    return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1, std::multiplies<dnnl::memory::dim>());
}

inline dnnl::memory::dims to_dnnl_dims(const TensorShape& shape)
{
    // nchw
    const auto dc = shape.get_dims_count();
    assert(dc != 3);
    assert(dc != 5);
    dnnl::memory::dims dims;
    if (dc >= 1)
    {
        dims.push_back(shape.n);
    }
    if (dc >= 2)
    {
        dims.push_back(shape.c);
    }
    if (dc == 4)
    {
        dims.push_back(shape.h);
        dims.push_back(shape.w);
    }

    return dims;
}

inline dnnl::memory::format_tag to_dnnl_format(const DataLayout l)
{
    switch (l)
    {
    case DataLayout::eNCHW: return dnnl::memory::format_tag::nchw;
    case DataLayout::eNHWC: return dnnl::memory::format_tag::nhwc;
    case DataLayout::eW:
    case DataLayout::eO: return dnnl::memory::format_tag::a;
    case DataLayout::eWeightsLayoutStart: return dnnl::memory::format_tag::any; break;
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

inline dnnl::memory::desc to_dnnl_mem_desc(const TensorShape& shape, const DataLayout& l, const DataType& t)
{
    if (is_data_layout_unpacked(l))
    {
        return dnnl::memory::desc{ to_dnnl_dims(shape), to_dnnl_data_type(t), to_dnnl_dims(data_layout_to_strides(shape, l)) };
    }
    return dnnl::memory::desc{ to_dnnl_dims(shape), to_dnnl_data_type(t), to_dnnl_format(l) };
}

inline dnnl::memory::desc gemm_matrix_transpose(const dnnl::memory::desc& nchw_desc, const DataType& t)
{
    auto dims = nchw_desc.get_dims();
    auto ndims = dims.size();
    assert(ndims == 4);

    std::swap(dims[2], dims[3]); // Swap channels and height

    // Create and return a new memory descriptor with NCWH format
    return dnnl::memory::desc(dims, to_dnnl_data_type(t), dnnl::memory::format_tag::abdc);
}

inline void copy_to_dnnl_memory(dnnl::memory& dst_memory, const std::byte* input_data)
{
    const auto desc = dst_memory.get_desc();
    auto dest_ptr = dst_memory.map_data<uint8_t>();
    const auto copy_size = desc.get_size();
    std::memcpy(dest_ptr, input_data, copy_size);
    dst_memory.unmap_data(dest_ptr);
}


inline void dump_buffer_to_file(const dnnl::memory& memory, const std::string& file_name)
{
    if (memory.get_desc().is_zero())
    {
        return;
    }
    const auto copy_size = memory.get_desc().get_size();
    std::vector<std::byte> ret(copy_size);
    auto* mapped_out_filter = memory.map_data<uint8_t>();
    std::memcpy(ret.data(), mapped_out_filter, copy_size);
    memory.unmap_data(mapped_out_filter);

    std::ofstream fout(file_name, std::ios::out | std::ios::binary);
    fout.write((char*)ret.data(), ret.size());
    fout.close();
}

inline dnnl::memory create_dnnl_memory(const dnnl_utils::binding_t binding, dnnl::engine& engine)
{
    const auto dims = dnnl_utils::to_dnnl_dims(binding.shape);
    const auto dt = dnnl_utils::to_dnnl_data_type(binding.dt);
    const auto ft = dnnl_utils::to_dnnl_format(binding.layout);
    auto ret = dnnl::memory({ dims, dt, ft }, engine);
    if (binding.data)
    {
        copy_to_dnnl_memory(ret, binding.data);
    }
    return ret;
}

inline dnnl::algorithm to_dnnl_activation_type(const ActivationType& type)
{
    switch (type)
    {
    case ActivationType::eRelu: return dnnl::algorithm::eltwise_relu;
    case ActivationType::eLeakyRelu: return dnnl::algorithm::eltwise_relu;  // for dnnl it's RELU, but it uses alpha to scale negative values
    case ActivationType::eClip: return dnnl::algorithm::eltwise_clip;
    case ActivationType::eGelu: return dnnl::algorithm::eltwise_gelu_erf;
    case ActivationType::eSigmoid: return dnnl::algorithm::eltwise_logistic;
    case ActivationType::eLinear: return dnnl::algorithm::eltwise_linear;
    case ActivationType::eTanh: return dnnl::algorithm::eltwise_tanh;
    default:
        assert(!"unknown activation type!");
    }
    return dnnl::algorithm::undef;
}

}// namespace dnnl_utils