#pragma once
#include <span>
#include <string>
#include <cassert>
#include <cstdint>
#include <istream>
#include <vector>
#include <random>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include "tensor_shape.h"

inline bool is_power_of_2(std::size_t n)
{
    return (n & (n - 1)) == 0;
}

template<typename T>
inline constexpr T round_up_next_multiple(T N, T M)
{
    return ((N + M - 1) / M) * M;
}

inline int64_t align(const int64_t value, const int64_t alignment)
{
    assert(alignment >= 1);
    return ((value + alignment - 1ll) / alignment) * alignment;
}



enum class DataType
{
    eFp32 = 0,
    eFp16 = 1,
    eCount
};

inline std::uint8_t get_data_type_bytes_width(DataType dt)
{
    switch (dt)
    {
    case DataType::eFp32: return sizeof(float);
    case DataType::eFp16: return sizeof(std::uint16_t);
    default:
        assert(false && "Unknown data type.");
    }
    return 0;
}


inline std::string get_data_type_str(DataType dt)
{
    switch (dt)
    {
    case DataType::eFp32: return "FP32";
    case DataType::eFp16: return "FP16";
    default:
        assert(false && "Unknown data type.");
    }
    return "UNKNOWN";
}

enum class ActivationType
{
    eUnknown = 0,
    eRelu = 1,
    eLeakyRelu,
    eClip,
    eGelu,
    eSigmoid,
    eLinear,
    eTanh,
    // we can add more here
    //..
    //..
    eCount
};

inline std::string get_activation_type_str(ActivationType type)
{
    switch (type)
    {
    case ActivationType::eRelu: return "relu";
    case ActivationType::eLeakyRelu: return "leakyrelu";
    case ActivationType::eClip: return "clip";
    case ActivationType::eGelu: return "gelu";
    case ActivationType::eSigmoid: return "sigmoid";
    case ActivationType::eLinear: return "linear";
    case ActivationType::eTanh: return "tanh";
    default:
        assert("!unknown act type. cant get string name for it");
    }
    return "";
}

inline ActivationType to_activation_type(std::string str)
{
    // lower case
    std::transform(str.begin(), str.end(), str.begin(),
        [](unsigned char c) { return std::tolower(c); });
    // remove underscore
    str.erase(std::remove(str.begin(), str.end(), '_'), str.end());

    auto is_str_matching_type = [](const auto& str, const auto& type)
    {
        if (str.compare(get_activation_type_str(type)) == 0)
        {
            return type;
        }
        return ActivationType::eUnknown;
    };  

    for (auto i = 0; i < static_cast<std::size_t>(ActivationType::eCount); i++)
    {
        const auto t = static_cast<ActivationType>(i);
        if (ActivationType::eUnknown != is_str_matching_type(str, t))
        {
            return t;
        }
    }
    return ActivationType::eUnknown;
}

struct ActivationSettings
{
    ActivationType type = ActivationType::eUnknown;
    float alpha = 0.0f;
    float beta = 0.0f;
};

inline bool lexical_cast(const std::string& input, ActivationSettings& act)
{
    std::stringstream stream;
    stream << input;

    std::string type = "";
    std::string alpha = "0.0f";
    std::string beta = "0.0f";

    std::getline(stream, type, ',');
    std::getline(stream, alpha, ',');
    std::getline(stream, beta, ',');

    act.type = to_activation_type(type);
    act.alpha = std::stof(alpha);
    act.beta = std::stof(beta);

    return true;
}

enum class DataLayout
{
    eNCHW = 0,
    eNHWC = 1,
    eNCHW_AlignW320,  // example layout for unpacked tensor, ToDo: refactor cross runner to work with strides instead of hardcoded data layouts
    eNHWC_AlignH48,  // example layout for unpacked tensor, ToDo: refactor cross runner to work with strides instead of hardcoded data layouts
    eCHW,      // 3d dims for GEMMS
    eW,


    // ..
    // ..

    // weights layouts
    eWeightsLayoutStart = 1000,
    eO, // for bias
    eOIYX,          // nchw and oiyx layouts are the same format, this is just to express it with proper name
    eIO_i8_o8_i2,  // layout for 1x1 fp16 CM simd8 dpas kernel

    eOYXI_o8,   // layout for non dpas CM kernel for simd8 mad
    eOYXI_o16,  // layout for non dpas CM kernel for simd16 mad

    // ..
    // ..

    eCount
};

inline std::string data_layout_name(DataLayout l)
{
    switch (l)
    {
    case DataLayout::eNCHW: return "NCHW";
    case DataLayout::eNCHW_AlignW320: return "NCHW_AlignW320";
    case DataLayout::eNHWC_AlignH48: return "eNHWC_AlignH48";
    case DataLayout::eNHWC: return "NHWC";
    case DataLayout::eCHW:  return "CHW";
    case DataLayout::eW:    return "W";
    case DataLayout::eOIYX: return "OIYX";
    case DataLayout::eIO_i8_o8_i2: return "IO_i8_o8_i2";
    case DataLayout::eOYXI_o8:  return "OYXI_o8";
    case DataLayout::eOYXI_o16: return "OYXI_o16";
    default:
        assert(false && "Unknown data layout name.");
        return "";
    }
    return "";

}

inline std::uint8_t data_layout_dimensions_count(DataLayout l)
{
    switch (l)
    {
    case DataLayout::eNCHW:
    case DataLayout::eNCHW_AlignW320:
    case DataLayout::eNHWC_AlignH48:
    case DataLayout::eNHWC:
        return 4;
    case DataLayout::eCHW:
        return 3;
    case DataLayout::eW:
        return 1;
    default:
        return 0;
    }
    return 0;
}

inline bool is_data_layout_unpacked(const DataLayout l)
{
    if (DataLayout::eNCHW_AlignW320 == l ||
        DataLayout::eNHWC_AlignH48 == l)
    {
        return true;
    }
    return false;
}

inline std::size_t data_layout_w_alignment(const DataLayout l)
{
    if (DataLayout::eNCHW_AlignW320 == l)
    {
        return 320ull;
    }
    return 1ull;
}

inline std::size_t data_layout_h_alignment(const DataLayout l)
{
    if (DataLayout::eNHWC_AlignH48 == l)
    {
        return 48ull;
    }
    return 1ull;
}

inline TensorShape data_layout_to_strides(TensorShape shape, DataLayout l)
{
    const auto c = shape.c;
    const auto h = static_cast<std::uint32_t>(align(shape.h, data_layout_h_alignment(l)));
    const auto w = static_cast<std::uint32_t>(align(shape.w, data_layout_w_alignment(l)));


    TensorShape ret{};
    switch (l)
    {
    case DataLayout::eNCHW_AlignW320:
    case DataLayout::eNCHW:
    {
        ret = TensorShape(
            c * h * w,
            h * w,
            w,
            1);
        break;
    }
    case DataLayout::eNHWC:
    case DataLayout::eNHWC_AlignH48:
    {
        ret = TensorShape(
            c * h * w,
            1,
            c * w,
            c);
        break;
    }
    default:
        assert("!unsupported right now");
    }
    return ret;
}


inline std::size_t get_tensor_elements_count(const TensorShape& ts, DataLayout l)
{
    if (ts.get_dims_count() == 0)
    {
        return 0ull;
    }
    const auto strides_ts = data_layout_to_strides(ts, l);

    return strides_ts.n * ts.n;
}




inline float cast_to_float(Half v)
{
    return DirectX::PackedVector::XMConvertHalfToFloat(v);
}

inline float cast_to_float(float v)
{
    return v;
}

inline void randomize_linear_container_float(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::byte> container)
{
    using Dt = float;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = static_cast<Dt>(dist(gen));
    }
}


inline void fill_with_constant_linear_container_float(std::span<std::byte> container, float value)
{
    using Dt = float;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = value;
    }
}

inline void randomize_linear_container_half(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::byte> container)
{
    using Dt = Half;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = DirectX::PackedVector::XMConvertFloatToHalf(dist(gen));
    }
}

inline void fill_with_constant_linear_container_half(std::span<std::byte> container, Half value)
{
    using Dt = Half;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = value;
    }
}

inline auto add_data_type_cli_option(CLI::App* opts, std::string_view opt_name, DataType& dt)
{
    return opts->add_option(opt_name.data(), dt)->check(CLI::IsMember({DataType::eFp32, DataType::eFp16}))
        ->transform(CLI::Transformer(std::map<std::string, DataType>{
            {"fp32", DataType::eFp32}, { "fp16", DataType::eFp16 }
    }, CLI::ignore_case, CLI::ignore_underscore));
}

inline auto add_data_layout_cli_option(CLI::App* opts, std::string_view opt_name, DataLayout& layout)
{
    return opts->add_option(opt_name.data(), layout)->check(CLI::IsMember({DataLayout::eNCHW, DataLayout::eNHWC, DataLayout::eW, DataLayout::eCHW, DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48 }))
        ->transform(CLI::Transformer(std::map<std::string, DataLayout>{
            {"nchw", DataLayout::eNCHW}, { "nhwc", DataLayout::eNHWC }, { "w", DataLayout::eW }, { "chw", DataLayout::eCHW }, { "nchw_alignw320", DataLayout::eNCHW_AlignW320 }, { "nhwc_alignh48", DataLayout::eNHWC_AlignH48 },
    }, CLI::ignore_case, CLI::ignore_underscore));
}
