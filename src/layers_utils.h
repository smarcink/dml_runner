#pragma once
#include <span>
#include <string>
#include <cassert>
#include <cstdint>
#include <istream>
#include <vector>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"


struct TensorShape
{
    std::uint32_t n = 0;
    std::uint32_t c = 0;
    std::uint32_t d = 0; // for 5d tensors
    std::uint32_t h = 0;
    std::uint32_t w = 0;

    TensorShape() = default;

    TensorShape(std::uint32_t n, std::uint32_t c, std::uint32_t h, std::uint32_t w)
        : n(n), c(c), h(h), w(w)
    {
    }

    TensorShape(std::span<std::uint32_t> in_v)
    {
        assert((in_v.size() == 2 || in_v.size() == 4 || in_v.size() == 5) && "Not supported shape!");
        std::int32_t current_idx = static_cast<std::int32_t>(in_v.size()) - 1;
        w = in_v[current_idx--];
        h = in_v[current_idx--];
        if (in_v.size() == 5)
        {
            d = in_v[current_idx--];
        }
        if (in_v.size() > 2)
        {
            c = in_v[current_idx--];
            n = in_v[current_idx--];
        }
        assert(current_idx == -1 && "Current idex should be equal -1 (parsed all dimensions).");
    }

    inline std::size_t get_elements_count() const
    {
        std::size_t acc = 1;
        acc *= n ? n : 1;
        acc *= c ? c : 1;
        acc *= d ? d : 1;
        acc *= h ? h : 1;
        acc *= w ? w : 1;
        return acc;
    }
};


inline bool lexical_cast(const std::string& input, TensorShape& ts)
{
    std::vector<std::uint32_t> data;
    constexpr const auto buffer_size = 128;
    std::string line(buffer_size, ' ');
    std::stringstream stream;
    stream << input;
    while (stream.getline(line.data(), buffer_size, ','))
    {
        data.push_back(std::stoi(line));
    }
    ts = TensorShape(data);
    return true;
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

enum class DataLayout
{
    eNCHW = 0,
    eNHWC = 1,
    eCount
};

template<typename T>
inline constexpr T round_up_next_multiple(T N, T M) 
{
    return ((N + M - 1) / M) * M;
}


struct ConformanceResult
{
    bool passed = true;
    float epsilon = 0.0f;
    float biggest_difference = 0.0f;
    float node_value = 0.0f;
    float reference_value = 0.0f;
    std::uint32_t index = 0;
    std::size_t tested_samples_count = 0;
};

inline float cast_to_float(Half v)
{
    return DirectX::PackedVector::XMConvertHalfToFloat(v);
}

inline float cast_to_float(float v)
{
    return v;
}

enum class NodeType
{
    eGemm,
    eConvDml,
    eConvCm,
    eSoftmax,
    eMvnDml,
    eMvnCm,
    eCount
};

inline void randomize_linear_container_float(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::byte> container)
{
    using Dt = float;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = static_cast<Dt>(dist(gen));
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
    return opts->add_option("--data_type", dt)->check(CLI::IsMember({ DataType::eFp32, DataType::eFp16 }))
        ->transform(CLI::Transformer(std::map<std::string, DataType>{
            {"fp32", DataType::eFp32}, { "fp16", DataType::eFp16 }
    }, CLI::ignore_case, CLI::ignore_underscore));
}

inline auto add_data_layout_cli_option(CLI::App* opts, std::string_view opt_name, DataLayout& layout)
{
    return opts->add_option("--layout", layout)->check(CLI::IsMember({ DataLayout::eNCHW, DataLayout::eNHWC }))
        ->transform(CLI::Transformer(std::map<std::string, DataLayout>{
            {"nchw", DataLayout::eNCHW}, { "nhwc", DataLayout::eNHWC },
    }, CLI::ignore_case, CLI::ignore_underscore));
}


template<typename Dt>
inline ConformanceResult run_conformance_check(const std::vector<std::byte>& gpu_untyped_result, const std::vector<std::byte>& dnnl_untyped_result, float epsilon)
{
    const auto* gpu_typed_result = reinterpret_cast<const Dt*>(gpu_untyped_result.data());
    const auto* dnnl_typed_result = reinterpret_cast<const Dt*>(dnnl_untyped_result.data());

    // compare results
    ConformanceResult ret;
    ret.epsilon = epsilon;
    for (std::uint32_t i = 0; i < gpu_untyped_result.size() / sizeof(Dt); i++)
    {
        ret.node_value = cast_to_float(gpu_typed_result[i]);
        ret.reference_value = cast_to_float(dnnl_typed_result[i]);

        const auto abs_diff = std::abs(ret.node_value - ret.reference_value);

        if (abs_diff > ret.epsilon)
        {
            ret.passed = false;

            std::cout << std::format("Mismatch, gpu: {}, cpu: {}, at index: {}. Absolute differece: {} \n", ret.node_value, ret.reference_value, i, abs_diff);
        }
        ret.biggest_difference = std::max(ret.biggest_difference, abs_diff);
        ret.tested_samples_count++;
    }
    return ret;
}

class NodeDispatcher
{
public:
    virtual std::uint32_t get_total_descriptor_count() = 0;
    virtual void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) = 0;
    virtual void execute(ID3D12GraphicsCommandList* cmd_list) = 0;

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) = 0;

    virtual ~NodeDispatcher() = default;
};
