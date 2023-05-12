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
    eW,


    // ..
    // ..

    // weights layouts
    eWeightsLayoutStart = 1000,
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
    case DataLayout::eNHWC: return "NHWC";
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
    case DataLayout::eNHWC:
        return 4;
    case DataLayout::eW:
        return 1;
    default:
        return 0;
    }
    return 0;
}

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
    eGemmDml,
    eGemmCm,
    eConvDml,
    eConvCm,
    eSoftmaxDml,
    eSoftmaxCm,
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
    return opts->add_option(opt_name.data(), dt)->check(CLI::IsMember({DataType::eFp32, DataType::eFp16}))
        ->transform(CLI::Transformer(std::map<std::string, DataType>{
            {"fp32", DataType::eFp32}, { "fp16", DataType::eFp16 }
    }, CLI::ignore_case, CLI::ignore_underscore));
}

inline auto add_data_layout_cli_option(CLI::App* opts, std::string_view opt_name, DataLayout& layout)
{
    return opts->add_option(opt_name.data(), layout)->check(CLI::IsMember({DataLayout::eNCHW, DataLayout::eNHWC, DataLayout::eW }))
        ->transform(CLI::Transformer(std::map<std::string, DataLayout>{
            {"nchw", DataLayout::eNCHW}, { "nhwc", DataLayout::eNHWC }, { "w", DataLayout::eW },
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


enum class DescType
{
    eSrv,
    eUav
};

inline ComPtr<ID3D12RootSignature> create_root_signature(ID3D12Device* d3d12_device, std::span<const DescType> desc_list)
{
    const auto bindings_size = desc_list.size();
    std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;
    std::vector<CD3DX12_ROOT_PARAMETER1> root_params;
    ranges.reserve(bindings_size);
    root_params.reserve(bindings_size + 1); // + 1 beacuse of the CM driver path

    std::uint32_t srv_range_reg = 0;
    std::uint32_t uav_range_reg = 0;
    std::uint32_t cbv_range_reg = 0;

    {
        // driver thing
        CD3DX12_ROOT_PARAMETER1 rp{};
        rp.InitAsConstants(1, cbv_range_reg++);
        root_params.push_back(rp);
    }

    auto add_desc_table = [&](DescType type)
    {
        if (type == DescType::eSrv)
        {
            ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, srv_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE });
        }
        else
        {
            ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, uav_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE });
        }
        CD3DX12_ROOT_PARAMETER1 rp{};
        rp.InitAsDescriptorTable(1u, &ranges.back());
        root_params.push_back(rp);
    };

    for (const auto d : desc_list)
    {
        add_desc_table(d);
    }

    if (root_params.size() == 0)
    {
        throw std::runtime_error("Something gone wrong. Why kernel has 0 root params?");
    }

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC compute_root_signature_desc;
    compute_root_signature_desc.Init_1_1(static_cast<UINT>(root_params.size()), root_params.data(), 0, nullptr);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    throw_if_failed(D3DX12SerializeVersionedRootSignature(
        &compute_root_signature_desc,
        D3D_ROOT_SIGNATURE_VERSION_1_1,
        &signature,
        &error), "D3DX12SerializeVersionedRootSignature failed.");

    if (error)
    {
        throw_with_msg("Failed to create root signature, error:" + std::string((LPCSTR)error->GetBufferPointer()));
    }
    ComPtr<ID3D12RootSignature> ret;
    throw_if_failed(d3d12_device->CreateRootSignature(
        0,
        signature->GetBufferPointer(),
        signature->GetBufferSize(),
        IID_PPV_ARGS(&ret)), "CreateRootSignature(...) failed.");
    return ret;
}

inline std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> create_resource_views_and_handles(ID3D12Device* d3d12_device, std::span<const std::pair<DescType, ID3D12Resource*>> resources_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
{
    const auto desc_heap_incrs_size = d3d12_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    const auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
    const auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles;
    gpu_handles.reserve(resources_list.size());

    for (std::size_t i = 0; i < resources_list.size(); i++)
    {
        auto cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(base_cpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size);
        gpu_handles.push_back(CD3DX12_GPU_DESCRIPTOR_HANDLE(base_gpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size));

        auto& resource_view_type = resources_list[i].first;
        auto& resource = resources_list[i].second;
        assert(resource != nullptr);
        const auto res_desc = resource->GetDesc();
        assert(res_desc.Dimension == D3D12_RESOURCE_DIMENSION::D3D12_RESOURCE_DIMENSION_BUFFER);

        if (resource_view_type == DescType::eSrv)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R8_UINT;
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
            desc.Buffer.FirstElement = 0;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

            d3d12_device->CreateShaderResourceView(resource, &desc, cpu_handle);
        }
        else
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R8_UINT;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;

            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
            desc.Buffer.FirstElement = 0;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

            d3d12_device->CreateUnorderedAccessView(resource, nullptr, &desc, cpu_handle);
        }
    }

    return gpu_handles;
}

inline void dispatch_kernel(ID3D12GraphicsCommandList* cmd_list, ID3D12PipelineState* pso, ID3D12RootSignature* root_signature, std::span<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles, std::uint32_t thg_x, std::uint32_t thg_y, std::uint32_t thg_z)
{
    assert(thg_x > 0);
    assert(thg_y > 0);
    assert(thg_z > 0);
    assert(cmd_list);
    assert(root_signature);
    assert(pso);
    assert(!gpu_handles.empty());

    cmd_list->SetComputeRootSignature(root_signature);
    cmd_list->SetPipelineState(pso);

    uint32_t root_index = 1; // start with 1, beacuse Cross compiler CM driver path needs that
    for (uint32_t i = 0; i < gpu_handles.size(); i++)
    {
        const auto gpu_heap_handle = gpu_handles[i];
        cmd_list->SetComputeRootDescriptorTable(root_index++, gpu_heap_handle);
    }

    cmd_list->Dispatch(thg_x, thg_y, thg_z);
}