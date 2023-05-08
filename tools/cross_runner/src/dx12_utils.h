#pragma once
#include <stdexcept>
#include <optional>
#include <span>

#include <dxgi1_4.h>
#include <d3d12.h>
#include <wrl/client.h>
#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h>
#include "DirectMLX.h"
#include "d3dx12.h"
#include <DirectXMath.h>
#include <DirectXPackedVector.h>

#undef max
#undef min

#define INTC_IGDEXT_D3D12
#include <igdext.h>

using Microsoft::WRL::ComPtr;
using Half = DirectX::PackedVector::HALF;

inline void throw_with_msg(std::string_view msg)
{
    throw std::runtime_error(msg.data());
}

inline void throw_if_failed(HRESULT hr, std::string_view msg)
{
    if (FAILED(hr))
    {
        throw_with_msg(msg);
    }
}

inline void initalize_d3d12(ComPtr<ID3D12Device>& d3D12_device, ComPtr<ID3D12CommandQueue>& command_queue, ComPtr<ID3D12CommandAllocator>& command_allocator, ComPtr<ID3D12GraphicsCommandList>& command_list)
{
#if defined(_DEBUG)
    ComPtr<ID3D12Debug> d3D12Debug;
    if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(d3D12Debug.ReleaseAndGetAddressOf()))))
    {
        // The D3D12 debug layer is missing - you must install the Graphics Tools optional feature
        //winrt::throw_hresult(DXGI_ERROR_SDK_COMPONENT_MISSING);
        throw_if_failed(DXGI_ERROR_SDK_COMPONENT_MISSING, "Cant enable debug build.");
    }
    d3D12Debug->EnableDebugLayer();
#endif
    ComPtr<IDXGIFactory4> dxgi_factory;
    throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(dxgi_factory.ReleaseAndGetAddressOf())), "dxgi factory");

    ComPtr<IDXGIAdapter> dxgiAdapter;
    UINT adapterIndex{};
    HRESULT hr{};
    do
    {
        dxgiAdapter = nullptr;
        throw_if_failed(dxgi_factory->EnumAdapters(adapterIndex, dxgiAdapter.GetAddressOf()), "enum adapters");
        ++adapterIndex;

        hr = ::D3D12CreateDevice(
            dxgiAdapter.Get(),
            D3D_FEATURE_LEVEL_12_0,
            IID_PPV_ARGS(d3D12_device.ReleaseAndGetAddressOf()));
        if (hr == DXGI_ERROR_UNSUPPORTED)
        {
            continue;
        }
        throw_if_failed(hr, "create device");
    } while (hr != S_OK);

    D3D12_COMMAND_QUEUE_DESC command_queue_desc{};
    command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    command_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    throw_if_failed(d3D12_device->CreateCommandQueue(
        &command_queue_desc, IID_PPV_ARGS(command_queue.ReleaseAndGetAddressOf())), "create command queue");

    throw_if_failed(d3D12_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf())), "create command allocator");

    throw_if_failed(d3D12_device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        command_allocator.Get(),
        nullptr, IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf())), "create command list");

}

class IntelExtension
{
public:
    IntelExtension(ID3D12Device* d3d12_device)
        :ext_ctx_(nullptr)
    {
        assert(d3d12_device != nullptr);

        try
        {
            // create extension    
            throw_if_failed(INTC_LoadExtensionsLibrary(true), "Intel Plugin Extension ERROR: INTC_LoadExtensionsLibrary failed");


            uint32_t supported_ext_version_count = 0;
            throw_if_failed(INTC_D3D12_GetSupportedVersions(d3d12_device, nullptr, &supported_ext_version_count), "Intel Plugin Extension ERROR: GetSupportedVersions");

            //Next, use returned value for supported_ext_version_count to allocate space for the supported extensions
            std::vector<INTCExtensionVersion> supported_ext_versions(supported_ext_version_count);
            const INTCExtensionVersion required_version = { 1,2,0 }; //version 1.2.0
            INTCExtensionInfo intc_extension_info = {};

            throw_if_failed(INTC_D3D12_GetSupportedVersions(d3d12_device, supported_ext_versions.data(), &supported_ext_version_count),
                "Intel Plugin Extension ERROR: GetSupportedVersions");

            for (uint32_t i = 0; i < supported_ext_version_count; i++)
            {
                if ((supported_ext_versions[i].HWFeatureLevel >= required_version.HWFeatureLevel) &&
                    (supported_ext_versions[i].APIVersion >= required_version.APIVersion) &&
                    (supported_ext_versions[i].Revision >= required_version.Revision))
                {
                    intc_extension_info.RequestedExtensionVersion = supported_ext_versions[i];
                    break;
                }
            }

            throw_if_failed(INTC_D3D12_CreateDeviceExtensionContext(d3d12_device, &ext_ctx_, &intc_extension_info, nullptr),
                "Intel Plugin Extension ERROR: CreateExtensionContext failed");

        }
        catch (...)
        {
            ext_ctx_ = nullptr;
        }
    }
    IntelExtension(const IntelExtension& rhs) = delete;
    IntelExtension(IntelExtension&& rhs) 
    {
        std::swap(ext_ctx_, rhs.ext_ctx_);
    }
    IntelExtension& operator=(const IntelExtension& rhs) = delete;
    IntelExtension& operator=(IntelExtension&& rhs)
    {
        if (this != &rhs)
        {
            std::swap(ext_ctx_, rhs.ext_ctx_);
        }
        return *this;
    }

    ~IntelExtension()
    {
        if (ext_ctx_)
        {
            throw_if_failed(INTC_DestroyDeviceExtensionContext(&ext_ctx_), "Intel Plugin Extension ERROR: DestroyDeviceExtensionContext failed");
        }
    }

    INTCExtensionContext* get() { return ext_ctx_; }

    ComPtr<ID3D12PipelineState> create_pipeline(const CD3DX12_SHADER_BYTECODE& shader_byte_code, std::string_view build_opts, ID3D12RootSignature* root_signature, INTC_D3D12_SHADER_INPUT_TYPE lang)
    {
        if (!ext_ctx_)
        {
            throw_with_msg("Intel extension context is missing. Cant create pipeline.");
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC compute_pso_desc = {};
        compute_pso_desc.pRootSignature = root_signature;
        compute_pso_desc.CS = CD3DX12_SHADER_BYTECODE(nullptr, 0);

        INTC_D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc_csext = {};
        pso_desc_csext.pD3D12Desc = &compute_pso_desc;

        pso_desc_csext.CS = shader_byte_code;
        pso_desc_csext.CompileOptions = (void*)build_opts.data();
        pso_desc_csext.InternalOptions = nullptr;// driver folks addes (void*)"-xess"; in xefx //ToDo: what it gives?
        pso_desc_csext.ShaderInputType = lang;

        ComPtr<ID3D12PipelineState> ret;
        throw_if_failed(INTC_D3D12_CreateComputePipelineState(ext_ctx_, &pso_desc_csext, IID_PPV_ARGS(&ret)),
            "INTC_D3D12_CreateComputePipelineState failed. Most probably compilation issue or root signature with kernels args mismatch!");
        //ret->SetName(name);  //ToDo: add naming
        return ret;
    }


private:
    INTCExtensionContext* ext_ctx_{nullptr};
};

inline ComPtr<IDMLDevice> create_dml_device(ID3D12Device* d3d12_device)
{
    DML_CREATE_DEVICE_FLAGS dml_create_device_flags = DML_CREATE_DEVICE_FLAG_NONE;
#if defined (_DEBUG)
    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dml_create_device_flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
    ComPtr<IDMLDevice> dml_device;
    DMLCreateDevice(d3d12_device, dml_create_device_flags, IID_PPV_ARGS(dml_device.ReleaseAndGetAddressOf()));
    return dml_device;
}

inline int64_t align(const int64_t value, const int64_t alignment)
{
    assert(alignment >= 1);
    return ((value + alignment - 1ll) / alignment) * alignment;
}

inline ComPtr<ID3D12DescriptorHeap> create_descriptor_heap(ID3D12Device* d3d12_device, uint32_t descriptors_count)
{
    // Create descriptor heaps.
    D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap_desc{};
    descriptor_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptor_heap_desc.NumDescriptors = descriptors_count;
    descriptor_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    throw_if_failed(d3d12_device->CreateDescriptorHeap(
        &descriptor_heap_desc, IID_PPV_ARGS(descriptor_heap.ReleaseAndGetAddressOf())), "create descriptor heap");
    return descriptor_heap;
}

inline ComPtr<ID3D12Resource> create_buffer(ID3D12Device* d3d12_device, std::size_t bytes_width, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES init_state, D3D12_RESOURCE_FLAGS resource_flag = D3D12_RESOURCE_FLAG_NONE)
{
    ComPtr<ID3D12Resource> ret;
    const auto heap_props = CD3DX12_HEAP_PROPERTIES(heap_type);
    const auto buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(align(bytes_width, 256), resource_flag);
    throw_if_failed(d3d12_device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &buffer_desc,
        init_state,
        nullptr, IID_PPV_ARGS(ret.ReleaseAndGetAddressOf())), "create commited resource");
    return ret;
}


inline void close_execute_reset_wait(ID3D12Device* d3d12_device, ID3D12CommandQueue* command_queue,
    ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
{
    throw_if_failed(command_list->Close(), "cmd list close");

    ID3D12CommandList* command_lists[] = { command_list };
    command_queue->ExecuteCommandLists(1, command_lists);

    ComPtr<ID3D12Fence> d3d12_fence;
    throw_if_failed(d3d12_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(d3d12_fence.ReleaseAndGetAddressOf())), "create fence");

    HANDLE fence_event_handle = ::CreateEvent(nullptr, true, false, nullptr);
    throw_if_failed(d3d12_fence->SetEventOnCompletion(1, fence_event_handle), "set event on completion");

    throw_if_failed(command_queue->Signal(d3d12_fence.Get(), 1), "command queue signal");
    ::WaitForSingleObjectEx(fence_event_handle, INFINITE, FALSE);

    throw_if_failed(command_allocator->Reset(), "command allocator reset");
    throw_if_failed(command_list->Reset(command_allocator, nullptr), "command list reset");
}

struct PerfCollectorDX12
{
    ComPtr<ID3D12QueryHeap> timestamp_query_heap;
    ComPtr<ID3D12Resource> timestamp_readback_buffer;
    uint64_t* timestamp_readback{};
    uint32_t timestamp_index = 0;

    inline void add_timestamp(ID3D12GraphicsCommandList* cmd_list)
    {
        cmd_list->EndQuery(timestamp_query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, timestamp_index++);
    }
};

inline PerfCollectorDX12 initialize_d3d12_performance_collector(ID3D12Device* device, std::uint32_t max_iters)
{
    PerfCollectorDX12 ret;
    D3D12_QUERY_HEAP_DESC query_heap_desc{};
    query_heap_desc.Count = 2 * max_iters;
    query_heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;

    throw_if_failed(device->CreateQueryHeap(&query_heap_desc, IID_PPV_ARGS(&ret.timestamp_query_heap)),
        "Failed to create query heap");

    ret.timestamp_readback_buffer = create_buffer(device, query_heap_desc.Count * sizeof(uint64_t),
        D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);

    throw_if_failed(ret.timestamp_readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&ret.timestamp_readback)),
        "Failed to map timestamp readback buffer");
    return ret;
}

template<typename TimeType = std::chrono::microseconds>
std::vector<TimeType> get_timestamps_timings_from_ptr(uint64_t timestamp_frequency, const uint64_t* readback, const uint32_t readback_size)
{
    const double frq_dbl = static_cast<double>(timestamp_frequency);
    std::vector<TimeType> exec_times(readback_size);
    for (uint32_t i = 0; i < exec_times.size(); i++)
    {
        const double t0 = readback[i] / frq_dbl;
        exec_times[i] = TimeType(static_cast<uint64_t>(1e6 * t0));
    }
    return exec_times;
}