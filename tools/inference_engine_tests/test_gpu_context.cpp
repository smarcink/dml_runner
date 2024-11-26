#include "test_d3d12_context.h"

#include <iostream>
#include <cassert>
#include <vector>

#include <dxgi1_4.h>
#include <d3d12.h>
#include "d3dx12.h"
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#undef max
#undef min

#define INTC_IGDEXT_D3D12
#include <igdext.h>

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

inline void initalize_d3d12(ComPtr<ID3D12Device>& d3D12_device, ComPtr<ID3D12CommandQueue>& command_queue, ComPtr<ID3D12CommandAllocator>& command_allocator, ComPtr<ID3D12GraphicsCommandList>& command_list, bool use_rcs)
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

        DXGI_ADAPTER_DESC desc{};
        throw_if_failed(dxgiAdapter->GetDesc(&desc), "get adapter desc");
        if (desc.VendorId != 0x8086) // intel
        {
            hr = S_FALSE;
            continue;
        }

        hr = ::D3D12CreateDevice(
            dxgiAdapter.Get(),
            D3D_FEATURE_LEVEL_12_0,
            IID_PPV_ARGS(d3D12_device.ReleaseAndGetAddressOf()));
        if (hr == DXGI_ERROR_UNSUPPORTED)
        {
            std::cout << "[Error] Tried to create DX12 device with D3D_FEATURE_LEVEL_12_0 on Intel HW, but it has failed." << std::endl;
            continue;
        }
        throw_if_failed(hr, "create device");
    } while (hr != S_OK);

    const auto queue_type = use_rcs ? D3D12_COMMAND_LIST_TYPE_DIRECT : D3D12_COMMAND_LIST_TYPE_COMPUTE;

    D3D12_COMMAND_QUEUE_DESC command_queue_desc{};
    command_queue_desc.Type = queue_type;
    command_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    throw_if_failed(d3D12_device->CreateCommandQueue(
        &command_queue_desc, IID_PPV_ARGS(command_queue.ReleaseAndGetAddressOf())), "create command queue");

    throw_if_failed(d3D12_device->CreateCommandAllocator(
        queue_type,
        IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf())), "create command allocator");

    throw_if_failed(d3D12_device->CreateCommandList(
        0,
        queue_type,
        command_allocator.Get(),
        nullptr, IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf())), "create command list");

}

class IntelExtension
{
public:
    IntelExtension() = default;

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

            throw_if_failed(INTC_D3D12_GetSupportedVersions(d3d12_device, supported_ext_versions.data(), &supported_ext_version_count),
                "Intel Plugin Extension ERROR: GetSupportedVersions");

            for (uint32_t i = 0; i < supported_ext_version_count; i++)
            {
                if ((supported_ext_versions[i].HWFeatureLevel >= required_version.HWFeatureLevel) &&
                    (supported_ext_versions[i].APIVersion >= required_version.APIVersion) &&
                    (supported_ext_versions[i].Revision >= required_version.Revision))
                {
                    intc_extension_info_.RequestedExtensionVersion = supported_ext_versions[i];
                    break;
                }
            }

            throw_if_failed(INTC_D3D12_CreateDeviceExtensionContext(d3d12_device, &ext_ctx_, &intc_extension_info_, nullptr),
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
        std::swap(intc_extension_info_, rhs.intc_extension_info_);
    }
    IntelExtension& operator=(const IntelExtension& rhs) = delete;
    IntelExtension& operator=(IntelExtension&& rhs)
    {
        if (this != &rhs)
        {
            std::swap(ext_ctx_, rhs.ext_ctx_);
            std::swap(intc_extension_info_, rhs.intc_extension_info_);
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
    INTCExtensionInfo get_info() { return intc_extension_info_; }

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
    INTCExtensionContext* ext_ctx_{ nullptr };
    INTCExtensionInfo intc_extension_info_ = {};
};

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

struct Dx12Engine
{
    ComPtr<ID3D12Device> d3d12_device;
    ComPtr<ID3D12CommandQueue> command_queue;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;

    IntelExtension intel_extension_d3d12;

    Dx12Engine()
    {
        initalize_d3d12(d3d12_device, command_queue, command_allocator, command_list, false);
        assert(d3d12_device && command_queue && command_allocator && command_list);
        // init extension
        intel_extension_d3d12 = IntelExtension(d3d12_device.Get());
    }

    void wait_for_execution()
    {
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());
    }
};



static inference_engine_kernel_t gpu_device_create_kernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options)
{
    std::cout << "Dummy callback gpu_device_create_kernel" << std::endl;
    return nullptr;
}

static inference_engine_resource_t gpu_device_allocate_resource(inference_engine_device_t device, size_t size)
{
    std::cout << "Dummy callback gpu_device_allocate_resource" << std::endl;
    return nullptr;
}

static void gpu_kernel_set_arg_resource(inference_engine_kernel_t kernel, uint32_t index, inference_engine_resource_t resource)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_resource" << std::endl;
}

static void gpu_kernel_set_arg_uint32(inference_engine_kernel_t kernel, uint32_t index, uint32_t value)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_uint32" << std::endl;
}

static void gpu_kernel_set_arg_float(inference_engine_kernel_t kernel, uint32_t index, float value)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_float" << std::endl;
}

static void gpu_stream_execute_kernel(inference_engine_stream_t stream, inference_engine_kernel_t kernel, uint32_t gws[3], uint32_t lws[3])
{
    std::cout << "Dummy callback gpu_stream_execute_kernel" << std::endl;
}

static void gpu_stream_fill_memory(inference_engine_stream_t stream, inference_engine_resource_t dst_resource, size_t size, inference_engine_event_t* out_event, inference_engine_event_t* dep_events, size_t dep_events_count)
{
    std::cout << "Dummy callback gpu_stream_fill_memory" << std::endl;
}

test_ctx::TestGpuContext::TestGpuContext()
{
    static Dx12Engine dx12_engine{};
    device_ = reinterpret_cast<inference_engine_device_t>(dx12_engine.d3d12_device.Get());

    inference_engine_context_callbacks_t callbacks{};

    callbacks.fn_gpu_device_create_kernel = &gpu_device_create_kernel;
    callbacks.fn_gpu_device_allocate_resource = &gpu_device_allocate_resource;

    callbacks.fn_gpu_kernel_set_arg_resource = &gpu_kernel_set_arg_resource;
    callbacks.fn_gpu_kernel_set_arg_uint32 = &gpu_kernel_set_arg_uint32;
    callbacks.fn_gpu_kernel_set_arg_float = &gpu_kernel_set_arg_float;

    callbacks.fn_gpu_stream_execute_kernel = &gpu_stream_execute_kernel;
    callbacks.fn_gpu_stream_fill_memory = &gpu_stream_fill_memory;

    ctx_ = inferenceEngineCreateContext(INFERENCE_ENGINE_ACCELERATOR_TYPE_GPU, device_, callbacks);
}

test_ctx::TestGpuContext::~TestGpuContext()
{
    inferenceEngineDestroyContext(ctx_);
}
