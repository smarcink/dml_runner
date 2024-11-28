#pragma once
#include <inference_engine.h>

#include <utility>
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

void close_execute_reset_wait(ID3D12Device* d3d12_device, ID3D12CommandQueue* command_queue,
    ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list);

void initalize_d3d12(ComPtr<ID3D12Device>& d3D12_device, ComPtr<ID3D12CommandQueue>& command_queue, ComPtr<ID3D12CommandAllocator>& command_allocator, ComPtr<ID3D12GraphicsCommandList>& command_list, bool use_rcs);

ComPtr<ID3D12DescriptorHeap> create_descriptor_heap(ID3D12Device* d3d12_device, uint32_t descriptors_count);

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* d3d12_device, std::size_t bytes_width, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES init_state, D3D12_RESOURCE_FLAGS resource_flag = D3D12_RESOURCE_FLAG_NONE);

void dispatch_resource_barrier(ID3D12GraphicsCommandList* command_list, const std::vector<CD3DX12_RESOURCE_BARRIER>& barriers);

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
inline Dx12Engine G_DX12_ENGINE{};

namespace dx12_callbacks
{

inline inference_engine_kernel_t gpu_device_create_kernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
{
    std::cout << "dummy gpu_device_create_kernel" << std::endl;
    return nullptr;
}

inline inference_engine_resource_t gpu_device_allocate_resource(inference_engine_device_t device, size_t size)
{
    std::cout << "[callback] gpu_device_allocate_resource, size: " << size << std::endl;
    auto dx12_device = reinterpret_cast<ID3D12Device*>(device);
    auto dx_buffer = create_buffer(dx12_device, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto handle = reinterpret_cast<inference_engine_resource_t>(dx_buffer.Detach());
    return handle;
}

inline void gpu_device_destroy_resource(inference_engine_resource_t handle)
{
    std::cout << "[callback] gpu_device_destroy_resource" << std::endl;
    auto rsc = reinterpret_cast<ID3D12Resource*>(handle);
    rsc->Release();
}

inline void gpu_kernel_set_arg_resource(inference_engine_kernel_t kernel, uint32_t index, inference_engine_resource_t resource)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_resource" << std::endl;
}

inline void gpu_kernel_set_arg_uint32(inference_engine_kernel_t kernel, uint32_t index, uint32_t value)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_uint32" << std::endl;
}

inline void gpu_kernel_set_arg_float(inference_engine_kernel_t kernel, uint32_t index, float value)
{
    std::cout << "Dummy callback gpu_kernel_set_arg_float" << std::endl;
}

inline void gpu_stream_execute_kernel(inference_engine_stream_t stream, inference_engine_kernel_t kernel, uint32_t gws[3], uint32_t lws[3])
{
    std::cout << "Dummy callback gpu_stream_execute_kernel" << std::endl;
}

inline void gpu_stream_fill_memory(inference_engine_stream_t stream, inference_engine_resource_t dst_resource, size_t size)
{
    std::cout << "Dummy callback gpu_stream_fill_memory" << std::endl;
}

inline void gpu_stream_resource_barrier(inference_engine_stream_t stream, inference_engine_resource_t* rsc_list, size_t rsc_list_size)
{
    std::cout << "[callback]  gpu_stream_resource_barrier" << std::endl;

    auto cmd_list = reinterpret_cast<ID3D12GraphicsCommandList*>(stream);
    std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
    std::vector<ID3D12Resource*> rscs(rsc_list_size);
    for (auto i = 0; i < rsc_list_size; i++)
    {
        rscs[i] = reinterpret_cast<ID3D12Resource*>(rsc_list[i]);
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::UAV(rscs[i]));
    }   
    dispatch_resource_barrier(cmd_list, barriers);
}

} // namespace dx12 callbacks

inline inference_engine_context_callbacks_t fill_with_dx12_callbacks()
{
    inference_engine_context_callbacks_t callbacks{};

    callbacks.fn_gpu_device_create_kernel = &dx12_callbacks::gpu_device_create_kernel;
    callbacks.fn_gpu_device_allocate_resource = &dx12_callbacks::gpu_device_allocate_resource;

    callbacks.fn_gpu_kernel_set_arg_resource = &dx12_callbacks::gpu_kernel_set_arg_resource;
    callbacks.fn_gpu_kernel_set_arg_uint32 = &dx12_callbacks::gpu_kernel_set_arg_uint32;
    callbacks.fn_gpu_kernel_set_arg_float = &dx12_callbacks::gpu_kernel_set_arg_float;

    callbacks.fn_gpu_stream_execute_kernel = &dx12_callbacks::gpu_stream_execute_kernel;
    callbacks.fn_gpu_stream_fill_memory = &dx12_callbacks::gpu_stream_fill_memory;
    callbacks.fn_gpu_stream_resource_barrier = &dx12_callbacks::gpu_stream_resource_barrier;

    return callbacks;
}