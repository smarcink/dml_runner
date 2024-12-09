#pragma once
#include <inference_engine.h>
#include <inference_engine.hpp>

#include <utility>
#include <iostream>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <variant>

#include <dxgi1_4.h>
#include <d3d12.h>
#include "d3dx12.h"
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#undef max
#undef min

#define INTC_IGDEXT_D3D12
#include <igdext.h>

//////////////////////////////////////////////////////////////////////////
// Custom Metacommand
// {9C365CB6-AF13-49B6-BA9C-4B74E10FCDE1}
static constexpr GUID GUID_CUSTOM =
{ 0x9c365cb6, 0xaf13, 0x49b6,{ 0xba, 0x9c, 0x4b, 0x74, 0xe1, 0xf, 0xcd, 0xe1 } };

//////////////////////////////////////////////////////////////////////////
enum META_COMMAND_CUSTOM_SHADER_LANGUAGE : UINT64
{
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE = 0,
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL,
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL_STATELESS,
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_ZEBIN_ELF,
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_CM,
};

//////////////////////////////////////////////////////////////////////////
enum META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE : UINT64
{
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_NONE = 0,
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_HANDLE,
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_ADDRESS
};

//////////////////////////////////////////////////////////////////////////
struct META_COMMAND_CREATE_CUSTOM_DESC
{
    UINT64 ShaderSourceCode;
    UINT64 ShaderSourceCodeSize;
    UINT64 BuildOptionString;
    UINT64 BuildOptionStringSize;
    META_COMMAND_CUSTOM_SHADER_LANGUAGE ShaderLanguage;
};

//////////////////////////////////////////////////////////////////////////
struct META_COMMAND_INITIALIZE_CUSTOM_DESC
{
    D3D12_GPU_DESCRIPTOR_HANDLE Resources[10];
};

//////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
struct META_COMMAND_EXECUTE_CUSTOM_DESC
{
    UINT64                      ResourceCount;
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE ResourceBindType[50];
    UINT64                      ResourceBindOffset[50];
    D3D12_GPU_DESCRIPTOR_HANDLE Resources[50];            // use address or handles
    D3D12_GPU_VIRTUAL_ADDRESS   ResourcesAddress[50];     // use address or handles
    UINT64                      ResourcesByteOffset[50];  // works only in stateless mode

    UINT64 RuntimeConstants;      // buffer with constants
    UINT64 RuntimeConstantsCount; // how many runtime constants in total
    UINT64 RuntimeConstantsBindOffsets[40];  // offsets in bindings
    UINT64 RuntimeConstantsMemorySizes[40];   // how much bytes to copy
    UINT64 RuntimeConstantsMemoryOffsets[40]; // bytes offset into "RuntimeConstants" buffer

    UINT64 DispatchGlobalWorkSize[3];
    UINT64 DispatchLocalWorkSize[3];
    UINT64 SharedLocalMemorySize;
};


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

struct MetaCommandWrapper
{
    ID3D12MetaCommand* mc;
    std::unordered_map<std::size_t, std::pair<inference_engine_resource_t, std::size_t>> resources;
    std::unordered_map<std::size_t, std::variant<float, std::uint32_t>> scalars;
    std::unordered_map<std::size_t, std::size_t> locals;
    std::string name;
};

namespace dx12_callbacks
{

inline inference_engine_kernel_t gpu_device_create_kernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
{
    std::cout << "[callback]  gpu_device_create_kernel" << std::endl;

    auto d3d12_dev = reinterpret_cast<ID3D12Device*>(device);
    ID3D12Device5* dev5 = nullptr;
    throw_if_failed(d3d12_dev->QueryInterface(&dev5), "cant cast d3d12 device to ID3D12Device5");

    if (kernel_code == nullptr || kernel_code_size == 0)
    {
        throw std::runtime_error("Code string is empty. Please provide valid kernel/binary data.\n");
    }

    META_COMMAND_CREATE_CUSTOM_DESC create_desc{};
    create_desc.ShaderSourceCode = reinterpret_cast<UINT64>(kernel_code);
    create_desc.ShaderSourceCodeSize = kernel_code_size;
    create_desc.BuildOptionString = reinterpret_cast<UINT64>(build_options);
    create_desc.BuildOptionStringSize = build_options ? std::strlen(build_options) : 0ull;

    switch (language)
    {
    case inference_engine_kernel_language_t::INFERENCE_ENGINE_KERNEL_LANGUAGE_OCL:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL_STATELESS;
        break;
    default:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE;
    }
    assert(create_desc.ShaderLanguage != META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE);
    auto mcw = new MetaCommandWrapper{};
    throw_if_failed(dev5->CreateMetaCommand(GUID_CUSTOM, 0, &create_desc, sizeof(create_desc),
        IID_PPV_ARGS(&mcw->mc)), "Cant create custom metacommand");
    if (!mcw->mc)
    {
        delete mcw;
        assert(!"Creation of custom MC failed.");
    }

    return reinterpret_cast<inference_engine_kernel_t>(mcw);
}

inline void gpu_device_destroy_kernel(inference_engine_kernel_t handle)
{
    std::cout << "[callback] gpu_device_destroy_kernel" << std::endl;
    auto mcw = reinterpret_cast<MetaCommandWrapper*>(handle);
    mcw->mc->Release();
    delete mcw;
}

inline inference_engine_resource_t gpu_device_allocate_resource(inference_engine_device_t device, size_t size)
{
    std::cout << "[callback] gpu_device_allocate_resource, size: " << size << std::endl;
    auto dx12_device = reinterpret_cast<ID3D12Device*>(device);
    auto dx_buffer = create_buffer(dx12_device, size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto handle = reinterpret_cast<inference_engine_resource_t>(dx_buffer.Detach());
    return handle;
}

inline void gpu_device_destroy_resource(inference_engine_resource_t handle)
{
    std::cout << "[callback] gpu_device_destroy_resource" << std::endl;
    auto rsc = reinterpret_cast<ID3D12Resource*>(handle);
    rsc->Release();
}

inline void gpu_kernel_set_arg_resource(inference_engine_kernel_t kernel, uint32_t index, inference_engine_resource_t resource, std::size_t offset)
{
    std::cout << "[callback]  callback gpu_kernel_set_arg_resource" << std::endl;
    auto mcw = reinterpret_cast<MetaCommandWrapper*>(kernel);
    mcw->resources[index] = { resource, offset };
}

inline void gpu_kernel_set_arg_uint32(inference_engine_kernel_t kernel, uint32_t index, uint32_t value)
{
    std::cout << "[callback]  callback gpu_kernel_set_arg_uint32" << std::endl;
    auto mcw = reinterpret_cast<MetaCommandWrapper*>(kernel);
    mcw->scalars[index] = value;
}

inline void gpu_kernel_set_arg_float(inference_engine_kernel_t kernel, uint32_t index, float value)
{
    std::cout << "[callback]  callback gpu_kernel_set_arg_float" << std::endl;
    auto mcw = reinterpret_cast<MetaCommandWrapper*>(kernel);
    mcw->scalars[index] = value;
}

inline void gpu_stream_execute_kernel(inference_engine_stream_t stream, inference_engine_kernel_t kernel, uint32_t gws[3], uint32_t lws[3])
{
    std::cout << "[callback] callback gpu_stream_execute_kernel" << std::endl;
    std::cout << "\t gws: " << gws[0] << ", " << gws[1] << ", " << gws[2] << std::endl;
    auto cmd_list = reinterpret_cast<ID3D12GraphicsCommandList4*>(stream);

    auto mcw = reinterpret_cast<MetaCommandWrapper*>(kernel);

    // [0] Calculate dispatch thread size
    META_COMMAND_EXECUTE_CUSTOM_DESC exec_desc{};
    for (std::size_t i = 0; i < 3; i++)
    {
        if (gws[i] == 0 || lws[i] == 0)
        {
            assert(!"Unexpected gws and/or lws sizes");
        }
        exec_desc.DispatchGlobalWorkSize[i] = gws[i];
        exec_desc.DispatchLocalWorkSize[i] = lws[i];
    }

    exec_desc.ResourceCount = mcw->resources.size();
    if (exec_desc.ResourceCount >= std::size(exec_desc.Resources))
    {
        assert(!"Please extend number of supported resources for custom metacommand!");
    }

    // [1] Prepare resources pointer handles 
    for (std::size_t idx = 0; const auto & [bind_indx, rsc] : mcw->resources)
    {
        exec_desc.ResourceBindOffset[idx] = bind_indx;

        const auto [rsc_handle, base_offset] = rsc;
        if (rsc_handle)
        {
            // set offset no matter what type of resource
            auto mem_ptr = reinterpret_cast<ID3D12Resource*>(rsc_handle);
            exec_desc.ResourcesByteOffset[idx] = base_offset;
            exec_desc.ResourceBindType[idx] = META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_ADDRESS;
            exec_desc.ResourcesAddress[idx] = mem_ptr->GetGPUVirtualAddress();
        }
        idx++;
    }

    // [2] Build execution time constants 
    std::vector<std::byte> execution_time_constants;
    for (std::size_t i = 0; const auto & [idx, scalar] : mcw->scalars)
    {
        //exec_desc.RuntimeConstantsBindOffsets[i] = idx;
        //exec_desc.RuntimeConstantsMemorySizes[i] = scalar.size;
        //exec_desc.RuntimeConstantsMemoryOffsets[i] = execution_time_constants.size();;
        //execution_time_constants.resize(execution_time_constants.size() + scalar.size);
        i++;
    }
    auto* ptr_to_copy_data = execution_time_constants.data();
    for (const auto& [idx, scalar] : mcw->scalars)
    {
        //std::memcpy(ptr_to_copy_data, scalar.data, scalar.size);
        //ptr_to_copy_data += scalar.size;
    }
    exec_desc.RuntimeConstantsCount = mcw->scalars.size();
    exec_desc.RuntimeConstants = reinterpret_cast<UINT64>(execution_time_constants.data());

    // [3] Build slm
    if (mcw->locals.size() > 1)
    {
        assert("!Unsupported case. Please remove this check and test - if it fails most probably driver need changes!");
    }

    for (const auto& [idx, slm_size] : mcw->locals)
    {
        exec_desc.SharedLocalMemorySize += slm_size;
    }

    cmd_list->ExecuteMetaCommand(mcw->mc, &exec_desc, sizeof(exec_desc));
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
    callbacks.fn_gpu_kernel_destroy = &dx12_callbacks::gpu_device_destroy_kernel;
    callbacks.fn_gpu_device_allocate_resource = &dx12_callbacks::gpu_device_allocate_resource;

    callbacks.fn_gpu_kernel_set_arg_resource = &dx12_callbacks::gpu_kernel_set_arg_resource;
    callbacks.fn_gpu_kernel_set_arg_uint32 = &dx12_callbacks::gpu_kernel_set_arg_uint32;
    callbacks.fn_gpu_kernel_set_arg_float = &dx12_callbacks::gpu_kernel_set_arg_float;

    callbacks.fn_gpu_stream_execute_kernel = &dx12_callbacks::gpu_stream_execute_kernel;
    callbacks.fn_gpu_stream_fill_memory = &dx12_callbacks::gpu_stream_fill_memory;
    callbacks.fn_gpu_stream_resource_barrier = &dx12_callbacks::gpu_stream_resource_barrier;

    return callbacks;
}

class ResourceDX12 : public inference_engine::Resource
{
public:
    ResourceDX12(ComPtr<ID3D12Resource> resource)
        : rsc_(resource)
    {
    }

    ID3D12Resource* get() { return rsc_.Get(); }

private:
    ComPtr<ID3D12Resource> rsc_;
};


class KernelDX12 : public inference_engine::Resource
{
public:
    KernelDX12(ID3D12Device* d3d12_dev, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
        : name_(kernel_name)
    {
        ID3D12Device5* dev5 = nullptr;
        throw_if_failed(d3d12_dev->QueryInterface(&dev5), "cant cast d3d12 device to ID3D12Device5");

        if (kernel_code == nullptr || kernel_code_size == 0)
        {
            throw std::runtime_error("Code string is empty. Please provide valid kernel/binary data.\n");
        }

        META_COMMAND_CREATE_CUSTOM_DESC create_desc{};
        create_desc.ShaderSourceCode = reinterpret_cast<UINT64>(kernel_code);
        create_desc.ShaderSourceCodeSize = kernel_code_size;
        create_desc.BuildOptionString = reinterpret_cast<UINT64>(build_options);
        create_desc.BuildOptionStringSize = build_options ? std::strlen(build_options) : 0ull;

        switch (language)
        {
        case inference_engine_kernel_language_t::INFERENCE_ENGINE_KERNEL_LANGUAGE_OCL:
            create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL_STATELESS;
            break;
        default:
            create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE;
        }
        assert(create_desc.ShaderLanguage != META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE);
        auto mcw = new MetaCommandWrapper{};
        throw_if_failed(dev5->CreateMetaCommand(GUID_CUSTOM, 0, &create_desc, sizeof(create_desc),
            IID_PPV_ARGS(&mcw->mc)), "Cant create custom metacommand");
        if (!mcw->mc)
        {
            delete mcw;
            assert(!"Creation of custom MC failed.");
        }
    }

    ID3D12MetaCommand* get() { return mc_.Get(); }

    void set_arg(std::uint32_t idx, ResourceDX12* rsc, std::size_t offset = 0)
    {
        resources_[idx] = { rsc, offset };
    }

    void set_arg(std::uint32_t idx, std::uint32_t u32)
    {
        scalars_[idx] = u32;
    }

    void set_arg(std::uint32_t idx, float f32)
    {
        scalars_[idx] = f32;
    }

    void execute(ID3D12GraphicsCommandList4* cmd_list, std::uint32_t gws[3], std::uint32_t lws[3])
    {
        std::cout << "[callback] callback gpu_stream_execute_kernel" << std::endl;
        std::cout << "\t gws: " << gws[0] << ", " << gws[1] << ", " << gws[2] << std::endl;;

        // [0] Calculate dispatch thread size
        META_COMMAND_EXECUTE_CUSTOM_DESC exec_desc{};
        for (std::size_t i = 0; i < 3; i++)
        {
            if (gws[i] == 0 || lws[i] == 0)
            {
                assert(!"Unexpected gws and/or lws sizes");
            }
            exec_desc.DispatchGlobalWorkSize[i] = static_cast<std::uint64_t>(gws[i]);
            exec_desc.DispatchLocalWorkSize[i] = static_cast<std::uint64_t>(lws[i]);
        }

        exec_desc.ResourceCount = resources_.size();
        if (exec_desc.ResourceCount >= std::size(exec_desc.Resources))
        {
            assert(!"Please extend number of supported resources for custom metacommand!");
        }

        // [1] Prepare resources pointer handles 
        for (std::size_t idx = 0; const auto & [bind_indx, rsc] : resources_)
        {
            exec_desc.ResourceBindOffset[idx] = bind_indx;

            const auto [rsc_handle, base_offset] = rsc;
            if (rsc_handle)
            {
                // set offset no matter what type of resource
                auto mem_ptr = reinterpret_cast<ID3D12Resource*>(rsc_handle);
                exec_desc.ResourcesByteOffset[idx] = base_offset;
                exec_desc.ResourceBindType[idx] = META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_ADDRESS;
                exec_desc.ResourcesAddress[idx] = mem_ptr->GetGPUVirtualAddress();
            }
            idx++;
        }

        // [2] Build execution time constants 
        std::vector<std::byte> execution_time_constants;
        for (std::size_t i = 0; const auto & [idx, scalar] : scalars_)
        {
            //exec_desc.RuntimeConstantsBindOffsets[i] = idx;
            //exec_desc.RuntimeConstantsMemorySizes[i] = scalar.size;
            //exec_desc.RuntimeConstantsMemoryOffsets[i] = execution_time_constants.size();;
            //execution_time_constants.resize(execution_time_constants.size() + scalar.size);
            i++;
        }
        auto* ptr_to_copy_data = execution_time_constants.data();
        for (const auto& [idx, scalar] : scalars_)
        {
            //std::memcpy(ptr_to_copy_data, scalar.data, scalar.size);
            //ptr_to_copy_data += scalar.size;
        }
        exec_desc.RuntimeConstantsCount = scalars_.size();
        exec_desc.RuntimeConstants = reinterpret_cast<UINT64>(execution_time_constants.data());

        // [3] Build slm
        if (locals_.size() > 1)
        {
            assert("!Unsupported case. Please remove this check and test - if it fails most probably driver need changes!");
        }

        for (const auto& [idx, slm_size] : locals_)
        {
            exec_desc.SharedLocalMemorySize += slm_size;
        }

        cmd_list->ExecuteMetaCommand(mc_.Get(), &exec_desc, sizeof(exec_desc));
    }

private:
    ComPtr<ID3D12MetaCommand> mc_;
    std::unordered_map<std::size_t, std::pair<ResourceDX12*, std::size_t>> resources_;
    std::unordered_map<std::size_t, std::variant<float, std::uint32_t>> scalars_;
    std::unordered_map<std::size_t, std::size_t> locals_;
    std::string name_;
};


class StreamDX12 : public inference_engine::Stream<StreamDX12>
{
public:
    StreamDX12(ComPtr<ID3D12GraphicsCommandList> cmd_list)
        : cmd_list_(cmd_list)
    {}

    void disaptch_resource_barrier(std::vector<ResourceDX12*> rscs_list)
    {
        std::vector<CD3DX12_RESOURCE_BARRIER> barriers(rscs_list.size());
        for (auto i = 0; i < barriers.size(); i++)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::UAV(rscs_list[i]->get()));
        }
        cmd_list_->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    void dispatch_kernel(KernelDX12& kernel, std::uint32_t gws[3], std::uint32_t lws[3])
    {
        ID3D12GraphicsCommandList4* cmd4 = nullptr;
        throw_if_failed(cmd_list_->QueryInterface(&cmd4), "cant cast cmd_list_ to ID3D12GraphicsCommandList");
        kernel.execute(cmd4, gws, lws);
    }

private:
    ComPtr<ID3D12GraphicsCommandList> cmd_list_ = nullptr;
};

class DeviceDX12 : public inference_engine::Device<DeviceDX12>
{
public:
    DeviceDX12(ComPtr<ID3D12Device> device)
        : device_(device)
    {}

    KernelDX12 create_kernel(const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
    {
        return KernelDX12(device_.Get(), kernel_name, kernel_code, kernel_code_size, build_options, language);
    }

    ResourceDX12 allocate_resource(std::size_t size)
    {
        return ResourceDX12(create_buffer(device_.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    }

private:
    ComPtr<ID3D12Device> device_ = nullptr;
};

using ContextDX12 = inference_engine::Context<DeviceDX12, StreamDX12, ResourceDX12, KernelDX12>;