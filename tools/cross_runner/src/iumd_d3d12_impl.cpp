#include "iumd_d3d12_impl.h"

#include <cassert>

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
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL_STATELESS
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
struct META_COMMAND_EXECUTE_CUSTOM_DESC
{
    D3D12_GPU_DESCRIPTOR_HANDLE Resources[20];
    UINT64                      ResourcesByteOffsets[20];  // works only in stateless mode

    UINT64 ResourceCount;
    UINT64 RuntimeConstants;
    UINT64 RuntimeConstantsCount;

    UINT64 DispatchThreadGroup[3];
};


UmdD3d12Device::UmdD3d12Device(ID3D12Device* device, INTCExtensionInfo extension_info)
    : impl_(device)
{
    assert(device);
    ComPtr<IDXGIFactory4> dxgi_factory;
    throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(dxgi_factory.ReleaseAndGetAddressOf())), "dxgi factory");
    dxgi_factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter_));
    const auto adapter_desc = get_adapter_desc();

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    if (SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
        sku_.max_simd_size = options1.WaveLaneCountMax;
    }
    const auto wstr_name = std::wstring(adapter_desc.Description);
    sku_.name = std::string(wstr_name.begin(), wstr_name.end());  // not ideal, enough for now
    sku_.eu_count = extension_info.IntelDeviceInfo.EUCount;
    sku_.igfx = [](const auto name)
    {
        if (std::wcscmp(name, L"Tigerlake") == 0)
        {
            return UMD_IGFX_TIGERLAKE_LP;
        }
        else if (std::wcscmp(name, L"Meteorlake") == 0)
        {
            return UMD_IGFX_METEORLAKE;
        }
        else if (std::wcscmp(name, L"DG2") == 0)
        {
            return UMD_IGFX_DG2;
        }
        return UMD_IGFX_UNKNOWN;
    }(extension_info.IntelDeviceInfo.GTGenerationName);

    if (sku_.igfx == UMD_IGFX_DG2 || sku_.igfx == UMD_IGFX_METEORLAKE)
    {
        sku_.eu_per_dss = 16;
        sku_.threads_per_eu = 8;

    }
    else if (sku_.igfx == UMD_IGFX_TIGERLAKE_LP)
    {
        sku_.eu_per_dss = 16;
        sku_.threads_per_eu = 7;
    }
}

UmdD3d12PipelineStateObject::UmdD3d12PipelineStateObject(UmdD3d12Device* device, const char* kernel_name, const char* code_string, const char* build_options, UMD_SHADER_LANGUAGE language)
{
    auto d3d12_dev = device->get_d3d12_device();
    ID3D12Device5* dev5 = nullptr;
    throw_if_failed(d3d12_dev->QueryInterface(&dev5), "cant cast d3d12 device to ID3D12Device5");

    META_COMMAND_CREATE_CUSTOM_DESC create_desc{};
    create_desc.ShaderSourceCode = reinterpret_cast<UINT64>(code_string);
    create_desc.ShaderSourceCodeSize = std::strlen(code_string);
    create_desc.BuildOptionString = reinterpret_cast<UINT64>(build_options);
    create_desc.BuildOptionStringSize = std::strlen(build_options);
    switch (language)
    {
    case UMD_SHADER_LANGUAGE_OCL:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL;
        break;
    case UMD_SHADER_LANGUAGE_OCL_STATELESS:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL_STATELESS;
        break;
    default:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE;
        
    }
    assert(create_desc.ShaderLanguage != META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE);

    throw_if_failed(dev5->CreateMetaCommand(GUID_CUSTOM, 0, &create_desc, sizeof(create_desc), IID_PPV_ARGS(mc_.ReleaseAndGetAddressOf())), "cant create custom metacommand");

    //D3D12_FEATURE_DATA_QUERY_META_COMMAND query{};
    //dev5->CheckFeatureSupport(D3D12_FEATURE_QUERY_META_COMMAND, &query, sizeof(query));
}

bool UmdD3d12PipelineStateObject::set_kernel_arg(std::size_t index, const IUMDMemory* memory, std::size_t offset)
{
    auto typed_mem = memory ? dynamic_cast<const UmdD3d12Memory*>(memory) : nullptr;
    resources[index] = { typed_mem, offset };
    return true;
}

bool UmdD3d12PipelineStateObject::set_kernel_arg(std::size_t index, IUMDPipelineStateObject::ScalarArgType scalar)
{
    scalars[index] = scalar;
    return true;
}

bool UmdD3d12PipelineStateObject::execute(ID3D12GraphicsCommandList4* cmd_list, const std::array<std::size_t, 3>& gws, const std::array<std::size_t, 3>& lws)
{
    assert(gws.size() == lws.size());

    // [0] Calculate dispatch thread size
    META_COMMAND_EXECUTE_CUSTOM_DESC exec_desc{};
    for (std::size_t i = 0; i < gws.size(); i++)
    {
        if (gws[i] == 0 || lws[i] == 0)
        {
            return false;
        }
        exec_desc.DispatchThreadGroup[i] = gws[i] / lws[i];
    }

    // [1] Prepare resoruces pointer handles
    for (const auto& [idx, memory] : resources)
    {
        if (idx >= std::size(exec_desc.Resources))
        {
            assert(!"Please extend number of supported resources for custom metacommand!");
            return false;
        }
        const auto& [mem_ptr, runtime_offset] = memory;
        if (mem_ptr)
        {
            exec_desc.Resources[idx] = mem_ptr->get_gpu_descriptor_handle();
            exec_desc.ResourcesByteOffsets[idx] = mem_ptr->get_base_offset() + runtime_offset;
        }
        exec_desc.ResourceCount++;
    }

    // [2] Build execution time constants 
    std::vector<std::byte> execution_time_constants;
    for (const auto& [idx, scalar] : scalars)
    {
        execution_time_constants.resize(execution_time_constants.size() + scalar.size);
    }
    if (execution_time_constants.size() % 4 != 0)
    {
        assert(!"Please use 4 byte aligned scalars for metacommand. ToDo: This can be workaround with proper padding and alignments!");
        return false;
    }
    auto* ptr_to_copy_data = execution_time_constants.data();
    for (const auto& [idx, scalar] : scalars)
    {
        std::memcpy(ptr_to_copy_data, scalar.data, scalar.size);
        ptr_to_copy_data += scalar.size;
    }
    exec_desc.RuntimeConstantsCount += execution_time_constants.size() / 4;  // how many 4 bytes are packed in the buffer
    exec_desc.RuntimeConstants = reinterpret_cast<UINT64>(execution_time_constants.data());

    cmd_list->ExecuteMetaCommand(mc_.Get(), &exec_desc, sizeof(exec_desc));
    return true;
}

bool UmdD3d12CommandList::dispatch(IUMDPipelineStateObject* pso, const std::array<std::size_t, 3>& gws, const std::array<std::size_t, 3>& lws, const std::vector<IUMDEvent*>& deps, std::shared_ptr<IUMDEvent>* out)
{
    if (!deps.empty())
    {
        // Single global barrier is enough for now, because we use b.UAV.pResource = nullptr;
        // If we would support concrete resources barriers, than we need a way to specify resource pointer to b.UAV.pResource
        constexpr const bool single_barrier_is_enough = true;  
        std::vector<D3D12_RESOURCE_BARRIER> barriers(single_barrier_is_enough ? 1u : deps.size());
        for (auto& b : barriers)
        {
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            b.UAV.pResource = nullptr;
        }
        impl_->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    auto typed_pso = dynamic_cast<UmdD3d12PipelineStateObject*>(pso);
    const auto result = typed_pso->execute(impl_, gws, lws);
 
    if (out)
    {
        *out = std::make_shared<UmdD3d12Event>();
    }
    return result;
}