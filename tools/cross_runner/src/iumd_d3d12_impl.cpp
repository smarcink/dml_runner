#include "iumd_d3d12_impl.h"

#include <cassert>

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

    throw_if_failed(dev5->CreateMetaCommand(GUID_CUSTOM, 0, &create_desc, sizeof(create_desc), IID_PPV_ARGS(mc_.ReleaseAndGetAddressOf())), "cant create custom metacommand");

    //D3D12_FEATURE_DATA_QUERY_META_COMMAND query{};
    //dev5->CheckFeatureSupport(D3D12_FEATURE_QUERY_META_COMMAND, &query, sizeof(query));
}