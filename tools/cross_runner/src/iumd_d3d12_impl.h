#pragma once


#include <oneapi/dnnl/iumd/iumd.h>

#include <d3d12.h>
#define INTC_IGDEXT_D3D12
#include <igdext.h>

class UmdD3d12Memory : public IUMDMemory
{
public:

};

class UmdD3d12Device : public IUMDDevice
{
public:
    UmdD3d12Device(ID3D12Device* device, INTCExtensionInfo extension_info)
        : impl_(device)
    {
        assert(device);
        ComPtr<IDXGIFactory4> dxgi_factory;
        throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(dxgi_factory.ReleaseAndGetAddressOf())), "dxgi factory");
        dxgi_factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter_));


        D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
        if (SUCCEEDED(device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
            //exec_size = options1.WaveLaneCountMin; // i.e.: 8 for DG2, 16 for ELG
        }

        sku_.eu_count = extension_info.IntelDeviceInfo.EUCount;
        sku_.igfx = [](const auto name)
        {
            if (name == L"Tigerlake")
            {
                return UMD_IGFX_TIGERLAKE_LP;
            }
            else if (name == L"Meteorlake")
            {
                return UMD_IGFX_METEORLAKE;
            }
            else if (name == L"DG2")
            {
                return UMD_IGFX_DG2;
            }
            return UMD_IGFX_UNKNOWN;
        }(extension_info.IntelDeviceInfo.GTGenerationName);
    }

    UMD_IGFX get_umd_igfx() const override
    {
        return sku_.igfx;
    }

    bool can_use_systolic() const override
    {
        if (sku_.igfx == UMD_IGFX_DG2)
        {
            return true;
        }
        return false;
    }

    virtual std::uint32_t get_vendor_id() const override
    {
        const auto desc = get_adapter_desc();
        return desc.VendorId;
    }

private:
    DXGI_ADAPTER_DESC get_adapter_desc() const
    {
        DXGI_ADAPTER_DESC desc{};
        throw_if_failed(adapter_->GetDesc(&desc), "get adapter desc");
        return desc;
    }

private:
    struct SKU
    {
        std::uint32_t eu_count = 0;
        UMD_IGFX igfx = UMD_IGFX_UNKNOWN;
    };

private:
    ID3D12Device* impl_;
    IDXGIAdapter* adapter_;
    SKU sku_{};
    
};

class UmdD3d12CommandList : public IUMDCommandList
{
public:
    UmdD3d12CommandList(ID3D12CommandList* cmd_list) 
        : impl_(cmd_list)
    {}
    bool supports_out_of_order() const { return true; }
    bool supports_profiling() const { return false; }

private:
    ID3D12CommandList* impl_;
};