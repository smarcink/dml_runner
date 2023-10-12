#pragma once

#include <d3d12.h>
#include <oneapi/dnnl/iumd/iumd.h>


class UmdD3d12Memory : public IUMDMemory
{
public:

};


class UmdD3d12Device : public IUMDDevice
{
public:
    UmdD3d12Device(ID3D12Device* device)
        : impl_(device)
    {

    }

    bool is_intel_platform() const override
    {
        // dummy, this should check vendorID from adapter or sth like that
        return true;
    }
private:
    ID3D12Device* impl_;
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