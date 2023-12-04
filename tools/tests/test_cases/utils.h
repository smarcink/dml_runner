#pragma once

#include <gtest/gtest.h>

#include <dx12_utils.h>

struct Dx12Engine
{
    ComPtr<ID3D12Device> d3d12_device;
    ComPtr<ID3D12CommandQueue> command_queue;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;

    IntelExtension intel_extension_d3d12;

    ComPtr<IDMLDevice> dml_device;
    ComPtr<IDMLCommandRecorder> dml_command_recorder;

    Dx12Engine()
    {
        initalize_d3d12(d3d12_device, command_queue, command_allocator, command_list);
        assert(d3d12_device && command_queue && command_allocator && command_list);
        // init extension
        intel_extension_d3d12 = IntelExtension(d3d12_device.Get());
        // init dml objects
        dml_device = create_dml_device(d3d12_device.Get());
        throw_if_failed(dml_device->CreateCommandRecorder(IID_PPV_ARGS(dml_command_recorder.ReleaseAndGetAddressOf())), "create dml command recorder");
    }

    void wait_for_execution()
    {
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());
    }
};