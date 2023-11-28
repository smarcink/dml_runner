#pragma once

#include <gtest/gtest.h>

#include <dx12_utils.h>

#include <dml_base_node.h>

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


class NodeDispatcherBase
{
public:
    virtual void run()
    {
        auto node = create_dispatcher_impl();

        // wait for any potential uploads
        g_dx12_engine.wait_for_execution();

        // bind descriptor heap
        const auto descriptors_count = node->get_total_descriptor_count();
        auto descriptor_heap = create_descriptor_heap(g_dx12_engine.d3d12_device.Get(), descriptors_count);
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        g_dx12_engine.command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        // initalize
        node->initialize(g_dx12_engine.command_list.Get(), descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        g_dx12_engine.wait_for_execution();

        // Bind and execute node
        g_dx12_engine.command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
        node->execute(g_dx12_engine.command_list.Get());
        g_dx12_engine.wait_for_execution();

        // finally validate conformance
        const auto conformance_result = node->validate_conformance(g_dx12_engine.command_queue.Get(), g_dx12_engine.command_allocator.Get(), g_dx12_engine.command_list.Get(), false);

        // we expect perfect match
        // comaprision have to be done vs dnnl!
        // vs HLSL there can be differences
        const auto perfect_match = conformance_result.biggest_difference == 0.0f;
        if (!perfect_match && conformance_result.passed)
        {
            std::cout << "Conformance has passed, but it wasn't perfect match. Was the tested validate vs dnnl?" << std::endl;
        }
        EXPECT_TRUE(perfect_match);
    }

protected:
    inline static Dx12Engine g_dx12_engine = Dx12Engine(); // create single engine to be reused across tests!
    virtual std::unique_ptr<NodeDispatcher> create_dispatcher_impl() = 0;
};