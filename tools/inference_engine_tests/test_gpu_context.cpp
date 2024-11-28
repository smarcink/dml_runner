#include "test_gpu_context.h"

void initalize_d3d12(ComPtr<ID3D12Device>& d3D12_device, ComPtr<ID3D12CommandQueue>& command_queue, ComPtr<ID3D12CommandAllocator>& command_allocator, ComPtr<ID3D12GraphicsCommandList>& command_list, bool use_rcs)
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

ComPtr<ID3D12DescriptorHeap> create_descriptor_heap(ID3D12Device* d3d12_device, uint32_t descriptors_count)
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

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* d3d12_device, std::size_t bytes_width, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES init_state, D3D12_RESOURCE_FLAGS resource_flag)
{
    auto align_size = [](const auto value, const auto alignment)
        {
            assert(alignment >= 1);
            return ((value + alignment - 1) / alignment) * alignment;
        };

    ComPtr<ID3D12Resource> ret;
    const auto heap_props = CD3DX12_HEAP_PROPERTIES(heap_type);
    const auto buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(align_size(bytes_width, 4), resource_flag);
    throw_if_failed(d3d12_device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &buffer_desc,
        init_state,
        nullptr, IID_PPV_ARGS(ret.ReleaseAndGetAddressOf())), "create commited resource");
    return ret;
}

void dispatch_resource_barrier(ID3D12GraphicsCommandList* command_list, const std::vector<CD3DX12_RESOURCE_BARRIER>& barriers)
{
    command_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
}

void close_execute_reset_wait(ID3D12Device* d3d12_device, ID3D12CommandQueue* command_queue,
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