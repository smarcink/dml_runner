#include "utils.h"

#include "iumd_d3d12_impl.h"


struct UmdEngine
{
    Dx12Engine dx12_engine;
    iumd::custom_metacommand::UmdD3d12Device device;
    iumd::custom_metacommand::UmdD3d12CommandList cmd_list;


    UmdEngine(Dx12Engine&& engine)
        : dx12_engine(std::move(engine))
        , device(dx12_engine.d3d12_device.Get(), dx12_engine.intel_extension_d3d12.get_info())
        , cmd_list(dx12_engine.command_list.Get())
    {

    }
    static inline UmdEngine create_umd_engine()
    {
        return UmdEngine(Dx12Engine());
    }
};

inline UmdEngine g_umd_engine = UmdEngine::create_umd_engine();

TEST(ApiTest, ResourceAllocation)
{
    auto mem_ptr = g_umd_engine.device.allocate_memory(1024);
    EXPECT_TRUE(mem_ptr != nullptr);
}

TEST(ApiTest, MemoryFillWithPattern)
{
    const std::size_t buffer_size = 1024;

    auto mem_ptr = g_umd_engine.device.allocate_memory(buffer_size);
    unsigned char pattern = 7;
    const auto result = g_umd_engine.device.fill_memory(&g_umd_engine.cmd_list, mem_ptr.get(), 1024, &pattern, sizeof(pattern));
    EXPECT_TRUE(result == true);

    g_umd_engine.dx12_engine.wait_for_execution();

    auto readback_buffer = create_buffer(g_umd_engine.dx12_engine.d3d12_device.Get(), buffer_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    auto typed_mem_ptr = static_cast<iumd::custom_metacommand::UmdD3d12Memory*>(mem_ptr.get());
    auto output_buffer = typed_mem_ptr->get_resource();

    auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(typed_mem_ptr->get_resource(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    g_umd_engine.dx12_engine.command_list->ResourceBarrier(1, &readback_output_barrirer);
    g_umd_engine.dx12_engine.command_list->CopyResource(readback_buffer.Get(), output_buffer);
    g_umd_engine.dx12_engine.wait_for_execution();

    std::vector<std::byte> data_out(buffer_size);
    std::byte* readback_mapped_ptr = nullptr;
    readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
    std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
    readback_buffer->Unmap(0, nullptr);

    // result validation
    for (const auto v : data_out)
    {
        EXPECT_EQ(static_cast<unsigned char>(v), pattern);
    }

}