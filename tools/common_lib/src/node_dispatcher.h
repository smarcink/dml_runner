#pragma once
#include "layers_utils.h"
#include "dx12_utils.h"
#include "conformance_check_helper.h"

enum class NodeType
{
    eGemmDml,
    eGemmCm,
    eGemmUmdD3d12,
    eConvDml,
    eConvCm,
    eConvUmdD3d12,
    eSoftmaxDml,
    eSoftmaxCm,
    eMvnDml,
    eMvnCm,
    eMhaDml,
    eMhaCm,
    eMemoryBandwidth,
    eQuantGemmDml,
    eQuantGemmUmdD3d12,
    eCount
};

class NodeDispatcher
{
public:
    virtual std::uint32_t get_total_descriptor_count() = 0;
    virtual void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) = 0;
    virtual void execute(ID3D12GraphicsCommandList* cmd_list) = 0;

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches, std::size_t reference_dispatch_iterations) = 0;

    virtual bool is_needing_descriptor_heap(){
        return true;
    }

    virtual ~NodeDispatcher() = default;
};


