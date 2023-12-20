#pragma once
#include "dx12_utils.h"
#include "layers_utils.h"
#include "node_dispatcher.h"

namespace
{
    inline static dml::TensorProperties compute_nchw_tensor_policy(
        DML_TENSOR_DATA_TYPE dataType,
        DML_TENSOR_FLAGS /*flags*/,
        std::span<const uint32_t> sizes)
    {
        uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
        dml::TensorStrides strides(dimension_count);

        uint32_t stride = 1;
        for (std::uint32_t i = dimension_count; i > 0; i--)
        {
            strides[i - 1] = stride;
            stride *= sizes[i - 1];
        }

        dml::TensorProperties props;
        props.strides = std::move(strides);
        props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimension_count, sizes.data(), props.strides->data());
        props.guaranteedBaseOffsetAlignment = 0;
        return props;
    }

    inline static dml::TensorProperties compute_w_tensor_policy(
        DML_TENSOR_DATA_TYPE dataType,
        DML_TENSOR_FLAGS /*flags*/,
        std::span<const uint32_t> sizes)
    {
        uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
        dml::TensorStrides strides(dimension_count);

        for (auto& s : strides)
        {
            s = 0;
        }
        strides.back() = 1;

        dml::TensorProperties props;
        props.strides = std::move(strides);
        props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimension_count, sizes.data(), props.strides->data());
        props.guaranteedBaseOffsetAlignment = 0;
        return props;
    }

    inline static dml::TensorProperties compute_nchw_alignw320_tensor_policy(
        DML_TENSOR_DATA_TYPE dataType,
        DML_TENSOR_FLAGS /*flags*/,
        std::span<const uint32_t> sizes)
    {
        const uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
        dml::TensorStrides strides(dimension_count);

        dml::TensorDimensions dims(dimension_count);
        for (std::uint32_t i = 0; i < dimension_count; i++)
        {
            dims[i] = sizes[i];
        }
        dims.back() = align(dims.back(), 320);

        uint32_t stride = 1;
        for (std::uint32_t i = dimension_count; i > 0; i--)
        {
            strides[i - 1] = stride;
            stride *= dims[i - 1];
        }

        dml::TensorProperties props;
        props.strides = std::move(strides);
        props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimension_count, sizes.data(), props.strides->data());
        props.guaranteedBaseOffsetAlignment = 0;
        return props;
    }
}

inline std::string to_string(const std::string& value) { return value; }

inline DML_TENSOR_DATA_TYPE to_dml_data_type(DataType dt)
{
    switch (dt)
    {
    case DataType::eFp32: return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::eFp16: return DML_TENSOR_DATA_TYPE_FLOAT16;
    default:
        assert(false && "Unknown data type.");
    }
    return DML_TENSOR_DATA_TYPE_UNKNOWN;
}

inline dml::TensorPolicy to_dml_tensor_policy(DataLayout layout)
{
    switch (layout)
    {
    case DataLayout::eCHW:
    case DataLayout::eNCHW: return dml::TensorPolicy::Default();
    case DataLayout::eNHWC: return dml::TensorPolicy::InterleavedChannel();
    case DataLayout::eW: return dml::TensorPolicy(compute_w_tensor_policy);
    case DataLayout::eNCHW_AlignW320: return dml::TensorPolicy(compute_nchw_alignw320_tensor_policy);
    default:
        assert(false && "Unknown data layout.");
    }
    return dml::TensorPolicy::Default();
}

namespace gpu_op
{
class DirectMlBaseNode
{
public:
    DirectMlBaseNode(IDMLDevice* dml_device, ID3D12Device* d3d12_device)
        : dml_device_(dml_device)
        , d3d12_device_(d3d12_device)
    {
    }

    virtual void record_initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
    {
        const auto initialize_binding_properties = dml_op_initializer_->GetBindingProperties();
        if (initialize_binding_properties.TemporaryResourceSize > 0 && temporary_buffer_)
        {
            DML_BUFFER_BINDING buffer_binding{ temporary_buffer_.Get(), 0, temporary_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_init_binding_table->BindTemporaryResource(&binding_desc);
        }

        if (persistent_buffer_)
        {
            // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
            DML_BUFFER_BINDING buffer_binding{ persistent_buffer_.Get(), 0, persistent_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_init_binding_table->BindOutputs(1, &binding_desc);
        }

        dml_cmd_recorder->RecordDispatch(
            cmd_list,
            dml_op_initializer_.Get(),
            dml_init_binding_table.Get());
    }

    virtual uint32_t get_total_descriptor_count() const
    {
        const auto initialize_binding_properties = dml_op_initializer_->GetBindingProperties();
        const auto execute_binding_properties = dml_op_executor_->GetBindingProperties();
        return initialize_binding_properties.RequiredDescriptorCount +
            execute_binding_properties.RequiredDescriptorCount;
    }

    virtual void create_binding_tables(D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        const auto desc_inc = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const auto init_desc_count = dml_op_initializer_->GetBindingProperties().RequiredDescriptorCount;
        const auto exec_desc_count = dml_op_executor_->GetBindingProperties().RequiredDescriptorCount;

        {
            DML_BINDING_TABLE_DESC dml_binding_table_desc{};
            dml_binding_table_desc.Dispatchable = dml_op_initializer_.Get();
            dml_binding_table_desc.CPUDescriptorHandle = cpu_handle;
            dml_binding_table_desc.GPUDescriptorHandle = gpu_handle;
            dml_binding_table_desc.SizeInDescriptors = dml_op_initializer_->GetBindingProperties().RequiredDescriptorCount;
            throw_if_failed(dml_device_->CreateBindingTable(
                &dml_binding_table_desc, IID_PPV_ARGS(dml_init_binding_table.ReleaseAndGetAddressOf())), "dml create init binding table");
        }

        {
            DML_BINDING_TABLE_DESC dml_binding_table_desc{};
            dml_binding_table_desc.Dispatchable = dml_op_executor_.Get();
            dml_binding_table_desc.CPUDescriptorHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu_handle, init_desc_count, desc_inc);
            dml_binding_table_desc.GPUDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu_handle, init_desc_count, desc_inc);
            dml_binding_table_desc.SizeInDescriptors = exec_desc_count;
            throw_if_failed(dml_device_->CreateBindingTable(
                &dml_binding_table_desc, IID_PPV_ARGS(dml_exec_binding_table.ReleaseAndGetAddressOf())), "dml create exec binding table");
        }
    }

protected:
    virtual void create_operator_impl()
    {
        assert(dml_op_executor_ != nullptr);
        IDMLCompiledOperator* dml_compiled_operators[] = { dml_op_executor_.Get() };
        // initlaizer operator
        dml_device_->CreateOperatorInitializer(1, dml_compiled_operators, IID_PPV_ARGS(dml_op_initializer_.ReleaseAndGetAddressOf()));

        // resources
        const auto initialize_binding_properties = dml_op_initializer_->GetBindingProperties();
        const auto execute_binding_properties = dml_op_executor_->GetBindingProperties();
        const auto temporary_resource_size = std::max(
            initialize_binding_properties.TemporaryResourceSize,
            execute_binding_properties.TemporaryResourceSize);
        const auto persistent_resource_size = execute_binding_properties.PersistentResourceSize;

        if (temporary_resource_size != 0)
        {
            const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            const auto buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(temporary_resource_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            throw_if_failed(d3d12_device_->CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(temporary_buffer_.ReleaseAndGetAddressOf())), "create buffer resource");
        }

        if (persistent_resource_size != 0)
        {
            const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            const auto buffder_desc = CD3DX12_RESOURCE_DESC::Buffer(persistent_resource_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            throw_if_failed(d3d12_device_->CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &buffder_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(persistent_buffer_.ReleaseAndGetAddressOf())), "create buffer resource");
        }
    }

    virtual void record_execute_impl(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, 
        const std::vector<DML_BINDING_DESC>& input_bindings, const std::vector<DML_BINDING_DESC>& output_bindings)
    {
        const auto execute_binding_properties = dml_op_executor_->GetBindingProperties();
        if (execute_binding_properties.TemporaryResourceSize > 0 && temporary_buffer_)
        {
            DML_BUFFER_BINDING buffer_binding{ temporary_buffer_.Get(), 0, temporary_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_exec_binding_table->BindTemporaryResource(&binding_desc);
        }

        if (execute_binding_properties.PersistentResourceSize > 0 && persistent_buffer_)
        {
            // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
            DML_BUFFER_BINDING buffer_binding{ persistent_buffer_.Get(), 0, persistent_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_exec_binding_table->BindPersistentResource(&binding_desc);
        }

        dml_exec_binding_table->BindInputs(static_cast<std::uint32_t>(input_bindings.size()), input_bindings.data());
        dml_exec_binding_table->BindOutputs(static_cast<std::uint32_t>(output_bindings.size()), output_bindings.data());

        dml_cmd_recorder->RecordDispatch(cmd_list, dml_op_executor_.Get(), dml_exec_binding_table.Get());
    }

protected:
    IDMLDevice* dml_device_;
    ID3D12Device* d3d12_device_;

    ComPtr<IDMLCompiledOperator> dml_op_executor_ = nullptr;
    ComPtr<IDMLOperatorInitializer> dml_op_initializer_ = nullptr;

    ComPtr<ID3D12Resource> temporary_buffer_;
    ComPtr<ID3D12Resource> persistent_buffer_;

    ComPtr<IDMLBindingTable> dml_init_binding_table;
    ComPtr<IDMLBindingTable> dml_exec_binding_table;

};
}