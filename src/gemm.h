#pragma once

#include <iostream>
#include <optional>
#include <span>
#include <format>


namespace gpu_op
{
class Gemm
{
public:
    Gemm(const dml::TensorDimensions& a_dims, const dml::TensorDimensions& b_dims, IDMLDevice* dml_device, ID3D12Device* d3d12_device)
        : dml_device_(dml_device)
        , d3d12_device_(d3d12_device)
        , graph_(dml_device)
    {
        tensor_a_ = dml::InputTensor(graph_, 0, dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, a_dims, dml::TensorPolicy::InterleavedChannel()));
        tensor_b_ = dml::InputTensor(graph_, 1, dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, b_dims, dml::TensorPolicy::InterleavedChannel()));
        tensor_out_ = dml::Gemm(tensor_a_, tensor_b_, {}, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, 1.0f, 1.0f, dml::FusedActivation::None());

        // compiled operator
        DML_EXECUTION_FLAGS exec_flags = DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        std::vector<dml::Expression> graph_outputs{ tensor_out_ };
        dml_op_executor_ = graph_.Compile(exec_flags, graph_outputs);
        IDMLCompiledOperator* dml_compiled_operators[] = { dml_op_executor_.Get() };

        // initlaizer operator
        dml_device->CreateOperatorInitializer(1, dml_compiled_operators, IID_PPV_ARGS(dml_op_initializer_.ReleaseAndGetAddressOf()));

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
            throw_if_failed(d3d12_device->CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(temporary_buffer_.ReleaseAndGetAddressOf())), "create buffer resource");
        }

        if (persistent_resource_size != 0)
        {
            const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            const auto buffder_desc = CD3DX12_RESOURCE_DESC::Buffer(persistent_resource_size);
            throw_if_failed(d3d12_device->CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &buffder_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(persistent_buffer_.ReleaseAndGetAddressOf())), "create buffer resource");
        }
    }

    dml::TensorDesc get_tensor_a_desc() const
    {
        return tensor_a_.GetOutputDesc();
    }

    dml::TensorDesc get_tensor_b_desc() const
    {
        return tensor_b_.GetOutputDesc();
    }

    dml::TensorDesc get_tensor_out_desc() const
    {
        return tensor_out_.GetOutputDesc();
    }

    void record_initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
    {
        if (temporary_buffer_)
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

    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, ID3D12Resource* resource_out, ID3D12Resource* resource_a, ID3D12Resource* resource_b)
    {
        if (temporary_buffer_)
        {
            DML_BUFFER_BINDING buffer_binding{ temporary_buffer_.Get(), 0, temporary_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_exec_binding_table->BindTemporaryResource(&binding_desc);
        }

        if (persistent_buffer_)
        {
            // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
            DML_BUFFER_BINDING buffer_binding{ persistent_buffer_.Get(), 0, persistent_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_exec_binding_table->BindPersistentResource(&binding_desc);
        }

        DML_BUFFER_BINDING input_buffer_a_binding{ resource_a, 0, resource_a->GetDesc().Width};
        DML_BUFFER_BINDING input_buffer_b_binding{ resource_b, 0, resource_b->GetDesc().Width};

        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(2);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_buffer_a_binding });
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_buffer_b_binding });
        dml_exec_binding_table->BindInputs(static_cast<std::uint32_t>(input_bindings.size()), input_bindings.data());

        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width};
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };
        dml_exec_binding_table->BindOutputs(1u, &output_binding_desc);

        dml_cmd_recorder->RecordDispatch(cmd_list, dml_op_executor_.Get(), dml_exec_binding_table.Get());
    }

    uint32_t get_total_descriptor_count() const
    {
        const auto initialize_binding_properties = dml_op_initializer_->GetBindingProperties();
        const auto execute_binding_properties = dml_op_executor_->GetBindingProperties();
        return initialize_binding_properties.RequiredDescriptorCount +
            execute_binding_properties.RequiredDescriptorCount;
    }

    void create_binding_tables(D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
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

private:
    IDMLDevice* dml_device_;
    ID3D12Device* d3d12_device_;

    dml::Graph graph_;
    dml::Expression tensor_a_;
    dml::Expression tensor_b_;
    dml::Expression tensor_out_;

    ComPtr<IDMLCompiledOperator> dml_op_executor_;
    ComPtr<IDMLOperatorInitializer> dml_op_initializer_;

    ComPtr<ID3D12Resource> temporary_buffer_;
    ComPtr<ID3D12Resource> persistent_buffer_;

    ComPtr<IDMLBindingTable> dml_init_binding_table;
    ComPtr<IDMLBindingTable> dml_exec_binding_table;
};
}


namespace cpu_op
{

template<typename T>
void gemm(std::uint32_t M, std::uint32_t K, std::uint32_t N, float alpha, float beta, T* src0, T* src1, T* src2, T* dst)
{
    for (std::uint32_t y = 0; y < M; ++y)
    {
        for (std::uint32_t x = 0; x < N; ++x)
        {
            T sum = T(0);
            for (std::uint32_t k = 0; k < K; ++k)
            {
                sum += src0[k + y * K] * src1[k * N + x];
            }
            dst[x + y * N] = alpha * sum + (src2 ? (beta * src2[x + y * N]) : T(0));
        }
    }
}
}