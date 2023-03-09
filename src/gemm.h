#pragma once

#include <iostream>
#include <optional>
#include <span>
#include <format>
#include <random>

#include "layers_utils.h"

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



class GemmDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        std::uint32_t M;
        std::uint32_t K;
        std::uint32_t N;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            opts->add_option("M", params.M)->required();
            opts->add_option("K", params.K)->required();
            opts->add_option("N", params.N)->required();
        }
    };
public:
    GemmDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , gemm_(dml::TensorDimensions{ 1, 1, params_.M, params_.K }, dml::TensorDimensions{ 1, 1, params_.K, params_.N }, dml_device, d3d12_device)
        , d3d12_device_(d3d12_device)
        , input_data_a_(params_.M* params_.K)
        , input_data_b_(params_.K* params_.N)
    {
        const auto tensor_a_desc = gemm_.get_tensor_a_desc();
        const auto tensor_b_desc = gemm_.get_tensor_b_desc();
        const auto tensor_out_desc = gemm_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_a_desc.totalTensorSizeInBytes;
        const auto tensor_b_bytes_width = tensor_b_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;


        upload_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width + tensor_b_bytes_width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_a_ = create_buffer(d3d12_device, tensor_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        input_buffer_b_ = create_buffer(d3d12_device, tensor_b_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);

        for (auto& a : input_data_a_)
        {
            a = uniform_distribution(random_generator);
        }
        for (auto& b : input_data_b_)
        {
            b = uniform_distribution(random_generator);
        }

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::memcpy(upload_mapped_ptr, input_data_a_.data(), tensor_a_bytes_width);
        std::memcpy(upload_mapped_ptr + tensor_a_bytes_width, input_data_b_.data(), tensor_b_bytes_width);
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);


        cmd_list->CopyBufferRegion(input_buffer_a_.Get(), 0, upload_buffer_.Get(), 0, tensor_a_bytes_width);
        cmd_list->CopyBufferRegion(input_buffer_b_.Get(), 0, upload_buffer_.Get(), tensor_a_bytes_width, tensor_b_bytes_width);

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_a_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_b_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return gemm_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        gemm_.create_binding_tables(cpu_handle, gpu_handle);

        // Record execution of the operator initializer.
        gemm_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        gemm_.record_execute(dml_cmd_recorder_, cmd_list, output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
    {
        const auto tensor_a_desc = gemm_.get_tensor_a_desc();
        const auto tensor_b_desc = gemm_.get_tensor_b_desc();
        const auto tensor_out_desc = gemm_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_a_desc.totalTensorSizeInBytes;
        const auto tensor_b_bytes_width = tensor_b_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;

        // readback data and validate
        auto readback_buffer = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> data_out(tensor_out_bytes_width);
        std::byte* readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
        readback_buffer->Unmap(0, nullptr);

        std::vector<float> cpu_data_f32(params_.M * params_.N);
        cpu_op::gemm<float>(params_.M, params_.K, params_.N, 1.0f, 1.0f, input_data_a_.data(), input_data_b_.data(), nullptr, cpu_data_f32.data());

        const auto* gpu_data_out_f32 = reinterpret_cast<const float*>(data_out.data());
        // compare results
        ConformanceResult ret;
        ret.epsilon = 0.001f;
        for (std::uint32_t i = 0; i < params_.M * params_.N; i++)
        {
            ret.node_value = gpu_data_out_f32[i];
            ret.reference_value = cpu_data_f32[i];
            const auto abs_diff = std::abs(ret.node_value - ret.reference_value);

            if (abs_diff > ret.epsilon)
            {
                ret.passed = false;

                std::cout << std::format("Mismatch, gpu: {}, cpu: {}, at index: {}. Absolute differece: \n", ret.node_value, ret.reference_value, i, abs_diff);
            }
            ret.biggest_difference = std::max(ret.biggest_difference, abs_diff);
            ret.tested_samples_count++;
        }
        return ret;
    }

private:
    create_params_t params_;
    gpu_op::Gemm gemm_;
    ID3D12Device* d3d12_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<float> input_data_a_;
    ComPtr<ID3D12Resource> input_buffer_a_;
    std::vector<float> input_data_b_;
    ComPtr<ID3D12Resource> input_buffer_b_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};