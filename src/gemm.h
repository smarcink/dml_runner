#pragma once

#include <iostream>
#include <optional>
#include <span>
#include <format>
#include <random>

#include "dml_base_node.h"

std::string to_string(const std::string& value) { return value; }

namespace gpu_op
{
class Gemm : public DirectMlBaseNode
{
public:
    Gemm(const DML_TENSOR_DATA_TYPE data_type, const TensorShape& shape_a, const dml::TensorPolicy& shape_a_tensor_policy, bool transform_a,
        const TensorShape& shape_b, const dml::TensorPolicy& shape_b_tensor_policy, bool transform_b,
        bool use_c_tensor, const TensorShape& shape_c, const dml::TensorPolicy& shape_c_tensor_policy,
        const TensorShape& shape_out, const dml::TensorPolicy& shape_out_tensor_policy,
        float alpha, float beta, IDMLDevice* dml_device, ID3D12Device* d3d12_device, bool disable_mc = false)
        : DirectMlBaseNode(dml_device, d3d12_device)
    {
        const dml::TensorDimensions input_a_dims{ shape_a.n, shape_a.c, shape_a.h, shape_a.w };
        const dml::TensorDimensions input_b_dims{ shape_b.n, shape_b.c, shape_b.h, shape_b.w };
        const dml::TensorDimensions input_c_dims{ shape_c.n, shape_c.c, shape_c.h, shape_c.w };
        const dml::TensorDimensions output_dims{ shape_out.n, shape_out.c, shape_out.h, shape_out.w };

        tensor_input_a_desc_.DataType = data_type;
        tensor_input_a_desc_.Flags = DML_TENSOR_FLAG_NONE;
        tensor_input_a_desc_.DimensionCount = static_cast<std::uint32_t>(input_a_dims.size());
        tensor_input_a_desc_.Sizes = input_a_dims.data();
        const auto tensor_a_properites = shape_a_tensor_policy.Get(tensor_input_a_desc_.DataType, tensor_input_a_desc_.Flags, input_a_dims);
        tensor_input_a_desc_.Strides = tensor_a_properites.strides.has_value() ? tensor_a_properites.strides->data() : nullptr;
        tensor_input_a_desc_.TotalTensorSizeInBytes = tensor_a_properites.totalTensorSizeInBytes;

        tensor_input_b_desc_.DataType = data_type;
        tensor_input_b_desc_.Flags = DML_TENSOR_FLAG_NONE;
        tensor_input_b_desc_.DimensionCount = static_cast<std::uint32_t>(input_b_dims.size());
        tensor_input_b_desc_.Sizes = input_b_dims.data();
        const auto tensor_b_properites = shape_b_tensor_policy.Get(tensor_input_b_desc_.DataType, tensor_input_b_desc_.Flags, input_b_dims);
        tensor_input_b_desc_.Strides = tensor_b_properites.strides.has_value() ? tensor_b_properites.strides->data() : nullptr;
        tensor_input_b_desc_.TotalTensorSizeInBytes = tensor_b_properites.totalTensorSizeInBytes;

        tensor_output_desc_.DataType = data_type;
        tensor_output_desc_.Flags = DML_TENSOR_FLAG_NONE;
        tensor_output_desc_.DimensionCount = static_cast<std::uint32_t>(output_dims.size());
        tensor_output_desc_.Sizes = output_dims.data();
        const auto tensor_out_properites = shape_b_tensor_policy.Get(tensor_output_desc_.DataType, tensor_output_desc_.Flags, output_dims);
        tensor_output_desc_.Strides = tensor_out_properites.strides.has_value() ? tensor_out_properites.strides->data() : nullptr;
        tensor_output_desc_.TotalTensorSizeInBytes = tensor_out_properites.totalTensorSizeInBytes;

        dml::TensorProperties tensor_c_properites;
        if (use_c_tensor)
        {
            tensor_input_c_desc_.emplace(DML_BUFFER_TENSOR_DESC{});
            tensor_input_c_desc_->DataType = data_type;
            tensor_input_c_desc_->Flags = DML_TENSOR_FLAG_NONE;
            tensor_input_c_desc_->DimensionCount = static_cast<std::uint32_t>(input_c_dims.size());
            tensor_input_c_desc_->Sizes = input_c_dims.data();
            tensor_c_properites = shape_c_tensor_policy.Get(tensor_input_c_desc_->DataType, tensor_input_c_desc_->Flags, input_c_dims);
            tensor_input_c_desc_->Strides = tensor_c_properites.strides.has_value() ? tensor_c_properites.strides->data() : nullptr;
            tensor_input_c_desc_->TotalTensorSizeInBytes = tensor_c_properites.totalTensorSizeInBytes;
        }

        DML_TENSOR_DESC input_a_desc{};
        input_a_desc.Desc = &tensor_input_a_desc_;
        input_a_desc.Type = DML_TENSOR_TYPE_BUFFER;

        DML_TENSOR_DESC input_b_desc{};
        input_b_desc.Desc = &tensor_input_b_desc_;
        input_b_desc.Type = DML_TENSOR_TYPE_BUFFER;

        DML_TENSOR_DESC input_c_desc{};
        if (use_c_tensor)
        {
            input_c_desc.Desc = &tensor_input_c_desc_.value();
            input_c_desc.Type = DML_TENSOR_TYPE_BUFFER;
        }

        DML_TENSOR_DESC input_out_desc{};
        input_out_desc.Desc = &tensor_output_desc_;
        input_out_desc.Type = DML_TENSOR_TYPE_BUFFER;

        DML_GEMM_OPERATOR_DESC gemm_op_desc{};
        gemm_op_desc.Alpha = alpha;
        gemm_op_desc.Beta = beta;
        gemm_op_desc.ATensor = &input_a_desc;
        gemm_op_desc.TransA = transform_a ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
        gemm_op_desc.BTensor = &input_b_desc;
        gemm_op_desc.TransB = transform_b ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
        gemm_op_desc.OutputTensor = &input_out_desc;
        if (use_c_tensor)
        {
            gemm_op_desc.CTensor = &input_c_desc;
        }

        DML_OPERATOR_DESC dml_operator_desc{};
        dml_operator_desc.Type = DML_OPERATOR_GEMM;
        dml_operator_desc.Desc = &gemm_op_desc;

        throw_if_failed(dml_device->CreateOperator(
            &dml_operator_desc, IID_PPV_ARGS(dml_operator_.ReleaseAndGetAddressOf())), "create softmax operator");

        auto exec_flags = DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        if (data_type == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            exec_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        }
        if (disable_mc)
        {
            exec_flags |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
        }

        throw_if_failed(dml_device->CompileOperator(
            dml_operator_.Get(),
            exec_flags,
            IID_PPV_ARGS(dml_op_executor_.ReleaseAndGetAddressOf())), "create softmax compiled operator");
        create_operator_impl();
    }

    dml::TensorDesc get_tensor_a_desc() const
    {
        return tensor_input_a_desc_;
    }

    dml::TensorDesc get_tensor_b_desc() const
    {
        return tensor_input_b_desc_;
    }

    dml::TensorDesc get_tensor_c_desc() const
    {
        if (tensor_input_c_desc_.has_value())
        {
            return tensor_input_c_desc_.value();
        }
        return dml::TensorDesc{};
    }

    dml::TensorDesc get_tensor_out_desc() const
    {
        return tensor_output_desc_;
    }

    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list,
        ID3D12Resource* resource_out, ID3D12Resource* resource_a, ID3D12Resource* resource_b, ID3D12Resource* resource_c)
    {
        assert(resource_a);
        assert(resource_b);
        assert(resource_out);
        DML_BUFFER_BINDING input_a_buffer_binding{ resource_a, 0, resource_a->GetDesc().Width };
        DML_BUFFER_BINDING input_b_buffer_binding{ resource_b, 0, resource_b->GetDesc().Width };
        DML_BUFFER_BINDING input_c_buffer_binding{ nullptr, 0, 0};
  
        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(3);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_a_buffer_binding });
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_b_buffer_binding });
        if (resource_c)
        {
            input_c_buffer_binding.Buffer = resource_c;
            input_c_buffer_binding.SizeInBytes = resource_c->GetDesc().Width;
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_c_buffer_binding });
        }
        else
        {
            input_bindings.push_back({ DML_BINDING_TYPE_NONE, nullptr });  //C tensor not supported yet
        }


        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_binding_desc);
    }


private:
    DML_BUFFER_TENSOR_DESC tensor_input_a_desc_{};
    DML_BUFFER_TENSOR_DESC tensor_input_b_desc_{};
    std::optional<DML_BUFFER_TENSOR_DESC> tensor_input_c_desc_{};
    DML_BUFFER_TENSOR_DESC tensor_output_desc_{};
    ComPtr<IDMLOperator> dml_operator_;
};
}


class GemmBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout_a;
        DataLayout layout_b;
        DataLayout layout_c = DataLayout::eCount;
        DataLayout layout_out;

        std::uint32_t B = 1; // batch
        std::uint32_t C = 1; // channels
        std::uint32_t M;
        std::uint32_t K;
        std::uint32_t N;

        bool use_c_tensor = false;

        bool transform_a = false;
        bool transform_b = false;

        float alpha = 1.0f;
        float beta = 0.0f;



        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout_a", params.layout_a)->required();
            add_data_layout_cli_option(opts, "--layout_b", params.layout_b)->required();
            add_data_layout_cli_option(opts, "--layout_c", params.layout_c);
            add_data_layout_cli_option(opts, "--layout_out", params.layout_out)->required();
            opts->add_option("--B", params.B);
            opts->add_option("--C", params.C);
            opts->add_option("--M", params.M)->required();
            opts->add_option("--K", params.K)->required();
            opts->add_option("--N", params.N)->required();
            opts->add_option("--alpha", params.alpha);
            opts->add_option("--beta", params.beta);
            opts->add_flag("--transform_a", params.transform_a);
            opts->add_flag("--transform_b", params.transform_b);
        }
    };
public:
    GemmBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , dml_device_(dml_device)
        , d3d12_device_(d3d12_device)
        , input_data_a_(params_.B * params_.C * params_.M * params_.K * get_data_type_bytes_width(params_.dt))
        , input_data_b_(params_.B * params_.C * params_.K * params_.N * get_data_type_bytes_width(params_.dt))
    {
        if (use_c_tensor())
        {
            const auto c_tensor_size = data_layout_dimensions_count(params_.layout_c) == 1 ? params_.N : (params_.B * params_.C * params_.M * params_.N);
            input_data_c_.resize(c_tensor_size * get_data_type_bytes_width(params_.dt));
        }
        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_a_);
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_b_);
            if (use_c_tensor())
            {
                randomize_linear_container_float(random_generator, uniform_distribution, input_data_c_);
            }
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_a_);
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_b_);
            if (use_c_tensor())
            {
                randomize_linear_container_half(random_generator, uniform_distribution, input_data_c_);
            }
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        const auto tensor_input_a_bytes_width = input_data_a_.size();
        const auto tensor_input_b_bytes_width = input_data_b_.size();
        const auto tensor_input_c_bytes_width = input_data_c_.size();
        const auto tensor_output_bytes_width = params_.B * params_.C * params_.M * params_.N * get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_a_bytes_width + tensor_input_b_bytes_width + tensor_input_c_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_a_ = create_buffer(d3d12_device, tensor_input_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        input_buffer_b_ = create_buffer(d3d12_device, tensor_input_b_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (use_c_tensor())
        {
            input_buffer_c_ = create_buffer(d3d12_device, tensor_input_c_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        output_buffer_ = create_buffer(d3d12_device, tensor_output_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_a_.data(), tensor_input_a_bytes_width);
        memcopy_offset += tensor_input_a_bytes_width;
        std::memcpy(upload_mapped_ptr + memcopy_offset, input_data_b_.data(), tensor_input_b_bytes_width);
        memcopy_offset += tensor_input_b_bytes_width;
        if (use_c_tensor())
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, input_data_c_.data(), tensor_input_c_bytes_width);
            memcopy_offset += tensor_input_c_bytes_width;
        }
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_a_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_a_bytes_width);
        memcopy_offset += tensor_input_a_bytes_width;
        cmd_list->CopyBufferRegion(input_buffer_b_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_input_b_bytes_width);
        memcopy_offset += tensor_input_b_bytes_width;
        if (use_c_tensor())
        {
            cmd_list->CopyBufferRegion(input_buffer_c_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_input_c_bytes_width);
            memcopy_offset += tensor_input_c_bytes_width;
        }

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_a_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_b_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (params_.use_c_tensor)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_c_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }

        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue, ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
    {
        const auto tensor_out_bytes_width = output_buffer_->GetDesc().Width;

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

        //
        //  calc reference with dml non-mc 
        //
        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        command_list->ResourceBarrier(1, &readback_output_barrirer);

        gpu_op::Gemm gemm_ref(to_dml_data_type(params_.dt), get_shape_input_a(), to_dml_tensor_policy(params_.layout_a), params_.transform_a,
            get_shape_input_b(), to_dml_tensor_policy(params_.layout_b), params_.transform_b,
            use_c_tensor(), get_shape_input_c(), to_dml_tensor_policy(use_c_tensor() ? params_.layout_c : DataLayout::eNCHW),
            get_shape_output(), to_dml_tensor_policy(params_.layout_out), params_.alpha, params_.beta, dml_device_, d3d12_device_, true);

        // bind descriptor heap
        auto descriptor_heap = create_descriptor_heap(d3d12_device_, gemm_ref.get_total_descriptor_count());
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        gemm_ref.create_binding_tables(descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        gemm_ref.record_initialize(dml_cmd_recorder_, command_list);
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
        gemm_ref.record_execute(dml_cmd_recorder_, command_list, output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get(), input_buffer_c_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> ref_untyped_result(tensor_out_bytes_width);
        readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(ref_untyped_result.data(), readback_mapped_ptr, ref_untyped_result.size());
        readback_buffer->Unmap(0, nullptr);

        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, ref_untyped_result, 0.001f);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, ref_untyped_result, 0.05f);
        }
        assert(false && "Unsupported output data type!");
        ConformanceResult ret{};
        return ret;

    }

protected:
    inline bool use_c_tensor() const
    {
        return params_.layout_c != DataLayout::eCount;
    }

    TensorShape get_shape_input_a() const
    {
        TensorShape ret{};
        ret.n = params_.B;
        ret.c = params_.C;
        ret.h = params_.M;
        ret.w = params_.K;

        if (params_.transform_a)
        {
            std::swap(ret.h, ret.w);
        }

        return ret;
    }

    TensorShape get_shape_input_b() const
    {
        TensorShape ret{};
        ret.n = params_.B;
        ret.c = params_.C;
        ret.h = params_.K;
        ret.w = params_.N;

        if (params_.transform_b)
        {
            std::swap(ret.h, ret.w);
        }

        return ret;
    }

    TensorShape get_shape_input_c() const
    {
        TensorShape ret{};
        if (!use_c_tensor())
        {
            return ret;
        }
        ret.n = params_.B;
        ret.c = params_.C;
        ret.h = params_.M;
        ret.w = params_.N;
        return ret;
    }

    TensorShape get_shape_output() const
    {
        TensorShape ret{};
        ret.n = params_.B;
        ret.c = params_.C;
        ret.h = params_.M;
        ret.w = params_.N;

        return ret;
    }

protected:
    create_params_t params_;
    ID3D12Device* d3d12_device_;
    IDMLDevice* dml_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<std::byte> input_data_a_;
    std::vector<std::byte> input_data_b_;
    std::vector<std::byte> input_data_c_;

    ComPtr<ID3D12Resource> input_buffer_a_;
    ComPtr<ID3D12Resource> input_buffer_b_;
    ComPtr<ID3D12Resource> input_buffer_c_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class GemmDmlDispatcher : public GemmBaseDispatcher
{
public:
    GemmDmlDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : GemmBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , gemm_(to_dml_data_type(params_.dt), get_shape_input_a(), to_dml_tensor_policy(params_.layout_a), params_.transform_a,
            get_shape_input_b(), to_dml_tensor_policy(params_.layout_b), params_.transform_b,
            use_c_tensor(), get_shape_input_c(), to_dml_tensor_policy(use_c_tensor() ? params_.layout_c : DataLayout::eNCHW),
            get_shape_output(), to_dml_tensor_policy(params_.layout_out), params_.alpha, params_.beta, dml_device, d3d12_device, false)
    {

    }


    std::uint32_t get_total_descriptor_count() override
    {
        return gemm_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        gemm_.create_binding_tables(cpu_handle, gpu_handle);
        gemm_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        gemm_.record_execute(dml_cmd_recorder_, cmd_list,
            output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get(), input_buffer_c_.Get());
    }

private:
    gpu_op::Gemm gemm_;
};

class GemmCmDispatcher : public GemmBaseDispatcher
{
public:
    struct cm_params_t
    {
        bool dump_asm;
        bool large_grf;
        bool print_reg_usage;
        std::array<std::uint32_t, 3> lws{ 1u, 1u, 1u };

        inline static void add_cli_options(CLI::App* opts, cm_params_t& params)
        {
            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
        }
    };

public:
    GemmCmDispatcher(create_params_t&& params, cm_params_t&& cm_params, IntelExtension& intc_ext, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : GemmBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , intc_ext_(intc_ext)
        , cm_params_(std::move(cm_params))
    {
        //validate
        assert(params_.dt == DataType::eFp16);

        {
            std::vector< DescType> desc_list =
            {
                DescType::eSrv, // input a
                DescType::eSrv, // input b
                DescType::eUav // output
            };
            if (use_c_tensor())
            {
                desc_list.push_back(DescType::eSrv);
            }
            root_signature_ = create_root_signature(d3d12_device, desc_list);
        }


        // kernel jits
        std::string build_options = "";
        const std::string pre_jit = "-D";
        const std::string post_jit = " ";
        const std::string between_name_and_value = "=";

        auto add_define = [&](const std::string& name, auto value) {
            using namespace std;
            std::string value_str;
            if (std::is_floating_point<decltype(value)>::value)
            {// to_*string precision is not enough to ensure good match betweeen GPU and CPU or pytorch execution results:
                value_str = (std::stringstream() << std::setiosflags(std::ios_base::showpoint | std::ios_base::fixed) << std::setprecision((std::numeric_limits<decltype(value)>::max_digits10 + 1)) << value).str();
            }
            else
            { // fine for other types:
                value_str = to_string(value);
            }

            build_options += pre_jit + name + between_name_and_value + value_str + post_jit;
        };

        add_define("SIZE_B", params_.B);
        add_define("SIZE_C", params_.C);
        add_define("SIZE_M", params_.M);
        add_define("SIZE_K", params_.K);
        add_define("SIZE_N", params_.N);

        add_define("USE_C", use_c_tensor());

        add_define("DT", "half");



        // kernel compilation
        const auto dump_asm_str = cm_params_.dump_asm ? " -mdump_asm" : "";
        const auto large_grf_str = cm_params_.large_grf ? " -Qxcm_doubleGRF" : "";
        const auto print_reg_str = cm_params_.print_reg_usage ? " -mCM_printregusage" : "";
        const auto lws_x = " -DLWS_SIZE_X=" + std::to_string(cm_params_.lws[0]);
        const auto lws_y = " -DLWS_SIZE_Y=" + std::to_string(cm_params_.lws[1]);
        const auto lws_z = " -DLWS_SIZE_Z=" + std::to_string(cm_params_.lws[2]);
        const auto build_options_final = " -I \" \" " + build_options + dump_asm_str + large_grf_str + print_reg_str + lws_x + lws_y + lws_z;

        if (cm_params_.dump_asm)
        {
            std::cout << build_options_final << std::endl;
        }

        auto kernel_source_content = []()
        {
            const auto path = "gemm_nchw_fp16.cpp";
            std::fstream file(path);
            if (!file.is_open())
            {
                const auto msg = std::format("Kernel file cant be opened:{} \n.", path);
                throw std::runtime_error(msg);
            }
            return std::string((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
        }();

        CD3DX12_SHADER_BYTECODE byte_code;
        byte_code.pShaderBytecode = kernel_source_content.data();
        byte_code.BytecodeLength = kernel_source_content.size();
        pso_ = intc_ext_.create_pipeline(byte_code, build_options_final, root_signature_.Get());

    }


    std::uint32_t get_total_descriptor_count() override
    {
        // input_a, input_b, output
        std::uint32_t descriptor_count = 3;
        if (use_c_tensor())
        {
            descriptor_count++;
        }
        return descriptor_count;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {

        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ DescType::eSrv, input_buffer_a_.Get() });
        resources_list.push_back({ DescType::eSrv, input_buffer_b_.Get() });
        resources_list.push_back({ DescType::eUav, output_buffer_.Get() });
        if (use_c_tensor())
        {
            resources_list.push_back({ DescType::eSrv, input_buffer_c_.Get() });
        }

        gpu_handles_ = create_resource_views_and_handles(d3d12_device_, resources_list, cpu_handle, gpu_handle);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        cmd_list->SetComputeRootSignature(root_signature_.Get());
        cmd_list->SetPipelineState(pso_.Get());

        uint32_t root_index = 1; // start with 1, beacuse Cross compiler CM driver path needs that
        for (uint32_t i = 0; i < gpu_handles_.size(); i++)
        {
            const auto gpu_heap_handle = gpu_handles_[i];
            cmd_list->SetComputeRootDescriptorTable(root_index++, gpu_heap_handle);
        }

        const auto gws_x = 1;
        const auto gws_y = 1;
        const auto gws_z = 1;

        assert(gws_x % cm_params_.lws[0] == 0);
        assert(gws_y % cm_params_.lws[1] == 0);
        assert(gws_z % cm_params_.lws[2] == 0);

        const auto thg_x = gws_x / cm_params_.lws[0];
        const auto thg_y = gws_y / cm_params_.lws[1];
        const auto thg_z = gws_z / cm_params_.lws[2];
        cmd_list->Dispatch(thg_x, thg_y, thg_z);
    }

private:
    cm_params_t cm_params_;
    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;
};