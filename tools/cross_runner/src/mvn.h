#pragma once
#include <vector>
#include <random>
#include "dml_base_node.h"

namespace gpu_op
{
class Mvn : public DirectMlBaseNode
{
public:
    Mvn(const TensorShape& shape, const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy,
        bool no_scale, bool no_bias, float epsilon,
        IDMLDevice* dml_device, ID3D12Device* d3d12_device, bool disable_mc = false)
        : DirectMlBaseNode(dml_device, d3d12_device)
    {
        const dml::TensorDimensions input_dims{ shape.n, shape.c, shape.h, shape.w };


        tensor_input_desc_.DataType = data_type;
        tensor_input_desc_.Flags = DML_TENSOR_FLAG_NONE;
        tensor_input_desc_.DimensionCount = static_cast<std::uint32_t>(input_dims.size());
        tensor_input_desc_.Sizes = input_dims.data();
        const auto tensor_properites = tensor_policy.Get(tensor_input_desc_.DataType, tensor_input_desc_.Flags, input_dims);
        tensor_input_desc_.Strides = tensor_properites.strides.has_value() ? tensor_properites.strides->data() : nullptr;
        tensor_input_desc_.TotalTensorSizeInBytes = tensor_properites.totalTensorSizeInBytes;

        // output tensor desc equals input tensor desc
        tensor_output_desc_ = tensor_input_desc_;

        DML_TENSOR_DESC inp_desc{};
        inp_desc.Desc = &tensor_input_desc_;
        inp_desc.Type = DML_TENSOR_TYPE_BUFFER;

        const dml::TensorDimensions scale_bias_dims{ 1, shape.c, 1, 1};
        DML_TENSOR_DESC scale_desc{};
        dml::TensorProperties scale_tensor_properites{};
        if (!no_scale)
        {
            tensor_scale_desc_ = DML_BUFFER_TENSOR_DESC{};
            tensor_scale_desc_->DataType = data_type;
            tensor_scale_desc_->Flags = DML_TENSOR_FLAG_NONE;
            tensor_scale_desc_->DimensionCount = static_cast<std::uint32_t>(scale_bias_dims.size());
            tensor_scale_desc_->Sizes = scale_bias_dims.data();

            scale_tensor_properites = tensor_policy.Get(tensor_scale_desc_->DataType, tensor_scale_desc_->Flags, scale_bias_dims);
            tensor_scale_desc_->Strides = scale_tensor_properites.strides.has_value() ? scale_tensor_properites.strides->data() : nullptr;
            tensor_scale_desc_->TotalTensorSizeInBytes = scale_tensor_properites.totalTensorSizeInBytes;

            scale_desc.Desc = &tensor_scale_desc_;
            scale_desc.Type = DML_TENSOR_TYPE_BUFFER;
        }

        DML_TENSOR_DESC bias_desc{};
        dml::TensorProperties bias_tensor_properites{};
        if (!no_bias)
        {
            tensor_bias_desc_ = DML_BUFFER_TENSOR_DESC{};
            tensor_bias_desc_->DataType = data_type;
            tensor_bias_desc_->Flags = DML_TENSOR_FLAG_NONE;
            tensor_bias_desc_->DimensionCount = static_cast<std::uint32_t>(scale_bias_dims.size());
            tensor_bias_desc_->Sizes = scale_bias_dims.data();

            bias_tensor_properites = tensor_policy.Get(tensor_bias_desc_->DataType, tensor_bias_desc_->Flags, scale_bias_dims);
            tensor_bias_desc_->Strides = bias_tensor_properites.strides.has_value() ? bias_tensor_properites.strides->data() : nullptr;
            tensor_bias_desc_->TotalTensorSizeInBytes = bias_tensor_properites.totalTensorSizeInBytes;

            bias_desc.Desc = &tensor_bias_desc_;
            bias_desc.Type = DML_TENSOR_TYPE_BUFFER;
        }

        DML_TENSOR_DESC out_desc{};
        out_desc.Desc = &tensor_input_desc_;
        out_desc.Type = DML_TENSOR_TYPE_BUFFER;

        DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC mvn_operators_desc{};
        mvn_operators_desc.InputTensor = &inp_desc;
        mvn_operators_desc.OutputTensor = &out_desc;
        mvn_operators_desc.CrossChannel = 0;
        mvn_operators_desc.NormalizeVariance = 1;
        mvn_operators_desc.ScaleTensor = no_scale ? nullptr : &scale_desc;
        mvn_operators_desc.BiasTensor = no_bias ? nullptr : &bias_desc;
        mvn_operators_desc.Epsilon = epsilon;

        DML_OPERATOR_DESC dml_operator_desc{};
        dml_operator_desc.Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION;
        dml_operator_desc.Desc = &mvn_operators_desc;

        throw_if_failed(dml_device->CreateOperator(
            &dml_operator_desc, IID_PPV_ARGS(dml_operator_.ReleaseAndGetAddressOf())), "create softmax operator");

        auto exec_flags = DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE | DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
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

    dml::TensorDesc get_tensor_input_desc() const
    {
        return tensor_input_desc_;
    }

    dml::TensorDesc get_tensor_scale_desc() const
    {
        return tensor_scale_desc_.value_or(DML_BUFFER_TENSOR_DESC{});
    }

    dml::TensorDesc get_tensor_bias_desc() const
    {
        return tensor_bias_desc_.value_or(DML_BUFFER_TENSOR_DESC{});
    }

    dml::TensorDesc get_tensor_out_desc() const
    {
        return tensor_output_desc_;
    }

    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list,
        ID3D12Resource* resource_out, ID3D12Resource* resource_input, ID3D12Resource* resource_scale, ID3D12Resource* resource_bias)
    {
        assert(resource_input);
        DML_BUFFER_BINDING input_buffer_binding{ resource_input, 0, resource_input->GetDesc().Width };
        DML_BUFFER_BINDING scale_buffer_binding{};
        DML_BUFFER_BINDING bias_buffer_binding{};

        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(3);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_buffer_binding });
        if (tensor_scale_desc_)
        {
            assert(resource_scale);
            scale_buffer_binding = { resource_scale, 0, resource_scale->GetDesc().Width };
        }
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &scale_buffer_binding });

        if (tensor_bias_desc_)
        {
            assert(resource_bias);
            bias_buffer_binding = { resource_bias, 0, resource_bias->GetDesc().Width };
        }
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &bias_buffer_binding });

        assert(resource_out);
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_binding_desc);
    }

private:
    DML_BUFFER_TENSOR_DESC tensor_input_desc_{};
    DML_BUFFER_TENSOR_DESC tensor_output_desc_{};
    std::optional<DML_BUFFER_TENSOR_DESC> tensor_scale_desc_{};
    std::optional<DML_BUFFER_TENSOR_DESC> tensor_bias_desc_{};
    ComPtr<IDMLOperator> dml_operator_;
};

}  // namespace gpu_op

namespace cpu_op
{

std::vector<std::byte> mvn(const TensorShape& in_out_shape, DataLayout in_out_layout, DataType in_out_datatype,
    const std::byte* input_data, const std::byte* scale_data, const std::byte* bias_data, const float epsilon);
}  // namespace cpu_op


class MvnBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout;
        TensorShape shape;
        bool no_scale = false;
        bool no_bias = false;
        float epsilon = 0.00005f;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();
            opts->add_option("--shape", params.shape, "shape: <n,c,h,w>")->required();
            opts->add_flag("--no_scale", params.no_scale);
            opts->add_flag("--no_bias", params.no_bias);
        }
    };
public:
    MvnBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , d3d12_device_(d3d12_device)
        , dml_device_(dml_device)
        , input_data_(params_.shape.get_elements_count() * get_data_type_bytes_width(params_.dt))
    {
        if (use_bias())
        {
            bias_data_.resize(params_.shape.c * get_data_type_bytes_width(params_.dt));
        }
        if (use_scale())
        {
            scale_data_.resize(params_.shape.c * get_data_type_bytes_width(params_.dt));
        }

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(0.0f, 5.0f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_);
            if (use_bias())
            {
                randomize_linear_container_float(random_generator, uniform_distribution, bias_data_);
            }
            if (use_scale())
            {
                randomize_linear_container_float(random_generator, uniform_distribution, scale_data_);
            }
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_);
            if (use_bias())
            {
                randomize_linear_container_half(random_generator, uniform_distribution, bias_data_);
                //fill_with_constant_linear_container_half(bias_data_, DirectX::PackedVector::XMConvertFloatToHalf(0.0f));
            }
            if (use_scale())
            {
                randomize_linear_container_half(random_generator, uniform_distribution, scale_data_);
                //fill_with_constant_linear_container_half(scale_data_, DirectX::PackedVector::XMConvertFloatToHalf(1.0f));
            }
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        const auto tensor_input_bytes_width = input_data_.size();
        const auto tensor_bias_bytes_width = bias_data_.size();
        const auto tensor_scale_bytes_width = scale_data_.size();
        const auto tensor_out_bytes_width = tensor_input_bytes_width;

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_bytes_width + tensor_bias_bytes_width + tensor_scale_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_input_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (use_bias())
        {
            bias_buffer_ = create_buffer(d3d12_device, tensor_bias_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        if (use_scale())
        {
            scale_buffer_ = create_buffer(d3d12_device, tensor_scale_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);


        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        if (use_bias())
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, bias_data_.data(), tensor_bias_bytes_width);
            memcopy_offset += tensor_bias_bytes_width;
        }
        if (use_scale())
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, scale_data_.data(), tensor_scale_bytes_width);
            memcopy_offset += tensor_scale_bytes_width;
        }
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        if (use_bias())
        {
            cmd_list->CopyBufferRegion(bias_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_bias_bytes_width);
            memcopy_offset += tensor_bias_bytes_width;
        }
        if (use_scale())
        {
            cmd_list->CopyBufferRegion(scale_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_scale_bytes_width);
            memcopy_offset += tensor_scale_bytes_width;
        }

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (use_bias())
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(bias_buffer_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        if (use_scale())
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(scale_buffer_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }


    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
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
        //  calc reference with dml non-mc mvn
        //
        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        command_list->ResourceBarrier(1, &readback_output_barrirer);

        gpu_op::Mvn mvn_ref(params_.shape, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout),
            params_.no_scale, params_.no_bias, params_.epsilon, dml_device_, d3d12_device_, true /*disable mc for ref calc*/);
        // bind descriptor heap
        auto descriptor_heap = create_descriptor_heap(d3d12_device_, mvn_ref.get_total_descriptor_count());
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        mvn_ref.create_binding_tables(descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        mvn_ref.record_initialize(dml_cmd_recorder_, command_list);
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
        mvn_ref.record_execute(dml_cmd_recorder_, command_list,
            output_buffer_.Get(), input_buffer_.Get(), scale_buffer_.Get(), bias_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> dnnl_untyped_result(tensor_out_bytes_width);
        readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(dnnl_untyped_result.data(), readback_mapped_ptr, dnnl_untyped_result.size());
        readback_buffer->Unmap(0, nullptr);

        // dnnl seems to be broken, use mvn non-mc path
        //const auto dnnl_untyped_result = cpu_op::mvn(params_.shape, params_.layout, params_.dt, input_data_.data(), scale_data_.data(), bias_data_.data(), params_.epsilon);

        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, dnnl_untyped_result, 0.001f);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, dnnl_untyped_result, 0.05f);
        }
        assert(false && "Unsupported output data type!");
        ConformanceResult ret{};
        return ret;
    }

protected:
    inline bool use_bias() const
    {
        return !params_.no_bias;
    }

    inline bool use_scale() const
    {
        return !params_.no_scale;
    }

protected:
    create_params_t params_;
    ID3D12Device* d3d12_device_;
    IDMLDevice* dml_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<std::byte> input_data_;
    std::vector<std::byte> bias_data_;
    std::vector<std::byte> scale_data_;

    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> scale_buffer_;
    ComPtr<ID3D12Resource> bias_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class MvnDmlDispatcher : public MvnBaseDispatcher
{
public:
    MvnDmlDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : MvnBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , mvn_(params_.shape, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout),
            params_.no_scale, params_.no_bias, params_.epsilon, dml_device, d3d12_device)
    {
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return mvn_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        mvn_.create_binding_tables(cpu_handle, gpu_handle);
        mvn_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        mvn_.record_execute(dml_cmd_recorder_, cmd_list,
            output_buffer_.Get(), input_buffer_.Get(), scale_buffer_.Get(), bias_buffer_.Get());
    }
private:
    gpu_op::Mvn mvn_;
};

class MvnCmDispatcher : public MvnBaseDispatcher
{
public:
    struct mvn_cm_params_t
    {
        bool dump_asm;
        bool large_grf;
        bool print_reg_usage;
        std::array<std::uint32_t, 3> lws{ 1u, 1u, 1u };
        const std::uint32_t items_per_hw_th = 128;

        inline static void add_cli_options(CLI::App* opts, mvn_cm_params_t& params)
        {
            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
        }
    };
public:
    MvnCmDispatcher(create_params_t&& params, mvn_cm_params_t&& cm_params, IntelExtension& intc_ext, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : MvnBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , intc_ext_(intc_ext)
        , cm_params_(std::move(cm_params))
    {
        const auto dataset_size = params_.shape.h * params_.shape.w;
        const auto dataset_groups = dataset_size / cm_params_.items_per_hw_th;
        cm_params_.lws[2] = dataset_groups;

        // root signature
        {
            const auto bindings_size = get_total_descriptor_count();
            std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;
            std::vector<CD3DX12_ROOT_PARAMETER1> root_params;
            ranges.reserve(bindings_size);
            root_params.reserve(bindings_size + 1); // + 1 beacuse of the CM driver path

            std::uint32_t srv_range_reg = 0;
            std::uint32_t uav_range_reg = 0;
            std::uint32_t cbv_range_reg = 0;

            {
                // driver thing
                CD3DX12_ROOT_PARAMETER1 rp{};
                rp.InitAsConstants(1, cbv_range_reg++);
                root_params.push_back(rp);
            }
            enum class DescType
            {
                eSrv,
                eUav
            };
            auto add_desc_table = [&](DescType type)
            {
                if (type == DescType::eSrv)
                {
                    ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, srv_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE });
                }
                else
                {
                    ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, uav_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE });
                }
                CD3DX12_ROOT_PARAMETER1 rp{};
                rp.InitAsDescriptorTable(1u, &ranges.back());
                root_params.push_back(rp);
            };

            // input
            add_desc_table(DescType::eSrv);
            // output uav
            add_desc_table(DescType::eUav);
            if (use_bias())
            {
                add_desc_table(DescType::eSrv);
            }
            if (use_scale())
            {
                add_desc_table(DescType::eSrv);
            }

            if (root_params.size() == 0)
            {
                throw std::runtime_error("Something gone wrong. Why kernel has 0 root params?");
            }

            CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC compute_root_signature_desc;
            compute_root_signature_desc.Init_1_1(static_cast<UINT>(root_params.size()), root_params.data(), 0, nullptr);

            ComPtr<ID3DBlob> signature;
            ComPtr<ID3DBlob> error;
            throw_if_failed(D3DX12SerializeVersionedRootSignature(
                &compute_root_signature_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_1,
                &signature,
                &error), "D3DX12SerializeVersionedRootSignature failed.");

            if (error)
            {
                throw_with_msg("Failed to create root signature, error:" + std::string((LPCSTR)error->GetBufferPointer()));
            }
            throw_if_failed(d3d12_device_->CreateRootSignature(
                0,
                signature->GetBufferPointer(),
                signature->GetBufferSize(),
                IID_PPV_ARGS(&root_signature_)), "CreateRootSignature(...) failed.");
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

        add_define("INOUT_WIDTH", params_.shape.w);
        add_define("INOUT_HEIGHT", params_.shape.h);
        add_define("INOUT_CHANNELS", params_.shape.c);
        add_define("INOUT_BATCH", params_.shape.n);

        add_define("USE_BIAS", use_bias());
        add_define("USE_SCALE", use_scale());
        add_define("EPSILON", params_.epsilon);
        add_define("ITEMNUM", cm_params_.items_per_hw_th);

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
            const auto path = "mvn_nchw.cpp";
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
        pso_ = intc_ext_.create_pipeline(byte_code, build_options_final, root_signature_.Get(), INTC_D3D12_SHADER_INPUT_TYPE::CM);
    }

    std::uint32_t get_total_descriptor_count() override
    {
        // input, output
        std::uint32_t descriptor_count = 3;
        if (use_bias())
        {
            descriptor_count++;
        }
        if (use_scale())
        {
            descriptor_count++;
        }
        return descriptor_count;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        const auto desc_heap_incrs_size = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        const auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
        const auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

        enum class ViewType
        {
            eSrv = 0,
            eUav = 1
        };

        std::vector<std::pair<ViewType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ ViewType::eSrv, input_buffer_.Get() });
        resources_list.push_back({ ViewType::eUav, output_buffer_.Get() });
        if (use_bias())
        {
            resources_list.push_back({ ViewType::eSrv, bias_buffer_.Get() });
        }
        if (use_scale())
        {
            resources_list.push_back({ ViewType::eSrv, scale_buffer_.Get() });
        }

        // reserve handles
        gpu_handles_.reserve(get_total_descriptor_count());
        // create handles

        for (std::size_t i = 0; i < resources_list.size(); i++)
        {
            auto cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(base_cpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size);
            gpu_handles_.push_back(CD3DX12_GPU_DESCRIPTOR_HANDLE(base_gpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size));

            auto& resource_view_type = resources_list[i].first;
            auto& resource = resources_list[i].second;
            const auto res_desc = resource->GetDesc();
            assert(res_desc.Dimension == D3D12_RESOURCE_DIMENSION::D3D12_RESOURCE_DIMENSION_BUFFER);

            if (resource_view_type == ViewType::eSrv)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                desc.Format = DXGI_FORMAT_R8_UINT;
                desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

                desc.Buffer.StructureByteStride = 0;
                desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
                desc.Buffer.FirstElement = 0;
                desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

                d3d12_device_->CreateShaderResourceView(resource, &desc, cpu_handle);
            }
            else
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
                desc.Format = DXGI_FORMAT_R8_UINT;
                desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;

                desc.Buffer.StructureByteStride = 0;
                desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
                desc.Buffer.FirstElement = 0;
                desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

                d3d12_device_->CreateUnorderedAccessView(resource, nullptr, &desc, cpu_handle);
            }
        }
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        cmd_list->SetComputeRootSignature(root_signature_.Get());
        cmd_list->SetPipelineState(pso_.Get());

        uint32_t root_index = 1; // start with 1, beacuse Cross compiler CM driver path needs that
        for (uint32_t i = 0; i < gpu_handles_.size(); i++)
        {
            const auto gpu_heap_handle = gpu_handles_[i];
            cmd_list->SetComputeRootDescriptorTable(root_index++, gpu_heap_handle);
        }

        const auto dataset_size = params_.shape.w * params_.shape.h;

        const auto gws_x = params_.shape.n;
        const auto gws_y = params_.shape.c;
        const auto gws_z = dataset_size / cm_params_.items_per_hw_th;

        assert(gws_x % cm_params_.lws[0] == 0);
        assert(gws_y % cm_params_.lws[1] == 0);
        assert(gws_z % cm_params_.lws[2] == 0);

        const auto thg_x = gws_x / cm_params_.lws[0];
        const auto thg_y = gws_y / cm_params_.lws[1];
        const auto thg_z = gws_z / cm_params_.lws[2];
        cmd_list->Dispatch(thg_x, thg_y, thg_z);
    }

private:
    mvn_cm_params_t cm_params_;
    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;
};