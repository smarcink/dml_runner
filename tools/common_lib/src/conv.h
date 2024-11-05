#pragma once
#include <vector>
#include <random>
#include "dml_base_node.h"

#include "dnnl_utils.h"

#include "iumd_d3d12_impl.h"
#include <dnnl_iumd.h>
#include <dnnl.hpp>
#include <dnnl_iumd.hpp>
#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl_conv_op
{
struct bindings_t
{
    dnnl_utils::binding_t input;
    dnnl_utils::binding_t filter;
    dnnl_utils::binding_t bias;
};

struct opts_t
{
    std::uint32_t inp_pad;
    std::uint32_t out_pad;
    TensorShape stride;
    TensorShape dilates;
    TensorShape output_shape;
    DataType out_dt = DataType::eCount;
    bool use_fp32_accu = false;
    ActivationSettings activation;
    DataLayout out_layout = DataLayout::eCount;
    std::uint32_t groups = 1u;
    bool force_winograd = false;
    bool dump_weights = false;
    bool dump_scratchpad = false;
    bool cache_blob = false;

    std::size_t execution_iterations = 1ul; // set it to bigger value to run more iterations
};
std::vector<std::byte> convolution(const bindings_t& bindings, opts_t opts);
std::vector<std::byte> deconvolution(const bindings_t& bindings, opts_t opts);
}


namespace gpu_op
{
class Convolution : public DirectMlBaseNode
{
public:
    Convolution(const TensorShape& input_shape, const TensorShape& filter_shape, const TensorShape& output_shape,
        const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& input_tensor_policy, const dml::TensorPolicy& filter_tensor_policy, const dml::TensorPolicy& output_tensor_policy,
        const TensorShape& stride_shape, const TensorShape& dilation_shape, std::uint32_t input_pad, std::uint32_t output_pad, std::uint32_t group_count,
        bool use_bias, bool allow_fp16_computations, bool transposed, const ActivationSettings& activation_settings, bool managed_weights,
        IDMLDevice* dml_device, ID3D12Device* d3d12_device, bool disable_mc = false, bool allow_descriptors_volatile = true)
        : DirectMlBaseNode(dml_device, d3d12_device)
    {

        const dml::TensorDimensions input_dims{ input_shape.n, input_shape.c, input_shape.h, input_shape.w };
        const dml::TensorDimensions filter_dims{ filter_shape.n, filter_shape.c / group_count, filter_shape.h, filter_shape.w };
        const dml::TensorDimensions output_dims{ output_shape.n, output_shape.c, output_shape.h, output_shape.w };

        const dml::TensorDimensions bias_dims{ 1, output_shape.c, 1, 1 };

        const std::array<std::uint32_t, 2> strides = { stride_shape.h, stride_shape.w };
        const std::vector<std::uint32_t> dilations = { 1u + dilation_shape.h, 1u + dilation_shape.w }; // for some reason DML default is 1
        const std::vector<std::uint32_t> start_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> end_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> out_pad = { output_pad, output_pad };

        dml::TensorProperties input_tensor_properites{};
        {
            tensor_input_desc_.DataType = data_type;
            tensor_input_desc_.Flags = DML_TENSOR_FLAG_NONE;
            tensor_input_desc_.DimensionCount = static_cast<std::uint32_t>(input_dims.size());
            tensor_input_desc_.Sizes = input_dims.data();

            input_tensor_properites = input_tensor_policy.Get(tensor_input_desc_.DataType, tensor_input_desc_.Flags, input_dims);
            tensor_input_desc_.Strides = input_tensor_properites.strides.has_value() ? input_tensor_properites.strides->data() : nullptr;
            tensor_input_desc_.TotalTensorSizeInBytes = input_tensor_properites.totalTensorSizeInBytes;
            tensor_input_desc_.GuaranteedBaseOffsetAlignment = input_tensor_properites.guaranteedBaseOffsetAlignment;
        }

        dml::TensorProperties filter_tensor_properites{};
        {
            tensor_filter_desc_.DataType = data_type;
            tensor_filter_desc_.Flags = managed_weights ? DML_TENSOR_FLAG_OWNED_BY_DML :  DML_TENSOR_FLAG_NONE;
            tensor_filter_desc_.DimensionCount = static_cast<std::uint32_t>(filter_dims.size());
            tensor_filter_desc_.Sizes = filter_dims.data();

            filter_tensor_properites = filter_tensor_policy.Get(tensor_filter_desc_.DataType, tensor_filter_desc_.Flags, filter_dims);
            tensor_filter_desc_.Strides = filter_tensor_properites.strides.has_value() ? filter_tensor_properites.strides->data() : nullptr;;
            tensor_filter_desc_.TotalTensorSizeInBytes = filter_tensor_properites.totalTensorSizeInBytes;
            tensor_filter_desc_.GuaranteedBaseOffsetAlignment = filter_tensor_properites.guaranteedBaseOffsetAlignment;
        }

        dml::TensorProperties bias_tensor_properites;
        if(use_bias)
        {
            DML_BUFFER_TENSOR_DESC bias_desc{};
            bias_desc.DataType = data_type;
            bias_desc.Flags = managed_weights ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE;
            bias_desc.DimensionCount = static_cast<std::uint32_t>(bias_dims.size());
            bias_desc.Sizes = bias_dims.data();
            bias_tensor_properites = filter_tensor_policy.Get(bias_desc.DataType, bias_desc.Flags, bias_dims);
            bias_desc.TotalTensorSizeInBytes = bias_tensor_properites.totalTensorSizeInBytes;
            bias_desc.GuaranteedBaseOffsetAlignment = bias_tensor_properites.guaranteedBaseOffsetAlignment;
            tensor_bias_desc_.emplace(std::move(bias_desc));
        }

        dml::TensorProperties output_tensor_properites;
        {
            tensor_out_desc_.DataType = data_type;
            tensor_out_desc_.Flags = DML_TENSOR_FLAG_NONE;
            tensor_out_desc_.DimensionCount = static_cast<std::uint32_t>(output_dims.size());
            tensor_out_desc_.Sizes = output_dims.data();

            output_tensor_properites = output_tensor_policy.Get(tensor_out_desc_.DataType, tensor_out_desc_.Flags, output_dims);
            tensor_out_desc_.Strides = output_tensor_properites.strides.has_value() ? output_tensor_properites.strides->data() : nullptr;
            tensor_out_desc_.TotalTensorSizeInBytes = output_tensor_properites.totalTensorSizeInBytes;
            tensor_out_desc_.GuaranteedBaseOffsetAlignment = output_tensor_properites.guaranteedBaseOffsetAlignment;
        }

        DML_TENSOR_DESC input_desc{};
        input_desc.Desc = &tensor_input_desc_;
        input_desc.Type = DML_TENSOR_TYPE_BUFFER;
        
        DML_TENSOR_DESC filter_desc{};
        filter_desc.Desc = &tensor_filter_desc_;
        filter_desc.Type = DML_TENSOR_TYPE_BUFFER;

        DML_TENSOR_DESC bias_desc{};
        if (use_bias)
        {
            bias_desc.Desc = &tensor_bias_desc_;
            bias_desc.Type = DML_TENSOR_TYPE_BUFFER;
        }

        DML_TENSOR_DESC output_desc{};
        output_desc.Desc = &tensor_out_desc_;
        output_desc.Type = DML_TENSOR_TYPE_BUFFER;

        DML_CONVOLUTION_OPERATOR_DESC desc = {};
        desc.InputTensor = &input_desc;
        desc.FilterTensor = &filter_desc;
        desc.BiasTensor = use_bias ? &bias_desc : nullptr;
        desc.OutputTensor = &output_desc;
        desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
        desc.Direction = transposed ? DML_CONVOLUTION_DIRECTION_BACKWARD : DML_CONVOLUTION_DIRECTION_FORWARD;
        desc.DimensionCount = 2;
        desc.Strides = strides.data();
        desc.Dilations = dilations.data();
        desc.StartPadding = start_pad.data();
        desc.EndPadding = end_pad.data();
        desc.OutputPadding = out_pad.data();
        desc.GroupCount = group_count;
        const auto activation = to_dml_activation_setting(activation_settings);
        desc.FusedActivation = activation.desc.Type != DML_OPERATOR_INVALID ? &activation.desc : nullptr;


        DML_OPERATOR_DESC dml_operator_desc{};
        dml_operator_desc.Type = DML_OPERATOR_CONVOLUTION;
        dml_operator_desc.Desc = &desc;


        throw_if_failed(dml_device->CreateOperator(
            &dml_operator_desc, IID_PPV_ARGS(dml_operator_.ReleaseAndGetAddressOf())), "create convolution operator");

        DML_EXECUTION_FLAGS exec_flags = DML_EXECUTION_FLAG_NONE;
        
        if (allow_descriptors_volatile)
        {
            exec_flags |= DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        }
        if (allow_fp16_computations)
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

    dml::TensorDesc get_tensor_input_desc() const
    {
        return tensor_input_desc_;
    }

    dml::TensorDesc get_tensor_filter_desc() const
    {
        return tensor_filter_desc_;
    }

    dml::TensorDesc get_tensor_bias_desc() const
    {
        if(tensor_bias_desc_.has_value())
        {
            return tensor_bias_desc_.value();
        }
        return {};
    }

    dml::TensorDesc get_tensor_out_desc() const
    {
        return tensor_out_desc_;
    }

    void record_initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, ID3D12Resource* resource_filter, ID3D12Resource* resource_bias)
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

        std::vector<DML_BUFFER_BINDING> input_binds{};
        input_binds.push_back({ nullptr, 0, 0 });  // input
        if (tensor_filter_desc_.Flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_binds.push_back({ resource_filter, 0, resource_filter->GetDesc().Width });
        }
        else
        {
            input_binds.push_back({ nullptr, 0, 0 });
        }

        if (tensor_bias_desc_.has_value() && tensor_bias_desc_->Flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_binds.push_back({ resource_bias, 0, resource_bias->GetDesc().Width });
        }
        else
        {
            input_binds.push_back({ nullptr, 0, 0 });
        }

        DML_BUFFER_ARRAY_BINDING input_bind{};
        input_bind.BindingCount = static_cast<UINT>(input_binds.size());
        input_bind.Bindings = input_binds.data();
        if (!input_binds.empty())
        {
            DML_BINDING_DESC binding{};
            binding.Type = DML_BINDING_TYPE_BUFFER_ARRAY;
            binding.Desc = &input_bind;
            dml_init_binding_table->BindInputs(1, &binding);
        }

        dml_cmd_recorder->RecordDispatch(
            cmd_list,
            dml_op_initializer_.Get(),
            dml_init_binding_table.Get());
    }

    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, ID3D12Resource* resource_out,
        ID3D12Resource* resource_input, ID3D12Resource* resource_filter, ID3D12Resource* resource_bias, ID3D12Resource* resource_constant)
    {
        assert(((resource_bias != nullptr)== tensor_bias_desc_.has_value()) && "bias resources is not matching what was expected.");

        DML_BUFFER_BINDING input_buffer_binding{ resource_input, 0, resource_input->GetDesc().Width };
        DML_BUFFER_BINDING filter_buffer_binding{};
        DML_BUFFER_BINDING bias_buffer_binding{};

        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(3);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_buffer_binding });
        if (tensor_filter_desc_.Flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_bindings.push_back({ DML_BINDING_TYPE_NONE, nullptr });
        }
        else
        {
            filter_buffer_binding = { resource_filter, 0, resource_filter->GetDesc().Width };
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &filter_buffer_binding });
        }

        if (!resource_bias || (tensor_bias_desc_.has_value() && tensor_bias_desc_->Flags == DML_TENSOR_FLAG_OWNED_BY_DML))
        {
            input_bindings.push_back({ DML_BINDING_TYPE_NONE, nullptr });
        }
        else
        {
            bias_buffer_binding = { resource_bias, 0, resource_bias->GetDesc().Width };
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &bias_buffer_binding });
        }

        std::vector<DML_BINDING_DESC> output_bindings;
        output_bindings.reserve(1);
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };
        output_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &output_buffer_binding });

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_bindings);
    }


private:
    ComPtr<IDMLOperator> dml_operator_;
    DML_BUFFER_TENSOR_DESC tensor_input_desc_;
    DML_BUFFER_TENSOR_DESC tensor_filter_desc_;
    std::optional<DML_BUFFER_TENSOR_DESC> tensor_bias_desc_;
    DML_BUFFER_TENSOR_DESC tensor_out_desc_;
};
}

class ConvolutionBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout input_layout;
        DataLayout output_layout = DataLayout::eNCHW;
        DataLayout filter_layout = DataLayout::eNCHW;
        TensorShape input_shape;
        TensorShape filter_shape;
        std::uint32_t in_pad = 0u;
        std::uint32_t out_pad = 0u;
        std::uint32_t groups = 1u;
        ActivationSettings activation{};
        TensorShape stride;
        TensorShape dilation{};  // DML non-dilation is 1, OneDNN non-dilation is 0 -> so for DML path we force to add 1 to dialtions
        bool no_bias = false;
        bool allow_fp16_computations = false;
        bool managed_weights = false;
        bool algo_winograd = false;
        bool transposed = false;

        bool dump_weights = false;
        bool use_dnnl_for_reference_calculations = false;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--input_layout", params.input_layout)->required();
            add_data_layout_cli_option(opts, "--output_layout", params.output_layout);
            add_data_layout_cli_option(opts, "--filter_layout", params.filter_layout);
            opts->add_option("--input_shape", params.input_shape, "speciify list: <n, ic, h, w")->required();
            opts->add_option("--filter_shape", params.filter_shape, "speciify list: <oc, ic, kh, kw")->required();
            opts->add_option("--in_pad", params.in_pad)->required();
            opts->add_option("--out_pad", params.out_pad)->required();
            opts->add_option("--groups", params.groups)->default_val(1u);
            opts->add_option("--stride", params.stride, "speciify list: <stride_h, stride_w>")->required();
            opts->add_option("--dilation", params.dilation, "speciify list: <dilation_h, dilation_w>");
            opts->add_flag("--no_bias", params.no_bias);
            opts->add_flag("--allow_fp16_computations", params.allow_fp16_computations);
            opts->add_flag("--managed_weights", params.managed_weights);
            opts->add_option("--activation", params.activation);
            opts->add_flag("--algo_winograd", params.algo_winograd);
            opts->add_flag("--transposed", params.transposed);
            opts->add_flag("--dnnl_reference", params.use_dnnl_for_reference_calculations)->default_val(false);

            opts->add_flag("--dump_weights", params.dump_weights);
        }
    };

    ConvolutionBaseDispatcher(const create_params_t& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(params)
        , d3d12_device_(d3d12_device)
        , dml_cmd_recorder_(dml_cmd_recorder)
        , dml_device_(dml_device)
        , input_data_(get_tensor_elements_count(params_.input_shape, params_.input_layout)* (std::uint8_t)get_data_type_bytes_width(params_.dt))
        , filter_data_(get_tensor_elements_count(params_.filter_shape, params_.filter_layout)* (std::uint8_t)get_data_type_bytes_width(params_.dt))

    {
        if (params_.transposed)
        {
            assert(params_.input_shape.c == params_.filter_shape.n);
        }
        else
        {
            assert(params_.input_shape.c == params_.filter_shape.c);
        }
        assert(params_.groups >= 1);

        const auto output_shape = get_output_shape();
        prepare_constant_data();

        if (!params_.no_bias)
        {
            bias_data_ = std::vector<std::byte>(output_shape.c * (std::uint8_t)get_data_type_bytes_width(params_.dt));
        }
        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);
        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_);
            randomize_linear_container_float(random_generator, uniform_distribution, filter_data_);
            if (use_bias())
            {
                randomize_linear_container_float(random_generator, uniform_distribution, bias_data_);
            }
        }
        else if (params_.dt == DataType::eFp16)
        {
            //fill_with_constant_linear_container_half(input_data_, DirectX::PackedVector::XMConvertFloatToHalf(1.0f));
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_);
            //fill_with_constant_linear_container_half(input_data_, DirectX::PackedVector::XMConvertFloatToHalf(1.0f));
            randomize_linear_container_half(random_generator, uniform_distribution, filter_data_);
            //Half* ptr = reinterpret_cast<Half*>(filter_data_.data());
            //for (int i = 0; i < params_.filter_shape.get_elements_count(); i++)
            //{
            //    ptr[i] = DirectX::PackedVector::XMConvertFloatToHalf(float(i));

            //}
            //fill_with_constant_linear_container_half(filter_data_, DirectX::PackedVector::XMConvertFloatToHalf(1.0f));
            if (use_bias())
            {
                randomize_linear_container_half(random_generator, uniform_distribution, bias_data_);
            }
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        const auto tensor_constant_bytes_width = constant_data_.size();
        const auto tensor_input_bytes_width = input_data_.size();
        const auto tensor_filter_bytes_width = filter_data_.size();
        const auto tensor_bias_bytes_width = bias_data_.size();
        const auto tensor_out_bytes_width = get_tensor_elements_count(output_shape, params_.output_layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_bytes_width + tensor_filter_bytes_width + tensor_bias_bytes_width + tensor_constant_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_input_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        filter_buffer_ = create_buffer(d3d12_device, tensor_filter_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (use_bias())
        {
            bias_buffer_ = create_buffer(d3d12_device, tensor_bias_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        constant_buffer_ = create_buffer(d3d12_device, tensor_constant_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        const auto tensor_out_bytes_width_1 = output_buffer_->GetDesc().Width;

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        std::memcpy(upload_mapped_ptr + memcopy_offset, filter_data_.data(), tensor_filter_bytes_width);
        memcopy_offset += tensor_filter_bytes_width;
        if (use_bias())
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, bias_data_.data(), tensor_bias_bytes_width);
            memcopy_offset += tensor_bias_bytes_width;
        }
        std::memcpy(upload_mapped_ptr + memcopy_offset, constant_data_.data(), tensor_constant_bytes_width);
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        cmd_list->CopyBufferRegion(filter_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_filter_bytes_width);
        memcopy_offset += tensor_filter_bytes_width;
        if (use_bias())
        {
            cmd_list->CopyBufferRegion(bias_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_bias_bytes_width);
            memcopy_offset += tensor_bias_bytes_width;
        }
        cmd_list->CopyBufferRegion(constant_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_constant_bytes_width);

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(filter_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (use_bias())
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(bias_buffer_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(constant_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches, std::size_t reference_dispatch_iterations) override
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

        std::vector<std::byte> ref_untyped_result;
        if (params_.use_dnnl_for_reference_calculations)
        {
            ref_untyped_result = get_dnnl_result(reference_dispatch_iterations);
        }
        else
        {
            ref_untyped_result = get_dml_results(tensor_out_bytes_width, command_queue, command_allocator, command_list);
        }


        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, ref_untyped_result, 0.001f, print_mismatches);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, ref_untyped_result, 0.05f, print_mismatches);
        }
        assert(false && "Unsupported output data type!");
        ConformanceResult ret{};
        return ret;
    }

protected:
    virtual bool use_persitent_cache_for_dnnl_reference() const
    {
        return false;
    }

    inline TensorShape get_output_shape() const
    {
        TensorShape ret;
        ret.n = params_.input_shape.n;
        ret.d = 0;

        const auto dkh = 1 + (params_.filter_shape.h - 1) * (params_.dilation.h + 1);
        const auto dkw = 1 + (params_.filter_shape.w - 1) * (params_.dilation.w + 1);
        if (params_.transposed)
        {
            ret.c = params_.filter_shape.c; // input channels (transposed case)
            ret.h = (params_.stride.h * (params_.input_shape.h - 1)) + dkh - params_.in_pad - params_.in_pad;
            ret.w = (params_.stride.w * (params_.input_shape.w - 1)) + dkw - params_.in_pad - params_.in_pad;
        }
        else
        {

            ret.c = params_.filter_shape.n; // output channels
            ret.h = (params_.input_shape.h - dkh + params_.in_pad + params_.in_pad) / params_.stride.h + 1;
            ret.w = (params_.input_shape.w - dkw + params_.in_pad + params_.in_pad) / params_.stride.w + 1;
        }

        return ret;
    }

    void prepare_constant_data()
    {
        constant_data_ = std::vector<std::byte>(100);
        const auto output_shape = get_output_shape();
        using Dt = int;
        auto* ptr = reinterpret_cast<Dt*>(constant_data_.data());
        *ptr++ = static_cast<Dt>(params_.input_shape.h);
        *ptr++ = static_cast<Dt>(params_.input_shape.w);
        *ptr++ = static_cast<Dt>(params_.in_pad);
        *ptr++ = static_cast<Dt>(output_shape.c);
        *ptr++ = static_cast<Dt>(output_shape.h);
        *ptr++ = static_cast<Dt>(output_shape.w);
        *ptr++ = static_cast<Dt>(params_.stride.h);
        *ptr++ = static_cast<Dt>(params_.filter_shape.c);
        *ptr++ = static_cast<Dt>(params_.filter_shape.n);

        //todo fix this
        *ptr++ = static_cast<Dt>(0);
        *ptr++ = static_cast<Dt>(0);
        *ptr++ = static_cast<Dt>(0);
        //*ptr++ = static_cast<Dt>(params_.act_type);
        //*ptr++ = static_cast<Dt>(params_.act_alpha);
        //*ptr++ = static_cast<Dt>(params_.act_beta);

        *ptr++ = static_cast<Dt>(params_.output_layout == DataLayout::eNCHW ? 0 : 1);
        *ptr++ = static_cast<Dt>(params_.input_layout == DataLayout::eNCHW ? 0 : 1);
        *ptr++ = static_cast<Dt>(params_.input_shape.c);
        *ptr++ = static_cast<Dt>(params_.filter_layout == DataLayout::eNCHW ? 0 : 1);
    }
    inline bool use_bias() const
    {
        return !params_.no_bias;
    }

    std::vector<std::byte> get_dnnl_result(std::size_t reference_dispatch_iterations) const
    {
        const auto output_shape = get_output_shape();

        dnnl_conv_op::bindings_t bindings{};
        {
            bindings.input.data = input_data_.data();
            bindings.input.dt = params_.dt;
            bindings.input.layout = params_.input_layout;
            bindings.input.shape = params_.input_shape;
        }

        {
            bindings.filter.data = filter_data_.data();
            bindings.filter.dt = params_.dt;
            bindings.filter.layout = params_.filter_layout;
            bindings.filter.shape = params_.filter_shape;
        }
        if (use_bias())
        {
            bindings.bias.data = bias_data_.data();
            bindings.bias.dt = params_.dt;
            bindings.bias.layout = DataLayout::eO;
            bindings.bias.shape = TensorShape(output_shape.c, 0u, 0u, 0u);
        }

        dnnl_conv_op::opts_t opts{};
        opts.output_shape = output_shape;
        opts.inp_pad = params_.in_pad;
        opts.out_pad = params_.out_pad;
        opts.stride = params_.stride;
        opts.dilates = params_.dilation;
        opts.out_layout = params_.output_layout;
        opts.out_dt = params_.dt;
        opts.activation = params_.activation;
        opts.force_winograd = params_.algo_winograd;
        opts.dump_weights = dump_weights();
        opts.dump_scratchpad = dump_weights();
        opts.use_fp32_accu = params_.dt == DataType::eFp16 && !params_.allow_fp16_computations;
        opts.groups = params_.groups;
        opts.cache_blob = use_persitent_cache_for_dnnl_reference();
        if (params_.transposed)
        {
            return dnnl_conv_op::deconvolution(bindings, opts);
        }
        opts.execution_iterations = reference_dispatch_iterations;
        return dnnl_conv_op::convolution(bindings, opts);
    }

    std::vector<std::byte> get_dml_results(const std::size_t tensor_out_bytes_width, ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) const
    {
        auto readback_buffer = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);

        auto conv_ref = gpu_op::Convolution(params_.input_shape, params_.filter_shape, get_output_shape(),
            to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.input_layout),
            to_dml_tensor_policy(params_.filter_layout), to_dml_tensor_policy(params_.output_layout),
            params_.stride, params_.dilation, params_.in_pad, params_.out_pad, params_.groups, !params_.no_bias, params_.allow_fp16_computations,
            params_.transposed, params_.activation, params_.managed_weights, dml_device_, d3d12_device_, true /* disable_mc*/);

        // bind descriptor heap
        auto descriptor_heap = create_descriptor_heap(d3d12_device_, conv_ref.get_total_descriptor_count());
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        conv_ref.create_binding_tables(descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        conv_ref.record_initialize(dml_cmd_recorder_, command_list, filter_buffer_.Get(), bias_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
        conv_ref.record_execute(dml_cmd_recorder_, command_list, output_buffer_.Get(), input_buffer_.Get(), filter_buffer_.Get(), bias_buffer_.Get(), constant_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        auto readback_buffer_ref = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer_ref.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> ref_untyped_result(tensor_out_bytes_width);
        void* readback_mapped_ptr_ref = nullptr;
        readback_buffer_ref->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr_ref));
        std::memcpy(ref_untyped_result.data(), readback_mapped_ptr_ref, ref_untyped_result.size());
        readback_buffer_ref->Unmap(0, nullptr);
        return ref_untyped_result;
    }

protected:
    virtual bool dump_weights() const
    {
        return params_.dump_weights;
    }

    virtual bool dump_scratchpad() const
    {
        return params_.dump_weights;
    }

protected:
    ID3D12Device* d3d12_device_;
    IDMLDevice* dml_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;
    create_params_t params_;

    std::vector<std::byte> input_data_;
    std::vector<std::byte> filter_data_;
    std::vector<std::byte> bias_data_;
    std::vector<std::byte> constant_data_;

    ComPtr<ID3D12Resource> constant_buffer_;
    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> filter_buffer_;
    ComPtr<ID3D12Resource> bias_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class ConvolutionDirectMLDispatcher : public ConvolutionBaseDispatcher
{
public:
    ConvolutionDirectMLDispatcher(create_params_t&& params, bool allow_descriptors_volatile, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : ConvolutionBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , dml_cmd_recorder_(dml_cmd_recorder)
        , conv_(params_.input_shape, params_.filter_shape, get_output_shape(),
            to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.input_layout),
            to_dml_tensor_policy(params_.filter_layout), to_dml_tensor_policy(params_.output_layout),
            params_.stride, params_.dilation, params_.in_pad, params_.out_pad, params_.groups, !params_.no_bias,
            params_.allow_fp16_computations, params_.transposed, params_.activation, params_.managed_weights,
            dml_device, d3d12_device, false, allow_descriptors_volatile)
    {
    }

    std::uint32_t get_total_descriptor_count()override
    {
        return conv_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        conv_.create_binding_tables(cpu_handle, gpu_handle);
        conv_.record_initialize(dml_cmd_recorder_, cmd_list, filter_buffer_.Get(), bias_buffer_.Get());
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        conv_.record_execute(dml_cmd_recorder_, cmd_list, output_buffer_.Get(), input_buffer_.Get(), filter_buffer_.Get(), bias_buffer_.Get(), constant_buffer_.Get());
    }

private:
    gpu_op::Convolution conv_;
    IDMLCommandRecorder* dml_cmd_recorder_;
};

class ConvolutionUmdD3d12Dispatcher : public ConvolutionBaseDispatcher
{
public:
    struct conv_umdd3d12_params_t
    {
        std::uint32_t verbose_mode = 0;  // 0: disabled; 1: execution; 2: creation and execution
        bool verbose_dump_to_file = false;
        bool cache_blob = false;

        inline static void add_cli_options(CLI::App* opts, conv_umdd3d12_params_t& params)
        {
            opts->add_option("--verbose_mode", params.verbose_mode)->default_val(0);
            opts->add_flag("--verbose_file", params.verbose_dump_to_file)->default_val(false);
            opts->add_flag("--cache_blob", params.cache_blob, "Use to test persistent cache blob.")->default_val(false);
        }
    };

public:
    ConvolutionUmdD3d12Dispatcher(const create_params_t& params, const conv_umdd3d12_params_t& umdd3d12_params, IntelExtension& intc_ext, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : ConvolutionBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , device_(d3d12_device, intc_ext.get_info())
        , umdd3d12_params_(std::move(umdd3d12_params))
    {      

        const auto t0 = std::chrono::high_resolution_clock::now();
        dnnl_engine_ = dnnl::iumd_interop::make_engine(&device_);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Engine create time: " << diff << std::endl;

        dnnl::set_verbose(umdd3d12_params_.verbose_mode);

        //if (umdd3d12_params_.verbose_dump_to_file)
        //{
        //    try
        //    {
        //        dnnl::iumd_interop::attach_verbose_attach_printf_callback(dnnl_utils::dump_onednn_logs_to_file);
        //    }
        //    catch (...)
        //    {
        //        // do nothing, callback was already attached
        //    }

        //}

        if (params_.transposed)
        {
            create_convolution<dnnl::deconvolution_forward>();
        }
        else
        {
            create_convolution<dnnl::convolution_forward>();
        }
    }

    std::uint32_t get_total_descriptor_count()override
    {
        // input, output, weights, bias
        std::uint32_t ret = 3;
        if (bias_memory_desc_.has_value())
        {
            ret++;
        }
        if (persistent_buffer_)
        {
            ret++;
        }
        if (temporary_buffer_)
        {
            ret++;
        }
        return ret;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        ID3D12GraphicsCommandList4* cmd_list4 = nullptr;
        throw_if_failed(cmd_list->QueryInterface(&cmd_list4), "cant cast d3d12 device to ID3D12Device5");
        iumd::custom_metacommand::UmdD3d12CommandList cmd(cmd_list4);
        dnnl::stream stream = dnnl::iumd_interop::make_stream(dnnl_engine_, &cmd);

        base_cpu_handle_ = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
        base_gpu_handle_ = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

        if (!reorder_weights_ && !reorder_bias_)
        {
            // early exit, as no reordering needed
            return;
        }

        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(2);
        resources_list.push_back({ DescType::eUav, filter_buffer_.Get() });
        resources_list.push_back({ DescType::eUav, persistent_buffer_.Get() });
        if (use_bias())
        {
            resources_list.push_back({ DescType::eUav, bias_buffer_.Get() });
        }
        const auto gpu_handles = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle_, base_gpu_handle_);

        // weights reorder
        if (reorder_weights_)
        { 
            auto umd_filter_input_mem = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[0]);
            auto umd_filter_reorder_mem = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[1]);

            dnnl::memory filer_input_memory = create_dnnl_memory(get_dnnl_filter_input_memory_desc(false), umd_filter_input_mem);
            dnnl::memory filer_reorder_memory = create_dnnl_memory(filter_memory_desc_, umd_filter_reorder_mem);

            std::unordered_map<int, dnnl::memory> args;
            args.insert({ DNNL_ARG_SRC, filer_input_memory });
            args.insert({ DNNL_ARG_DST, filer_reorder_memory });

            reorder_weights_.execute(stream, args);
        }

        if (use_bias() && reorder_bias_)
        {
            const auto persitent_resoruce_base_offset = filter_memory_desc_.get_size();

            auto umd_bias_input_mem = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[2]);   // bias input
            auto umd_bias_reorder_mem = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[1]); // persitent output
            const auto bias_desc = dnnl_utils::to_dnnl_mem_desc(TensorShape{ params_.filter_shape.n, 1, 1, 1 }, DataLayout::eNCHW, params_.dt);
            // this is just a copy from user provided bias to metacommand manged persitent resource 
            dnnl::memory bias_input_memory = create_dnnl_memory(bias_desc, umd_bias_input_mem);
            dnnl::memory bias_reorder_memory = create_dnnl_memory(bias_desc, umd_bias_reorder_mem, persitent_resoruce_base_offset);

            std::unordered_map<int, dnnl::memory> args;
            args.insert({ DNNL_ARG_SRC, bias_input_memory });
            args.insert({ DNNL_ARG_DST, bias_reorder_memory });

            reorder_bias_.execute(stream, args);
        }

    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ DescType::eUav, input_buffer_.Get() });
        resources_list.push_back({ DescType::eUav, reorder_weights_ ? persistent_buffer_.Get() : filter_buffer_.Get() });
        resources_list.push_back({ DescType::eUav, output_buffer_.Get() });
        if (use_bias() && bias_buffer_ && !reorder_bias_)
        {
            resources_list.push_back({ DescType::eUav,  bias_buffer_.Get() });
        }
        if (temporary_buffer_)
        {
            resources_list.push_back({ DescType::eUav, temporary_buffer_.Get() });
        }
        const auto gpu_handles = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle_, base_gpu_handle_);

        std::size_t res_idx = 0;
        auto umd_input_memory = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
        auto umd_filter_memory = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
        auto umd_output_memory = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
        auto umd_bias_memory = iumd::custom_metacommand::UmdD3d12Memory(); // use_bias() ? umd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]) :
        if (use_bias() && persistent_buffer_ && reorder_bias_)
        {
            // persitent resources shader with filter memory;
            umd_bias_memory = umd_filter_memory;
        }
        else if (use_bias() && bias_buffer_ && !reorder_bias_)
        {
            umd_bias_memory = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
        }
        auto umd_scratchpad_memory = temporary_buffer_ ? iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]) : iumd::custom_metacommand::UmdD3d12Memory();

        // stream is created in execute(...), because in MetaCommand cmd list object can be different from execute-to-execute
        ID3D12GraphicsCommandList4* cmd_list4 = nullptr;
        throw_if_failed(cmd_list->QueryInterface(&cmd_list4), "cant cast d3d12 device to ID3D12Device5");
        iumd::custom_metacommand::UmdD3d12CommandList cmd(cmd_list4);
        dnnl::stream stream = dnnl::iumd_interop::make_stream(dnnl_engine_, &cmd);

        // memory resources are created in execute(...), because in MetaCommand these objects can be different from execute-to-execute
        dnnl::memory input_memory = create_dnnl_memory(input_memory_desc_, umd_input_memory);
        dnnl::memory filter_memory = create_dnnl_memory(filter_memory_desc_, umd_filter_memory);
        std::optional<dnnl::memory> bias_memory;
        if (use_bias())
        {
            bias_memory.emplace(create_dnnl_memory(bias_memory_desc_.value(), umd_bias_memory, filter_memory_desc_.get_size()));
        }
        std::optional<dnnl::memory> scratchpad_memory;
        if (scratchpad_memory_desc_)
        {
            scratchpad_memory.emplace(create_dnnl_memory(scratchpad_memory_desc_.value(), umd_scratchpad_memory));
        }
        dnnl::memory output_memory = create_dnnl_memory(output_memory_desc_, umd_output_memory);

        std::unordered_map<int, dnnl::memory> args;
        args.insert( { DNNL_ARG_SRC, input_memory });
        args.insert( { DNNL_ARG_WEIGHTS, filter_memory });
        args.insert( { DNNL_ARG_DST, output_memory} );
        if (use_bias())
        {
            args.insert({ DNNL_ARG_BIAS, bias_memory.value()});
        }
        if (scratchpad_memory_desc_)
        {
            args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_memory.value() });
        }

        auto exec_untyped = [&](const auto& conv)
        {
            conv.execute(stream, args);
        };

        if (params_.transposed)
        {
            exec_untyped(std::get<dnnl::deconvolution_forward>(convolution_));
        }
        else
        {
            exec_untyped(std::get<dnnl::convolution_forward>(convolution_));
        }
    }

private:
    dnnl::memory::desc get_dnnl_filter_input_memory_desc(bool any_format)
    {
        using namespace dnnl_utils;
        const auto dt = to_dnnl_data_type(params_.dt);
        const auto fmt = [&]()
        {
            if (any_format)
            {
                return dnnl::memory::format_tag::any;
            }
            const auto& fl = params_.filter_layout;
            if (params_.transposed)
            {
                if (fl == DataLayout::eNCHW)
                {
                    return params_.groups > 1 ? dnnl::memory::format_tag::giohw : dnnl::memory::format_tag::iohw;
                }
                else if (fl == DataLayout::eNHWC)
                {
                    return params_.groups > 1 ? dnnl::memory::format_tag::acdeb : dnnl::memory::format_tag::ihwo;
                }
            }
            else
            {
                if (fl == DataLayout::eNCHW)
                {
                    return params_.groups > 1 ? dnnl::memory::format_tag::goihw : dnnl::memory::format_tag::oihw;
                }
                else if (fl == DataLayout::eNHWC)
                {
                    return params_.groups > 1 ? dnnl::memory::format_tag::gohwi : dnnl::memory::format_tag::ohwi;
                }
            }
            return dnnl::memory::format_tag::undef;
        }();          

        const auto dims = [&]()
        {
            dnnl::memory::dims dims = to_dnnl_dims(params_.filter_shape);
            if (params_.transposed)
            {
                std::swap(dims[0], dims[1]);
            }
            if (params_.groups != 1)
            {
                dims[0] /= params_.groups;
                dims[1] /= params_.groups;

                dnnl::memory::dims dims_temp{ params_.groups };
                dims_temp.insert(dims_temp.end(), dims.begin(), dims.end());
                dims = dims_temp;
            }
            return dims;
        }();

        return dnnl::memory::desc(dims, dt, fmt);
    }

    template<typename T>
    void create_convolution()
    {
        using namespace dnnl_utils;
        const dnnl::memory::dims pad{ params_.in_pad, params_.in_pad };
        const dnnl::memory::dims stride{ params_.stride.h, params_.stride.w };
        const dnnl::memory::dims dilates{ params_.dilation.h, params_.dilation.w };

        const dnnl::primitive_attr attr = [](const ActivationSettings& activation, bool use_fp32_accu)
        {
            // create a post-op with relu
            dnnl::post_ops ops;
            dnnl::primitive_attr attr;

            // sanity check
            assert(attr.get_scratchpad_mode() == dnnl::scratchpad_mode::library);
            // set scratchpad mode to user provided
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

            if (use_fp32_accu)
            {
                attr.set_accumulation_mode(dnnl::accumulation_mode::strict);
            }

            if (activation.type != ActivationType::eUnknown)
            {
                ops.append_eltwise(to_dnnl_activation_type(activation.type), activation.alpha, activation.beta);
                attr.set_post_ops(ops);
            }

            return attr;
        }(params_.activation, params_.dt == DataType::eFp16 && !params_.allow_fp16_computations);

        input_memory_desc_ = to_dnnl_mem_desc(params_.input_shape, params_.input_layout, params_.dt);
        output_memory_desc_ = to_dnnl_mem_desc(get_output_shape(), params_.output_layout, params_.dt);

        if (!params_.no_bias)
        {
            bias_memory_desc_.emplace(to_dnnl_mem_desc(TensorShape{ get_output_shape().c, 0, 0, 0}, DataLayout::eO, params_.dt));
        }

        const auto conv_algorithm = [&]()
        {
            if constexpr (std::is_same_v<T, dnnl::deconvolution_forward>)
            {
                return params_.algo_winograd ? dnnl::algorithm::deconvolution_winograd : dnnl::algorithm::deconvolution_direct;
            }
            return params_.algo_winograd ? dnnl::algorithm::convolution_winograd : dnnl::algorithm::convolution_direct;
        }();

        // this doesnt respect cmd line option: "managed_weights" ToDo: add non managed support
        const auto t00 = std::chrono::high_resolution_clock::now();
        const auto conv_desc = T::primitive_desc(
            dnnl_engine_,
            dnnl::prop_kind::forward_inference,
            conv_algorithm,
            input_memory_desc_,
            get_dnnl_filter_input_memory_desc(true),
            bias_memory_desc_ ? bias_memory_desc_.value() : dnnl::memory::desc{},
            output_memory_desc_,
            stride,
            dilates,
            pad,
            pad,
            attr
        );
        const auto t11 = std::chrono::high_resolution_clock::now();
        const auto diff1 = std::chrono::duration_cast<std::chrono::milliseconds>(t11 - t00);
        std::cout << "Primitive-desc create time: " << diff1 << std::endl;
        std::cout << "dnnl-umd kernel impl: " << conv_desc.impl_info_str() << std::endl;

        filter_memory_desc_ = conv_desc.query_md(dnnl::query::weights_md);

        const auto persistent_resource_size = [&]()
        {
            auto ret = filter_memory_desc_.get_size();
            if (use_bias())
            {
                ret += bias_memory_desc_->get_size();
            }
            return ret;
        }();
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

        assert(conv_desc.query_s64(dnnl::query::memory_consumption_s64) == 0);  // we provide scratchpad, so sanity check that primitive does not require any "hidden" memory
        scratchpad_memory_desc_.emplace(conv_desc.query_md(dnnl::query::scratchpad_md));
        const auto temporary_resoruce_size = scratchpad_memory_desc_->get_size();
        if (temporary_resoruce_size != 0)
        {
            const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            const auto buffder_desc = CD3DX12_RESOURCE_DESC::Buffer(temporary_resoruce_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            throw_if_failed(d3d12_device_->CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &buffder_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(temporary_buffer_.ReleaseAndGetAddressOf())), "create buffer resource");
        }

        // create convolution primitive
        std::ifstream in_key_file("onednn_persistent_cache.key", std::ofstream::in | std::ifstream::binary);
        std::ifstream in_value_file("onednn_persistent_cache.value", std::ofstream::in | std::ifstream::binary);
        std::vector<std::uint8_t> buffer_key;
        std::vector<std::uint8_t> buffer_value;
        const auto conv_blob_key = conv_desc.get_cache_blob_id();
        if (umdd3d12_params_.cache_blob && in_key_file.is_open())
        {
            buffer_key = std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in_key_file), {});
        }  
        if (buffer_key == conv_blob_key)
        {
            std::cout << "Found persistent cache blob files. Using them to create convolution primitive!" << std::endl;
            assert(in_value_file.is_open());  // Proper file  with key value exists, but file with cache blob (value) does not exist. Delete file with key and rerun application.
            buffer_value = std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in_value_file), {});       
        }
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (buffer_value.empty())
        {
            convolution_ = T(conv_desc);
        }
        else
        {
            convolution_ = T(conv_desc, buffer_value);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Primitive create time: " << diff << std::endl;

        if (umdd3d12_params_.cache_blob && buffer_value.empty())
        {
            std::cout << "Storing persistent cache blob files for." << std::endl;
            auto store_binary_data_to_file = [](const auto& file_name, const auto& data)
                {
                    std::ofstream out_file(file_name, std::ofstream::out | std::ofstream::binary);
                    std::copy(data.begin(), data.end(), std::ostream_iterator<std::uint8_t>(out_file));
                    out_file.close();
                };
            const auto cache_blob_id = conv_desc.get_cache_blob_id();
            store_binary_data_to_file("onednn_persistent_cache.key", cache_blob_id);

            const auto cache_blob = std::get<T>(convolution_).get_cache_blob();
            store_binary_data_to_file("onednn_persistent_cache.value", cache_blob);
        }

        // compile weights reorder kernel and create reorder primitive
        // ToDo: check if reorder needs scratchpad memory??
        dnnl::reorder::primitive_desc reorder_desc(dnnl_engine_, get_dnnl_filter_input_memory_desc(false), dnnl_engine_, filter_memory_desc_);
        reorder_weights_ = dnnl::reorder(reorder_desc);

        if (use_bias())
        {
            assert(bias_memory_desc_.has_value());
            // mimic copy shader from metacommand
            dnnl::reorder::primitive_desc reorder_desc(dnnl_engine_, *bias_memory_desc_, dnnl_engine_, *bias_memory_desc_);
            reorder_bias_ = dnnl::reorder(reorder_desc);
        }
    }

    dnnl::memory create_dnnl_memory(const auto& desc, auto& umd_mem, std::size_t offset = 0)
    {
        return dnnl::iumd_interop::make_memory(desc, dnnl_engine_, &umd_mem, offset);
    };

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatche, std::size_t reference_dispatch_iterations) override
    {
        auto dump_buffer_to_file = [&](const auto& buffer, const auto& file_name)
        {
            if (!buffer)
            {
                return;
            }
            const auto bytes_width = buffer->GetDesc().Width;
            // readback data and validate
            auto readback_buffer = create_buffer(d3d12_device_, bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(buffer.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            command_list->ResourceBarrier(1, &readback_output_barrirer);
            command_list->CopyResource(readback_buffer.Get(), persistent_buffer_.Get());
            close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

            std::vector<std::byte> data_out(bytes_width);
            std::byte* readback_mapped_ptr = nullptr;
            readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
            std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
            readback_buffer->Unmap(0, nullptr);

            std::ofstream fout(file_name, std::ios::out | std::ios::binary);
            fout.write((char*)data_out.data(), data_out.size());
            fout.close();
        };

        if (dump_weights())
        {
            dump_buffer_to_file(persistent_buffer_, "umd_weights_data.dat");
        }

        const bool dump_scratchpads_data = true;
        if (dump_scratchpad())
        {
            dump_buffer_to_file(temporary_buffer_, "umd_scratchpad_data.dat");
        }

        const auto ret = ConvolutionBaseDispatcher::validate_conformance(command_queue, command_allocator, command_list, print_mismatche, reference_dispatch_iterations);
        return ret;
    }

protected:
    bool use_persitent_cache_for_dnnl_reference() const override
    {
        return umdd3d12_params_.cache_blob;
    }

private:
    iumd::custom_metacommand::UmdD3d12Device device_;
    conv_umdd3d12_params_t umdd3d12_params_;
    dnnl::engine dnnl_engine_;

    std::variant<dnnl::convolution_forward, dnnl::deconvolution_forward> convolution_;
    dnnl::reorder reorder_weights_;
    dnnl::reorder reorder_bias_;  // metacommand copy bias data to persistent buffer

    dnnl::memory::desc input_memory_desc_;
    dnnl::memory::desc filter_memory_desc_;
    std::optional<dnnl::memory::desc> bias_memory_desc_;
    std::optional<dnnl::memory::desc> scratchpad_memory_desc_;
    dnnl::memory::desc output_memory_desc_;


    ComPtr<ID3D12Resource> temporary_buffer_;
    ComPtr<ID3D12Resource> persistent_buffer_;

    CD3DX12_CPU_DESCRIPTOR_HANDLE base_cpu_handle_;
    CD3DX12_GPU_DESCRIPTOR_HANDLE base_gpu_handle_;
};

class ConvolutionCmDispatcher : public ConvolutionBaseDispatcher
{
public:
    struct conv_cm_params_t
    {
        bool dump_asm;
        bool large_grf;
        bool print_reg_usage;
        std::array<std::uint32_t, 3> lws{ 1u, 1u, 1u };
        std::uint32_t block_w = 16;
        const std::uint32_t block_h = 1;  //ToDo: make configurable if needed
        std::uint32_t block_oc = 16;
        std::uint32_t block_batch = 1;  // block batch
        std::uint32_t slice_ic = 1;
        bool reorder_weights = true;
        bool dispatch_only_weights_reorder = false;

        inline static void add_cli_options(CLI::App* opts, conv_cm_params_t& params)
        {
            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
            opts->add_option("--block_w", params.block_w)->default_val(16);
            opts->add_option("--block_oc", params.block_oc)->default_val(16);
            opts->add_option("--block_batch", params.block_batch)->default_val(1);
            opts->add_option("--slice_ic", params.slice_ic, "How many HW threads cooperate to compute final output. Setting to 1 is equal to having this feature disabled. It increases thread group size (lws) in X dimension.")->default_val(1);
            opts->add_option("--lws", params.lws)->delimiter(',');
            opts->add_flag("--reorder_weights,!--no_reorder_weights", params.reorder_weights);
            opts->add_flag("--dispatch_only_weights_reorder", params.dispatch_only_weights_reorder);
        }
    };
public:
    ConvolutionCmDispatcher(create_params_t&& params, conv_cm_params_t&& cm_params, IntelExtension& intc_ext, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : ConvolutionBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , intc_ext_(intc_ext)
        , d3d12_device_(d3d12_device)
        , cm_params_(std::move(cm_params))
        , output_shape_(get_output_shape())
    {
        assert(params_.filter_shape.h == params_.filter_shape.w);

        // weights reoder
        if(cm_params_.reorder_weights)
        {
            WeightsReorder::create_params_t wr_params{};
            wr_params.input_dt = params_.dt;
            wr_params.output_dt = params_.dt;
            wr_params.ic = params_.filter_shape.c;
            wr_params.oc = params_.filter_shape.n;
            wr_params.k_size = params_.filter_shape.w;
            wr_params.input_layout = DataLayout::eOIYX;

            /*if (params_.dt == DataType::eFp16 && params_.filter_shape.w == 1 && params_.filter_shape.h == 1)
            {
                wr_params.output_layout = DataLayout::eIO_i8_o8_i2;
            }
            else if (params_.dt == DataType::eFp16  && params_.filter_shape.w != 1 && params_.filter_shape.h != 1)
            {*/
                if (cm_params_.block_oc == 8)
                {
                    wr_params.output_layout = DataLayout::eOYXI_o8;
                }
                if (cm_params_.block_oc == 16)
                {
                    wr_params.output_layout = DataLayout::eOYXI_o16;
                }
            //}

            weights_reorder_.emplace(WeightsReorder(std::move(wr_params), filter_buffer_, constant_buffer_, intc_ext, d3d12_device, cmd_list));
        }

        // root signature
        {
            // input, filter
            std::vector<DescType> desc_list = { DescType::eSrv, DescType::eSrv };
            if (!params_.no_bias)
            {
                desc_list.push_back(DescType::eSrv);
            }
            desc_list.push_back(DescType::eSrv);
            // output 
            desc_list.push_back(DescType::eUav);
            root_signature_ = create_root_signature(d3d12_device_, desc_list);
            assert(root_signature_);
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
        add_define("DT", static_cast<uint32_t>(params_.dt));
        //add_define("INPUT_WIDTH", params_.input_shape.w);
        //add_define("INPUT_HEIGHT", params_.input_shape.h);
        //add_define("INPUT_CHANNELS", params_.input_shape.c);

        //add_define("OUTPUT_WIDTH", output_shape_.w);
        //add_define("OUTPUT_HEIGHT", output_shape_.h);
        //("OUTPUT_CHANNELS", output_shape_.c);

        //add_define("BATCH", params_.input_shape.n);
        //add_define("INPUT_PAD", params_.in_pad);
        //add_define("OUTPUT_PAD", params_.out_pad);
        add_define("USE_BIAS", !params_.no_bias);
        add_define("KERNEL_SIZE", params_.filter_shape.h);
        add_define("STRIDE_W", params_.stride.w);
        //add_define("STRIDE_H", params_.stride.h);

        //add_define("SLICE_IC", cm_params_.slice_ic);
        //add_define("BLOCK_W", cm_params_.block_w);
        //add_define("BLOCK_H", cm_params_.block_h);
        //add_define("BLOCK_OC", cm_params_.block_oc);
        //add_define("BLOCK_BATCH", cm_params_.block_batch);

        //add_define("WEIGHTS_IN_OPTIMAL_FORMAT", cm_params.reorder_weights);

        // kernel compilation
        const auto dump_asm_str = cm_params_.dump_asm ? " -mdump_asm" : "";
        const auto large_grf_str = cm_params_.large_grf ? " -Qxcm_doubleGRF" : "";
        const auto print_reg_str = cm_params_.print_reg_usage ? " -mCM_printregusage" : "";
        const auto lws_x = " -DLWS_SIZE_X=" + std::to_string(cm_params_.lws[0] * cm_params_.slice_ic);
        const auto lws_y = " -DLWS_SIZE_Y=" + std::to_string(cm_params_.lws[1]);
        const auto lws_z = " -DLWS_SIZE_Z=" + std::to_string(cm_params_.lws[2]);
        const auto build_options_final = " -I \" \" " + build_options + dump_asm_str + large_grf_str + print_reg_str + lws_x + lws_y + lws_z;

        if (cm_params_.dump_asm)
        {
            std::cout << build_options_final << std::endl;
        }

        auto kernel_source_content = [](const auto kernel_size)
        {
            std::string path = "";
            if (false/*kernel_size == 1*/)
            {
                path = "conv_1x1_nchw_fp16.cpp";
            }
            else
            {
                path = "conv_nchw_fp16.cpp";
            }

            std::fstream file(path);
            if (!file.is_open())
            {
                const auto msg = std::format("Kernel file cant be opened:{} \n.", path);
                throw std::runtime_error(msg);
            }
            std::cout << std::format("Read kernel file: {} \n", path);
            return std::string((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
        }(params_.filter_shape.w);

        CD3DX12_SHADER_BYTECODE byte_code;
        byte_code.pShaderBytecode = kernel_source_content.data();
        byte_code.BytecodeLength = kernel_source_content.size();

        pso_ = intc_ext_.create_pipeline(byte_code, build_options_final, root_signature_.Get(), INTC_D3D12_SHADER_INPUT_TYPE::CM);
        assert(pso_);
    }

    std::uint32_t get_total_descriptor_count() override
    {
        // input, weights, output
        std::uint32_t descriptor_count = 4;
        if (!params_.no_bias)
        {
            descriptor_count++;
        }

        if (weights_reorder_.has_value())
        {
            descriptor_count += weights_reorder_->get_total_descriptor_count();
        }

        return descriptor_count;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        const auto desc_heap_incrs_size = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // i.e. add weights reorder

        auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
        auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

        if (weights_reorder_.has_value())
        {
            weights_reorder_->initialize(cmd_list, cpu_handle, gpu_handle);
            base_cpu_handle.Offset(weights_reorder_->get_total_descriptor_count(), desc_heap_incrs_size);
            base_gpu_handle.Offset(weights_reorder_->get_total_descriptor_count(), desc_heap_incrs_size);
        }

        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ DescType::eSrv, input_buffer_.Get() });
        resources_list.push_back({ DescType::eSrv, weights_reorder_.has_value() ? weights_reorder_->get_output_resource() : filter_buffer_.Get() });
        if (bias_buffer_)
        {
            resources_list.push_back({ DescType::eSrv, bias_buffer_.Get() });
        }
        resources_list.push_back({ DescType::eSrv, constant_buffer_.Get() });

        const auto tensor_out_bytes_width = output_buffer_->GetDesc().Width;
        resources_list.push_back({ DescType::eUav, output_buffer_.Get() });

        gpu_handles_ = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle, base_gpu_handle);
        assert(!gpu_handles_.empty());

        // dispatch weights reorder here in initalized if weights are managed
        if (params_.managed_weights && weights_reorder_.has_value())
        {
            weights_reorder_->execute(cmd_list);

            auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(weights_reorder_->get_output_resource());
            cmd_list->ResourceBarrier(1, &barrier);
        }
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        // dispatch weights reorder if needed (non managed case)
        if (!params_.managed_weights && weights_reorder_.has_value())
        {
            weights_reorder_->execute(cmd_list);
            auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(weights_reorder_->get_output_resource());
            cmd_list->ResourceBarrier(1, &barrier);

            if (cm_params_.dispatch_only_weights_reorder)
            {
                return;
            }
        }

        const uint32_t out_ch_size = static_cast<uint32_t>(std::ceil(params_.filter_shape.n / (double)(cm_params_.block_oc)));
        const auto gws_x = cm_params_.slice_ic * (round_up_next_multiple(output_shape_.w, cm_params_.block_w) / cm_params_.block_w);
        const auto gws_y = round_up_next_multiple(output_shape_.h, cm_params_.block_h) / cm_params_.block_h;
        const auto gws_z = (params_.input_shape.n / cm_params_.block_batch) * out_ch_size;

        assert(gws_x % cm_params_.lws[0] == 0);
        assert(gws_y % cm_params_.lws[1] == 0);
        assert(gws_z % cm_params_.lws[2] == 0);

        const auto thg_x = gws_x / (cm_params_.slice_ic * cm_params_.lws[0]);
        const auto thg_y = gws_y / cm_params_.lws[1];
        const auto thg_z = gws_z / cm_params_.lws[2];

        //std::cout << std::format("gws: {}, {}, {}, thg: {}, {}, {}\n", gws_x, gws_y, gws_z, thg_x, thg_y, thg_z);

        dispatch_kernel(cmd_list, pso_.Get(), root_signature_.Get(), gpu_handles_, thg_x, thg_y, thg_z);
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches, std::size_t reference_dispatch_iterations) override
    {
        if (weights_reorder_.has_value())
        {
            weights_reorder_->validate_conformance(command_queue, command_allocator, command_list, print_mismatches, reference_dispatch_iterations);
        }
        const auto ret = ConvolutionBaseDispatcher::validate_conformance(command_queue, command_allocator, command_list, print_mismatches, reference_dispatch_iterations);
        return ret;
    }

private:
    class WeightsReorder : public NodeDispatcher
    {
    public:
        struct create_params_t
        {
            DataType input_dt = DataType::eCount;
            DataType output_dt = DataType::eCount;

            DataLayout input_layout = DataLayout::eWeightsLayoutStart;
            DataLayout output_layout = DataLayout::eWeightsLayoutStart;

            std::uint32_t ic = 0;
            std::uint32_t oc = 0;
            std::uint32_t k_size = 0;

            std::array<std::uint32_t, 3> lws{ 1u, 1u, 1u };

            std::array<std::uint32_t, 3> get_gws() const
            {
                std::uint32_t gws_x = 0;
                std::uint32_t gws_y = 0;
                std::uint32_t gws_z = 0;
                if (output_layout == DataLayout::eIO_i8_o8_i2)
                {
                    const std::uint32_t ic_chunks_per_hw_thread = 8;
                    const std::uint32_t exec_size = 8;
                    const std::uint32_t dpas_depth = 8;
                    const std::uint32_t out_dt_size = (std::uint32_t)get_data_type_bytes_width(output_dt);
                    gws_x = oc / exec_size;
                    gws_y = ic / (ic_chunks_per_hw_thread * dpas_depth * out_dt_size);
                    gws_z = 1;
                }
                else if (output_layout == DataLayout::eOYXI_o8)
                {
                    gws_x = static_cast<uint32_t>(std::ceil(oc / 8.0));
                    gws_y = ic;
                    gws_z = k_size;  // kernel size Y
                }
                else if (output_layout == DataLayout::eOYXI_o16)
                {
                    gws_x = static_cast<uint32_t>(std::ceil(oc / 16.0));
                    gws_y = ic;
                    gws_z = k_size;  // kernel size Y
                }
                else
                {
                    assert(false && "Unknown data layout for weights reorder CM kernel.");
                }
                return { gws_x, gws_y, gws_z };
            }
        };
    public:
        WeightsReorder(create_params_t&& params, ComPtr<ID3D12Resource> input_resource, ComPtr<ID3D12Resource> constant_resource, IntelExtension& intc_ext, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list)
            : params_(std::move(params))
            , intc_ext_(intc_ext)
            , d3d12_device_(d3d12_device)
            , input_buffer_(input_resource)
            , constant_buffer_(constant_resource)
        {
            assert(params_.input_dt != DataType::eCount);
            assert(params_.output_dt != DataType::eCount);
            assert(params_.input_layout > DataLayout::eWeightsLayoutStart);
            assert(params_.output_layout > DataLayout::eWeightsLayoutStart);

            // root signature
            {
                // input, filter
                const std::vector<DescType> desc_list = { DescType::eSrv, DescType::eSrv, DescType::eUav };
                root_signature_ = create_root_signature(d3d12_device_, desc_list);
                assert(root_signature_);
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
            if (params_.input_dt == DataType::eFp16 && params_.output_dt == DataType::eFp16)
            {
                add_define("DT", "half");
            }
            else
            {
                add_define("DT", "float");
            }
            //add_define("WEI_OFFSET", 0);
            //add_define("IC", params_.ic);
            //add_define("OC", params_.oc);
            add_define("K_SIZE", params_.k_size);

            /*for (std::int32_t i = static_cast<std::int32_t>(DataLayout::eWeightsLayoutStart) + 1; i < static_cast<std::int32_t>(DataLayout::eCount); i++)
            {
                add_define("LAYOUT_" + data_layout_name(static_cast<DataLayout>(i)), i);
            }*/
            add_define("INPUT_LAYOUT", static_cast<std::int32_t>(params_.input_layout));
            add_define("OUTPUT_LAYOUT", static_cast<std::int32_t>(params_.output_layout));

            auto kernel_source_content = []()
            {
                const auto path = "reorder_weights.cpp";
                std::fstream file(path);
                if (!file.is_open())
                {
                    const auto msg = std::format("Kernel file cant be opened:{} \n.", path);
                    throw std::runtime_error(msg);
                }
                return std::string((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
            }();

            // kernel compilation
            const auto dump_asm_str = " -mdump_asm";
            const auto print_reg_str = " -mCM_printregusage";

            const auto lws_x = " -DLWS_SIZE_X=" + std::to_string(params_.lws[0]);
            const auto lws_y = " -DLWS_SIZE_Y=" + std::to_string(params_.lws[1]);
            const auto lws_z = " -DLWS_SIZE_Z=" + std::to_string(params_.lws[2]);
            const auto build_options_final = " -I \" \" " + build_options + dump_asm_str + print_reg_str + lws_x + lws_y + lws_z;

            CD3DX12_SHADER_BYTECODE byte_code;
            byte_code.pShaderBytecode = kernel_source_content.data();
            byte_code.BytecodeLength = kernel_source_content.size();

            //if (cm_params_.dump_asm)
            {
                std::cout << build_options_final << std::endl;
            }

            pso_ = intc_ext_.create_pipeline(byte_code, build_options_final, root_signature_.Get(), INTC_D3D12_SHADER_INPUT_TYPE::CM);
            assert(pso_);

            uint32_t filter_channels = 0;
            if (params_.output_layout == DataLayout::eOYXI_o8)
            {
                filter_channels = (params_.oc % 8 == 0) ? params_.oc : (params_.oc + (8 - params_.oc % 8));
            }
            else if (params_.output_layout == DataLayout::eOYXI_o16)
            {
                filter_channels = (params_.oc % 16 == 0) ? params_.oc : (params_.oc + (16 - params_.oc % 16));
            }
            else
            {
                filter_channels = params_.oc;
            }
            // compiled success -> create buffer
            {
                const auto size_bytes = params_.ic * filter_channels * params_.k_size * params_.k_size * (std::uint8_t)get_data_type_bytes_width(params_.output_dt);
                output_buffer_ = create_buffer(d3d12_device, size_bytes,
                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            }
        }

        ID3D12Resource* get_output_resource()
        {
            return output_buffer_.Get();
        }

        void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
        {
            assert(input_buffer_);
            assert(output_buffer_);
            assert(constant_buffer_);

            const auto desc_heap_incrs_size = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            // i.e. add weights reorder

            auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
            auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

            std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
            resources_list.reserve(get_total_descriptor_count());
            resources_list.push_back({ DescType::eSrv, input_buffer_.Get() });
            resources_list.push_back({ DescType::eSrv, constant_buffer_.Get() });
            resources_list.push_back({ DescType::eUav, output_buffer_.Get() });

            gpu_handles_ = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle, base_gpu_handle);
        }

        std::uint32_t get_total_descriptor_count() override
        {
            return 3;
        }

        void execute(ID3D12GraphicsCommandList* cmd_list) override
        {
            assert(cmd_list);
            assert(pso_);
            assert(root_signature_);
            assert(!gpu_handles_.empty());

            const auto gws_xyz = params_.get_gws();
            const auto gws_x = gws_xyz.at(0);
            const auto gws_y = gws_xyz.at(1);
            const auto gws_z = gws_xyz.at(2);

            assert(gws_x % params_.lws[0] == 0);
            assert(gws_x % params_.lws[1] == 0);
            assert(gws_x % params_.lws[2] == 0);
                   
            const auto thg_x = gws_x / params_.lws[0];
            const auto thg_y = gws_y / params_.lws[1];
            const auto thg_z = gws_z / params_.lws[2];
            //std::cout << std::format("thg: {}, {}, {} \n", thg_x, thg_y, thg_z);
            dispatch_kernel(cmd_list, pso_.Get(), root_signature_.Get(), gpu_handles_, thg_x, thg_y, thg_z);
        }


        ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
            ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches, std::size_t reference_dispatch_iterations) override
        {
            // optional weights conformance check
            if (false)
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

                const Half* data_ptr = reinterpret_cast<const Half*>(data_out.data());
                for (std::int32_t i = 0; i < tensor_out_bytes_width / (std::uint8_t)get_data_type_bytes_width(params_.output_dt); i++)
                {
                    const auto f = DirectX::PackedVector::XMConvertHalfToFloat(data_ptr[i]);
                    std::cout << (std::int32_t)f << " ";

                    if (params_.output_layout == DataLayout::eIO_i8_o8_i2)
                    {
                        if ((i + 1) % 16 == 0 && i != 0)
                        {
                            std::cout << std::endl;
                        }
                    }
                    else if (params_.output_layout == DataLayout::eOYXI_o16)
                    {
                        if (i != 0)
                        {
                            const auto oc_block = 16;
                            if ((i + 1) % oc_block == 0)
                            {
                                std::cout << std::endl;
                            }
                            if ((i + 1) % (oc_block * params_.ic) == 0)
                            {
                                std::cout << std::endl;
                            }
                        }
                    }
                    else if (params_.output_layout == DataLayout::eOYXI_o8)
                    {
                        if (i != 0)
                        {
                            const auto oc_block = 8;
                            if ((i + 1) % oc_block == 0)
                            {
                                std::cout << std::endl;
                            }
                            if ((i + 1) % (oc_block * params_.ic) == 0)
                            {
                                std::cout << std::endl;
                            }
                        }
                    }
                    else
                    {
                        std::cout << std::endl;
                    }
                }
            }

            return ConformanceResult{};
        }

    private:
        create_params_t params_;
        IntelExtension& intc_ext_;
        ID3D12Device* d3d12_device_;
        ComPtr<ID3D12Resource> input_buffer_;
        ComPtr<ID3D12Resource> constant_buffer_;
        ComPtr<ID3D12Resource> output_buffer_;
        ComPtr<ID3D12PipelineState> pso_;
        ComPtr<ID3D12RootSignature> root_signature_;
        std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;
    };

private:
    conv_cm_params_t cm_params_;
    ID3D12Device* d3d12_device_;
    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;

    std::optional<WeightsReorder> weights_reorder_;

    const TensorShape output_shape_;
};
