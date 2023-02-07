#pragma once
#include <vector>
#include "dml_base_node.h"

namespace gpu_op
{
class Convolution : public DirectMlBaseNode
{
public:
    Convolution(const dml::TensorDimensions& input_dims, const dml::TensorDimensions& filter_dims,
        const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy,
            std::uint32_t stride, std::uint32_t input_pad, std::uint32_t output_pad,
            bool use_bias, bool allow_fp16_computations, 
            IDMLDevice* dml_device, ID3D12Device* d3d12_device)
        : DirectMlBaseNode(dml_device, d3d12_device)
    {
        const std::vector<std::uint32_t> strides = { stride, stride };
        const std::vector<std::uint32_t> dilations = { 0u, 0u };
        const std::vector<std::uint32_t> start_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> end_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> out_pad = { output_pad, output_pad };

        const dml::TensorDimensions bias_dims {1, filter_dims[0], 1, 1 };

        const dml::TensorDimensions output_dims{
            input_dims[0],
            filter_dims[0],
            (input_dims[2] - filter_dims[2] + start_pad[0] + end_pad[0]) / strides[0] + 1,
            (input_dims[3] - filter_dims[3] + start_pad[1] + end_pad[1]) / strides[1] + 1,
        };

        dml::TensorProperties input_tensor_properites{};
        {
            tensor_input_desc_.DataType = data_type;
            tensor_input_desc_.Flags = DML_TENSOR_FLAG_NONE;
            tensor_input_desc_.DimensionCount = static_cast<std::uint32_t>(input_dims.size());
            tensor_input_desc_.Sizes = input_dims.data();
            input_tensor_properites = tensor_policy.Get(tensor_input_desc_.DataType, tensor_input_desc_.Flags, input_dims);
           // tensor_input_desc_.Strides = input_tensor_properites.strides.has_value() ? input_tensor_properites.strides->data() : nullptr;
            tensor_input_desc_.Strides = nullptr;
            tensor_input_desc_.TotalTensorSizeInBytes = input_tensor_properites.totalTensorSizeInBytes;
            tensor_input_desc_.GuaranteedBaseOffsetAlignment = input_tensor_properites.guaranteedBaseOffsetAlignment;
        }

        dml::TensorProperties filter_tensor_properites{};
        {
            tensor_filter_desc_.DataType = data_type;
            tensor_filter_desc_.Flags = DML_TENSOR_FLAG_NONE;
            tensor_filter_desc_.DimensionCount = static_cast<std::uint32_t>(filter_dims.size());
            tensor_filter_desc_.Sizes = filter_dims.data();
            filter_tensor_properites = tensor_policy.Get(tensor_filter_desc_.DataType, tensor_filter_desc_.Flags, filter_dims);
            tensor_filter_desc_.Strides = nullptr;
            tensor_filter_desc_.TotalTensorSizeInBytes = filter_tensor_properites.totalTensorSizeInBytes;
            tensor_filter_desc_.GuaranteedBaseOffsetAlignment = filter_tensor_properites.guaranteedBaseOffsetAlignment;
        }

        dml::TensorProperties bias_tensor_properites;
        if(use_bias)
        {
            DML_BUFFER_TENSOR_DESC bias_desc{};
            bias_desc.DataType = data_type;
            bias_desc.Flags = DML_TENSOR_FLAG_NONE;
            bias_desc.DimensionCount = static_cast<std::uint32_t>(bias_dims.size());
            bias_desc.Sizes = bias_dims.data();
            bias_tensor_properites = tensor_policy.Get(bias_desc.DataType, bias_desc.Flags, bias_dims);
            bias_desc.Strides = nullptr;
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

            output_tensor_properites = tensor_policy.Get(tensor_out_desc_.DataType, tensor_out_desc_.Flags, output_dims);
            //tensor_out_desc_.Strides = output_tensor_properites.strides.has_value() ? output_tensor_properites.strides->data() : nullptr;
            tensor_out_desc_.Strides = nullptr;
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
        desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        desc.DimensionCount = 2;
        desc.Strides = strides.data();
        desc.Dilations = dilations.data();
        desc.StartPadding = start_pad.data();
        desc.EndPadding = end_pad.data();
        desc.OutputPadding = out_pad.data();
        desc.GroupCount = 1u;
        desc.FusedActivation = nullptr;

        DML_OPERATOR_DESC dml_operator_desc{};
        dml_operator_desc.Type = DML_OPERATOR_CONVOLUTION;
        dml_operator_desc.Desc = &desc;

        throw_if_failed(dml_device->CreateOperator(
            &dml_operator_desc, IID_PPV_ARGS(dml_operator_.ReleaseAndGetAddressOf())), "create convolution operator");

        DML_EXECUTION_FLAGS exec_flags = DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        if (allow_fp16_computations)
        {
            exec_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
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


    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, ID3D12Resource* resource_out,
        ID3D12Resource* resource_input, ID3D12Resource* resource_filter, ID3D12Resource* resource_bias)
    {
        assert(((resource_bias != nullptr)== tensor_bias_desc_.has_value()) && "bias resources is not matching what was expected.");

        DML_BUFFER_BINDING input_buffer_binding{ resource_input, 0, resource_input->GetDesc().Width };
        DML_BUFFER_BINDING filter_buffer_binding{ resource_filter, 0, resource_filter->GetDesc().Width };
        DML_BUFFER_BINDING bias_buffer_binding;

        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(3);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_buffer_binding });
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &filter_buffer_binding });
        if (resource_bias)
        {
            bias_buffer_binding = { resource_bias, 0, resource_bias->GetDesc().Width };
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &bias_buffer_binding });
        }
        else
        {
            input_bindings.push_back({ DML_BINDING_TYPE_NONE, nullptr});
        }
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_binding_desc);
    }


private:
    ComPtr<IDMLOperator> dml_operator_;
    DML_BUFFER_TENSOR_DESC tensor_input_desc_;
    DML_BUFFER_TENSOR_DESC tensor_filter_desc_;
    std::optional<DML_BUFFER_TENSOR_DESC> tensor_bias_desc_;
    DML_BUFFER_TENSOR_DESC tensor_out_desc_;
};
}

namespace cpu_op
{

struct binding_t
{
    const std::byte* data = nullptr;
    DataType dt = DataType::eCount;
    DataLayout layout = DataLayout::eCount;
    std::vector<std::uint32_t> dims;
};

struct bindings_t
{
    binding_t input;
    binding_t filter;
    binding_t bias;
};

struct opts_t
{
    std::uint32_t inp_pad;
    std::uint32_t out_pad;
    std::uint32_t stride;

    DataType out_dt = DataType::eCount;
    DataLayout out_layout = DataLayout::eCount;
};
std::vector<std::byte> convolution(const bindings_t& bindings, opts_t opts);
}