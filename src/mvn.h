#pragma once
#include <vector>
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