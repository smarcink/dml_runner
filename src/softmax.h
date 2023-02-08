#pragma once
#include <vector>
#include "dml_base_node.h"

namespace gpu_op
{
class Softmax : public DirectMlBaseNode
{
public:
    Softmax(std::uint32_t axis, const dml::TensorDimensions& input_dims, const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy, 
        IDMLDevice* dml_device, ID3D12Device* d3d12_device)
        : DirectMlBaseNode(dml_device, d3d12_device)
    {
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

        DML_TENSOR_DESC out_desc{};
        out_desc.Desc = &tensor_input_desc_;
        out_desc.Type = DML_TENSOR_TYPE_BUFFER;

        DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC softmax1_operator_desc{};
        softmax1_operator_desc.AxisCount = 1;
        softmax1_operator_desc.Axes = &axis;
        softmax1_operator_desc.InputTensor = &inp_desc;
        softmax1_operator_desc.OutputTensor = &out_desc;

        DML_OPERATOR_DESC dml_operator_desc{};
        dml_operator_desc.Type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
        dml_operator_desc.Desc = &softmax1_operator_desc;

        throw_if_failed(dml_device->CreateOperator(
            &dml_operator_desc, IID_PPV_ARGS(dml_operator_.ReleaseAndGetAddressOf())), "create softmax operator");

        throw_if_failed(dml_device->CompileOperator(
            dml_operator_.Get(),
            DML_EXECUTION_FLAG_NONE,
            IID_PPV_ARGS(dml_op_executor_.ReleaseAndGetAddressOf())), "create softmax compiled operator");
        create_operator_impl();
    }

    dml::TensorDesc get_tensor_input_desc() const
    {
        return tensor_input_desc_;
    }

    dml::TensorDesc get_tensor_out_desc() const
    {
        return tensor_output_desc_;
    }

    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list,
        ID3D12Resource* resource_out,
        ID3D12Resource* resource_input)
    {
        DML_BUFFER_BINDING input_buffer_binding{ resource_input, 0, resource_input->GetDesc().Width };

        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(1);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_buffer_binding });
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_binding_desc);
    }

private:
    DML_BUFFER_TENSOR_DESC tensor_input_desc_{};
    DML_BUFFER_TENSOR_DESC tensor_output_desc_{};
    ComPtr<IDMLOperator> dml_operator_;
};

}  // namespace gpu_op

namespace cpu_op
{
std::vector<std::byte> softmax(std::uint32_t axis, const std::byte* in_data, const TensorShape& in_out_shape, DataType in_out_datatype, DataLayout in_out_layout);
}  // namespace cpu_op