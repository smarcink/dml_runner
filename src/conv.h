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
            std::uint32_t stride, std::uint32_t input_pad, std::uint32_t output_pad, bool use_bias,   
            IDMLDevice* dml_device, ID3D12Device* d3d12_device)
        : DirectMlBaseNode(dml_device, d3d12_device)
        , graph_(dml_device)
    {
        const std::vector<std::uint32_t> strides = { stride, stride };
        const std::vector<std::uint32_t> start_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> end_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> out_pad = { output_pad, output_pad };

        //dml::Tensor
        tensor_input_ = dml::InputTensor(graph_, 0, dml::TensorDesc(data_type, input_dims, tensor_policy));
        tensor_filter_ = dml::InputTensor(graph_, 1, dml::TensorDesc(data_type, filter_dims, tensor_policy));
        if (use_bias)
        {
            tensor_bias_ = dml::InputTensor(graph_, 2, dml::TensorDesc(data_type, dml::TensorDimensions{ 1, filter_dims[0], 1, 1 }, tensor_policy));
        }

        tensor_out_ = dml::ConvolutionBuilder(tensor_input_, tensor_filter_, tensor_bias_)
            .Strides(strides)
            .StartPadding(start_pad)
            .EndPadding(end_pad)
            .OutputPadding(out_pad)
            .Build();

        // compiled operator
        DML_EXECUTION_FLAGS exec_flags = DML_EXECUTION_FLAG_NONE;
        std::vector<dml::Expression> graph_outputs{ tensor_out_ };
        dml_op_executor_ = graph_.Compile(exec_flags, graph_outputs);
        create_operator_impl();
    }

    dml::TensorDesc get_tensor_input_desc() const
    {
        return tensor_input_.GetOutputDesc();
    }

    dml::TensorDesc get_tensor_filter_desc() const
    {
        return tensor_filter_.GetOutputDesc();
    }

    dml::TensorDesc get_tensor_bias_desc() const
    {
        if(tensor_bias_.has_value())
        {
            return tensor_bias_->GetOutputDesc();
        }
        return {};
    }

    dml::TensorDesc get_tensor_out_desc() const
    {
        return tensor_out_.GetOutputDesc();
    }


    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, ID3D12Resource* resource_out,
        ID3D12Resource* resource_input, ID3D12Resource* resource_filter, ID3D12Resource* resource_bias)
    {
        assert(((resource_bias != nullptr)== tensor_bias_.has_value()) && "bias resources is not matching what was expected.");

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
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_binding_desc);
    }


private:
    dml::Graph graph_;
    dml::Expression tensor_input_;
    dml::Expression tensor_filter_;
    std::optional<dml::Expression> tensor_bias_;
    dml::Expression tensor_out_;
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