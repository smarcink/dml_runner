#pragma once
#include <vector>
#include <random>
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


class SoftmaxDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout;
        std::uint32_t batch;
        std::uint32_t ic;
        std::uint32_t in_width;
        std::uint32_t in_height;
        std::uint32_t axis;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();
            opts->add_option("--batch", params.batch)->required();
            opts->add_option("--ic", params.ic)->required();
            opts->add_option("--in_width", params.in_width)->required();
            opts->add_option("--in_height", params.in_height)->required();
            opts->add_option("--axis", params.axis, "axis represents the axis of which the SoftMax is calculated.")->required();
        }
    };
public:
    SoftmaxDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , softmax_(params_.axis, dml::TensorDimensions{ params_.batch, params_.ic, params_.in_width, params.in_height },
            to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), dml_device, d3d12_device)
        , d3d12_device_(d3d12_device)
        , input_data_(params_.batch* params_.ic* params_.in_width* params.in_height* get_data_type_bytes_width(params_.dt))
    {
        const auto tensor_in_desc = softmax_.get_tensor_input_desc();
        const auto tensor_out_desc = softmax_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_in_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;


        upload_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(0.0f, 5.0f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_);
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_);
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_a_bytes_width);
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_a_bytes_width);

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return softmax_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        softmax_.create_binding_tables(cpu_handle, gpu_handle);
        softmax_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        softmax_.record_execute(dml_cmd_recorder_, cmd_list, output_buffer_.Get(), input_buffer_.Get());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
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

        const auto dnnl_untyped_result = cpu_op::softmax(params_.axis, input_data_.data(),
            { params_.batch, params_.ic, params_.in_height, params_.in_width }, params_.dt, params_.layout);

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


private:
    create_params_t params_;
    gpu_op::Softmax softmax_;
    ID3D12Device* d3d12_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<std::byte> input_data_;

    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};