#pragma once

#include <iostream>
#include <optional>
#include <span>
#include <format>
#include <random>

#include "dml_base_node.h"

enum class GemmType
{
    GemmType_AB = 0,
    // qkv
    GemmType_QK_QKV = 1,
    GemmType_SV_S_QKV = 2,

    // q + kv
    GemmType_QK_Q_KV,
    GemmType_SV_S_KV,
};

namespace
{
// a bit of hack :)>
dml::Expression dml_transpose(dml::Expression input_in, dml::TensorDimensions out_dims, dml::TensorPolicy out_tensor_policy)
{
    auto input = dml::Reinterpret(input_in, out_dims, out_tensor_policy.Get(input_in.GetOutputDesc().dataType, input_in.GetOutputDesc().flags, input_in.GetOutputDesc().sizes).strides);

    dml::detail::GraphBuilder* builder = input.Impl()->GetGraphBuilder();
    dml::TensorDesc input_tensor = input.GetOutputDesc();

    dml::TensorDesc output_tensor(input_tensor.dataType, out_dims);

    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC desc = {};
    desc.InputTensor = input_tensor.AsPtr<DML_TENSOR_DESC>();
    desc.OutputTensor = output_tensor.AsPtr<DML_TENSOR_DESC>();

    dml::detail::NodeOutput* const inputs[] = { input.Impl() };
    dml::detail::NodeID node = builder->CreateOperatorNode(DML_OPERATOR_ELEMENT_WISE_IDENTITY, &desc, inputs);
    dml::detail::NodeOutput* output = builder->CreateNodeOutput(node, 0, std::move(output_tensor));

    return output;
}

inline dml::TensorProperties compute_transpose_nchw_to_nhcw(
    DML_TENSOR_DATA_TYPE dataType,
    DML_TENSOR_FLAGS /*flags*/,
    std::span<const uint32_t> sizes)
{
    uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
    assert(dimension_count == 4);
    dml::TensorStrides strides(dimension_count);

    strides[3] = 1;
    strides[2] = sizes[2] * sizes[3];
    strides[1] = sizes[3];
    strides[0] = sizes[1] * sizes[2] * sizes[3];

    std::vector<uint32_t> new_sizes(dimension_count);
    new_sizes[3] = sizes[3];
    new_sizes[2] = sizes[1];
    new_sizes[1] = sizes[2];
    new_sizes[0] = sizes[0];

    dml::TensorProperties props;
    props.strides = std::move(strides);
    props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimension_count, new_sizes.data(), props.strides->data());
    props.guaranteedBaseOffsetAlignment = 0;
    return props;
}

}

namespace gpu_op
{

class Gemm : public DirectMlBaseNode
{
public:
    Gemm(GemmType gemm_type, const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy, const TensorShape& shape_a,
        const TensorShape& shape_b, const TensorShape& shape_out, float alpha, float beta,
        IDMLDevice* dml_device, ID3D12Device* d3d12_device, bool disable_mc = false)
        : DirectMlBaseNode(dml_device, d3d12_device)
        , type_(gemm_type)
        , graph_(dml_device)
    {
        outputs_.resize(1);
        if (type_ == GemmType::GemmType_AB)
        {
            dml::TensorDesc::Dimensions dimensions_0;
            dimensions_0.push_back(shape_a.n);
            dimensions_0.push_back(shape_a.c);
            dimensions_0.push_back(shape_a.h);
            dimensions_0.push_back(shape_a.w);
            dml::TensorDesc desc_input_0 = { data_type, dimensions_0 };
            input_0_ = dml::InputTensor(graph_, 0, desc_input_0);

            dml::TensorDesc::Dimensions dimensions_1;
            dimensions_1.push_back(shape_b.n);
            dimensions_1.push_back(shape_b.c);
            dimensions_1.push_back(shape_b.h);
            dimensions_1.push_back(shape_b.w);
            dml::TensorDesc desc_input_1 = { data_type, dimensions_1 };
            input_1_ = dml::InputTensor(graph_, 1, desc_input_1);

            outputs_[0] = dml::Gemm(input_0_, input_1_, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta);
        }
        else if (type_ == GemmType::GemmType_QK_QKV)
        {
            dml::TensorDesc::Dimensions dimensions_0;
            dimensions_0.push_back(shape_a.n);
            dimensions_0.push_back(shape_a.c);
            dimensions_0.push_back(shape_a.d);
            dimensions_0.push_back(shape_a.h);
            dimensions_0.push_back(shape_a.w);
            dml::TensorDesc desc_input_0 = { data_type, dimensions_0 };
            input_0_ = dml::InputTensor(graph_, 0, desc_input_0);

            // split the single input
            std::vector<std::uint32_t> after_split_dims = { 1, 1, 1 };
            auto split_outputs = dml::Split(input_0_, 3, after_split_dims);

            // reshape, we care only about Q and K for this case
            dml::TensorDimensions reshaped_dimss{ shape_a.n, shape_a.c, shape_a.d, shape_a.w };
            decltype(split_outputs) reshaped_splits(2);
            for (auto i = 0; i < reshaped_splits.size(); i++)
            {
                auto& sout = split_outputs[i];
                reshaped_splits[i] = dml::Reinterpret(sout, reshaped_dimss, dml::NullOpt);
            }

            const auto batch = reshaped_dimss[0];
            const auto seq = reshaped_dimss[1];
            const auto head_count = reshaped_dimss[2];
            const auto head_size = reshaped_dimss[3];

            // transpose logical
            const auto head_size_stride = 1;
            const auto head_count_stride = head_size * head_size_stride;
            const auto seq_stride = head_count * head_count_stride;
            const auto batch_stride = seq * seq_stride;
            dml::TensorStrides strides_0 = { batch_stride, head_count_stride, seq_stride, head_size_stride };
            dml::TensorStrides strides_1 = { batch_stride, head_count_stride, head_size_stride, seq_stride };
            auto gemm_inp_a = dml::Reinterpret(reshaped_splits[0], dml::TensorDimensions{batch, head_count, seq, head_size }, strides_0);
            auto gemm_inp_b = dml::Reinterpret(reshaped_splits[1], dml::TensorDimensions{batch, head_count, head_size, seq }, strides_1);
            outputs_[0] = dml::Gemm(gemm_inp_a, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta);
        }
        else if (type_ == GemmType::GemmType_SV_S_QKV)
        {
            dml::TensorDesc::Dimensions dimensions_0;
            dimensions_0.push_back(shape_a.n);
            dimensions_0.push_back(shape_a.c);
            dimensions_0.push_back(shape_a.h);
            dimensions_0.push_back(shape_a.w);
            dml::TensorDesc desc_input_0 = { data_type, dimensions_0 };
            input_0_ = dml::InputTensor(graph_, 0, desc_input_0);

            dml::TensorDesc::Dimensions dimensions_1;
            dimensions_1.push_back(shape_b.n);
            dimensions_1.push_back(shape_b.c);
            dimensions_1.push_back(shape_b.d);
            dimensions_1.push_back(shape_b.h);
            dimensions_1.push_back(shape_b.w);
            dml::TensorDesc desc_input_1 = { data_type, dimensions_1 };
            input_1_ = dml::InputTensor(graph_, 1, desc_input_1);

            // split the 2nd input
            std::vector<std::uint32_t> after_split_dims = { 1, 1, 1 };
            auto split_outputs = dml::Split(input_1_, 3, after_split_dims);

            // reshape, we care only about V for this case
            dml::TensorDimensions reshaped_dims{ shape_b.n, shape_b.c, shape_b.d, shape_b.w };
            auto reshaped_split = dml::Reinterpret(split_outputs[2], reshaped_dims, dml::NullOpt);

            const auto batch = reshaped_dims[0];
            const auto seq = reshaped_dims[1];
            const auto head_count = reshaped_dims[2];
            const auto head_size = reshaped_dims[3];

            // transpose logical
            const auto head_size_stride = 1;
            const auto head_count_stride = head_size * head_size_stride;
            const auto seq_stride = head_count * head_count_stride;
            const auto batch_stride = seq * seq_stride;
            dml::TensorStrides strides_1 = { batch_stride, head_count_stride, seq_stride, head_size_stride };
            auto gemm_inp_b = dml::Reinterpret(reshaped_split,  dml::TensorDimensions{batch, head_count, seq, head_size }, strides_1);
            auto gemm_out = dml::Gemm(input_0_, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta);
            outputs_[0] = dml_transpose(gemm_out, dml::TensorDimensions{ batch, seq, head_count, head_size },
                dml::TensorPolicy(&compute_transpose_nchw_to_nhcw));
        }
        else if (type_ == GemmType::GemmType_QK_Q_KV)
        {
            dml::TensorDesc::Dimensions dimensions_0;
            dimensions_0.push_back(shape_a.n);
            dimensions_0.push_back(shape_a.c);
            dimensions_0.push_back(shape_a.h);
            dimensions_0.push_back(1);
            dml::TensorDesc desc_input_0 = { data_type, dimensions_0 };
            input_0_ = dml::InputTensor(graph_, 0, desc_input_0);

            dml::TensorDesc::Dimensions dimensions_1;
            dimensions_1.push_back(shape_b.n);
            dimensions_1.push_back(shape_b.c);
            dimensions_1.push_back(shape_b.d);
            dimensions_1.push_back(shape_b.h);
            dimensions_1.push_back(shape_b.w);
            dml::TensorDesc desc_input_1 = { data_type, dimensions_1 };
            input_1_ = dml::InputTensor(graph_, 1, desc_input_1);

            const auto batch = shape_a.n;
            const auto seq = shape_a.c;
            const auto head_count = shape_b.d;
            const auto head_size = shape_a.h / head_count;
            const auto N = shape_b.c;

            // reshape and transpose first input
            const auto head_size_stride = 1;
            const auto head_count_stride = head_size * head_size_stride;
            const auto seq_stride = head_count * head_count_stride;
            const auto batch_stride = seq * seq_stride;
            dml::TensorStrides input_0_strides = { batch_stride, head_count_stride, seq_stride, head_size_stride };
            auto gemm_inp_a = dml::Reinterpret(input_0_, dml::TensorDimensions{ batch, head_count, seq, head_size }, input_0_strides);

            // split the 2nd input
            std::vector<std::uint32_t> after_split_dims = { 1, 1};
            auto split_outputs = dml::Split(input_1_, 3, after_split_dims);

            // reshape, we care only about V for this case
            dml::TensorDimensions reshaped_dims{ batch, N, head_count, head_size };
            auto reshaped_split = dml::Reinterpret(split_outputs[0], reshaped_dims, dml::NullOpt);

            // transpose logical
            dml::TensorStrides input_1_strides = { N * head_count * head_size, head_size, 1, head_count * head_size };
            auto gemm_inp_b = dml::Reinterpret(reshaped_split, dml::TensorDimensions{ batch, head_count, head_size, N }, input_1_strides);
            outputs_[0] = dml::Gemm(gemm_inp_a, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta);
        }
        else if (type_ == GemmType::GemmType_SV_S_KV)
        {
            dml::TensorDesc::Dimensions dimensions_0;
            dimensions_0.push_back(shape_a.n);
            dimensions_0.push_back(shape_a.c);
            dimensions_0.push_back(shape_a.h);
            dimensions_0.push_back(shape_a.w);
            dml::TensorDesc desc_input_0 = { data_type, dimensions_0 };
            input_0_ = dml::InputTensor(graph_, 0, desc_input_0);

            dml::TensorDesc::Dimensions dimensions_1;
            dimensions_1.push_back(shape_b.n);
            dimensions_1.push_back(shape_b.c);
            dimensions_1.push_back(shape_b.d);
            dimensions_1.push_back(shape_b.h);
            dimensions_1.push_back(shape_b.w);
            dml::TensorDesc desc_input_1 = { data_type, dimensions_1 };
            input_1_ = dml::InputTensor(graph_, 1, desc_input_1);

            const auto batch = shape_b.n;
            const auto seq = shape_b.c;
            const auto head_count = shape_b.d;
            const auto head_size = shape_b.w;

            // split the 2nd input
            std::vector<std::uint32_t> after_split_dims = { 1, 1 };
            auto split_outputs = dml::Split(input_1_, 3, after_split_dims);

            // reshape, we care only about V for this case
            dml::TensorDimensions reshaped_dims{ batch, seq, head_count, head_size };
            auto reshaped_split = dml::Reinterpret(split_outputs[1], reshaped_dims, dml::NullOpt);

            // transpose logical
            dml::TensorStrides input_1_strides = { seq * head_count * head_size, head_size, head_count * head_size , 1};
            auto gemm_inp_b = dml::Reinterpret(reshaped_split, dml::TensorDimensions{ batch, head_count, seq, head_size }, input_1_strides);
            outputs_[0] = dml::Gemm(input_0_, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta);
        }
        else
        {
            assert(false && "Unsupported gemm type!");
        }

        DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        if (data_type == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            execution_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        }
        if (disable_mc)
        {
            execution_flags |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
        }
        assert(!outputs_.empty());
        dml_op_executor_ = graph_.Compile(execution_flags, outputs_);
        create_operator_impl();
    }

    void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list,
        ID3D12Resource* resource_out, ID3D12Resource* resource_a, ID3D12Resource* resource_b)
    {
        assert(resource_a);
        assert(resource_out);
        DML_BUFFER_BINDING input_a_buffer_binding{ resource_a, 0, resource_a->GetDesc().Width };
        DML_BUFFER_BINDING input_b_buffer_binding{ nullptr, 0, 0 }; 

  
        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(3);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_a_buffer_binding });
        if (resource_b)
        {
            input_b_buffer_binding = { resource_b, 0, resource_b->GetDesc().Width };
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_b_buffer_binding });
        }


        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_binding_desc);
    }


private:
    dml::Graph graph_;
    dml::Expression input_0_;
    dml::Expression input_1_;
    std::vector<dml::Expression> outputs_;

    GemmType type_{};
};
}


class GemmBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        GemmType type;
        DataType dt;
        DataLayout layout;

        TensorShape shape_a;
        TensorShape shape_b;


        float alpha = 1.0f;
        float beta = 0.0f;



        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();


            opts->add_option("--shape_a", params.shape_a)->required();
            opts->add_option("--shape_b", params.shape_b); 

            opts->add_option("--alpha", params.alpha);

            opts->add_option("--gemm_type", params.type, "Name of the type of GEMM to run.")
                ->check(CLI::IsMember({ GemmType::GemmType_AB, GemmType::GemmType_QK_QKV, GemmType::GemmType_SV_S_QKV, GemmType::GemmType_QK_Q_KV, GemmType::GemmType_SV_S_KV }))->
                transform(CLI::Transformer(std::map<std::string, GemmType>{
                    { "ab", GemmType::GemmType_AB },
                    { "qk_qkv", GemmType::GemmType_QK_QKV },
                    { "sv_qkv", GemmType::GemmType_SV_S_QKV },
                    { "qk_q_kv", GemmType::GemmType_QK_Q_KV },
                    { "sv_s_kv", GemmType::GemmType_SV_S_KV },
            }, CLI::ignore_case))->required();

        }
    };
public:
    GemmBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , dml_device_(dml_device)
        , d3d12_device_(d3d12_device)

    {
        input_data_a_.resize(params_.shape_a.get_elements_count() * get_data_type_bytes_width(params_.dt));
        input_data_b_.resize(params_.shape_b.get_elements_count() * get_data_type_bytes_width(params_.dt));

        if (params_.type == GemmType::GemmType_AB)
        {
            assert(params_.shape_a.get_dims_count() == 4);
            assert(params_.shape_b.get_dims_count() == 4);
  
            assert(!input_data_a_.empty());
            assert(!input_data_b_.empty());
        }
        else if (params_.type == GemmType::GemmType_QK_QKV)
        {
            assert(params_.shape_a.get_dims_count() == 5);
            assert(params_.shape_b.get_dims_count() == 0);

            assert(!input_data_a_.empty());
            assert(input_data_b_.empty());
        }
        else if (params_.type == GemmType::GemmType_SV_S_QKV)
        {
            assert(params_.shape_a.get_dims_count() == 4);  // softmax input
            assert(params_.shape_b.get_dims_count() == 5);  // qkv input 

            assert(!input_data_a_.empty());
            assert(!input_data_b_.empty());
        }
        else if (params_.type == GemmType::GemmType_QK_Q_KV)
        {
            assert(params_.shape_a.get_dims_count() == 3);  // q input
            assert(params_.shape_b.get_dims_count() == 5);  // q_kv input 

            assert(!input_data_a_.empty());
            assert(!input_data_b_.empty());

            assert(params_.shape_a.h == params_.shape_b.d * params_.shape_b.w); // K SIZE
        }
        else if (params_.type == GemmType::GemmType_SV_S_KV)
        {
            assert(params_.shape_a.get_dims_count() == 4);  // s input
            assert(params_.shape_b.get_dims_count() == 5);  // q_kv input 

            assert(!input_data_a_.empty());
            assert(!input_data_b_.empty());

            assert(params_.shape_a.w == params_.shape_b.c); // K SIZE
        }
        else
        {
            assert(false && "Not supported gemm type!");
        }

        const auto B = get_batch();
        const auto C = get_channels();
        const auto M = get_M();
        const auto K = get_K();
        const auto N = get_N();
        std::cout << std::format("Running [B, C, M, K, N]: [{}, {}, {}, {}, {}]\n", B, C, M, K, N);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-0.1f, 0.1f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_a_);
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_b_);
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_a_);
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_b_);
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        const auto tensor_input_a_bytes_width = input_data_a_.size();
        const auto tensor_input_b_bytes_width = input_data_b_.size();

        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = out_shape.get_elements_count() * get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_a_bytes_width + tensor_input_b_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_a_ = create_buffer(d3d12_device, tensor_input_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (tensor_input_b_bytes_width > 0)
        {
            input_buffer_b_ = create_buffer(d3d12_device, tensor_input_b_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }

        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_a_.data(), tensor_input_a_bytes_width);
        memcopy_offset += tensor_input_a_bytes_width;
        if (tensor_input_b_bytes_width > 0)
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, input_data_b_.data(), tensor_input_b_bytes_width);
            memcopy_offset += tensor_input_b_bytes_width;
        }
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_a_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_a_bytes_width);
        memcopy_offset += tensor_input_a_bytes_width;
        if (tensor_input_b_bytes_width > 0)
        {
            cmd_list->CopyBufferRegion(input_buffer_b_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_input_b_bytes_width);
            memcopy_offset += tensor_input_b_bytes_width;
        }
        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_a_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (input_buffer_b_)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_b_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue, ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
    {
        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = out_shape.get_elements_count() * get_data_type_bytes_width(params_.dt);

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

        gpu_op::Gemm gemm_ref(params_.type, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), params_.shape_a, params_.shape_b, get_shape_output(),
             params_.alpha, params_.beta, dml_device_, d3d12_device_, true);

        // bind descriptor heap
        auto descriptor_heap = create_descriptor_heap(d3d12_device_, gemm_ref.get_total_descriptor_count());
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        gemm_ref.create_binding_tables(descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        gemm_ref.record_initialize(dml_cmd_recorder_, command_list);
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
        gemm_ref.record_execute(dml_cmd_recorder_, command_list, output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get());
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
            return run_conformance_check<float>(data_out, ref_untyped_result, 0.05f);
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
    TensorShape get_shape_output() const
    {
        TensorShape ret{};
        ret.n = get_batch();
        ret.c = get_channels();
        ret.h = get_M();
        ret.w = get_N();
        return ret;
    }

    std::uint32_t get_batch() const
    {
        return params_.shape_a.n;
    }

    std::uint32_t get_channels() const
    {
        if (params_.type == GemmType::GemmType_AB || params_.type == GemmType::GemmType_SV_S_QKV || params_.type == GemmType::GemmType_SV_S_KV)
        {
            return params_.shape_a.c;
        }
        else if (params_.type == GemmType::GemmType_QK_Q_KV)
        {
            return params_.shape_b.d;
        }
        else
        {
            return params_.shape_a.d;
        }
        assert(false && "Not supported");
    }

    std::uint32_t get_M() const
    {
        if (params_.type == GemmType::GemmType_AB || params_.type == GemmType::GemmType_SV_S_QKV || params_.type == GemmType::GemmType_SV_S_KV)
        {
            return params_.shape_a.h;
        }
        else if (params_.type == GemmType::GemmType_QK_QKV || params_.type == GemmType::GemmType_QK_Q_KV)
        {
            return params_.shape_a.c;
        }
        assert(false && "Not supported");
        return 0;
    }

    std::uint32_t get_K() const
    {
        if (params_.type == GemmType::GemmType_AB || params_.type == GemmType::GemmType_SV_S_QKV || params_.type == GemmType::GemmType_SV_S_KV)
        {
            return params_.shape_a.w;
        }
        else if (params_.type == GemmType::GemmType_QK_QKV)
        {
            return params_.shape_a.w;
        }
        else if (params_.type == GemmType::GemmType_QK_Q_KV)
        {
            return params_.shape_b.w;
        }
        assert(false && "Not supported");
        return 0;
    }

    std::uint32_t get_N() const
    {
        if (params_.type == GemmType::GemmType_AB || params_.type == GemmType::GemmType_SV_S_QKV || params_.type == GemmType::GemmType_SV_S_KV)
        {
            return params_.shape_b.w;
        }
        else if (params_.type == GemmType::GemmType_QK_QKV)
        {
            return params_.shape_a.c;
        }
        else if (params_.type == GemmType::GemmType_QK_Q_KV)
        {
            return params_.shape_b.c;
        }
        assert(false && "Not supported");
        return 0;
    }

protected:
    create_params_t params_;
    ID3D12Device* d3d12_device_;
    IDMLDevice* dml_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<std::byte> input_data_a_;
    std::vector<std::byte> input_data_b_;

    ComPtr<ID3D12Resource> input_buffer_a_;
    ComPtr<ID3D12Resource> input_buffer_b_;

    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class GemmDmlDispatcher : public GemmBaseDispatcher
{
public:
    GemmDmlDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : GemmBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , gemm_(params_.type, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout),  params_.shape_a, params_.shape_b, get_shape_output(), params_.alpha, params_.beta,
            dml_device, d3d12_device, false)
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
            output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get());
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
        bool fp32_accu = false;
        std::array<std::uint32_t, 3> lws{ 1u, 1u, 1u };

        std::uint32_t tile_m = 0;
        std::uint32_t tile_k = 0;
        std::uint32_t tile_n = 0;

        std::uint32_t slice_k = 1;

        inline static void add_cli_options(CLI::App* opts, cm_params_t& params)
        {
            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
            opts->add_flag("--fp32_accu", params.fp32_accu)->default_val(false);

            opts->add_option("--tile_m", params.tile_m);
            opts->add_option("--tile_k", params.tile_k);
            opts->add_option("--tile_n", params.tile_n);
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

        // set dispatch params
        const std::vector<std::uint32_t> accepted_tile_m_sizes = { 1, 2};
        const std::vector<std::uint32_t> accepted_tile_k_sizes = { 16, 32, 64, 128 };
        const std::vector<std::uint32_t> accepted_tile_n_sizes = { 16, 32, 64, 128 };

        const auto B = get_batch();
        const auto C = get_channels();
        const auto M = get_M();
        const auto K = get_K();
        const auto N = get_N();

        if (cm_params_.tile_m == 0)
        {
            for (const auto tm : accepted_tile_m_sizes)
            {
                if (M % tm == 0)
                {
                    cm_params_.tile_m = tm;
                }
            }
        }
        assert(cm_params_.tile_m > 0);

        if (cm_params_.tile_k == 0)
        {
            for (const auto tk : accepted_tile_k_sizes)
            {
                if (K % tk == 0)
                {
                    cm_params_.tile_k = tk;
                }
            }
        }
        assert(cm_params_.tile_k > 0);

        if (cm_params_.tile_n == 0)
        {
            for (const auto tn : accepted_tile_n_sizes)
            {
                if (N % tn == 0)
                {
                    cm_params_.tile_n = tn;
                }
            }
        }
        assert(cm_params_.tile_n > 0);


        if (params_.type == GemmType::GemmType_SV_S_QKV)
        {
            cm_params_.lws[0] = cm_params_.tile_k;
        }
        else if (params_.type == GemmType::GemmType_QK_QKV)
        {
            cm_params_.slice_k = K / cm_params_.tile_k;
            cm_params_.lws[2] = cm_params_.slice_k;

            if (cm_params_.slice_k == 1)
            {
                //cm_params_.lws[0] = 16;
            }
        }
        else if(params_.type == GemmType::GemmType_QK_Q_KV)
        {
            cm_params_.slice_k = 1;
            cm_params_.lws[2] = 1;
        }
        else if (params_.type == GemmType::GemmType_SV_S_KV)
        {
            cm_params_.slice_k = 1;
            cm_params_.lws[2] = 1;
            cm_params_.lws[0] = 1;
        }
        assert(cm_params_.slice_k > 0);

        {
            std::vector< DescType> desc_list =
            {
                DescType::eSrv, // input a
                DescType::eSrv, // input b
                DescType::eUav // output
            };
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

        add_define("SIZE_B", B);
        add_define("SIZE_C", C);
        add_define("SIZE_M", M);
        add_define("SIZE_K", K);
        add_define("SIZE_N", N);


        if (params_.type == GemmType::GemmType_QK_QKV)
        {
            add_define("SIZE_BATCH", params_.shape_a.n);
            add_define("SIZE_SEQ_LEN", params_.shape_a.c);
            add_define("SIZE_NUM_HEADS", params_.shape_a.d);
            add_define("SIZE_STACKED_TENSORS", params_.shape_a.h);
            add_define("SIZE_HEAD_SIZE", params_.shape_a.w);
        } 
        else if (params_.type == GemmType::GemmType_SV_S_QKV || params_.type == GemmType::GemmType_QK_Q_KV || params_.type == GemmType::GemmType_SV_S_KV)
        {
            add_define("SIZE_BATCH", params_.shape_b.n);
            add_define("SIZE_SEQ_LEN", params_.shape_b.c);
            add_define("SIZE_NUM_HEADS", params_.shape_b.d);
            add_define("SIZE_STACKED_TENSORS", params_.shape_b.h);
            add_define("SIZE_HEAD_SIZE", params_.shape_b.w);
        }

        add_define("SCALE", params_.alpha);

        add_define("DT", "half");

        add_define("TILE_K", cm_params_.tile_k);
        add_define("TILE_N", cm_params_.tile_n);
        add_define("TILE_M", cm_params_.tile_m);
        add_define("SLICE_K", cm_params_.slice_k);

        add_define("ACCU_IS_FP32", cm_params_.fp32_accu);

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

        auto kernel_source_content = [](GemmType type)
        {
            std::string path = "";
            switch (type)
            {
            case GemmType::GemmType_AB: path = "gemm_nchw_fp16.cpp"; break;
            case GemmType::GemmType_QK_QKV: path = "mha_qk_qkv_gemm_fp16.cpp"; break;
            case GemmType::GemmType_SV_S_QKV: path = "mha_sv_s_qkv_gemm_fp16.cpp";  break;
            case GemmType::GemmType_SV_S_KV: path = "mha_sv_s_kv_gemm_fp16.cpp";  break;
            case GemmType::GemmType_QK_Q_KV: path = "mha_qk_q_kv_gemm_fp16.cpp";  break;
            default:
                assert(false && "Unsupported gemm type. Cant deduce JIT!.");
            }

            std::fstream file(path);
            if (!file.is_open())
            {
                const auto msg = std::format("Kernel file cant be opened:{} \n.", path);
                throw std::runtime_error(msg);
            }
            return std::string((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
        }(params_.type);

        CD3DX12_SHADER_BYTECODE byte_code;
        byte_code.pShaderBytecode = kernel_source_content.data();
        byte_code.BytecodeLength = kernel_source_content.size();
        pso_ = intc_ext_.create_pipeline(byte_code, build_options_final, root_signature_.Get(), INTC_D3D12_SHADER_INPUT_TYPE::CM);

        const auto gws = get_gws();
        const auto lws = cm_params_.lws;
        std::cout << std::format("gws: [{}, {}, {}], lws: [{}, {}, {}]\n", gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);
    }


    std::uint32_t get_total_descriptor_count() override
    {
        // input_a, input_b, output
        std::uint32_t descriptor_count = 3;
        return descriptor_count;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {

        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ DescType::eSrv, input_buffer_a_.Get() });
        if (input_buffer_b_)
        {
            resources_list.push_back({ DescType::eSrv, input_buffer_b_.Get() });
        }
        resources_list.push_back({ DescType::eUav, output_buffer_.Get() });

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

        const auto gws = get_gws();

        const auto gws_x = gws[0];
        const auto gws_y = gws[1];
        const auto gws_z = gws[2];

        assert(gws_x % cm_params_.lws[0] == 0);
        assert(gws_y % cm_params_.lws[1] == 0);
        assert(gws_z % cm_params_.lws[2] == 0);

        const auto thg_x = gws_x / cm_params_.lws[0];
        const auto thg_y = gws_y / cm_params_.lws[1];
        const auto thg_z = gws_z / cm_params_.lws[2];
        cmd_list->Dispatch(thg_x, thg_y, thg_z);
    }

    private:
        std::vector<std::uint32_t> get_gws() const
        {
            const std::uint32_t gws_x = get_M() / cm_params_.tile_m;
            const std::uint32_t gws_y = get_N() / cm_params_.tile_n;
            const std::uint32_t gws_z = [this]()
            {
                if (params_.type == GemmType::GemmType_SV_S_QKV)
                {
                    return 1u;
                }
                if (params_.type == GemmType::GemmType_QK_QKV)
                {
                    return get_batch() * get_channels() * cm_params_.slice_k;
                }
                return get_batch() * get_channels() * cm_params_.slice_k;
            }();
            
            
            // ;

            assert(gws_x != 0);
            assert(gws_y != 0);
            assert(gws_z != 0);
            return { gws_x, gws_y, gws_z };
        }

private:
    cm_params_t cm_params_;
    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;
};