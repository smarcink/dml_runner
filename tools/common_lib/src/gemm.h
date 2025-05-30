#pragma once

#include <iostream>
#include <optional>
#include <span>
#include <format>
#include <random>

#include "dml_base_node.h"
#include "softmax.h"

#include "dnnl_utils.h"

#include "iumd_d3d12_impl.h"
#include <dnnl_iumd.h>
#include <dnnl.hpp>
#include <dnnl_iumd.hpp>
#include "oneapi/dnnl/dnnl.hpp"

enum class GemmType
{
    GemmType_AB = 0,
    // qkv
    GemmType_QK_QKV = 1,
    GemmType_SV_S_QKV = 2,

    // q + kv
    GemmType_QK_Q_KV,
    GemmType_SV_S_KV
};

namespace dnnl_gemm_op
{
struct bindings_t
{
    dnnl_utils::binding_t input_a;
    dnnl_utils::binding_t input_b;
    dnnl_utils::binding_t input_c;
};

struct opts_t
{
    TensorShape output_shape;
    DataType out_dt = DataType::eCount;
    DataLayout out_layout = DataLayout::eCount;
    bool force_fp32_accumulator = false;
    ActivationSettings activation{};
    bool a_transposed = false;
    bool b_transposed = false;

    float alpha = 1.0f;
    float beta = 1.0f;

    std::size_t execution_iterations = 1ul; // set it to bigger value to run more iterations
};
std::vector<std::byte> gemm(const bindings_t& bindings, opts_t opts);
}


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
    Gemm(GemmType gemm_type, const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy_ab, const dml::TensorPolicy& tensor_policy_c,
        const TensorShape& shape_a, const TensorShape& shape_b, const TensorShape& shape_c, const TensorShape& shape_out,
        bool a_transposed, bool b_managed, bool b_transposed, bool c_managed, float alpha, float beta,
        bool allow_fp16_computations, const ActivationSettings& activation_settings,
        IDMLDevice* dml_device, ID3D12Device* d3d12_device, bool allow_descriptors_volatile, bool disable_mc = false)
        : DirectMlBaseNode(dml_device, d3d12_device)
        , type_(gemm_type)
        , graph_(dml_device)
    {

        const auto fused_act = [](const auto& activation_settings)
        {
            auto ret = dml::FusedActivation::None();
            if (activation_settings.type != ActivationType::eUnknown)
            {
                const auto activation = to_dml_activation_setting(activation_settings);
                ret = dml::FusedActivation(activation.desc.Type, activation_settings.alpha, activation_settings.beta);
            }
            return ret;
        }(activation_settings);

        outputs_.resize(1);
        if (type_ == GemmType::GemmType_AB)
        {
            dml::TensorDesc::Dimensions dimensions_0;
            dimensions_0.push_back(shape_a.n);
            dimensions_0.push_back(shape_a.c);
            dimensions_0.push_back(a_transposed ? shape_a.w : shape_a.h); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
            dimensions_0.push_back(a_transposed ? shape_a.h : shape_a.w); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
            dml::TensorDesc desc_input_0 = { data_type, dimensions_0 };
            input_0_ = dml::InputTensor(graph_, 0, desc_input_0);

            dml::TensorDesc::Dimensions dimensions_1;
            dimensions_1.push_back(shape_b.n);
            dimensions_1.push_back(shape_b.c);
            dimensions_1.push_back(b_transposed ? shape_b.w : shape_b.h); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
            dimensions_1.push_back(b_transposed ? shape_b.h : shape_b.w); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
            dml::TensorDesc desc_input_1 = { data_type, b_managed ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE, dimensions_1 };
            input_1_ = dml::InputTensor(graph_, 1, desc_input_1);

            if (shape_c.get_dims_count() > 0)
            {
                dml::TensorDesc::Dimensions dimensions_2;
                dimensions_2.push_back(shape_c.n);
                dimensions_2.push_back(shape_c.c);
                dimensions_2.push_back(shape_c.h);
                dimensions_2.push_back(shape_c.w);
                DML_BUFFER_TENSOR_DESC desc_input_2;
                dml::TensorProperties tensor_c_properites;
                {
                    desc_input_2.DataType = data_type;
                    desc_input_2.Flags =  c_managed ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE;
                    desc_input_2.DimensionCount = static_cast<std::uint32_t>(dimensions_2.size());
                    desc_input_2.Sizes = dimensions_2.data();

                    tensor_c_properites = tensor_policy_c.Get(desc_input_2.DataType, desc_input_2.Flags, dimensions_2);
                    desc_input_2.Strides = tensor_c_properites.strides.has_value() ? tensor_c_properites.strides->data() : nullptr;
                    desc_input_2.TotalTensorSizeInBytes = tensor_c_properites.totalTensorSizeInBytes;
                    desc_input_2.GuaranteedBaseOffsetAlignment = tensor_c_properites.guaranteedBaseOffsetAlignment;
                }
               
                input_2_ = dml::InputTensor(graph_, 2, desc_input_2);
            }

            outputs_[0] = dml::Gemm(input_0_, input_1_, input_2_,
                a_transposed ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE,
                b_transposed ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE,
                alpha, beta, fused_act);
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
            outputs_[0] = dml::Gemm(gemm_inp_a, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta, fused_act);
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
            auto gemm_out = dml::Gemm(input_0_, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta, fused_act);
            const auto& gemm_out_sizes = gemm_out.GetOutputDesc().sizes;
            outputs_[0] = dml_transpose(gemm_out, dml::TensorDimensions{ gemm_out_sizes[0], gemm_out_sizes[2], gemm_out_sizes[1], gemm_out_sizes[3]},
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
            outputs_[0] = dml::Gemm(gemm_inp_a, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta, fused_act);
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
            outputs_[0] = dml::Gemm(input_0_, gemm_inp_b, dml::NullOpt, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, alpha, beta, fused_act);
        }
        else
        {
            assert(false && "Unsupported gemm type!");
        }

        DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
        
        if (allow_descriptors_volatile)
        {
            execution_flags |= DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        }
        ;
        if (allow_fp16_computations)
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
        ID3D12Resource* resource_out, ID3D12Resource* resource_a, ID3D12Resource* resource_b, ID3D12Resource* resource_c)
    {
        assert(resource_a);
        assert(resource_out);
        DML_BUFFER_BINDING input_a_buffer_binding{ resource_a, 0, resource_a->GetDesc().Width };
        DML_BUFFER_BINDING input_b_buffer_binding{ nullptr, 0, 0 };
        DML_BUFFER_BINDING input_c_buffer_binding{ nullptr, 0, 0 };

        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(3);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_a_buffer_binding });
        if (resource_b)
        {
            if (input_1_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
            {
                input_b_buffer_binding = { nullptr, 0, 0 };
                input_bindings.push_back({ DML_BINDING_TYPE_NONE, &input_b_buffer_binding });
            }
            else
            {
                input_b_buffer_binding = { resource_b, 0, resource_b->GetDesc().Width };
                input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_b_buffer_binding });
            }
        }
        if (resource_c)
        {
            if (input_2_ && input_2_->GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
            {
                input_c_buffer_binding = { nullptr, 0, 0 };
                input_bindings.push_back({ DML_BINDING_TYPE_NONE, &input_c_buffer_binding });
            }
            else
            {
                input_c_buffer_binding = { resource_c, 0, resource_c->GetDesc().Width };
                input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_c_buffer_binding });
            }
        }

        std::vector<DML_BINDING_DESC> output_bindings;
        output_bindings.reserve(1);
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };
        output_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &output_buffer_binding });

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_bindings);
    }

    virtual void record_initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, ID3D12Resource* resource_b, ID3D12Resource* resource_c)
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
        input_binds.push_back({ nullptr, 0, 0 });  // tensor a

        //tensor b
        if (input_1_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_binds.push_back({ resource_b, 0, resource_b->GetDesc().Width });
        }
        else
        {
            input_binds.push_back({ nullptr, 0, 0 });
        }

        // tensor c 
        if (input_2_ && input_2_->GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            assert(resource_c != nullptr);
            input_binds.push_back({ resource_c, 0, resource_c->GetDesc().Width });
        }
        else if (input_2_)
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

private:
    dml::Graph graph_;
    dml::Expression input_0_;
    dml::Expression input_1_;
    dml::Optional<dml::Expression> input_2_;
    std::vector<dml::Expression> outputs_;

    GemmType type_{};
};

class QuantGemm : public DirectMlBaseNode
{
public:
    QuantGemm(const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy_ab, const dml::TensorPolicy& tensor_policy_c,
        const TensorShape& shape_a, const TensorShape& shape_b, const TensorShape& shape_c, const TensorShape& shape_out,
        const TensorShape& shape_scale, const TensorShape& shape_zeropoint, bool a_quantized, bool b_quantized, bool c_quantized, const uint32_t block_size, const DML_TENSOR_DATA_TYPE quantized_data_type, bool has_zero_point,
        bool a_transposed, bool b_managed, bool b_transposed, bool bs_managed, bool bz_managed, bool c_managed, float alpha, float beta,
        bool allow_fp16_computations, const ActivationSettings& activation_settings,
        IDMLDevice* dml_device, ID3D12Device* d3d12_device, bool allow_descriptors_volatile, bool disable_mc = false)
        : DirectMlBaseNode(dml_device, d3d12_device)
        , graph_(dml_device)
    {
        assert(!a_quantized && "A Quantized feature is unsupported");
        assert(!c_quantized && "C Quantized feature is unsupported");
        assert(b_quantized && "B has to be quantized to execute Quant GEMM");
        assert(has_zero_point && "Only Zero point supported");

        const auto fused_act = [](const auto& activation_settings)
        {
            auto ret = dml::FusedActivation::None();
            if (activation_settings.type != ActivationType::eUnknown)
            {
                const auto activation = to_dml_activation_setting(activation_settings);
                ret = dml::FusedActivation(activation.desc.Type, activation_settings.alpha, activation_settings.beta);
            }
                return ret;
        }(activation_settings);

        outputs_.resize(1);

        dml::TensorDesc::Dimensions dimensions_0;
        dimensions_0.push_back(shape_a.n);
        dimensions_0.push_back(shape_a.c);
        dimensions_0.push_back(a_transposed ? shape_a.w : shape_a.h); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
        dimensions_0.push_back(a_transposed ? shape_a.h : shape_a.w); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
        dml::TensorDesc desc_input_0 = { data_type, dimensions_0 };
        input_0_ = dml::InputTensor(graph_, 0, desc_input_0);

        dml::TensorDesc::Dimensions dimensions_1;
        dimensions_1.push_back(shape_b.n);
        dimensions_1.push_back(shape_b.c);
        dimensions_1.push_back(b_transposed ? shape_b.w : shape_b.h); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
        dimensions_1.push_back(b_transposed ? shape_b.h : shape_b.w); // to make cmd arguments have normal matrix with transpose internal to each dispatcher
        dml::TensorDesc desc_input_1 = { quantized_data_type, b_managed ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE, dimensions_1 };
        input_1_ = dml::InputTensor(graph_, 1, desc_input_1);

        uint32_t index = shape_c.get_dims_count() > 0 ? 3 : 2;

        dml::TensorDesc::Dimensions dimensions_scale;
        dimensions_scale.push_back(shape_scale.n);
        dimensions_scale.push_back(shape_scale.c);
        dimensions_scale.push_back(shape_scale.h);
        dimensions_scale.push_back(shape_scale.w);
        dml::TensorDesc desc_scale = { data_type, bs_managed ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE, dimensions_scale };
        input_1_scale_ = dml::InputTensor(graph_, index, desc_scale);

        dml::TensorDesc::Dimensions dimensions_zero_point;
        dimensions_zero_point.push_back(shape_zeropoint.n);
        dimensions_zero_point.push_back(shape_zeropoint.c);
        dimensions_zero_point.push_back(shape_zeropoint.h);
        dimensions_zero_point.push_back(shape_zeropoint.w);
        dml::TensorDesc desc_zeropoint = { quantized_data_type, bz_managed ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE, dimensions_zero_point };
        input_1_zeropoint_ = dml::InputTensor(graph_, index + 1, desc_zeropoint);

        std::vector<dml::Expression> tensor_b_quantization_params(2);
        tensor_b_quantization_params[0] = input_1_scale_;
        tensor_b_quantization_params[1] = input_1_zeropoint_;


        input_1_dequantized_ = dml::Dequantize(input_1_, tensor_b_quantization_params, has_zero_point ? DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT : DML_QUANTIZATION_TYPE_SCALE);

        if (shape_c.get_dims_count() > 0)
        {
            dml::TensorDesc::Dimensions dimensions_2;
            dimensions_2.push_back(shape_c.n);
            dimensions_2.push_back(shape_c.c);
            dimensions_2.push_back(shape_c.h);
            dimensions_2.push_back(shape_c.w);
            DML_BUFFER_TENSOR_DESC desc_input_2;
            dml::TensorProperties tensor_c_properites;
            {
                desc_input_2.DataType = data_type;
                desc_input_2.Flags = c_managed ? DML_TENSOR_FLAG_OWNED_BY_DML : DML_TENSOR_FLAG_NONE;
                desc_input_2.DimensionCount = static_cast<std::uint32_t>(dimensions_2.size());
                desc_input_2.Sizes = dimensions_2.data();

                tensor_c_properites = tensor_policy_c.Get(desc_input_2.DataType, desc_input_2.Flags, dimensions_2);
                desc_input_2.Strides = tensor_c_properites.strides.has_value() ? tensor_c_properites.strides->data() : nullptr;
                desc_input_2.TotalTensorSizeInBytes = tensor_c_properites.totalTensorSizeInBytes;
                desc_input_2.GuaranteedBaseOffsetAlignment = tensor_c_properites.guaranteedBaseOffsetAlignment;
            }

            input_2_ = dml::InputTensor(graph_, 2, desc_input_2);
        }

        outputs_[0] = dml::Gemm(input_0_, input_1_dequantized_, input_2_,
            a_transposed ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE,
            b_transposed ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE,
            alpha, beta, fused_act);

        DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;

        if (allow_descriptors_volatile)
        {
            execution_flags |= DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
        }
        ;
        if (allow_fp16_computations)
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
        ID3D12Resource* resource_out, ID3D12Resource* resource_a, ID3D12Resource* resource_b, ID3D12Resource* resource_c,
        ID3D12Resource* resource_scale, ID3D12Resource* resource_zeropoint)
    {
        assert(resource_a);
        assert(resource_out);
        assert(resource_scale);
        assert(resource_zeropoint);
        DML_BUFFER_BINDING input_a_buffer_binding{ resource_a, 0, resource_a->GetDesc().Width };
        DML_BUFFER_BINDING input_b_buffer_binding{ nullptr, 0, 0 };
        DML_BUFFER_BINDING input_c_buffer_binding{ nullptr, 0, 0 };
        DML_BUFFER_BINDING input_scale_buffer_binding{ resource_scale, 0, resource_scale->GetDesc().Width };
        DML_BUFFER_BINDING input_zeropoint_buffer_binding{ resource_zeropoint, 0, resource_zeropoint->GetDesc().Width };

        std::vector<DML_BINDING_DESC> input_bindings;
        input_bindings.reserve(5);
        input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_a_buffer_binding });
        if (resource_b)
        {
            if (input_1_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
            {
                input_b_buffer_binding = { nullptr, 0, 0 };
                input_bindings.push_back({ DML_BINDING_TYPE_NONE, &input_b_buffer_binding });
            }
            else
            {
                input_b_buffer_binding = { resource_b, 0, resource_b->GetDesc().Width };
                input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_b_buffer_binding });
            }
        }
        if (resource_c)
        {
            if (input_2_ && input_2_->GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
            {
                input_c_buffer_binding = { nullptr, 0, 0 };
                input_bindings.push_back({ DML_BINDING_TYPE_NONE, &input_c_buffer_binding });
            }
            else
            {
                input_c_buffer_binding = { resource_c, 0, resource_c->GetDesc().Width };
                input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_c_buffer_binding });
            }
        }

        if (input_1_scale_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_scale_buffer_binding = { nullptr, 0, 0 };
            input_bindings.push_back({ DML_BINDING_TYPE_NONE, &input_scale_buffer_binding });
        }
        else
        {
            input_scale_buffer_binding = { resource_scale, 0, resource_scale->GetDesc().Width };
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_scale_buffer_binding });
        }

        if (input_1_zeropoint_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_zeropoint_buffer_binding = { nullptr, 0, 0 };
            input_bindings.push_back({ DML_BINDING_TYPE_NONE, &input_zeropoint_buffer_binding });
        }
        else
        {
            input_zeropoint_buffer_binding = { resource_zeropoint, 0, resource_zeropoint->GetDesc().Width };
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_zeropoint_buffer_binding });
        }


        std::vector<DML_BINDING_DESC> output_bindings;
        output_bindings.reserve(1);
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };
        output_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &output_buffer_binding });

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_bindings);
    }

    virtual void record_initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, ID3D12Resource* resource_b, ID3D12Resource* resource_c,
        ID3D12Resource* resource_scale, ID3D12Resource* resource_zeropoint)
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
        input_binds.push_back({ nullptr, 0, 0 });  // tensor a

        //tensor b
        if (input_1_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_binds.push_back({ resource_b, 0, resource_b->GetDesc().Width });
        }
        else
        {
            input_binds.push_back({ nullptr, 0, 0 });
        }

        // tensor c 
        if (input_2_ && input_2_->GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            assert(resource_c != nullptr);
            input_binds.push_back({ resource_c, 0, resource_c->GetDesc().Width });
        }
        else if (input_2_)
        {
            input_binds.push_back({ nullptr, 0, 0 });
        }

        // tensor scale
        if (input_1_scale_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_binds.push_back({ resource_scale, 0, resource_scale->GetDesc().Width });
        }
        else
        {
            input_binds.push_back({ nullptr, 0, 0 });
        }

        // tensor zero point
        if (input_1_zeropoint_.GetOutputDesc().flags == DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            input_binds.push_back({ resource_zeropoint, 0, resource_zeropoint->GetDesc().Width });
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

private:
    dml::Graph graph_;
    dml::Expression input_0_;
    dml::Expression input_1_;
    dml::Optional<dml::Expression> input_2_;
    std::vector<dml::Expression> outputs_;

    dml::Expression input_1_dequantized_;
    dml::Expression input_1_scale_;
    dml::Expression input_1_zeropoint_;
};

}

class QuantGemmBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataType quant_dt;
        DataLayout layout;
        DataLayout layout_c = DataLayout::eNCHW;
        ActivationSettings activation{};

        TensorShape shape_a;
        TensorShape shape_b;
        TensorShape shape_c;

        TensorShape shape_scale;
        TensorShape shape_zeropoint;

        float alpha = 1.0f;
        float beta = 1.0f;

        bool b_managed = false;
        bool bs_managed = false;
        bool bz_managed = false;
        bool c_managed = false;
        bool a_transposed = false;
        bool b_transposed = false;

        bool a_quantized = false;
        bool b_quantized = true;
        bool c_quantized = false;
        bool has_zero_point = true;

        bool allow_fp16_computations = false;
        bool use_dnnl_for_reference_calculations = false;

        bool dump_resource = false;

        uint32_t block_size = 32;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_type_cli_option(opts, "--quantize_data_type", params.quant_dt)->default_val(DataType::eUint4);
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();
            add_data_layout_cli_option(opts, "--layout_c", params.layout_c);
            opts->add_option("--activation", params.activation);

            opts->add_option("--shape_a", params.shape_a)->required();
            opts->add_option("--shape_b", params.shape_b);
            opts->add_option("--shape_c", params.shape_c);

            opts->add_flag("--a_transposed", params.a_transposed)->default_val(false);
            opts->add_flag("--b_transposed", params.b_transposed)->default_val(false);
            opts->add_flag("--b_managed", params.b_managed)->default_val(false);
            opts->add_flag("--bs_managed", params.bs_managed)->default_val(false);
            opts->add_flag("--bz_managed", params.bz_managed)->default_val(false);
            opts->add_flag("--c_managed", params.c_managed)->default_val(false);
            opts->add_flag("--allow_fp16_computations", params.allow_fp16_computations);

            opts->add_option("--alpha", params.alpha)->default_val(1.0f);
            opts->add_option("--beta", params.beta)->default_val(1.0f);
            opts->add_option("--block_size", params.block_size)->default_val(32);

            opts->add_flag("--a_quantized", params.a_quantized)->default_val(false);
            opts->add_flag("--b_quantized", params.b_quantized)->default_val(true);
            opts->add_flag("--c_quantized", params.c_quantized)->default_val(false);

            opts->add_flag("--has_zeropoint", params.has_zero_point)->default_val(true);

            opts->add_flag("--dnnl_reference", params.use_dnnl_for_reference_calculations)->default_val(false);

            opts->add_flag("--dump_resource", params.dump_resource);

        }
    };
public:
    QuantGemmBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , dml_device_(dml_device)
        , d3d12_device_(d3d12_device)
    {
        
        uint32_t quant_param_h = params_.shape_b.w;
        uint32_t quant_param_w = params_.shape_b.h / params_.block_size;

        params_.shape_scale = TensorShape(params_.shape_b.n, params_.shape_b.c, quant_param_h, quant_param_w);
        params_.shape_zeropoint = TensorShape(params_.shape_b.n, params_.shape_b.c, quant_param_h, quant_param_w);

        input_data_a_.resize(get_tensor_elements_count(params_.shape_a, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt));
        input_data_b_.resize(static_cast<size_t>(get_tensor_elements_count(params_.shape_b, params_.layout) * get_data_type_bytes_width(params_.quant_dt)));
        if (params_.shape_c.get_dims_count() > 0)
        {
            input_data_c_.resize(get_tensor_elements_count(params_.shape_c, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt));
        }
        input_data_scale_.resize(get_tensor_elements_count(params_.shape_scale, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt));
        input_data_zeropoint_.resize(static_cast<size_t>(get_tensor_elements_count(params_.shape_zeropoint, params_.layout) * get_data_type_bytes_width(params_.quant_dt)));

        std::vector<std::byte> input_data_dequantized_b_;
        input_data_dequantized_b_.resize(get_tensor_elements_count(params_.shape_b, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt));

        assert(params_.shape_a.get_dims_count() == 4);
        assert(params_.shape_b.get_dims_count() == 4);

        assert(!input_data_a_.empty());
        assert(!input_data_b_.empty());

        // populate the gemm input tensors here
        const auto B = get_batch();
        const auto C = get_channels();
        const auto M = get_M();
        const auto K = get_K();
        const auto N = get_N();
        std::cout << std::format("Running [B, C, M, K, N]: [{}, {}, {}, {}, {}]\n", B, C, M, K, N);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-1.0f, 1.0f);
        //std::uniform_real_distribution<float> uniform_distribution(0.001f, 0.01f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_a_);
            if (params_.shape_c.get_dims_count() > 0)
            {
                randomize_linear_container_float(random_generator, uniform_distribution, input_data_c_);
            }
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_dequantized_b_);
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_a_);
            if (params_.shape_c.get_dims_count() > 0)
            {
                randomize_linear_container_half(random_generator, uniform_distribution, input_data_c_);
            }
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_dequantized_b_);
        }
        else
        {
            assert(false && "Unsupported data type in Quant GEMM dispatcher!");
        }

        auto chunk_size = params_.block_size;
        if (params_.dt == DataType::eFp32)
        {
            assert(false && "Unsupported data type in Quant GEMM dispatcher for data initialize!");
            //fill_quantized_data_float_to_uint4(input_data_b_, input_data_dequantized_b_, chunk_size, input_data_scale_, input_data_zeropoint_);
        }
        else
        {
            fill_quantized_data_half_to_uint4(input_data_b_, input_data_dequantized_b_, chunk_size, input_data_scale_, input_data_zeropoint_);
        }

        const auto tensor_input_a_bytes_width = input_data_a_.size();
        const auto tensor_input_b_bytes_width = input_data_b_.size();
        const auto tensor_input_c_bytes_width = input_data_c_.size();
        const auto tensor_input_scale_bytes_width = input_data_scale_.size();
        const auto tensor_input_zeropoint_bytes_width = input_data_zeropoint_.size();

        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = get_tensor_elements_count(out_shape, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_a_bytes_width + tensor_input_b_bytes_width + tensor_input_c_bytes_width + tensor_input_scale_bytes_width + tensor_input_zeropoint_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_a_ = create_buffer(d3d12_device_, tensor_input_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (tensor_input_b_bytes_width > 0)
        {
            input_buffer_b_ = create_buffer(d3d12_device_, tensor_input_b_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        if (tensor_input_c_bytes_width > 0)
        {
            input_buffer_c_ = create_buffer(d3d12_device_, tensor_input_c_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        if (tensor_input_scale_bytes_width > 0)
        {
            input_buffer_scale_ = create_buffer(d3d12_device_, tensor_input_scale_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        if (tensor_input_zeropoint_bytes_width > 0)
        {
            input_buffer_zeropoint_ = create_buffer(d3d12_device_, tensor_input_zeropoint_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }

        output_buffer_ = create_buffer(d3d12_device_, tensor_out_bytes_width,
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
        if (tensor_input_c_bytes_width > 0)
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, input_data_c_.data(), tensor_input_c_bytes_width);
            memcopy_offset += tensor_input_c_bytes_width;
        }
        if (tensor_input_scale_bytes_width > 0)
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, input_data_scale_.data(), tensor_input_scale_bytes_width);
            memcopy_offset += tensor_input_scale_bytes_width;
        }
        if (tensor_input_zeropoint_bytes_width > 0)
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, input_data_zeropoint_.data(), tensor_input_zeropoint_bytes_width);
            memcopy_offset += tensor_input_zeropoint_bytes_width;
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
        if (tensor_input_c_bytes_width > 0)
        {
            cmd_list->CopyBufferRegion(input_buffer_c_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_input_c_bytes_width);
            memcopy_offset += tensor_input_c_bytes_width;
        }
        if (tensor_input_scale_bytes_width > 0)
        {
            cmd_list->CopyBufferRegion(input_buffer_scale_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_input_scale_bytes_width);
            memcopy_offset += tensor_input_scale_bytes_width;
        }
        if (tensor_input_zeropoint_bytes_width > 0)
        {
            cmd_list->CopyBufferRegion(input_buffer_zeropoint_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_input_zeropoint_bytes_width);
            memcopy_offset += tensor_input_zeropoint_bytes_width;
        }

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_a_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (input_buffer_b_)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_b_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        if (input_buffer_c_ && tensor_input_c_bytes_width > 0)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_c_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        if (input_buffer_scale_)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_scale_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        if (input_buffer_zeropoint_)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_zeropoint_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue, ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches, std::size_t reference_dispatch_iterations)
    {
        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = get_tensor_elements_count(out_shape, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt);

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

        std::vector<std::byte> ref_untyped_result;
        if (params_.use_dnnl_for_reference_calculations)
        {
            dnnl_gemm_op::bindings_t bindings{};
            bindings.input_a.data = input_data_a_.data();
            bindings.input_a.dt = params_.dt;
            bindings.input_a.layout = params_.layout;
            bindings.input_a.shape = params_.shape_a;

            bindings.input_b.data = input_data_b_.data();
            bindings.input_b.dt = params_.dt;
            bindings.input_b.layout = params_.layout;
            bindings.input_b.shape = params_.shape_b;

            if (input_buffer_c_)
            {
                bindings.input_c.data = input_data_c_.data();
                bindings.input_c.dt = params_.dt;
                bindings.input_c.layout = params_.layout;
                bindings.input_c.shape = params_.shape_c;
            }

            dnnl_gemm_op::opts_t opts{};
            opts.out_dt = params_.dt;
            opts.out_layout = params_.layout;
            opts.output_shape = get_shape_output();
            opts.force_fp32_accumulator = params_.dt == DataType::eFp16 && !params_.allow_fp16_computations;
            opts.alpha = params_.alpha;
            opts.beta = params_.beta;
            opts.activation = params_.activation;
            opts.a_transposed = params_.a_transposed;
            opts.b_transposed = params_.b_transposed;
            opts.execution_iterations = reference_dispatch_iterations;
            ref_untyped_result = dnnl_gemm_op::gemm(bindings, opts);
        }
        else
        {
            gpu_op::QuantGemm gemm_ref(to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), to_dml_tensor_policy(params_.layout_c),
                params_.shape_a, params_.shape_b, params_.shape_c, get_shape_output(),
                get_shape_quant_param(), get_shape_quant_param(), params_.a_quantized, params_.b_quantized, params_.c_quantized, params_.block_size, to_dml_data_type(params_.quant_dt), params_.has_zero_point,
                params_.a_transposed, false /*params_.b_managed*/, params_.b_transposed, false /*params_.bs_managed*/, false /*params_.bz_managed*/, false /*params_.c_managed*/, params_.alpha, params_.beta,
                params_.allow_fp16_computations, params_.activation,
                dml_device_, d3d12_device_, false, true);
            // bind descriptor heap
            auto descriptor_heap = create_descriptor_heap(d3d12_device_, gemm_ref.get_total_descriptor_count());
            ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
            command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

            gemm_ref.create_binding_tables(descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
            gemm_ref.record_initialize(dml_cmd_recorder_, command_list, input_buffer_b_.Get(), input_buffer_c_.Get(), input_buffer_scale_.Get(), input_buffer_zeropoint_.Get());
            close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

            command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
            gemm_ref.record_execute(dml_cmd_recorder_, command_list, output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get(), input_buffer_c_.Get(), input_buffer_scale_.Get(), input_buffer_zeropoint_.Get());
            close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

            auto readback_buffer_ref = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            command_list->ResourceBarrier(1, &readback_output_barrirer);
            command_list->CopyResource(readback_buffer_ref.Get(), output_buffer_.Get());
            close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

            ref_untyped_result.resize(tensor_out_bytes_width);
            void* readback_mapped_ptr_ref = nullptr;
            readback_buffer_ref->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr_ref));
            std::memcpy(ref_untyped_result.data(), readback_mapped_ptr_ref, ref_untyped_result.size());
            readback_buffer_ref->Unmap(0, nullptr);
        }

        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, ref_untyped_result, 0.05f, print_mismatches);
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
    TensorShape get_shape_output() const
    {
        TensorShape ret{};
        ret.n = get_batch();
        ret.c = get_channels();
        ret.h = get_M();
        ret.w = get_N();
        return ret;
    }

    TensorShape get_shape_quant_param() const
    {
        TensorShape ret{};
        ret.n = get_batch();
        ret.c = get_channels();
        ret.h = params_.shape_b.w;
        ret.w = params_.shape_b.h / params_.block_size;
        return ret;
    }
        
    std::uint32_t get_batch() const
    {
        return params_.shape_a.n;
    }

    std::uint32_t get_channels() const
    {
        return params_.shape_a.c;
    }

    std::uint32_t get_M() const
    {
        return params_.shape_a.h;
    }

    std::uint32_t get_K() const
    {
        return params_.shape_a.w;
    }

    std::uint32_t get_N() const
    {
        return params_.shape_b.w;
    }

protected:
    create_params_t params_;
    ID3D12Device* d3d12_device_;
    IDMLDevice* dml_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<std::byte> input_data_a_;
    std::vector<std::byte> input_data_b_;
    std::vector<std::byte> input_data_c_;

    std::vector<std::byte> input_data_scale_;
    std::vector<std::byte> input_data_zeropoint_;

    ComPtr<ID3D12Resource> input_buffer_a_;
    ComPtr<ID3D12Resource> input_buffer_b_;
    ComPtr<ID3D12Resource> input_buffer_c_;

    ComPtr<ID3D12Resource> input_buffer_scale_;
    ComPtr<ID3D12Resource> input_buffer_zeropoint_;

    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class QuantGemmDmlDispatcher : public QuantGemmBaseDispatcher
{
public:
    QuantGemmDmlDispatcher(create_params_t&& params, bool allow_descriptors_volatile, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : QuantGemmBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , quantgemm_(to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), to_dml_tensor_policy(params_.layout_c), params_.shape_a, params_.shape_b, params_.shape_c, get_shape_output(),
            get_shape_quant_param(), get_shape_quant_param(), params_.a_quantized, params_.b_quantized, params_.c_quantized, params_.block_size, to_dml_data_type(params_.quant_dt), params_.has_zero_point,
            params_.a_transposed, params_.b_managed, params_.b_transposed, params_.bs_managed, params_.bz_managed, params_.c_managed,
            params_.alpha, params_.beta, params_.allow_fp16_computations, params_.activation,
            dml_device, d3d12_device, allow_descriptors_volatile, false)
    {
        if (params_.c_managed)
        {
            assert(params_.shape_c.get_dims_count() > 0 && "Cant use c_managed if shape of c is not defined!");
        }
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return quantgemm_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        quantgemm_.create_binding_tables(cpu_handle, gpu_handle);
        
        quantgemm_.record_initialize(dml_cmd_recorder_, cmd_list, input_buffer_b_.Get(), input_buffer_c_.Get(), input_buffer_scale_.Get(), input_buffer_zeropoint_.Get());
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        quantgemm_.record_execute(dml_cmd_recorder_, cmd_list,
                output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get(), input_buffer_c_.Get(), input_buffer_scale_.Get(), input_buffer_zeropoint_.Get());
    }

private:
    gpu_op::QuantGemm quantgemm_;
};

class GemmBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        GemmType type;
        DataType dt;
        DataLayout layout;
        DataLayout layout_c = DataLayout::eNCHW;
        ActivationSettings activation{};

        TensorShape shape_a;
        TensorShape shape_b;
        TensorShape shape_c;

        TensorShape shape_scale;
        TensorShape shape_zeropoint;

        float alpha = 1.0f;
        float beta = 1.0f;

        bool b_managed = false;
        bool c_managed = false;
        bool a_transposed = false;
        bool b_transposed = false;

        bool allow_fp16_computations = false;
        bool use_dnnl_for_reference_calculations = false;

        bool dump_resource = false;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();
            add_data_layout_cli_option(opts, "--layout_c", params.layout_c);
            opts->add_option("--activation", params.activation);

            opts->add_option("--shape_a", params.shape_a)->required();
            opts->add_option("--shape_b", params.shape_b); 
            opts->add_option("--shape_c", params.shape_c); 

            opts->add_flag("--a_transposed", params.a_transposed)->default_val(false);
            opts->add_flag("--b_transposed", params.b_transposed)->default_val(false);
            opts->add_flag("--b_managed", params.b_managed)->default_val(false);
            opts->add_flag("--c_managed", params.c_managed)->default_val(false);
            opts->add_flag("--allow_fp16_computations", params.allow_fp16_computations);

            opts->add_option("--alpha", params.alpha)->default_val(1.0f);
            opts->add_option("--beta", params.beta)->default_val(1.0f);

            opts->add_flag("--dnnl_reference", params.use_dnnl_for_reference_calculations)->default_val(false);

            opts->add_option("--gemm_type", params.type, "Name of the type of GEMM to run.")
                ->check(CLI::IsMember({ GemmType::GemmType_AB, GemmType::GemmType_QK_QKV, GemmType::GemmType_SV_S_QKV, GemmType::GemmType_QK_Q_KV, GemmType::GemmType_SV_S_KV }))->
                transform(CLI::Transformer(std::map<std::string, GemmType>{
                    { "ab", GemmType::GemmType_AB },
                    { "qk_qkv", GemmType::GemmType_QK_QKV },
                    { "sv_qkv", GemmType::GemmType_SV_S_QKV },
                    { "qk_q_kv", GemmType::GemmType_QK_Q_KV },
                    { "sv_s_kv", GemmType::GemmType_SV_S_KV },
            }, CLI::ignore_case))->required();

            opts->add_flag("--dump_resource", params.dump_resource);

        }
    };
public:

    GemmBaseDispatcher(const create_params_t& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(params)
        , dml_cmd_recorder_(dml_cmd_recorder)
        , dml_device_(dml_device)
        , d3d12_device_(d3d12_device)
    {
        input_data_a_.resize(get_tensor_elements_count(params_.shape_a, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt));
        input_data_b_.resize(get_tensor_elements_count(params_.shape_b, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt));
        input_data_c_.resize(get_tensor_elements_count(params_.shape_c, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt));

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
        std::uniform_real_distribution<float> uniform_distribution(-1.0f, 1.0f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_a_);
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_b_);
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_c_);
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_a_);
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_b_);
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_c_);
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        const auto tensor_input_a_bytes_width = input_data_a_.size();
        const auto tensor_input_b_bytes_width = input_data_b_.size();
        const auto tensor_input_c_bytes_width = input_data_c_.size();


        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = get_tensor_elements_count(out_shape, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_a_bytes_width + tensor_input_b_bytes_width + tensor_input_c_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_a_ = create_buffer(d3d12_device, tensor_input_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (tensor_input_b_bytes_width > 0)
        {
            input_buffer_b_ = create_buffer(d3d12_device, tensor_input_b_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        if (tensor_input_c_bytes_width > 0)
        {
            input_buffer_c_ = create_buffer(d3d12_device, tensor_input_c_bytes_width,
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
        if (tensor_input_c_bytes_width > 0)
        {
            std::memcpy(upload_mapped_ptr + memcopy_offset, input_data_c_.data(), tensor_input_c_bytes_width);
            memcopy_offset += tensor_input_c_bytes_width;
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
        if (tensor_input_c_bytes_width > 0)
        {
            cmd_list->CopyBufferRegion(input_buffer_c_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_input_c_bytes_width);
            memcopy_offset += tensor_input_c_bytes_width;
        }

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_a_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (input_buffer_b_)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_b_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        if (input_buffer_c_)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_c_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }

        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue, ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches, std::size_t reference_dispatch_iterations)
    {
        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = get_tensor_elements_count(out_shape, params_.layout) * (std::uint8_t)get_data_type_bytes_width(params_.dt);

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

        std::vector<std::byte> ref_untyped_result;
        if (params_.use_dnnl_for_reference_calculations)
        {
            dnnl_gemm_op::bindings_t bindings{};
            bindings.input_a.data = input_data_a_.data();
            bindings.input_a.dt = params_.dt;
            bindings.input_a.layout = params_.layout;
            bindings.input_a.shape = params_.shape_a;

            bindings.input_b.data = input_data_b_.data();
            bindings.input_b.dt = params_.dt;
            bindings.input_b.layout = params_.layout;
            bindings.input_b.shape = params_.shape_b;

            if (input_buffer_c_)
            {
                bindings.input_c.data = input_data_c_.data();
                bindings.input_c.dt = params_.dt;
                bindings.input_c.layout = params_.layout;
                bindings.input_c.shape = params_.shape_c;
            }

            dnnl_gemm_op::opts_t opts{};
            opts.out_dt = params_.dt;
            opts.out_layout = params_.layout;
            opts.output_shape = get_shape_output();
            opts.force_fp32_accumulator = params_.dt == DataType::eFp16 && !params_.allow_fp16_computations;
            opts.alpha = params_.alpha;
            opts.beta = params_.beta;
            opts.activation = params_.activation;
            opts.a_transposed = params_.a_transposed;
            opts.b_transposed = params_.b_transposed;
            opts.execution_iterations = reference_dispatch_iterations;
            ref_untyped_result = dnnl_gemm_op::gemm(bindings, opts);
        }
        else   
        {
            gpu_op::Gemm gemm_ref(params_.type, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), to_dml_tensor_policy(params_.layout_c),
                params_.shape_a, params_.shape_b, params_.shape_c, get_shape_output(),
                params_.a_transposed, false /*params_.b_managed*/, params_.b_transposed, false /*params_.c_managed*/, params_.alpha, params_.beta,
                params_.allow_fp16_computations, params_.activation,
                dml_device_, d3d12_device_, false, true);
            // bind descriptor heap
            auto descriptor_heap = create_descriptor_heap(d3d12_device_, gemm_ref.get_total_descriptor_count());
            ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
            command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

            gemm_ref.create_binding_tables(descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
            gemm_ref.record_initialize(dml_cmd_recorder_, command_list, input_buffer_b_.Get(), input_buffer_c_.Get());
            close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

            command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
            gemm_ref.record_execute(dml_cmd_recorder_, command_list, output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get(), input_buffer_c_.Get());
            close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

            auto readback_buffer_ref = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            command_list->ResourceBarrier(1, &readback_output_barrirer);
            command_list->CopyResource(readback_buffer_ref.Get(), output_buffer_.Get());
            close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

            ref_untyped_result.resize(tensor_out_bytes_width);
            void* readback_mapped_ptr_ref = nullptr;
            readback_buffer_ref->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr_ref));
            std::memcpy(ref_untyped_result.data(), readback_mapped_ptr_ref, ref_untyped_result.size());
            readback_buffer_ref->Unmap(0, nullptr);
        }

        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, ref_untyped_result, 0.05f, print_mismatches);
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
    GemmDmlDispatcher(create_params_t&& params, bool allow_descriptors_volatile, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : GemmBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , gemm_(params_.type, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), to_dml_tensor_policy(params_.layout_c), params_.shape_a, params_.shape_b, params_.shape_c, get_shape_output(), 
            params_.a_transposed, params_.b_managed, params_.b_transposed, params_.c_managed,
            params_.alpha, params_.beta, params_.allow_fp16_computations, params_.activation,
            dml_device, d3d12_device, allow_descriptors_volatile, false)
    {
        if (params_.c_managed)
        {
            assert(params_.shape_c.get_dims_count() > 0 && "Cant use c_managed if shape of c is not defined!");
        }
    }


    std::uint32_t get_total_descriptor_count() override
    {
        return gemm_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        gemm_.create_binding_tables(cpu_handle, gpu_handle);
        gemm_.record_initialize(dml_cmd_recorder_, cmd_list, input_buffer_b_.Get(), input_buffer_c_.Get());
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        gemm_.record_execute(dml_cmd_recorder_, cmd_list,
                output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get(), input_buffer_c_.Get());
    }

private:
    gpu_op::Gemm gemm_;
};


class GemmUmdD3d12Dispatcher : public GemmBaseDispatcher
{
public:
    struct gemm_umdd3d12_params_t
    {
        std::uint32_t verbose_mode = 0;  // 0: disabled; 1: execution; 2: creation and execution
        bool verbose_dump_to_file = false;
        bool cache_blob = false;

        inline static void add_cli_options(CLI::App* opts, gemm_umdd3d12_params_t& params)
        {
            opts->add_option("--verbose_mode", params.verbose_mode)->default_val(0);
            opts->add_flag("--verbose_file", params.verbose_dump_to_file)->default_val(false);
            opts->add_flag("--cache_blob", params.cache_blob, "Use to test persistent cache blob.")->default_val(false);
        }
    };
public:
    GemmUmdD3d12Dispatcher(const create_params_t& params, const gemm_umdd3d12_params_t& umdd3d12_param, IntelExtension& intc_ext, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : GemmBaseDispatcher(params, d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , device_(d3d12_device, intc_ext.get_info())
        , dnnl_engine_(dnnl::iumd_interop::make_engine(&device_))
    {
        using namespace dnnl_utils;

        dnnl::set_verbose(umdd3d12_param.verbose_mode);
        
        //input_a_memory_desc_ = to_dnnl_mem_desc(params_.a_transposed ? TensorShape{ params_.shape_a.n, params_.shape_a.c, params_.shape_a.w, params_.shape_a.h } : params_.shape_a, params_.layout, params_.dt);
        input_a_memory_desc_ = to_dnnl_mem_desc(params_.shape_a, params_.layout, params_.dt);
        if (params_.a_transposed)
        {
            input_a_memory_desc_ = convert_to_ncwh_format(input_a_memory_desc_);
        }
        // const auto input_b_memory_desc = to_dnnl_mem_desc(params_.b_transposed ? TensorShape{ params_.shape_b.n, params_.shape_b.c, params_.shape_b.w, params_.shape_b.h } : params_.shape_b, params_.b_managed ? DataLayout::eWeightsLayoutStart : params_.layout, params_.dt);
        if (params_.b_managed)
        {
            input_b_memory_desc_ = to_dnnl_mem_desc(params_.shape_b, DataLayout::eWeightsLayoutStart, params_.dt);
        }
        else
        {
            input_b_memory_desc_ = to_dnnl_mem_desc(params_.shape_b, params_.layout, params_.dt);
            if (params_.b_transposed)
            {
                input_b_memory_desc_ = convert_to_ncwh_format(input_b_memory_desc_);
            }
        }
        output_memory_desc_ = to_dnnl_mem_desc(get_shape_output(), params_.layout, params_.dt);

        if (has_c_tensor())
        {
            input_c_memory_desc_.emplace(to_dnnl_mem_desc(params_.shape_c, params_.layout, params_.dt));
        }
        
        const dnnl::primitive_attr attr = [this]()
        {
            // create a post-op with relu
            dnnl::post_ops ops;
            dnnl::primitive_attr attr;

            // sanity check
            assert(attr.get_scratchpad_mode() == dnnl::scratchpad_mode::library);
            // set scratchpad mode to user provided
            attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

            const bool force_fp32_accu = params_.dt == DataType::eFp16 && !params_.allow_fp16_computations;
            if (force_fp32_accu)
            {
                attr.set_accumulation_mode(dnnl::accumulation_mode::strict);
            }

            // alpha
            if (params_.alpha != 1.0f || has_beta_scaling_factors())
            {
               ops.append_eltwise(dnnl::algorithm::eltwise_linear, has_beta_scaling_factors() ? params_.alpha / params_.beta : params_.alpha , 0.0f);
            }

            if (has_c_tensor())
            {
                ops.append_binary(dnnl::algorithm::binary_add, input_c_memory_desc_.value());
            }

            if (has_beta_scaling_factors())
            {
                ops.append_eltwise(dnnl::algorithm::eltwise_linear, params_.beta, 0.0f);
            }

            if (params_.activation.type != ActivationType::eUnknown)
            {
                ops.append_eltwise(to_dnnl_activation_type(params_.activation.type), params_.activation.alpha, params_.activation.beta);
                attr.set_post_ops(ops);
            }

            attr.set_post_ops(ops);
            return attr;
        }();

        dnnl::matmul::primitive_desc matmul_desc(dnnl_engine_,
            input_a_memory_desc_,
            input_b_memory_desc_,
            output_memory_desc_,
            attr
        );
        std::cout << "dnnl-umd kernel impl: " << matmul_desc.impl_info_str() << std::endl;

        input_b_memory_desc_ = matmul_desc.query_md(dnnl::query::weights_md, 0);
        const auto persistent_resource_size = [&]()
        {
            std::size_t ret = 0ull;

            if (params_.b_managed)
            {
                ret += input_b_memory_desc_.get_size();
            }

            if (params_.c_managed)
            {
                assert(!"params_.c_managed is nt not tested option, most likely bugs hidden somewhere!");
                assert(input_c_memory_desc_.has_value());
                ret += input_c_memory_desc_->get_size();
            }
            return ret;
        }();

        if (persistent_resource_size != 0)
        {
            persistent_buffer_ = create_buffer(d3d12_device, persistent_resource_size,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }

        assert(matmul_desc.query_s64(dnnl::query::memory_consumption_s64) == 0);  // we provide scratchpad, so sanity check that primitive does not require any "hidden" memory
        scratchpad_memory_desc_.emplace(matmul_desc.query_md(dnnl::query::scratchpad_md));
        const auto temporary_resoruce_size = [&]()
        {
            return scratchpad_memory_desc_->get_size();
        }(); 
        if (temporary_resoruce_size != 0)
        {
            const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            temporary_buffer_ = create_buffer(d3d12_device, temporary_resoruce_size,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }

        // create convolution primitive


		std::ifstream in_key_file("onednn_persistent_cache.key", std::ofstream::in | std::ifstream::binary);
		std::ifstream in_value_file("onednn_persistent_cache.value", std::ofstream::in | std::ifstream::binary);
		std::vector<std::uint8_t> buffer_key;
		std::vector<std::uint8_t> buffer_value;
		const auto conv_blob_key = matmul_desc.get_cache_blob_id();
		if (umdd3d12_param.cache_blob && in_key_file.is_open())
		{
			buffer_key = std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in_key_file), {});
		}
		if (buffer_key == conv_blob_key)
		{
			std::cout << "Found persistent cache blob files. Using them to create gemm primitive!" << std::endl;
			assert(in_value_file.is_open());  // Proper file  with key value exists, but file with cache blob (value) does not exist. Delete file with key and rerun application.
			buffer_value = std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in_value_file), {});
		}
		const auto t0 = std::chrono::high_resolution_clock::now();
		if (buffer_value.empty())
		{
            gemm_ = dnnl::matmul(matmul_desc);
		}
		else
		{
            gemm_ = dnnl::matmul(matmul_desc, buffer_value);
		}
		const auto t1 = std::chrono::high_resolution_clock::now();
		const auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
		std::cout << "Primitive create time: " << diff << std::endl;

		if (umdd3d12_param.cache_blob && buffer_value.empty())
		{
			std::cout << "Storing persistent cache blob files for." << std::endl;
			auto store_binary_data_to_file = [](const auto& file_name, const auto& data)
				{
					std::ofstream out_file(file_name, std::ofstream::out | std::ofstream::binary);
					std::copy(data.begin(), data.end(), std::ostream_iterator<std::uint8_t>(out_file));
					out_file.close();
				};
			const auto cache_blob_id = matmul_desc.get_cache_blob_id();
			store_binary_data_to_file("onednn_persistent_cache.key", cache_blob_id);

			const auto cache_blob = gemm_.get_cache_blob();
			store_binary_data_to_file("onednn_persistent_cache.value", cache_blob);
		}

        if (params_.b_managed)
        {
            dnnl::reorder::primitive_desc reorder_desc(dnnl_engine_, to_dnnl_mem_desc(params_.shape_b, params_.layout, params_.dt), dnnl_engine_, input_b_memory_desc_);
            reorder_input_b_ = dnnl::reorder(reorder_desc);
        }

        if (params_.c_managed)
        {
            assert(input_c_memory_desc_.has_value());
            // its just a copy
            dnnl::reorder::primitive_desc reorder_desc(dnnl_engine_, input_c_memory_desc_.value(), dnnl_engine_, input_c_memory_desc_.value());
            reorder_input_c_ = dnnl::reorder(reorder_desc);
        }

    }

    std::uint32_t get_total_descriptor_count()override
    {
        // allocate enough descriptor upfront
        return 50u;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        ID3D12GraphicsCommandList4* cmd_list4 = nullptr;
        throw_if_failed(cmd_list->QueryInterface(&cmd_list4), "cant cast d3d12 device to ID3D12Device5");
        iumd::custom_metacommand::UmdD3d12CommandList cmd(cmd_list4);
        dnnl::stream stream = dnnl::iumd_interop::make_stream(dnnl_engine_, &cmd);

        base_cpu_handle_ = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
        base_gpu_handle_ = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

        if (!reorder_input_b_ && !reorder_input_c_ /* && !copy_alpha_shader_*/)
        {
            // early exit, as no reordering needed or copy shader for alpha value not needed
            return;
        }

        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(4);
        if (persistent_buffer_)
        {
            resources_list.push_back({ DescType::eUav, persistent_buffer_.Get() });
        }
        if (reorder_input_b_)
        {
            resources_list.push_back({ DescType::eUav, input_buffer_b_.Get() });
        }
        if (reorder_input_c_)
        {
            resources_list.push_back({ DescType::eUav, input_buffer_c_.Get() });
        }
        const auto gpu_handles = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle_, base_gpu_handle_);
        
        std::size_t rsc_idx = 0;
        auto umd_persistent_mem = persistent_buffer_ ? iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[rsc_idx++]) : iumd::custom_metacommand::UmdD3d12Memory{};
        std::size_t persistent_mem_offset = 0;

        // weights reorder
        if (reorder_input_b_)
        {  
            auto umd_input_mem = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[rsc_idx++]);

            auto input_memory_desc_ = dnnl_utils::to_dnnl_mem_desc(params_.shape_b, params_.layout, params_.dt);
            if (params_.b_transposed)
            {
                input_memory_desc_ = dnnl_utils::convert_to_ncwh_format(input_memory_desc_);
            }
            dnnl::memory input_memory = create_dnnl_memory(input_memory_desc_, umd_input_mem);
            dnnl::memory reorder_memory = create_dnnl_memory(input_b_memory_desc_, umd_persistent_mem, persistent_mem_offset);

            std::unordered_map<int, dnnl::memory> args;
            args.insert({ DNNL_ARG_SRC, input_memory });
            args.insert({ DNNL_ARG_DST, reorder_memory });

            reorder_input_b_.execute(stream, args);
            persistent_mem_offset += input_b_memory_desc_.get_size();

        }

        // weights reorder
        if (reorder_input_c_)
        {
            assert(input_c_memory_desc_.has_value());
            auto umd_input_mem = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[rsc_idx++]);

            dnnl::memory input_memory = create_dnnl_memory(input_c_memory_desc_.value(), umd_input_mem);
            dnnl::memory reorder_memory = create_dnnl_memory(input_c_memory_desc_.value(), umd_persistent_mem, persistent_mem_offset);

            std::unordered_map<int, dnnl::memory> args;
            args.insert({ DNNL_ARG_SRC, input_memory });
            args.insert({ DNNL_ARG_DST, reorder_memory });

            reorder_input_c_.execute(stream, args);
        }
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ DescType::eUav, input_buffer_a_.Get() });
        if (persistent_buffer_)
        {
            resources_list.push_back({ DescType::eUav, persistent_buffer_.Get() });
        }
        if (!reorder_input_b_)
        {
            resources_list.push_back({ DescType::eUav, input_buffer_b_.Get()});
        }

        resources_list.push_back({ DescType::eUav, output_buffer_.Get() });
        if (input_buffer_c_ && !reorder_input_c_)
        {
            resources_list.push_back({ DescType::eUav, input_buffer_c_.Get() });
        }
        if (temporary_buffer_)
        {
            resources_list.push_back({ DescType::eUav, temporary_buffer_.Get() });
        }
        const auto gpu_handles = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle_, base_gpu_handle_);

        std::size_t res_idx = 0;
        auto umd_input_a_memory_ = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
        auto umd_persitent_memory = persistent_buffer_ ? iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]) : iumd::custom_metacommand::UmdD3d12Memory();
        auto umd_input_b_memory_ = reorder_input_b_ ? umd_persitent_memory : iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
        auto umd_output_memory_ = iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
        auto umd_input_c_memory_ = [&]()
        {
            if (input_buffer_c_ && !reorder_input_c_)
            {
                return iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]);
            }
            else if (reorder_input_c_)
            {
                return umd_persitent_memory;
            }
            return iumd::custom_metacommand::UmdD3d12Memory{};
        }();

        //auto umd_alpha_beta_memory_ = umd_persitent_memory;
        auto umd_scratchpad_memory_ = temporary_buffer_ ? iumd::custom_metacommand::UmdD3d12Memory(gpu_handles[res_idx++]) : iumd::custom_metacommand::UmdD3d12Memory();

        // stream is created in execute(...), because in MetaCommand cmd list object can be different from execute-to-execute
        ID3D12GraphicsCommandList4* cmd_list4 = nullptr;
        throw_if_failed(cmd_list->QueryInterface(&cmd_list4), "cant cast d3d12 device to ID3D12Device5");
        iumd::custom_metacommand::UmdD3d12CommandList cmd(cmd_list4);
        dnnl::stream stream = dnnl::iumd_interop::make_stream(dnnl_engine_, &cmd);
        
        // memory resources are created in execute(...), because in MetaCommand these objects can be different from execute-to-execute
        dnnl::memory input_memory = create_dnnl_memory(input_a_memory_desc_, umd_input_a_memory_);

        std::size_t persistent_mem_offset = 0;
     
        dnnl::memory input_b_memory = [&]()
        {
            std::size_t offset = 0ull;
            return create_dnnl_memory(input_b_memory_desc_, umd_input_b_memory_, offset);
        }();

        dnnl::memory input_c_memory = [&]()
        {
            if (has_c_tensor())
            {
                return create_dnnl_memory(input_c_memory_desc_.value(), umd_input_c_memory_, reorder_input_c_ ? persistent_mem_offset : 0ull);
            }
            return dnnl::memory{};
        }();

        std::optional<dnnl::memory> scratchpad_memory;
        if(has_scratchpad_tensor())
        {
            scratchpad_memory.emplace(create_dnnl_memory(scratchpad_memory_desc_.value(), umd_scratchpad_memory_));
        }

        dnnl::memory output_memory = create_dnnl_memory(output_memory_desc_, umd_output_memory_);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, input_memory });
        args.insert({ DNNL_ARG_WEIGHTS, input_b_memory });
        std::size_t post_ops_idx = 0ull;
        if(params_.alpha!=1 || has_beta_scaling_factors())
        {
            post_ops_idx++;
        }

        if (input_c_memory)
        {
            args.insert({ static_cast<int>(DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_ops_idx) | DNNL_ARG_SRC_1), input_c_memory });
            post_ops_idx++;
        }

        if(has_beta_scaling_factors())
        {
            post_ops_idx++;
        }

        if (scratchpad_memory_desc_)
        {
            args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_memory.value() });
        }

        args.insert({ DNNL_ARG_DST, output_memory });

        gemm_.execute(stream, args);
    }


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

             // Assuming data_out now contains the float data
            float* float_ptr = reinterpret_cast<float*>(data_out.data());
            size_t num_floats = data_out.size() / sizeof(float);
            std::ofstream file(file_name, std::ios::out); // Open in text mode; use std::ios::binary for binary mode
            for (size_t i = 0; i < num_floats; ++i) {
                file << float_ptr[i] << std::endl; // Write in text format; for binary, use file.write(reinterpret_cast<const char*>(&float_ptr[i]), sizeof(float));
            }
            file.close();
        };

        if(params_.dump_resource)
        {
            dump_buffer_to_file(persistent_buffer_, "umd_gemm_data.txt");
        }
        
        const auto ret = GemmBaseDispatcher::validate_conformance(command_queue, command_allocator, command_list, print_mismatche, reference_dispatch_iterations);
        return ret;
    }
private:
    dnnl::memory create_dnnl_memory(const auto& desc, auto& umd_mem, std::size_t offset = 0)
    {
        return dnnl::iumd_interop::make_memory(desc, dnnl_engine_, &umd_mem, offset);
    };

private:
    bool has_beta_scaling_factors() const
    {
        // OneDNNL has a bit different GEMM API defintion: alpha*A*B + beta*C
        // DirectML: beta*(alpha/beta*(A*B)+C))
        // we will pass alpha as alpha/beta if beta value is effective
         return params_.beta != 0.0f && params_.beta != 1.0f;
    }

    bool has_c_tensor() const
    {
        return input_buffer_c_ != nullptr;
    }

    bool has_scratchpad_tensor() const
    {
        return scratchpad_memory_desc_.has_value();
    }

private:
    iumd::custom_metacommand::UmdD3d12Device device_;
    dnnl::engine dnnl_engine_;

    dnnl::matmul gemm_;
    dnnl::reorder reorder_input_b_;
    dnnl::reorder reorder_input_c_;

    iumd::IUMDPipelineStateObject::Ptr copy_alpha_shader_ = nullptr;

    dnnl::memory::desc input_a_memory_desc_;
    dnnl::memory::desc input_b_memory_desc_;
    dnnl::memory::desc output_memory_desc_;
    std::optional<dnnl::memory::desc> input_c_memory_desc_;
    std::optional<dnnl::memory::desc> scratchpad_memory_desc_;

    ComPtr<ID3D12Resource> temporary_buffer_;
    ComPtr<ID3D12Resource> persistent_buffer_;  // ToDo: input_b can be managed, than it should be used for that

    CD3DX12_CPU_DESCRIPTOR_HANDLE base_cpu_handle_;
    CD3DX12_GPU_DESCRIPTOR_HANDLE base_gpu_handle_;
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
        bool use_stateless = false;

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
            opts->add_flag("--use_stateless", params.use_stateless)->default_val(false);

            opts->add_option("--tile_m", params.tile_m);
            opts->add_option("--tile_k", params.tile_k);
            opts->add_option("--tile_n", params.tile_n);

            opts->add_option("--lws_x", params.lws[0]);
            opts->add_option("--lws_y", params.lws[1]);
            opts->add_option("--lws_z", params.lws[2]);

            opts->add_option("--slice_k", params.slice_k);
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

        const auto B = get_batch();
        const auto C = get_channels();
        const auto M = get_M();
        const auto K = get_K();
        const auto N = get_N();

        if (params_.type == GemmType::GemmType_SV_S_QKV)
        {
#if 0
            cm_params_.large_grf = false; // 128 "small" grf 
            cm_params_.tile_n = N == 40 ? 40 : 80;
            cm_params_.tile_m = cm_params_.tile_n == 40 ? 16 : 8;  // tile tile_n is big then we need to have tile_m smaller to not spill reigsters
            cm_params_.tile_k = ((K > 64) && (K % 16 == 0)) ? 16 : 8;

            assert(K % cm_params_.tile_k == 0);
            assert(N % cm_params_.tile_n == 0);
            assert(M % cm_params_.tile_m == 0);

            cm_params_.slice_k = 1;
            cm_params_.lws[0] = cm_params_.tile_k;
            cm_params_.lws[2] = cm_params_.slice_k;
#endif
        }
        else if (params_.type == GemmType::GemmType_QK_QKV)
        {
#if 0
            cm_params_.large_grf = true;
            cm_params_.tile_k = K == 40 ? 40 : 80;
            cm_params_.tile_n = 64;
            cm_params_.tile_m = M <= 256 ? 16 : 8;

            assert(K % cm_params_.tile_k == 0);
            assert(N % cm_params_.tile_n == 0);
            assert(M % cm_params_.tile_m == 0);

            cm_params_.slice_k = K / cm_params_.tile_k;
            cm_params_.lws[2] = cm_params_.slice_k;

            if (cm_params_.slice_k == 1)
            {
                cm_params_.lws[1] = 16;
            }
#endif
            //cm_params_.lws[0] = 32;
            cm_params_.lws[1] = 16;
        }
        else if(params_.type == GemmType::GemmType_QK_Q_KV)
        {
            cm_params_.large_grf = true;
            cm_params_.tile_k = K;
            cm_params_.tile_n = N; // SD1.5: 77
            cm_params_.tile_m = 8;
            
            assert(K % cm_params_.tile_k == 0);
            assert(N % cm_params_.tile_n == 0);
            assert(cm_params_.tile_n == 77 || (is_power_of_2(cm_params_.tile_n) && cm_params_.tile_n <= 128));
            assert(M % cm_params_.tile_m == 0);

            cm_params_.slice_k = 1;
            cm_params_.lws[2] = 1;
        }
        else if (params_.type == GemmType::GemmType_SV_S_KV)
        {
            cm_params_.large_grf = true;
            cm_params_.tile_k = K; // SD1.5: 77
            cm_params_.tile_n = N == 40 ? 40 : 80;
            cm_params_.tile_m = 8;

            assert(K % cm_params_.tile_k == 0);
            assert(cm_params_.tile_k == 77 || (is_power_of_2(cm_params_.tile_k) && cm_params_.tile_k <= 128));
            assert(N % cm_params_.tile_n == 0);
            assert(M % cm_params_.tile_m == 0);

            cm_params_.slice_k = 1;
            cm_params_.lws[2] = 1;
            cm_params_.lws[0] = 1;
        }

        assert(cm_params_.tile_m > 0);
        assert(cm_params_.tile_k > 0);
        assert(cm_params_.tile_n > 0);
        assert(cm_params_.slice_k > 0);

        // root signature
        {
            if(cm_params_.use_stateless)
            {
                root_signature_ = create_root_signature_stateless(d3d12_device, 3);
            }
            else
            {
                std::vector< DescType> desc_list =
                {
                    DescType::eSrv, // input a
                    DescType::eSrv, // input b
                    DescType::eUav // output
                };
                root_signature_ = create_root_signature(d3d12_device, desc_list);
            }
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
        add_define("FUSE_SOFTMAX", false);

        // kernel compilation
        const auto dump_asm_str = cm_params_.dump_asm ? " -mdump_asm" : "";
        const auto large_grf_str = cm_params_.large_grf ? " -Qxcm_doubleGRF" : "";
        const auto print_reg_str = cm_params_.print_reg_usage ? " -mCM_printregusage" : "";
        const auto lws_x = " -DLWS_SIZE_X=" + std::to_string(cm_params_.lws[0]);
        const auto lws_y = " -DLWS_SIZE_Y=" + std::to_string(cm_params_.lws[1]);
        const auto lws_z = " -DLWS_SIZE_Z=" + std::to_string(cm_params_.lws[2]);

        auto build_options_final = " -I \" \" " + build_options + dump_asm_str + large_grf_str + print_reg_str + lws_x + lws_y + lws_z;

        if(cm_params_.use_stateless)
        {
            build_options_final += " -DCM_STATELESS=1";
        }

        if (cm_params_.dump_asm)
        {
            std::cout << build_options_final << std::endl;
        }

        auto kernel_source_content = [&](GemmType type)
        {
            std::string path = "";
            switch (type)
            {
            case GemmType::GemmType_AB: 
                path = "gemm_nchw_fp16.cpp"; 
                if(cm_params_.use_stateless)
                {
                    path = "gemm_nchw_fp16_stateless.cpp";
                }
                break;
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

        if(cm_params_.use_stateless) // if in stateless mode, all buffers are set to UAV accoring to its GPU address
        {  
            // the root parameter index should start from 1, in order to skip first constant buffer.
            cmd_list->SetComputeRootUnorderedAccessView(1, input_buffer_a_->GetGPUVirtualAddress());
            cmd_list->SetComputeRootUnorderedAccessView(2, input_buffer_b_->GetGPUVirtualAddress());
            cmd_list->SetComputeRootUnorderedAccessView(3, output_buffer_->GetGPUVirtualAddress());
        } 
        else 
        {
            uint32_t root_index = 1; // start with 1, beacuse Cross compiler CM driver path needs that
            for (uint32_t i = 0; i < gpu_handles_.size(); i++)
            {
                const auto gpu_heap_handle = gpu_handles_[i];
                cmd_list->SetComputeRootDescriptorTable(root_index++, gpu_heap_handle);
            }

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

    bool is_needing_descriptor_heap() override {
        return !cm_params_.use_stateless;
    }

    private:
        std::vector<std::uint32_t> get_gws() const
        {
            std::uint32_t gws_x = 0;
            std::uint32_t gws_y = 0;
            std::uint32_t gws_z = 0;
            if (params_.type == GemmType::GemmType_SV_S_QKV)
            {
                gws_x = get_M() / cm_params_.tile_m;
                gws_y = get_N() / cm_params_.tile_n;
                gws_z = get_batch() * get_channels() * cm_params_.slice_k;
            }
            else if (params_.type == GemmType::GemmType_QK_QKV)
            {
                gws_x = get_N() / cm_params_.tile_n;  // n first
                gws_y = get_M() / cm_params_.tile_m;  // m second
                gws_z = get_batch() * get_channels() * cm_params_.slice_k;
            }
            else
            {
                gws_x = get_M() / cm_params_.tile_m;
                gws_y = get_N() / cm_params_.tile_n;
                gws_z = get_batch() * get_channels() * cm_params_.slice_k;
            }
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
