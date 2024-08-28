#pragma once
#include "dx12_utils.h"
#include "layers_utils.h"
#include "node_dispatcher.h"

#include <variant>

namespace
{
    inline static dml::TensorProperties compute_nchw_tensor_policy(
        DML_TENSOR_DATA_TYPE dataType,
        DML_TENSOR_FLAGS /*flags*/,
        std::span<const uint32_t> sizes)
    {
        uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
        dml::TensorStrides strides(dimension_count);

        uint32_t stride = 1;
        for (std::uint32_t i = dimension_count; i > 0; i--)
        {
            strides[i - 1] = stride;
            stride *= sizes[i - 1];
        }

        dml::TensorProperties props;
        props.strides = std::move(strides);
        props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimension_count, sizes.data(), props.strides->data());
        props.guaranteedBaseOffsetAlignment = 0;
        return props;
    }

    inline static dml::TensorProperties compute_w_tensor_policy(
        DML_TENSOR_DATA_TYPE dataType,
        DML_TENSOR_FLAGS /*flags*/,
        std::span<const uint32_t> sizes)
    {
        uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
        dml::TensorStrides strides(dimension_count);

        for (auto& s : strides)
        {
            s = 0;
        }
        strides.back() = 1;

        dml::TensorProperties props;
        props.strides = std::move(strides);
        props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimension_count, sizes.data(), props.strides->data());
        props.guaranteedBaseOffsetAlignment = 0;
        return props;
    }

    inline static dml::TensorProperties compute_nchw_alignw320_tensor_policy(
        DML_TENSOR_DATA_TYPE dataType,
        DML_TENSOR_FLAGS /*flags*/,
        std::span<const uint32_t> sizes)
    {
        const uint32_t dimension_count = static_cast<uint32_t>(sizes.size());
        dml::TensorStrides strides(dimension_count);

        dml::TensorDimensions dims(dimension_count);
        for (std::uint32_t i = 0; i < dimension_count; i++)
        {
            dims[i] = sizes[i];
        }
        dims.back() = align(dims.back(), 320);

        uint32_t stride = 1;
        for (std::uint32_t i = dimension_count; i > 0; i--)
        {
            strides[i - 1] = stride;
            stride *= dims[i - 1];
        }

        dml::TensorProperties props;
        props.strides = std::move(strides);
        props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimension_count, sizes.data(), props.strides->data());
        props.guaranteedBaseOffsetAlignment = 0;
        return props;
    }

    inline static dml::TensorProperties compute_nhwc_alignh40_tensor_policy(
        DML_TENSOR_DATA_TYPE dataType,
        DML_TENSOR_FLAGS /*flags*/,
        std::span<const uint32_t> sizes)
    {
        uint32_t dimensionCount = static_cast<uint32_t>(sizes.size());
        dml::TensorStrides strides(dimensionCount);

        enum Axes { N, C, H, W /* more dims*/ };

        dml::TensorDimensions dims(dimensionCount);
        for (std::uint32_t i = 0; i < dimensionCount; i++)
        {
            dims[i] = sizes[i];
        }
        dims[H] = align(dims[H], 40);

        // N dimension strides
        if (dimensionCount >= 1)
        {
            strides[N] = 1;
            for (uint32_t i = 1; i < dimensionCount; ++i)
            {
                strides[N] *= dims[i];
            }
        }

        // C dimension strides
        if (dimensionCount >= 2)
        {
            strides[C] = 1;
        }

        // Spatial dimension strides
        if (dimensionCount >= 3)
        {
            uint32_t stride = dims[C];
            for (uint32_t i = dimensionCount - 1; i >= 2; --i)
            {
                strides[i] = stride;
                stride *= dims[i];
            }
        }

        dml::TensorProperties props;
        props.strides = std::move(strides);
        props.totalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, dimensionCount, sizes.data(), props.strides->data());
        props.guaranteedBaseOffsetAlignment = 0;
        return props;
    }
}

inline std::string to_string(const std::string& value) { return value; }

inline DML_TENSOR_DATA_TYPE to_dml_data_type(DataType dt)
{
    switch (dt)
    {
    case DataType::eFp32: return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::eFp16: return DML_TENSOR_DATA_TYPE_FLOAT16;
    case DataType::eUint4: return DML_TENSOR_DATA_TYPE_UINT4;
    default:
        assert(false && "Unknown data type.");
    }
    return DML_TENSOR_DATA_TYPE_UNKNOWN;
}

inline dml::TensorPolicy to_dml_tensor_policy(DataLayout layout)
{
    switch (layout)
    {
    case DataLayout::eCHW:
    case DataLayout::eNCHW: return dml::TensorPolicy::Default();
    case DataLayout::eNHWC: return dml::TensorPolicy::InterleavedChannel();
    case DataLayout::eW: return dml::TensorPolicy(compute_w_tensor_policy);
    case DataLayout::eNCHW_AlignW320: return dml::TensorPolicy(compute_nchw_alignw320_tensor_policy);
    case DataLayout::eNHWC_AlignH48: return dml::TensorPolicy(compute_nhwc_alignh40_tensor_policy);
    default:
        assert(false && "Unknown data layout.");
    }
    return dml::TensorPolicy::Default();
}

struct DmlActivationSetting
{
    DML_OPERATOR_DESC desc{};

    // below concrete OP descs - needed to hold memory, which is pointed by above DML_OPERATOR_DESC desc
    std::unique_ptr<DML_ACTIVATION_RELU_OPERATOR_DESC> relu{};
    std::unique_ptr<DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC> leaky_relu{};
    std::unique_ptr<DML_ELEMENT_WISE_CLIP_OPERATOR_DESC> clip{};
    std::unique_ptr<DML_ACTIVATION_GELU_OPERATOR_DESC> gelu{};
    std::unique_ptr<DML_ACTIVATION_SIGMOID_OPERATOR_DESC> sigmoid{};
    std::unique_ptr<DML_ACTIVATION_LINEAR_OPERATOR_DESC> linear{};
    std::unique_ptr<DML_ACTIVATION_TANH_OPERATOR_DESC> tanh{};
};

inline DmlActivationSetting to_dml_activation_setting(const ActivationSettings& act)
{
    DmlActivationSetting ret{};
    ret.desc.Type = DML_OPERATOR_INVALID;
    switch (act.type)
    {
    case ActivationType::eRelu:
    {
        ret.desc.Type = DML_OPERATOR_ACTIVATION_RELU;
        ret.relu = std::make_unique<DML_ACTIVATION_RELU_OPERATOR_DESC>();
        ret.desc.Desc = ret.relu.get();
        break;
    }
    case ActivationType::eLeakyRelu:
    {
        ret.desc.Type = DML_OPERATOR_ACTIVATION_LEAKY_RELU;
        ret.leaky_relu = std::make_unique<DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC>();
        ret.leaky_relu->Alpha = act.alpha;
        ret.desc.Desc = ret.leaky_relu.get();
        break;
    }
    case ActivationType::eClip:
    {
        ret.desc.Type = DML_OPERATOR_ELEMENT_WISE_CLIP;
        ret.clip = std::make_unique<DML_ELEMENT_WISE_CLIP_OPERATOR_DESC>();
        ret.clip->Min = act.alpha;
        ret.clip->Max = act.beta;
        ret.desc.Desc = ret.clip.get();
        break;
    }
    case ActivationType::eGelu:
    {
        ret.desc.Type = DML_OPERATOR_ACTIVATION_GELU;
        ret.gelu = std::make_unique<DML_ACTIVATION_GELU_OPERATOR_DESC>();
        ret.desc.Desc = ret.gelu.get();
        break;
    }
    case ActivationType::eSigmoid:
    {
        ret.desc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
        ret.sigmoid = std::make_unique<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();
        ret.desc.Desc = ret.sigmoid.get();
        break;
    }
    case ActivationType::eLinear:
    {
        ret.desc.Type = DML_OPERATOR_ACTIVATION_LINEAR;
        ret.linear = std::make_unique<DML_ACTIVATION_LINEAR_OPERATOR_DESC>();
        ret.linear->Alpha = act.alpha;
        ret.linear->Beta = act.beta;
        ret.desc.Desc = ret.linear.get();
        break;
    }
    case ActivationType::eTanh:
    {
        ret.desc.Type = DML_OPERATOR_ACTIVATION_TANH;
        ret.tanh = std::make_unique<DML_ACTIVATION_TANH_OPERATOR_DESC>();
        ret.desc.Desc = ret.tanh.get();
        break;
    }
    default:
        assert("unknown fused activation type, add it to the switch above!");
    }
    return ret;
}

namespace gpu_op
{
class DirectMlBaseNode
{
public:
    DirectMlBaseNode(IDMLDevice* dml_device, ID3D12Device* d3d12_device)
        : dml_device_(dml_device)
        , d3d12_device_(d3d12_device)
    {
    }

    virtual void record_initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
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

        dml_cmd_recorder->RecordDispatch(
            cmd_list,
            dml_op_initializer_.Get(),
            dml_init_binding_table.Get());
    }

    virtual uint32_t get_total_descriptor_count() const
    {
        const auto initialize_binding_properties = dml_op_initializer_->GetBindingProperties();
        const auto execute_binding_properties = dml_op_executor_->GetBindingProperties();
        return initialize_binding_properties.RequiredDescriptorCount +
            execute_binding_properties.RequiredDescriptorCount;
    }

    virtual void create_binding_tables(D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        const auto desc_inc = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const auto init_desc_count = dml_op_initializer_->GetBindingProperties().RequiredDescriptorCount;
        const auto exec_desc_count = dml_op_executor_->GetBindingProperties().RequiredDescriptorCount;

        {
            DML_BINDING_TABLE_DESC dml_binding_table_desc{};
            dml_binding_table_desc.Dispatchable = dml_op_initializer_.Get();
            dml_binding_table_desc.CPUDescriptorHandle = cpu_handle;
            dml_binding_table_desc.GPUDescriptorHandle = gpu_handle;
            dml_binding_table_desc.SizeInDescriptors = dml_op_initializer_->GetBindingProperties().RequiredDescriptorCount;
            throw_if_failed(dml_device_->CreateBindingTable(
                &dml_binding_table_desc, IID_PPV_ARGS(dml_init_binding_table.ReleaseAndGetAddressOf())), "dml create init binding table");
        }

        {
            DML_BINDING_TABLE_DESC dml_binding_table_desc{};
            dml_binding_table_desc.Dispatchable = dml_op_executor_.Get();
            dml_binding_table_desc.CPUDescriptorHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(cpu_handle, init_desc_count, desc_inc);
            dml_binding_table_desc.GPUDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(gpu_handle, init_desc_count, desc_inc);
            dml_binding_table_desc.SizeInDescriptors = exec_desc_count;
            throw_if_failed(dml_device_->CreateBindingTable(
                &dml_binding_table_desc, IID_PPV_ARGS(dml_exec_binding_table.ReleaseAndGetAddressOf())), "dml create exec binding table");
        }
    }

protected:
    virtual void create_operator_impl()
    {
        assert(dml_op_executor_ != nullptr);
        IDMLCompiledOperator* dml_compiled_operators[] = { dml_op_executor_.Get() };
        // initlaizer operator
        dml_device_->CreateOperatorInitializer(1, dml_compiled_operators, IID_PPV_ARGS(dml_op_initializer_.ReleaseAndGetAddressOf()));

        // resources
        const auto initialize_binding_properties = dml_op_initializer_->GetBindingProperties();
        const auto execute_binding_properties = dml_op_executor_->GetBindingProperties();
        const auto temporary_resource_size = std::max(
            initialize_binding_properties.TemporaryResourceSize,
            execute_binding_properties.TemporaryResourceSize);
        const auto persistent_resource_size = execute_binding_properties.PersistentResourceSize;

        if (temporary_resource_size != 0)
        {
            temporary_buffer_ = create_buffer(d3d12_device_, temporary_resource_size,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }

        if (persistent_resource_size != 0)
        {           
            persistent_buffer_ = create_buffer(d3d12_device_, persistent_resource_size,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
    }

    virtual void record_execute_impl(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, 
        const std::vector<DML_BINDING_DESC>& input_bindings, const std::vector<DML_BINDING_DESC>& output_bindings)
    {
        const auto execute_binding_properties = dml_op_executor_->GetBindingProperties();
        if (execute_binding_properties.TemporaryResourceSize > 0 && temporary_buffer_)
        {
            DML_BUFFER_BINDING buffer_binding{ temporary_buffer_.Get(), 0, temporary_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_exec_binding_table->BindTemporaryResource(&binding_desc);
        }

        if (execute_binding_properties.PersistentResourceSize > 0 && persistent_buffer_)
        {
            // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
            DML_BUFFER_BINDING buffer_binding{ persistent_buffer_.Get(), 0, persistent_buffer_->GetDesc().Width };
            DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
            dml_exec_binding_table->BindPersistentResource(&binding_desc);
        }

        dml_exec_binding_table->BindInputs(static_cast<std::uint32_t>(input_bindings.size()), input_bindings.data());
        dml_exec_binding_table->BindOutputs(static_cast<std::uint32_t>(output_bindings.size()), output_bindings.data());

        dml_cmd_recorder->RecordDispatch(cmd_list, dml_op_executor_.Get(), dml_exec_binding_table.Get());
    }

protected:
    IDMLDevice* dml_device_;
    ID3D12Device* d3d12_device_;

    ComPtr<IDMLCompiledOperator> dml_op_executor_ = nullptr;
    ComPtr<IDMLOperatorInitializer> dml_op_initializer_ = nullptr;

    ComPtr<ID3D12Resource> temporary_buffer_;
    ComPtr<ID3D12Resource> persistent_buffer_;

    ComPtr<IDMLBindingTable> dml_init_binding_table;
    ComPtr<IDMLBindingTable> dml_exec_binding_table;

};
}