#pragma once
#include <vector>
#include <random>
#include "dml_base_node.h"

namespace gpu_op
{
class Convolution : public DirectMlBaseNode
{
public:
    Convolution(const TensorShape& input_shape, const TensorShape& filter_shape, const TensorShape& output_shape,
        const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy,
        const TensorShape& stride_shape, std::uint32_t input_pad, std::uint32_t output_pad,
            bool use_bias, bool allow_fp16_computations, 
            IDMLDevice* dml_device, ID3D12Device* d3d12_device)
        : DirectMlBaseNode(dml_device, d3d12_device)
    {

        const dml::TensorDimensions input_dims{ input_shape.n, input_shape.c, input_shape.h, input_shape.w };
        const dml::TensorDimensions filter_dims{ filter_shape.n, filter_shape.c, filter_shape.h, filter_shape.w };
        const dml::TensorDimensions output_dims{ output_shape.n, output_shape.c, output_shape.h, output_shape.w };
        const dml::TensorDimensions bias_dims{ 1, output_shape.c, 1, 1 };

        const std::array<std::uint32_t, 2> strides = { stride_shape.h, stride_shape.w };
        const std::vector<std::uint32_t> dilations = { 1u, 1u };
        const std::vector<std::uint32_t> start_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> end_pad = { input_pad, input_pad };
        const std::vector<std::uint32_t> out_pad = { output_pad, output_pad };

        dml::TensorProperties input_tensor_properites{};
        {
            tensor_input_desc_.DataType = data_type;
            tensor_input_desc_.Flags = DML_TENSOR_FLAG_NONE;
            tensor_input_desc_.DimensionCount = static_cast<std::uint32_t>(input_dims.size());
            tensor_input_desc_.Sizes = input_dims.data();

            input_tensor_properites = tensor_policy.Get(tensor_input_desc_.DataType, tensor_input_desc_.Flags, input_dims);
            tensor_input_desc_.Strides = input_tensor_properites.strides.has_value() ? input_tensor_properites.strides->data() : nullptr;
            tensor_input_desc_.TotalTensorSizeInBytes = input_tensor_properites.totalTensorSizeInBytes;
            tensor_input_desc_.GuaranteedBaseOffsetAlignment = input_tensor_properites.guaranteedBaseOffsetAlignment;
        }

        dml::TensorProperties filter_tensor_properites{};
        {
            tensor_filter_desc_.DataType = data_type;
            tensor_filter_desc_.Flags = DML_TENSOR_FLAG_NONE; // DML_TENSOR_FLAG_OWNED_BY_DML;
            tensor_filter_desc_.DimensionCount = static_cast<std::uint32_t>(filter_dims.size());
            tensor_filter_desc_.Sizes = filter_dims.data();

            filter_tensor_properites = tensor_policy.Get(tensor_filter_desc_.DataType, tensor_filter_desc_.Flags, filter_dims);
            tensor_filter_desc_.Strides = filter_tensor_properites.strides.has_value() ? filter_tensor_properites.strides->data() : nullptr;;
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
    TensorShape shape;
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
    TensorShape stride;
    TensorShape output_shape;

    DataType out_dt = DataType::eCount;
    DataLayout out_layout = DataLayout::eCount;
};
std::vector<std::byte> convolution(const bindings_t& bindings, opts_t opts);
}


class ConvolutionBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout;
        TensorShape input_shape;
        TensorShape filter_shape;
        std::uint32_t in_pad;
        std::uint32_t out_pad;
        TensorShape stride;
        bool no_bias = false;
        bool allow_fp16_computations = false;
        bool managaed_weights = false; // ToDo: pass it to DML class so its actually beigned used

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();
            opts->add_option("--input_shape", params.input_shape, "speciify list: <n, ic, h, w")->required();
            opts->add_option("--filter_shape", params.filter_shape, "speciify list: <oc, ic, kh, kw")->required();
            opts->add_option("--in_pad", params.in_pad)->required();
            opts->add_option("--out_pad", params.out_pad)->required();
            opts->add_option("--stride", params.stride, "speciify list: <stride_h, stride_w>")->required();
            opts->add_flag("--no_bias", params.no_bias);
            opts->add_flag("--allow_fp16_computations", params.allow_fp16_computations);

        }
    };

    ConvolutionBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , d3d12_device_(d3d12_device)
        , input_data_(params_.input_shape.get_elements_count()* get_data_type_bytes_width(params_.dt))
        , filter_data_(params_.filter_shape.get_elements_count()* get_data_type_bytes_width(params_.dt))

    {
        assert(params_.input_shape.c == params_.filter_shape.c);
        if (!params_.no_bias)
        {
            bias_data_ = std::vector<std::byte>(params_.filter_shape.n * get_data_type_bytes_width(params_.dt));
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

        const auto tensor_input_bytes_width = input_data_.size();
        const auto tensor_filter_bytes_width = filter_data_.size();
        const auto tensor_bias_bytes_width = bias_data_.size();
        const auto output_shape = get_output_shape();
        const auto tensor_out_bytes_width = output_shape.get_elements_count() * get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_bytes_width + tensor_filter_bytes_width + tensor_bias_bytes_width,
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
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        std::memcpy(upload_mapped_ptr + memcopy_offset, filter_data_.data(), tensor_filter_bytes_width);
        if (use_bias())
        {
            memcopy_offset += tensor_filter_bytes_width;
            std::memcpy(upload_mapped_ptr + memcopy_offset, bias_data_.data(), tensor_bias_bytes_width);
        }
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        cmd_list->CopyBufferRegion(filter_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_filter_bytes_width);
        if (use_bias())
        {
            memcopy_offset += tensor_filter_bytes_width;
            cmd_list->CopyBufferRegion(bias_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_bias_bytes_width);
        }

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
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) override
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

        const auto dnnl_untyped_result = get_dnnl_result();

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
    inline TensorShape get_output_shape() const
    {
        TensorShape ret;
        ret.n = params_.input_shape.n;
        ret.c = params_.filter_shape.n; // output channels
        ret.d = 0;
        ret.h = (params_.input_shape.h - params_.filter_shape.h + params_.in_pad + params_.in_pad) / params_.stride.h + 1;
        ret.w = (params_.input_shape.w - params_.filter_shape.w + params_.in_pad + params_.in_pad) / params_.stride.w + 1;
        return ret;
    }

    inline bool use_bias() const
    {
        return !params_.no_bias;
    }

    std::vector<std::byte> get_dnnl_result() const
    {
        cpu_op::bindings_t bindings{};
        {
            bindings.input.data = input_data_.data();
            bindings.input.dt = params_.dt;
            bindings.input.layout = params_.layout;
            bindings.input.shape = params_.input_shape;
        }

        {
            bindings.filter.data = filter_data_.data();
            bindings.filter.dt = params_.dt;
            bindings.filter.layout = params_.layout;
            bindings.filter.shape = params_.filter_shape;
        }
        if (use_bias())
        {
            bindings.bias.data = bias_data_.data();
            bindings.bias.dt = params_.dt;
            bindings.bias.layout = params_.layout;
            bindings.bias.shape = TensorShape(params_.filter_shape.n, 1u, 1u, 1u);
        }
        cpu_op::opts_t opts{};
        opts.output_shape = get_output_shape();
        opts.inp_pad = params_.in_pad;
        opts.out_pad = params_.out_pad;
        opts.stride = params_.stride;
        opts.out_layout = params_.layout;
        opts.out_dt = params_.dt;
        return cpu_op::convolution(bindings, opts);
    }

protected:
    ID3D12Device* d3d12_device_;
    create_params_t params_;

    std::vector<std::byte> input_data_;
    std::vector<std::byte> filter_data_;
    std::vector<std::byte> bias_data_;

    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> filter_buffer_;
    ComPtr<ID3D12Resource> bias_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class ConvolutionDirectMLDispatcher : public ConvolutionBaseDispatcher
{
public:

    ConvolutionDirectMLDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : ConvolutionBaseDispatcher(std::move(params), d3d12_device, cmd_list)
        , dml_cmd_recorder_(dml_cmd_recorder)
        , conv_(params_.input_shape, params_.filter_shape, get_output_shape(),
            to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout),
            params_.stride, params_.in_pad, params_.out_pad, !params_.no_bias, params_.allow_fp16_computations,
            dml_device, d3d12_device)
    {
    }

    std::uint32_t get_total_descriptor_count()override
    {
        return conv_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        conv_.create_binding_tables(cpu_handle, gpu_handle);
        conv_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        conv_.record_execute(dml_cmd_recorder_, cmd_list, output_buffer_.Get(), input_buffer_.Get(), filter_buffer_.Get(), bias_buffer_.Get());
    }

private:
    gpu_op::Convolution conv_;
    IDMLCommandRecorder* dml_cmd_recorder_;
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
        std::uint32_t block_w = 8;
        const std::uint32_t block_h = 1;  //ToDo: make configurable if needed
        std::uint32_t block_oc = 8;
        std::uint32_t slice_ic = 1;
        bool reorder_weights = true;
        bool dispatch_only_weights_reorder = false;

        inline static void add_cli_options(CLI::App* opts, conv_cm_params_t& params)
        {
            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
            opts->add_option("--block_w", params.block_w);
            opts->add_option("--block_oc", params.block_oc);
            opts->add_option("--slice_ic", params.slice_ic, "How many HW threads cooperate to compute final output. Setting to 1 is equal to having this feature disabled. It increases thread group size (lws) in X dimension.")->default_val(1);
            opts->add_option("--lws", params.lws)->delimiter(',');
            opts->add_flag("--reorder_weights,!--no_reorder_weights", params.reorder_weights);
            opts->add_flag("--dispatch_only_weights_reorder", params.dispatch_only_weights_reorder);
        }
    };
public:
    ConvolutionCmDispatcher(create_params_t&& params, conv_cm_params_t&& cm_params, IntelExtension& intc_ext, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list)
        : ConvolutionBaseDispatcher(std::move(params), d3d12_device, cmd_list)
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

            weights_reorder_.emplace(WeightsReorder(std::move(wr_params), filter_buffer_, intc_ext, d3d12_device, cmd_list));
        }

        // root signature
        {
            // input, filter
            std::vector<DescType> desc_list = { DescType::eSrv, DescType::eSrv };
            if (!params_.no_bias)
            {
                desc_list.push_back(DescType::eSrv);
            }
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

        add_define("INPUT_WIDTH", params_.input_shape.w);
        add_define("INPUT_HEIGHT", params_.input_shape.h);
        add_define("INPUT_CHANNELS", params_.input_shape.c);

        add_define("OUTPUT_WIDTH", output_shape_.w);
        add_define("OUTPUT_HEIGHT", output_shape_.h);
        add_define("OUTPUT_CHANNELS", output_shape_.c);

        add_define("BATCH", params_.input_shape.n);
        add_define("INPUT_PAD", params_.in_pad);
        add_define("OUTPUT_PAD", params_.out_pad);
        add_define("USE_BIAS", !params_.no_bias);
        add_define("KERNEL_SIZE", params_.filter_shape.h);
        add_define("STRIDE_W", params_.stride.w);
        add_define("STRIDE_H", params_.stride.h);

        add_define("SLICE_IC", cm_params_.slice_ic);
        add_define("BLOCK_W", cm_params_.block_w);
        add_define("BLOCK_H", cm_params_.block_h);
        add_define("BLOCK_OC", cm_params_.block_oc);

        add_define("WEIGHTS_IN_OPTIMAL_FORMAT", cm_params.reorder_weights);

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
            if (kernel_size == 1)
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
            std::cout << std::format("Read kernel file: \n", path);
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
        std::uint32_t descriptor_count = 3;
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
        resources_list.push_back({ DescType::eUav, output_buffer_.Get() });

        gpu_handles_ = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle, base_gpu_handle);
        assert(!gpu_handles_.empty());

        // dispatch weights reorder here in initalized if weights are managed
        if (params_.managaed_weights && weights_reorder_.has_value())
        {
            weights_reorder_->execute(cmd_list);

            auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(weights_reorder_->get_output_resource());
            cmd_list->ResourceBarrier(1, &barrier);
        }
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        // dispatch weights reorder if needed (non managed case)
        if (!params_.managaed_weights && weights_reorder_.has_value())
        {
            weights_reorder_->execute(cmd_list);
            auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(weights_reorder_->get_output_resource());
            cmd_list->ResourceBarrier(1, &barrier);

            if (cm_params_.dispatch_only_weights_reorder)
            {
                return;
            }
        }

        const auto gws_x = cm_params_.slice_ic * (round_up_next_multiple(output_shape_.w, cm_params_.block_w) / cm_params_.block_w);
        const auto gws_y = round_up_next_multiple(output_shape_.h, cm_params_.block_h) / cm_params_.block_h;
        const auto gws_z = params_.input_shape.n * (params_.filter_shape.n / cm_params_.block_oc);

        assert(gws_x % cm_params_.lws[0] == 0);
        assert(gws_y % cm_params_.lws[1] == 0);
        assert(gws_z % cm_params_.lws[2] == 0);

        const auto thg_x = gws_x / (cm_params_.slice_ic * cm_params_.lws[0]);
        const auto thg_y = gws_y / cm_params_.lws[1];
        const auto thg_z = gws_z / cm_params_.lws[2];

        dispatch_kernel(cmd_list, pso_.Get(), root_signature_.Get(), gpu_handles_, thg_x, thg_y, thg_z);
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) override
    {
        if (weights_reorder_.has_value())
        {
            weights_reorder_->validate_conformance(command_queue, command_allocator, command_list);
        }
        const auto ret = ConvolutionBaseDispatcher::validate_conformance(command_queue, command_allocator, command_list);
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
            std::uint32_t ic = 0;
            std::uint32_t oc = 0;
            std::uint32_t k_size = 0;
            std::uint32_t ic_chunks_per_hw_thread = 8;

            std::uint32_t dpas_depth = 8;
            std::uint32_t dpas_exec_size = 8;
            std::array<std::uint32_t, 3> lws{ 2u, 1u, 1u };

            inline std::uint32_t ic_chunk_size() const
            {
                return dpas_depth * (sizeof(uint32_t) / get_data_type_bytes_width(input_dt));
            }
        };
    public:
        WeightsReorder(create_params_t&& params, ComPtr<ID3D12Resource> input_resource, IntelExtension& intc_ext, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list)
            : params_(std::move(params))
            , intc_ext_(intc_ext)
            , d3d12_device_(d3d12_device)
            , input_buffer_(input_resource)
        {
            // root signature
            {
                // input, filter
                const std::vector<DescType> desc_list = { DescType::eSrv, DescType::eUav };
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

            add_define("INPUT_TYPE", "half");
            add_define("OUTPUT_TYPE", "half");
            add_define("WEI_OFFSET", 0);
            add_define("IC", params_.ic);
            add_define("OC", params_.oc);
            add_define("K_SIZE", params_.k_size);
            add_define("IC_CHUNK_SIZE", params_.ic_chunk_size());
            add_define("IC_CHUNKS_PER_HW_THREAD", params_.ic_chunks_per_hw_thread);

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

            // compiled success -> create buffer
            {
                const auto size_bytes = params_.ic * params_.oc * params_.k_size * params_.k_size * get_data_type_bytes_width(params_.output_dt);
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

            const auto desc_heap_incrs_size = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            // i.e. add weights reorder

            auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
            auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

            std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
            resources_list.reserve(get_total_descriptor_count());
            resources_list.push_back({ DescType::eSrv, input_buffer_.Get() });
            resources_list.push_back({ DescType::eUav, output_buffer_.Get() });

            gpu_handles_ = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle, base_gpu_handle);
        }

        std::uint32_t get_total_descriptor_count() override
        {
            return 2;
        }

        void execute(ID3D12GraphicsCommandList* cmd_list) override
        {
            assert(cmd_list);
            assert(pso_);
            assert(root_signature_);
            assert(!gpu_handles_.empty());

            const auto gws_x = params_.oc / params_.dpas_exec_size;
            const auto gws_y = params_.ic / (params_.ic_chunks_per_hw_thread * params_.ic_chunk_size());
            const auto gws_z = 1;

            assert(gws_x % params_.lws[0] == 0);
            assert(gws_x % params_.lws[1] == 0);
            assert(gws_x % params_.lws[2] == 0);
                   
            const auto thg_x = gws_x / params_.lws[0];
            const auto thg_y = gws_y / params_.lws[1];
            const auto thg_z = gws_z / params_.lws[2];

            dispatch_kernel(cmd_list, pso_.Get(), root_signature_.Get(), gpu_handles_, thg_x, thg_y, thg_z);
        }


        ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
            ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) override
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
                for (std::int32_t i = 0; i < tensor_out_bytes_width / get_data_type_bytes_width(params_.output_dt); i++)
                {
                    const auto f = DirectX::PackedVector::XMConvertHalfToFloat(data_ptr[i]);
                    std::cout << (std::int32_t)f << " ";

                    if ((i + 1) % 16 == 0 && i != 0)
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
