#pragma once
#include <vector>
#include <random>
#include "dml_base_node.h"

namespace gpu_op
{
class Softmax : public DirectMlBaseNode
{
public:
    Softmax(std::uint32_t axis, const TensorShape& shape, const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy,
        IDMLDevice* dml_device, ID3D12Device* d3d12_device)
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

        std::vector<DML_BINDING_DESC> output_bindings;
        output_bindings.reserve(1);
        DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
        DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };
        output_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &output_buffer_binding });

        record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_bindings);
    }

private:
    DML_BUFFER_TENSOR_DESC tensor_input_desc_{};
    DML_BUFFER_TENSOR_DESC tensor_output_desc_{};
    ComPtr<IDMLOperator> dml_operator_;
};

}  // namespace gpu_op

namespace dnnl_softmax_op
{
std::vector<std::byte> softmax(std::uint32_t axis, const std::byte* in_data, const TensorShape& in_out_shape, DataType in_out_datatype, DataLayout in_out_layout);
}  // namespace dnnl_op


class SoftmaxBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout;
        TensorShape shape;
        std::uint32_t axis;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();
            opts->add_option("--shape", params.shape, "shape: <n,c,h,w>")->required();
            opts->add_option("--axis", params.axis, "axis represents the axis of which the SoftMax is calculated.")->required();
        }
    };
public:
    SoftmaxBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , d3d12_device_(d3d12_device)
        , input_data_(get_tensor_elements_count(params_.shape, params_.layout) * get_data_type_bytes_width(params_.dt))
    {
        const auto tensor_a_bytes_width = input_data_.size();
        const auto tensor_out_bytes_width = input_data_.size();


        upload_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(50.0f, 505.0f);

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

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches)
    {
        const auto tensor_out_bytes_width = input_data_.size();

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

        const auto dnnl_untyped_result = dnnl_softmax_op::softmax(params_.axis, input_data_.data(), params_.shape, params_.dt, params_.layout);

        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, dnnl_untyped_result, 0.0001f, print_mismatches);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, dnnl_untyped_result, 0.005f, print_mismatches);
        }
        assert(false && "Unsupported output data type!");
        ConformanceResult ret{};
        return ret;
    }


protected:
    create_params_t params_;
    ID3D12Device* d3d12_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<std::byte> input_data_;

    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class SoftmaxDmlDispatcher : public SoftmaxBaseDispatcher
{
public:
    SoftmaxDmlDispatcher(SoftmaxBaseDispatcher::create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : SoftmaxBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , softmax_(params_.axis, params_.shape, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), dml_device, d3d12_device)
    {
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return softmax_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        softmax_.create_binding_tables(cpu_handle, gpu_handle);
        softmax_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        softmax_.record_execute(dml_cmd_recorder_, cmd_list, output_buffer_.Get(), input_buffer_.Get());
    }

private:
    gpu_op::Softmax softmax_;
};

class SoftmaxCmDispatcher : public SoftmaxBaseDispatcher
{
public:
    struct softmax_cm_params_t
    {
        bool dump_asm;
        bool large_grf;
        bool print_reg_usage;
        std::array<std::uint32_t, 3> lws{ 1u, 1u, 1u };

        inline static void add_cli_options(CLI::App* opts, softmax_cm_params_t& params)
        {
            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
        }
    };
public:
    SoftmaxCmDispatcher(create_params_t&& params, softmax_cm_params_t&& cm_params, IntelExtension& intc_ext, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : SoftmaxBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , intc_ext_(intc_ext)
        , cm_params_(std::move(cm_params))
    {
        const auto items_per_hw_th = get_items_per_hw();
        if(items_per_hw_th == 0)
        {
            throw std::runtime_error("Unsupported width for softmax operator for MHA layer!");
        }

        cm_params_.lws[0] = params_.shape.w / items_per_hw_th;
        cm_params_.lws[1] = 1;
        cm_params_.lws[2] = 1;

        // root signature
        {
            // input, filter
            std::vector<DescType> desc_list = { DescType::eSrv, DescType::eUav };
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

        add_define("INOUT_WIDTH", params_.shape.w);
        add_define("INOUT_HEIGHT", params_.shape.h);
        add_define("ITEMNUM_PER_HW", items_per_hw_th);
        add_define("LWS_SIZE_X_ALIGNED", align(cm_params_.lws[0], 8));

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

        auto kernel_source_content = []()
        {
            const auto path = "softmax_nchw.cpp";
            std::fstream file(path);
            if (!file.is_open())
            {
                const auto msg = std::format("Kernel file cant be opened:{} \n.", path);
                throw std::runtime_error(msg);
            }
            return std::string((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
        }();

        CD3DX12_SHADER_BYTECODE byte_code;
        byte_code.pShaderBytecode = kernel_source_content.data();
        byte_code.BytecodeLength = kernel_source_content.size();
        pso_ = intc_ext_.create_pipeline(byte_code, build_options_final, root_signature_.Get(), INTC_D3D12_SHADER_INPUT_TYPE::CM);
    }

    std::uint32_t get_total_descriptor_count() override
    {
        // input, output
        std::uint32_t descriptor_count = 2;
        return descriptor_count;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        const auto desc_heap_incrs_size = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // i.e. add weights reorder

        auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
        auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

        std::vector<std::pair<DescType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ DescType::eSrv, input_buffer_.Get() });
        resources_list.push_back({ DescType::eUav, output_buffer_.Get() });

        gpu_handles_ = create_resource_views_and_handles(d3d12_device_, resources_list, base_cpu_handle, base_gpu_handle);
        assert(!gpu_handles_.empty());
    }

    void execute(ID3D12GraphicsCommandList* cmd_list) override
    {
        cmd_list->SetComputeRootSignature(root_signature_.Get());
        cmd_list->SetPipelineState(pso_.Get());

        uint32_t root_index = 1; // start with 1, beacuse Cross compiler CM driver path needs that
        for (uint32_t i = 0; i < gpu_handles_.size(); i++)
        {
            const auto gpu_heap_handle = gpu_handles_[i];
            cmd_list->SetComputeRootDescriptorTable(root_index++, gpu_heap_handle);
        }

        const auto gws_x = params_.shape.w / get_items_per_hw();
        const auto gws_y = params_.shape.h;
        const auto gws_z = params_.shape.n * params_.shape.c;

        assert(gws_x % cm_params_.lws[0] == 0);
        assert(gws_y % cm_params_.lws[1] == 0);
        assert(gws_z % cm_params_.lws[2] == 0);

        const auto thg_x = gws_x / cm_params_.lws[0];
        const auto thg_y = gws_y / cm_params_.lws[1];
        const auto thg_z = gws_z / cm_params_.lws[2];
        cmd_list->Dispatch(thg_x, thg_y, thg_z);
    }

private:
    std::uint32_t get_items_per_hw()
    {
        std::uint32_t items_per_hw_th = params_.shape.w;
        if (params_.shape.w % 128 == 0)
        {
            items_per_hw_th = 128;
        }
        else if (params_.shape.w % 64 == 0)
        {
            items_per_hw_th = 64;
        }
        else if (params_.shape.w % 32 == 0)
        {
            items_per_hw_th = 32;
        }
        else if (params_.shape.w % 16 == 0)
        {
            items_per_hw_th = 16;
        }
        // tehnically bigger W would work, but not tested
        else if (params_.shape.w < 128)
        {
            items_per_hw_th = params_.shape.w;
        }
        // error
        return items_per_hw_th;
    }

private:
    softmax_cm_params_t cm_params_;
    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;
};