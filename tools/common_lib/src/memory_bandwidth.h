#pragma once
#include <vector>
#include <random>
#include "dml_base_node.h"

namespace gpu_op
{

class MemoryBandwidthDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        TensorShape shape;
        std::uint32_t items_per_hw = 128;
        std::uint32_t lws_x = 1;
        bool dump_asm;
        bool large_grf;
        bool print_reg_usage;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            opts->add_option("--shape", params.shape, "shape: <n,c,h,w>")->required();
            opts->add_option("--items_per_hw", params.items_per_hw)->required();
            opts->add_option("--lws_x", params.lws_x);

            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
        }
    };
public:
    MemoryBandwidthDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list, IntelExtension& intc_ext, bool use_stateless)
        : params_(std::move(params))
        , input_data_(get_tensor_elements_count(params_.shape, DataLayout::eNCHW) * (std::uint8_t)get_data_type_bytes_width(params_.dt))
        , intc_ext_(intc_ext)
        , d3d12_device_(d3d12_device)
        , use_stateless_(use_stateless)
    {
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

        const auto tensor_input_bytes_width = input_data_.size();
        const auto tensor_out_bytes_width = tensor_input_bytes_width;

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_input_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);


        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());


        // root signature
        {
            if(use_stateless_)
            {
                root_signature_ = create_root_signature_without_roottable(d3d12_device_, 2);
            } 
            else 
            {
                // input, filter
                std::vector<DescType> desc_list = { DescType::eSrv, DescType::eUav };
                root_signature_ = create_root_signature(d3d12_device_, desc_list);
                assert(root_signature_);
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

        add_define("ITEMS_PER_HW", params_.items_per_hw);

        // kernel compilation
        const auto dump_asm_str = params_.dump_asm ? " -mdump_asm" : "";
        const auto large_grf_str = params_.large_grf ? " -Qxcm_doubleGRF" : "";
        const auto print_reg_str = params_.print_reg_usage ? " -mCM_printregusage" : "";
        const auto lws_x = " -DLWS_SIZE_X=" + std::to_string(params_.lws_x);
        const auto lws_y = " -DLWS_SIZE_Y=1";
        const auto lws_z = " -DLWS_SIZE_Z=1";

        auto build_options_final = " -I \" \" " + build_options + dump_asm_str + large_grf_str + print_reg_str + lws_x + lws_y + lws_z;
        if(use_stateless_)
        {
            build_options_final += " -DCM_STATELESS=1";
        }

        std::cout << "Build options: " << build_options_final << std::endl;

        if (params_.dump_asm)
        {
            std::cout << build_options_final << std::endl;
        }

        auto kernel_source_content = [&]()
        {
            auto path = "memory_copy.cpp";
            if(use_stateless_)
            {
                path = "memory_copy_stateless.cpp";
            }
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
        return 2u;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        assert(!use_stateless_);
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
        if(use_stateless_) // if in stateless mode, all buffers are set to UAV accoring to its GPU address
        {
            // the root parameter index should start from 1, in order to skip first constant buffer.
            cmd_list->SetComputeRootUnorderedAccessView(1, input_buffer_->GetGPUVirtualAddress());
            cmd_list->SetComputeRootUnorderedAccessView(2, output_buffer_->GetGPUVirtualAddress());
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

        const auto gws_x = (params_.shape.n * params_.shape.c * params_.shape.h * params_.shape.w) / params_.items_per_hw;

        assert(gws_x % params_.lws_x == 0);

        const auto thg_x = gws_x / params_.lws_x;
        const auto thg_y = 1;
        const auto thg_z = 1;
        cmd_list->Dispatch(thg_x, thg_y, thg_z);
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches, std::size_t reference_dispatch_iterations) override
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

        //
        //  calc reference with dml non-mc mvn
        //
        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        command_list->ResourceBarrier(1, &readback_output_barrirer);


        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, input_data_, 0.0f, print_mismatches);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, input_data_, 0.0f, print_mismatches);
        }
        ConformanceResult ret{};
        return ret;
    }

protected:

protected:
    bool use_stateless_;
    create_params_t params_;
    ID3D12Device* d3d12_device_;

    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;

    std::vector<std::byte> input_data_;
    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};
}
