#include "dx12_utils.h"
#include "gemm.h"
#include "conv.h"
#include "softmax.h"
#include "layers_utils.h"

#include <iostream>
#include <optional>
#include <span>
#include <format>
#include <random>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

template<typename TimeType>
inline void print_performance_stats(const std::vector<TimeType>& timings)
{
    TimeType avg(0);
    TimeType best((std::numeric_limits<uint32_t>::max)());
    TimeType median(0);

    // avg and best
    {
        for (const auto& t : timings)
        {
            avg += t;
            if (t < best)
            {
                best = t;
            }
        }
        avg /= timings.size();
    }

    // median
    {
        auto timings_copy = timings;
        std::nth_element(timings_copy.begin(), timings_copy.begin() + timings_copy.size() / 2, timings_copy.end());
        median = timings_copy[timings_copy.size() / 2];
    }

    std::cout << "Avg: " << avg << std::endl;
    std::cout << "Median: " << avg << std::endl;
    std::cout << "Best: " << best << std::endl;
}

inline void randomize_linear_container_float(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::byte> container)
{
    using Dt = float;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = static_cast<Dt>(dist(gen));
    }
}

inline void randomize_linear_container_half(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::byte> container)
{
    using Dt = Half;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = DirectX::PackedVector::XMConvertFloatToHalf(dist(gen));
    }
}

inline void fill_with_constant_linear_container_half(std::span<std::byte> container, Half value)
{
    using Dt = Half;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = value;
    }
}

inline auto add_data_type_cli_option(CLI::App* opts, std::string_view opt_name, DataType& dt)
{
    return opts->add_option("--data_type", dt)->check(CLI::IsMember({ DataType::eFp32, DataType::eFp16 }))
        ->transform(CLI::Transformer(std::map<std::string, DataType>{
            {"fp32", DataType::eFp32}, { "fp16", DataType::eFp16 }
    }, CLI::ignore_case, CLI::ignore_underscore));
}

inline auto add_data_layout_cli_option(CLI::App* opts, std::string_view opt_name, DataLayout& layout)
{
    return opts->add_option("--layout", layout)->check(CLI::IsMember({ DataLayout::eNCHW, DataLayout::eNHWC }))
        ->transform(CLI::Transformer(std::map<std::string, DataLayout>{
            {"nchw", DataLayout::eNCHW}, { "nhwc", DataLayout::eNHWC }, 
    }, CLI::ignore_case, CLI::ignore_underscore));
}

struct ConformanceResult
{
    bool passed = true;
    float epsilon = 0.0f;
    float biggest_difference = 0.0f;
    float node_value = 0.0f;
    float reference_value = 0.0f;
    std::uint32_t index = 0;
    std::size_t tested_samples_count = 0;
};

inline float cast_to_float(Half v)
{
    return DirectX::PackedVector::XMConvertHalfToFloat(v);
}

inline float cast_to_float(float v)
{
    return v;
}

template<typename Dt>
inline ConformanceResult run_conformance_check(const std::vector<std::byte>& gpu_untyped_result, const std::vector<std::byte>& dnnl_untyped_result, float epsilon)
{
    const auto* gpu_typed_result = reinterpret_cast<const Dt*>(gpu_untyped_result.data());
    const auto* dnnl_typed_result = reinterpret_cast<const Dt*>(dnnl_untyped_result.data());

    // compare results
    ConformanceResult ret;
    ret.epsilon = epsilon;
    for (std::uint32_t i = 0; i < gpu_untyped_result.size() / sizeof(Dt); i++)
    {
        ret.node_value = cast_to_float(gpu_typed_result[i]);
        ret.reference_value = cast_to_float(dnnl_typed_result[i]);

        const auto abs_diff = std::abs(ret.node_value - ret.reference_value);

        if (abs_diff > ret.epsilon)
        {
            ret.passed = false;

            std::cout << std::format("Mismatch, gpu: {}, cpu: {}, at index: {}. Absolute differece: {} \n", ret.node_value, ret.reference_value, i, abs_diff);
        }
        ret.biggest_difference = std::max(ret.biggest_difference, abs_diff);
        ret.tested_samples_count++;
    }
    return ret;
}

enum class NodeType
{
    eGemm,
    eConvDml,
    eConvCm,
    eSoftmax,
    eCount
};

class NodeDispatcher
{
public:
    virtual std::uint32_t get_total_descriptor_count() = 0;
    virtual void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) = 0;
    virtual void execute(ID3D12GraphicsCommandList* cmd_list) = 0;

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) = 0;

    virtual ~NodeDispatcher() = default;
};

class GemmDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        std::uint32_t M;
        std::uint32_t K;
        std::uint32_t N;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            opts->add_option("M", params.M)->required();
            opts->add_option("K", params.K)->required();
            opts->add_option("N", params.N)->required();
        }
    };
public:
    GemmDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , gemm_(dml::TensorDimensions{ 1, 1, params_.M, params_.K }, dml::TensorDimensions{ 1, 1, params_.K, params_.N }, dml_device, d3d12_device)
        , d3d12_device_(d3d12_device)
        , input_data_a_(params_.M * params_.K)
        , input_data_b_(params_.K * params_.N)
    {
        const auto tensor_a_desc = gemm_.get_tensor_a_desc();
        const auto tensor_b_desc = gemm_.get_tensor_b_desc();
        const auto tensor_out_desc = gemm_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_a_desc.totalTensorSizeInBytes;
        const auto tensor_b_bytes_width = tensor_b_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;


        upload_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width + tensor_b_bytes_width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_a_ = create_buffer(d3d12_device, tensor_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        input_buffer_b_ = create_buffer(d3d12_device, tensor_b_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);

        for (auto& a : input_data_a_)
        {
            a = uniform_distribution(random_generator);
        }
        for (auto& b : input_data_b_)
        {
            b = uniform_distribution(random_generator);
        }

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::memcpy(upload_mapped_ptr, input_data_a_.data(), tensor_a_bytes_width);
        std::memcpy(upload_mapped_ptr + tensor_a_bytes_width, input_data_b_.data(), tensor_b_bytes_width);
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);


        cmd_list->CopyBufferRegion(input_buffer_a_.Get(), 0, upload_buffer_.Get(), 0, tensor_a_bytes_width);
        cmd_list->CopyBufferRegion(input_buffer_b_.Get(), 0, upload_buffer_.Get(), tensor_a_bytes_width, tensor_b_bytes_width);

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_a_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_b_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return gemm_.get_total_descriptor_count();
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        gemm_.create_binding_tables(cpu_handle, gpu_handle);

        // Record execution of the operator initializer.
        gemm_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        gemm_.record_execute(dml_cmd_recorder_, cmd_list, output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
    {
        const auto tensor_a_desc = gemm_.get_tensor_a_desc();
        const auto tensor_b_desc = gemm_.get_tensor_b_desc();
        const auto tensor_out_desc = gemm_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_a_desc.totalTensorSizeInBytes;
        const auto tensor_b_bytes_width = tensor_b_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;

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

        std::vector<float> cpu_data_f32(params_.M * params_.N);
        cpu_op::gemm<float>(params_.M, params_.K, params_.N, 1.0f, 1.0f, input_data_a_.data(), input_data_b_.data(), nullptr, cpu_data_f32.data());

        const auto* gpu_data_out_f32 = reinterpret_cast<const float*>(data_out.data());
        // compare results
        ConformanceResult ret;
        ret.epsilon = 0.001f;
        for (std::uint32_t i = 0; i < params_.M * params_.N; i++)
        {
            ret.node_value = gpu_data_out_f32[i];
            ret.reference_value = cpu_data_f32[i];
            const auto abs_diff = std::abs(ret.node_value - ret.reference_value);

            if (abs_diff > ret.epsilon)
            {
                ret.passed = false;

                std::cout << std::format("Mismatch, gpu: {}, cpu: {}, at index: {}. Absolute differece: \n", ret.node_value, ret.reference_value, i, abs_diff);
            }
            ret.biggest_difference = std::max(ret.biggest_difference, abs_diff);
            ret.tested_samples_count++;
        }
        return ret;
    }

private:
    create_params_t params_;
    gpu_op::Gemm gemm_;
    ID3D12Device* d3d12_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<float> input_data_a_;
    ComPtr<ID3D12Resource> input_buffer_a_;
    std::vector<float> input_data_b_;
    ComPtr<ID3D12Resource> input_buffer_b_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class ConvolutionBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout;
        std::uint32_t batch;
        std::uint32_t ic;
        std::uint32_t oc;
        std::uint32_t in_width;
        std::uint32_t in_height;
        std::uint32_t in_pad;
        std::uint32_t out_pad;
        std::uint32_t kernel_size;
        std::uint32_t stride;
        bool no_bias = false;
        bool allow_fp16_computations = false;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt)->required();
            add_data_layout_cli_option(opts, "--layout", params.layout)->required();
            opts->add_option("--batch", params.batch)->required();
            opts->add_option("--ic", params.ic)->required();
            opts->add_option("--oc", params.oc)->required();
            opts->add_option("--in_width", params.in_width)->required();
            opts->add_option("--in_height", params.in_height)->required();
            opts->add_option("--in_pad", params.in_pad)->required();
            opts->add_option("--out_pad", params.out_pad)->required();
            opts->add_option("--kernel_size", params.kernel_size)->required();
            opts->add_option("--stride", params.stride)->required();
            opts->add_flag("--no_bias", params.no_bias);
            opts->add_flag("--allow_fp16_computations", params.allow_fp16_computations);

        }
    };

    ConvolutionBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , d3d12_device_(d3d12_device)
        , input_data_(params_.batch* params_.ic* params_.in_height* params_.in_width * get_data_type_bytes_width(params_.dt))
        , filter_data_(params_.oc* params_.ic* params_.kernel_size* params_.kernel_size * get_data_type_bytes_width(params_.dt))

    {
        if (!params_.no_bias)
        {
            bias_data_ = std::vector<std::byte>(params_.oc * get_data_type_bytes_width(params_.dt));
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
        const auto [out_width, out_height] = get_output_sizes();
        const auto tensor_out_bytes_width = params_.batch * params_.oc * out_height * out_width * get_data_type_bytes_width(params_.dt);

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
    inline std::pair<std::uint32_t, std::uint32_t> get_output_sizes() const
    {
        const auto out_width = (params_.in_width - params_.kernel_size + params_.in_pad + params_.in_pad) / params_.stride + 1;
        const auto out_height = (params_.in_height - params_.kernel_size + params_.in_pad + params_.in_pad) / params_.stride + 1;
        return { out_width, out_height };
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
            bindings.input.dims = { params_.batch, params_.ic, params_.in_width, params_.in_height };
        }

        {
            bindings.filter.data = filter_data_.data();
            bindings.filter.dt = params_.dt;
            bindings.filter.layout = params_.layout;
            bindings.filter.dims = { params_.oc , params_.ic, params_.kernel_size, params_.kernel_size };
        }
        if(use_bias())
        {
            bindings.bias.data = bias_data_.data();
            bindings.bias.dt = params_.dt;
            bindings.bias.layout = params_.layout;
            bindings.bias.dims = { params_.oc , 1, 1, 1 };
        }
        cpu_op::opts_t opts{};
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
        , conv_(dml::TensorDimensions{params_.batch, params_.ic, params_.in_width, params_.in_height},
            dml::TensorDimensions{ params_.oc, params_.ic, params_.kernel_size, params_.kernel_size},
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
        std::array<std::uint32_t, 3> lws{ 1u, 1u, 1u}; 
        std::uint32_t block_w = 8;
        std::uint32_t block_h = 1;
        std::uint32_t block_oc = 8;

        inline static void add_cli_options(CLI::App* opts, conv_cm_params_t& params)
        {
            opts->add_flag("--dump_asm", params.dump_asm)->default_val(false);
            opts->add_flag("--large_grf", params.large_grf)->default_val(false);
            opts->add_flag("--print_reg_usage", params.print_reg_usage)->default_val(false);
            opts->add_option("--block_w", params.block_w);
            opts->add_option("--block_h", params.block_h);
            opts->add_option("--block_oc", params.block_oc);
            opts->add_option("--lws", params.lws)->delimiter(',');
        }
    };
public:
    ConvolutionCmDispatcher(create_params_t&& params, conv_cm_params_t&& cm_params, IntelExtension& intc_ext, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list)
        : ConvolutionBaseDispatcher(std::move(params), d3d12_device, cmd_list)
        , intc_ext_(intc_ext)
        , d3d12_device_(d3d12_device)
        , cm_params_(std::move(cm_params))
        , output_sizes_(get_output_sizes())
    {
        // root signature
        {
            const auto bindings_size = get_total_descriptor_count();
            std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;
            std::vector<CD3DX12_ROOT_PARAMETER1> root_params;
            ranges.reserve(bindings_size);
            root_params.reserve(bindings_size + 1); // + 1 beacuse of the CM driver path

            std::uint32_t srv_range_reg = 0;
            std::uint32_t uav_range_reg = 0;
            std::uint32_t cbv_range_reg = 0;

            {
                // driver thing
                CD3DX12_ROOT_PARAMETER1 rp{};
                rp.InitAsConstants(1, cbv_range_reg++); 
                root_params.push_back(rp);
            }
            enum class DescType
            {
                eSrv,
                eUav
            };
            auto add_desc_table = [&](DescType type)
            {
                if (type == DescType::eSrv)
                {
                    ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, srv_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE });
                }
                else
                {
                    ranges.push_back({ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, uav_range_reg++, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE });
                }
                CD3DX12_ROOT_PARAMETER1 rp{};
                rp.InitAsDescriptorTable(1u, &ranges.back());
                root_params.push_back(rp);
            };

            // input
            add_desc_table(DescType::eSrv);
            // filter
            add_desc_table(DescType::eSrv);
            if (!params_.no_bias)
            {
                // bias
                add_desc_table(DescType::eSrv);
            }   
            // output uav
            add_desc_table(DescType::eUav);

            if (root_params.size() == 0)
            {
                throw std::runtime_error("Something gone wrong. Why kernel has 0 root params?");
            }

            CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC compute_root_signature_desc;
            compute_root_signature_desc.Init_1_1(static_cast<UINT>(root_params.size()), root_params.data(), 0, nullptr);

            ComPtr<ID3DBlob> signature;
            ComPtr<ID3DBlob> error;
            throw_if_failed(D3DX12SerializeVersionedRootSignature(
                &compute_root_signature_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_1,
                &signature,
                &error), "D3DX12SerializeVersionedRootSignature failed.");

            if (error)
            {
                throw_with_msg("Failed to create root signature, error:" + std::string((LPCSTR)error->GetBufferPointer()));
            }
            throw_if_failed(d3d12_device_->CreateRootSignature(
                0,
                signature->GetBufferPointer(),
                signature->GetBufferSize(),
                IID_PPV_ARGS(&root_signature_)), "CreateRootSignature(...) failed.");
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

        add_define("INPUT_WIDTH", params_.in_width);
        add_define("INPUT_HEIGHT", params_.in_height);
        add_define("INPUT_CHANNELS", params_.ic);

        const auto [out_width, out_height] = get_output_sizes();
        add_define("OUTPUT_WIDTH", out_width);
        add_define("OUTPUT_HEIGHT", out_height);
        add_define("OUTPUT_CHANNELS", params_.oc);

        add_define("BATCH", params_.batch);
        add_define("INPUT_PAD", params_.in_pad);
        add_define("OUTPUT_PAD", params_.out_pad);
        add_define("USE_BIAS", !params_.no_bias);
        add_define("KERNEL_SIZE", params_.kernel_size);
        add_define("STRIDE", params_.stride);

        add_define("BLOCK_W", cm_params_.block_w);
        add_define("BLOCK_H", cm_params_.block_h);
        add_define("BLOCK_OC", cm_params_.block_oc);

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
            const auto path = "conv_1x1_nchw_b1_fp16.cpp";
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
        pso_ = intc_ext_.create_pipeline(byte_code, build_options_final, root_signature_.Get());
    }

    std::uint32_t get_total_descriptor_count() override
    {
        // input, weights, output
        std::uint32_t descriptor_count = 3;
        if (!params_.no_bias)
        {
            descriptor_count++;
        }
        return descriptor_count;
    }

    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        const auto desc_heap_incrs_size = d3d12_device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // i.e. add weights reorder


        const auto base_cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE{ cpu_handle };
        const auto base_gpu_handle = CD3DX12_GPU_DESCRIPTOR_HANDLE{ gpu_handle };

        enum class ViewType
        {
            eSrv = 0,
            eUav = 1
        };

        std::vector<std::pair<ViewType, ID3D12Resource*>> resources_list;
        resources_list.reserve(get_total_descriptor_count());
        resources_list.push_back({ ViewType::eSrv, input_buffer_.Get() });
        resources_list.push_back({ ViewType::eSrv, filter_buffer_.Get() });
        if (bias_buffer_)
        {
            resources_list.push_back({ ViewType::eSrv, bias_buffer_.Get() });
        }
        resources_list.push_back({ ViewType::eUav, output_buffer_.Get() });

        // reserve handles
        gpu_handles_.reserve(get_total_descriptor_count());
        // create handles

        for (std::size_t i = 0; i < resources_list.size(); i++)
        {
            auto cpu_handle = CD3DX12_CPU_DESCRIPTOR_HANDLE(base_cpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size);
            gpu_handles_.push_back(CD3DX12_GPU_DESCRIPTOR_HANDLE(base_gpu_handle, static_cast<int32_t>(i), desc_heap_incrs_size));

            auto& resource_view_type = resources_list[i].first;
            auto& resource = resources_list[i].second;
            const auto res_desc = resource->GetDesc();
            assert(res_desc.Dimension == D3D12_RESOURCE_DIMENSION::D3D12_RESOURCE_DIMENSION_BUFFER);

            if (resource_view_type == ViewType::eSrv)
            {
                D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                desc.Format = DXGI_FORMAT_R8_UINT;
                desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

                desc.Buffer.StructureByteStride = 0;
                desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
                desc.Buffer.FirstElement = 0;
                desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

                d3d12_device_->CreateShaderResourceView(resource, &desc, cpu_handle);
            }
            else
            {
                D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
                desc.Format = DXGI_FORMAT_R8_UINT;
                desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;

                desc.Buffer.StructureByteStride = 0;
                desc.Buffer.NumElements = static_cast<UINT>(res_desc.Width);
                desc.Buffer.FirstElement = 0;
                desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

                d3d12_device_->CreateUnorderedAccessView(resource, nullptr, &desc, cpu_handle);
            }
        }
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

        const auto gws_x = round_up_next_multiple(output_sizes_.first, cm_params_.block_w) / cm_params_.block_w;
        const auto gws_y = round_up_next_multiple(output_sizes_.second, cm_params_.block_h) / cm_params_.block_h;
        const auto gws_z = params_.oc / cm_params_.block_oc;

        assert(gws_x % cm_params_.lws[0] == 0);
        assert(gws_y % cm_params_.lws[1] == 0);
        assert(gws_z % cm_params_.lws[2] == 0);

        const auto thg_x = gws_x / cm_params_.lws[0];
        const auto thg_y = gws_y / cm_params_.lws[1];
        const auto thg_z = gws_z / cm_params_.lws[2];
        cmd_list->Dispatch(thg_x, thg_y, thg_z);
    }

private:
    conv_cm_params_t cm_params_;
    ID3D12Device* d3d12_device_;
    IntelExtension& intc_ext_;
    std::vector<CD3DX12_GPU_DESCRIPTOR_HANDLE> gpu_handles_;

    ComPtr<ID3D12PipelineState> pso_;
    ComPtr<ID3D12RootSignature> root_signature_;

    std::pair<std::uint32_t, std::uint32_t> output_sizes_;
};

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
        , input_data_(params_.batch * params_.ic * params_.in_width * params.in_height * get_data_type_bytes_width(params_.dt))
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

struct CliOptions
{
    NodeType node_type = NodeType::eCount;
    std::uint32_t dispatch_iterations = 1;
    bool no_conformance_check = false;

    // generic type of layers params
    GemmDispatcher::create_params_t gemm_opts{};
    ConvolutionBaseDispatcher::create_params_t conv_opts{};
    SoftmaxDispatcher::create_params_t softmax_opts{};

    // specific for implementation
    ConvolutionCmDispatcher::conv_cm_params_t conv_cm_params{};
};

int main()
{
    constexpr const std::uint32_t MAX_ITERATIONS = 10'000;

    CliOptions opts;
    CLI::App dml_runner_app{ "App to microbenchmark and developer dml kernels.", "DirectML runner." };
    dml_runner_app.add_option("--type", opts.node_type, "Name of the type of layer to run.")
        ->required()->check(CLI::IsMember({ NodeType::eConvDml, NodeType::eConvCm , NodeType::eGemm, NodeType::eSoftmax }))->
        transform(CLI::Transformer(std::map<std::string, NodeType>{
            { "conv_dml", NodeType::eConvDml },
            { "conv_cm", NodeType::eConvCm },
            { "gemm", NodeType::eGemm },
            { "softmax", NodeType::eSoftmax }
    }, CLI::ignore_case, CLI::ignore_underscore));
    dml_runner_app.add_option("--iters", opts.dispatch_iterations, "How many iterations to run.")->check(CLI::Range(1u, MAX_ITERATIONS));
    dml_runner_app.add_flag("--no_conform", opts.no_conformance_check);

    // generic type of layers options
    auto gemm_option_groups = dml_runner_app.add_subcommand("gemm_opts", "Options for genn layer.");
    GemmDispatcher::create_params_t::add_cli_options(gemm_option_groups, opts.gemm_opts);
    auto conv_option_groups = dml_runner_app.add_subcommand("conv_opts", "Options for convolution layer.");
    ConvolutionBaseDispatcher::create_params_t::add_cli_options(conv_option_groups, opts.conv_opts);
    auto softmax_option_groups = dml_runner_app.add_subcommand("softmax_opts", "Options for softmax layer.");
    SoftmaxDispatcher::create_params_t::add_cli_options(softmax_option_groups, opts.softmax_opts);

    // specific for implementation
    auto conv_cm_option_groups = dml_runner_app.add_subcommand("conv_cm_opts", "Options for convolution layer with CM implementation.");
    ConvolutionCmDispatcher::conv_cm_params_t::add_cli_options(conv_cm_option_groups, opts.conv_cm_params);

    try {
        dml_runner_app.parse();
    }
    catch (const CLI::ParseError& e) {
        return dml_runner_app.exit(e);
    }

    const auto dumped_config = dml_runner_app.config_to_str(true);
    std::cout << std::format("Running app with config:\n {}", dumped_config);

    assert(opts.node_type != NodeType::eCount);
    if ((opts.node_type == NodeType::eConvCm || opts.node_type == NodeType::eConvCm)
        && !conv_option_groups->parsed())
    {
        std::cout << "Convoltion options not set.\n";
        return -1;
    }
    if (opts.node_type == NodeType::eGemm && !gemm_option_groups->parsed())
    {
        std::cout << "Gemm options not set.\n";
        return -1;
    }
    if (opts.node_type == NodeType::eSoftmax && !softmax_option_groups->parsed())
    {
        std::cout << "Softmax options not set.\n";
        return -1;
    }

    try
    {
        ComPtr<ID3D12Device> d3d12_device;
        ComPtr<ID3D12CommandQueue> command_queue;
        ComPtr<ID3D12CommandAllocator> command_allocator;
        ComPtr<ID3D12GraphicsCommandList> command_list;
        initalize_d3d12(d3d12_device, command_queue, command_allocator, command_list);
        auto dml_device = create_dml_device(d3d12_device.Get());
        assert(opts.dispatch_iterations < MAX_ITERATIONS);
        auto performance_collector = initialize_d3d12_performance_collector(d3d12_device.Get(), MAX_ITERATIONS);

        auto intel_extension_d3d12 = IntelExtension(d3d12_device.Get());
        // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
        ComPtr<IDMLCommandRecorder> dml_command_recorder;
        throw_if_failed(dml_device->CreateCommandRecorder(IID_PPV_ARGS(dml_command_recorder.ReleaseAndGetAddressOf())), "create dml command recorder");

        std::unique_ptr<NodeDispatcher> node;
        if (opts.node_type == NodeType::eGemm)
        {
            node = std::make_unique<GemmDispatcher>(std::move(opts.gemm_opts), 
                d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eConvDml)
        {
            node = std::make_unique<ConvolutionDirectMLDispatcher>(std::move(opts.conv_opts),
                d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eConvCm)
        {
            node = std::make_unique<ConvolutionCmDispatcher>(std::move(opts.conv_opts), std::move(opts.conv_cm_params),
                intel_extension_d3d12, d3d12_device.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eSoftmax)
        {
            node = std::make_unique<SoftmaxDispatcher>(std::move(opts.softmax_opts),
                d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
        }
        else
        {
            assert(false && "Unknown node type!");
        }

        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());
        const auto descriptors_count = node->get_total_descriptor_count();
        
        // bind descriptor heap
        auto descriptor_heap = create_descriptor_heap(d3d12_device.Get(), descriptors_count);
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);


        // initalize
        node->initialize(command_list.Get(), descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        // 
        // Bind and execute the operator on the GPU.
        // 
        // 
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        for (std::uint32_t i = 0; i < opts.dispatch_iterations; ++i)
        {
            performance_collector.add_timestamp(command_list.Get());
            node->execute(command_list.Get());
            performance_collector.add_timestamp(command_list.Get());
        }
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        const auto device_remove_reason = d3d12_device->GetDeviceRemovedReason();
        if (device_remove_reason != S_OK)
        {
            std::cout << std::format("Device removal. Reason: {}\n", device_remove_reason);
        }

        if (opts.no_conformance_check)
        {
            std::cout << std::format("Skipping conformance check as requested by cmd line.\n");
        }
        else
        {
            const auto conformance_result = node->validate_conformance(command_queue.Get(), command_allocator.Get(), command_list.Get());
            std::cout << std::format("Conformance {}. Tested values (tensor out elements count): {} \n", conformance_result.passed, conformance_result.tested_samples_count);
            std::cout << std::format("Biggest difference in the output tensor: {}. It is in the epsilion range: {}. \n", conformance_result.biggest_difference, conformance_result.epsilon);
        }

        // Copy the timing data back
        command_list->ResolveQueryData(
            performance_collector.timestamp_query_heap.Get(),
            D3D12_QUERY_TYPE_TIMESTAMP,
            0,
            performance_collector.timestamp_index,
            performance_collector.timestamp_readback_buffer.Get(),
            0);
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        uint64_t timestamp_frequency = 0;
        command_queue->GetTimestampFrequency(&timestamp_frequency);

        const auto timestamps_timings = get_timestamps_timings_from_ptr<std::chrono::microseconds>(timestamp_frequency, performance_collector.timestamp_readback, performance_collector.timestamp_index);
        performance_collector.timestamp_index = 0;

        std::vector<std::chrono::microseconds> timings(timestamps_timings.size() / 2);
        for (uint32_t i = 0; i < timings.size(); i++)
        {
            const auto t0 = timestamps_timings[i * 2];
            const auto t1 = timestamps_timings[i * 2 + 1];
            timings[i] = t1 - t0;
        }

        print_performance_stats(timings);
    }
    catch (std::exception e)
    {
        std::cout << std::format("Exception caught: {} \n", e.what());
    }

    return 0;
}