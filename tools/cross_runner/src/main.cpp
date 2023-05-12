#include "dx12_utils.h"
#include "gemm.h"
#include "conv.h"
#include "softmax.h"
#include "mvn.h"
#include "layers_utils.h"

#include <dml_types.hpp>
#include <dml_convolution.hpp>

#include <iostream>
#include <optional>
#include <span>
#include <format>
#include <random>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>

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

struct CliOptions
{
    NodeType node_type = NodeType::eCount;
    std::uint32_t dispatch_iterations = 1;
    bool no_conformance_check = false;

    // generic type of layers params
    GemmBaseDispatcher::create_params_t gemm_opts{};
    ConvolutionBaseDispatcher::create_params_t conv_opts{};
    SoftmaxBaseDispatcher::create_params_t softmax_opts{};
    MvnBaseDispatcher::create_params_t mvn_opts{};

    // specific for implementation
    ConvolutionCmDispatcher::conv_cm_params_t conv_cm_params{};
    MvnCmDispatcher::mvn_cm_params_t mvn_cm_params{};
    SoftmaxCmDispatcher::softmax_cm_params_t softmax_cm_params{};
    GemmCmDispatcher::cm_params_t gemm_cm_params{};
};

int main()
{
    libdml::DeviceInfo device_info{};
    device_info.platform = libdml::HwPlatform::eDG2;
    device_info.eu_count = 512;

    libdml::ConvolutionDescriptor conv_desc{};
    const auto convs_impls = libdml::get_convolution_implementation_list(device_info, conv_desc);

    constexpr const std::uint32_t MAX_ITERATIONS = 10'000;

    CliOptions opts;
    CLI::App dml_runner_app{ "App to microbenchmark and developer dml kernels.", "DirectML runner." };
    dml_runner_app.add_option("--type", opts.node_type, "Name of the type of layer to run.")
        ->required()->check(CLI::IsMember({ NodeType::eConvDml, NodeType::eConvCm, NodeType::eGemmDml, NodeType::eGemmCm, NodeType::eSoftmaxDml, NodeType::eSoftmaxCm, NodeType::eMvnDml, NodeType::eMvnCm }))->
        transform(CLI::Transformer(std::map<std::string, NodeType>{
            { "conv_dml", NodeType::eConvDml },
            { "conv_cm", NodeType::eConvCm },
            { "gemm_dml", NodeType::eGemmDml },
            { "gemm_cm", NodeType::eGemmCm },
            { "softmax_dml", NodeType::eSoftmaxDml },
            { "softmax_cm", NodeType::eSoftmaxCm },
            { "mvn_dml", NodeType::eMvnDml },
            { "mvn_cm", NodeType::eMvnCm },
    }, CLI::ignore_case, CLI::ignore_underscore));
    dml_runner_app.add_option("--iters", opts.dispatch_iterations, "How many iterations to run.")->check(CLI::Range(1u, MAX_ITERATIONS));
    dml_runner_app.add_flag("--no_conform", opts.no_conformance_check);

    // generic type of layers options
    auto gemm_option_groups = dml_runner_app.add_subcommand("gemm_opts", "Options for genn layer.");
    GemmBaseDispatcher::create_params_t::add_cli_options(gemm_option_groups, opts.gemm_opts);
    auto conv_option_groups = dml_runner_app.add_subcommand("conv_opts", "Options for convolution layer.");
    ConvolutionBaseDispatcher::create_params_t::add_cli_options(conv_option_groups, opts.conv_opts);
    auto softmax_option_groups = dml_runner_app.add_subcommand("softmax_opts", "Options for softmax layer.");
    SoftmaxBaseDispatcher::create_params_t::add_cli_options(softmax_option_groups, opts.softmax_opts);
    auto mvn_option_groups = dml_runner_app.add_subcommand("mvn_opts", "Options for mvn layer.");
    MvnBaseDispatcher::create_params_t::add_cli_options(mvn_option_groups, opts.mvn_opts);

    // specific for implementation
    auto conv_cm_option_groups = dml_runner_app.add_subcommand("conv_cm_opts", "Options for convolution layer with CM implementation.");
    ConvolutionCmDispatcher::conv_cm_params_t::add_cli_options(conv_cm_option_groups, opts.conv_cm_params);
    auto mvn_cm_option_groups = dml_runner_app.add_subcommand("mvn_cm_opts", "Options for mvn layer with CM implementation.");
    MvnCmDispatcher::mvn_cm_params_t::add_cli_options(mvn_cm_option_groups, opts.mvn_cm_params);
    auto softmax_cm_option_groups = dml_runner_app.add_subcommand("softmax_cm_opts", "Options for softmax layer with CM implementation.");
    SoftmaxCmDispatcher::softmax_cm_params_t::add_cli_options(softmax_cm_option_groups, opts.softmax_cm_params);
    auto gemm_cm_option_groups = dml_runner_app.add_subcommand("gemm_cm_opts", "Options for gemm layer with CM implementation.");
    GemmCmDispatcher::cm_params_t::add_cli_options(gemm_cm_option_groups, opts.gemm_cm_params);

    try {
        dml_runner_app.parse();
    }
    catch (const CLI::ParseError& e) {
        return dml_runner_app.exit(e);
    }

    const auto dumped_config = dml_runner_app.config_to_str(true);
    std::cout << std::format("Running app with config:\n {}", dumped_config);

    assert(opts.node_type != NodeType::eCount);
    if ((opts.node_type == NodeType::eConvCm || opts.node_type == NodeType::eConvDml)
        && !conv_option_groups->parsed())
    {
        std::cout << "Convoltion options not set.\n";
        return -1;
    }
    if ((opts.node_type == NodeType::eGemmDml || opts.node_type == NodeType::eGemmCm) && !gemm_option_groups->parsed())
    {
        std::cout << "Gemm options not set.\n";
        return -1;
    }
    if ((opts.node_type == NodeType::eSoftmaxDml || opts.node_type == NodeType::eSoftmaxCm) && !softmax_option_groups->parsed())
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
        if (opts.node_type == NodeType::eGemmDml)
        {
            node = std::make_unique<GemmDmlDispatcher>(std::move(opts.gemm_opts), 
                d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eGemmCm)
        {
            node = std::make_unique<GemmCmDispatcher>(std::move(opts.gemm_opts), std::move(opts.gemm_cm_params),
                intel_extension_d3d12, d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
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
        else if (opts.node_type == NodeType::eSoftmaxDml)
        {
            node = std::make_unique<SoftmaxDmlDispatcher>(std::move(opts.softmax_opts),
                d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eSoftmaxCm)
        {
            node = std::make_unique<SoftmaxCmDispatcher>(std::move(opts.softmax_opts), std::move(opts.softmax_cm_params),
                intel_extension_d3d12, d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eMvnDml)
        {
            node = std::make_unique<MvnDmlDispatcher>(std::move(opts.mvn_opts),
                d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eMvnCm)
        {
            node = std::make_unique<MvnCmDispatcher>(std::move(opts.mvn_opts), std::move(opts.mvn_cm_params),
                intel_extension_d3d12, d3d12_device.Get(), dml_device.Get(), dml_command_recorder.Get(), command_list.Get());
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
        std::cerr << std::format("Exception caught: {} \n", e.what());
        return -1; 
    }
    return 0;
}