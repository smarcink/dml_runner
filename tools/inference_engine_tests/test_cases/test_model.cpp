#include "test_gpu_context.h"
#include "inference_engine.hpp"
#include "inference_engine_model.hpp"
#include "utils.h"

#include <numeric>
#include <vector>
#include <array>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl.hpp>

inline dnnl::graph::logical_tensor to_onednn_logical_tensor(std::size_t onednn_logical_tensor_id, const inference_engine::Tensor& tensor)
{
    dnnl::graph::logical_tensor::dims dims{};
    for (const auto& d : tensor.dims)
    {
        dims.push_back(static_cast<dnnl::graph::logical_tensor::dim>(d));
    }
    dnnl::graph::logical_tensor::data_type data_type = dnnl::graph::logical_tensor::data_type::undef;
    switch (tensor.data_type)
    {
    case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32:
    {
        data_type = dnnl::graph::logical_tensor::data_type::f32;
        break;
    }
    case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16:
    {
        data_type = dnnl::graph::logical_tensor::data_type::f16;
        break;
    }
    default:
        assert(!"add more data types support to to_onednn_logical_tensor() function");
    }
    return dnnl::graph::logical_tensor(onednn_logical_tensor_id, data_type, dims, dnnl::graph::logical_tensor::layout_type::strided);
}

inline dnnl::graph::tensor onednn_allocate_graph_mem(const dnnl::graph::logical_tensor& lt, void* data_buffer, const dnnl::engine& eng) 
{
    const auto mem_size = lt.get_mem_size();
    return dnnl::graph::tensor{ lt, eng, data_buffer };
}

TEST(ModelTest, Activation_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    inference_engine::ModelDescriptor md{};
    auto port_id = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP32 });
    auto out_node = md.add_activation(inference_engine_activation_desc_t{ port_id, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU });

    inference_engine::TensorMapping inputs{};
    inputs[port_id] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 2, 4, 4 });
    const auto& input = inputs[port_id];

    auto model = inference_engine::Model(ctx, stream, md, inputs);

    const auto tensor_elements_count = accumulate_tensor_dims(input);
    const auto tensor_size_bytes = tensor_elements_count * sizeof(float);
    std::vector<float> input_data_host(tensor_elements_count, 0.0f);

    // randomize data
    std::mt19937 random_generator(42); // static, create it once!
    std::uniform_real_distribution<float> uniform_distribution(-5.0f, 5.0f);
    randomize_linear_container_float(random_generator, uniform_distribution, input_data_host);

    auto input_buffer = device.allocate_resource(tensor_size_bytes);
    device.upload_data_to_resource<float>(input_buffer, input_data_host);

    auto output_buffer = device.allocate_resource(tensor_size_bytes);
    // set resources
    model.set_resource(port_id, input_buffer);
    model.set_resource(out_node, output_buffer);
    
    // ask model for output size (we know that there has to be 1 output in this test case)
    auto output = model.get_outputs()[out_node];
    ASSERT_EQ(output.data_type, input.data_type);
    ASSERT_EQ(output.dims, input.dims);

    // finally execute model
    model.execute(stream);

    // readback data and wait for execution
    const auto data_out = device.readback_data_from_resource<float>(output_buffer);
    std::vector<float> data_out_ref(data_out.size(), 0.0f);

    // conformance with OneDNN
    {
        dnnl::engine onednn_engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream onednn_stream{ onednn_engine };
        dnnl::set_verbose(2);

        dnnl::graph::logical_tensor onednn_input = to_onednn_logical_tensor(0, inputs[port_id]);
        dnnl::graph::logical_tensor onednn_output(1, dnnl::graph::logical_tensor::data_type::f32);
        dnnl::graph::op onednn_activ(0, dnnl::graph::op::kind::ReLU, { onednn_input }, { onednn_output });

        dnnl::graph::graph g(dnnl::engine::kind::cpu);
        g.add_op(onednn_activ);
        g.finalize();
        auto partitions = g.get_partitions();

        std::unordered_map<size_t, dnnl::graph::logical_tensor> id_to_queried_logical_tensors;
        std::vector<dnnl::graph::compiled_partition> cps;
        cps.reserve(partitions.size());
        for (auto& p : partitions)
        {
            std::vector<dnnl::graph::logical_tensor> onednn_inputs = p.get_input_ports();
            std::vector<dnnl::graph::logical_tensor> onednn_outputs = p.get_output_ports();
 
            // Update input logical tensors with concrete shape and layout
            for (auto& input : onednn_inputs) {
                const auto id = input.get_id();
                // If the tensor is an output of another partition, use the cached logical tensor
                if (id_to_queried_logical_tensors.find(id) != id_to_queried_logical_tensors.end())
                {
                    input = id_to_queried_logical_tensors[id];
                }
                else
                {
                    input = onednn_input;
                }
            }

            // Update output logical tensors with concrete shape and layout
            for (auto& output : onednn_outputs)
            {
                const auto id = output.get_id();
                output = dnnl::graph::logical_tensor{ id, output.get_data_type(), DNNL_GRAPH_UNKNOWN_NDIMS, dnnl::graph::logical_tensor::layout_type::strided };
            }

            // compile partition
            cps.push_back(p.compile(onednn_inputs, onednn_outputs, onednn_engine));

            // Update output logical tensors with queried one
            for (auto& output : onednn_outputs) {
                const auto id = output.get_id();
                output = cps.back().query_logical_tensor(id);
                id_to_queried_logical_tensors[id] = output;
            }

            // Allocate memory for the partition, and bind the data buffers with
            // input and output logical tensors
            std::vector<dnnl::graph::tensor> inputs_ts;
            for (auto i = 0; i < inputs.size(); i++)
            {
                inputs_ts.push_back(onednn_allocate_graph_mem(onednn_inputs[i], input_data_host.data(), onednn_engine));
            }
            std::vector<dnnl::graph::tensor> outputs_ts;
            for (auto i = 0; i < inputs.size(); i++)
            {
                outputs_ts.push_back(onednn_allocate_graph_mem(onednn_outputs[i], data_out_ref.data(), onednn_engine));
            }

            cps.back().execute(onednn_stream, inputs_ts, outputs_ts);
        }
    }
    // validate conformance
    for (int i = 0; i < tensor_elements_count; i++)
    {
        // relu activation reference
        const auto& reference = data_out_ref[i];
        const auto& real_data = data_out[i];
        // for now switch it off so we have the test passing and result is not polluted with fake fail
        ASSERT_FLOAT_EQ(real_data, reference) << "idx: " << i;
    }
}

void test_fusion_activation_impl(int num_activations)
{

    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const auto data_type = INFERENCE_ENGINE_DATA_TYPE_FP32;
    inference_engine::ModelDescriptor md{};
    auto input_a = md.add_port(inference_engine_port_desc_t{ data_type });
    auto input_b = md.add_port(inference_engine_port_desc_t{ data_type });
    auto port_matmul_out = md.add_matmul(inference_engine_matmul_desc_t{ input_a, input_b });
    std::vector<inference_engine::NodeID> activation_nodes;
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_LINEAR;
    activation_desc.params.linear.a = 2.0f;
    activation_desc.params.linear.b = 0.5f;
    for (int i = 0; i < num_activations; ++i)
    {
        activation_desc.input = i == 0 ? port_matmul_out : activation_nodes.back();
        activation_nodes.push_back(md.add_activation(activation_desc));
    }

    inference_engine::TensorMapping inputs{};
    inputs[input_a] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 1, 32, 32 });
    inputs[input_b] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 1, 32, 32 });

    auto model = inference_engine::Model(ctx, stream, md, inputs);

    const auto tensor_a_elements_count = accumulate_tensor_dims(inputs[input_a]);
    const auto tensor_a_bytes_width = tensor_a_elements_count * sizeof(float);
    const auto tensor_b_elements_count = accumulate_tensor_dims(inputs[input_b]);
    const auto tensor_b_bytes_width = tensor_b_elements_count * sizeof(float);
    std::vector<float> input_a_data_host(tensor_a_elements_count, 1.0f);
    std::vector<float> input_b_data_host(tensor_b_bytes_width, 1.0f);

    auto input_a_buffer = device.allocate_resource(tensor_a_bytes_width);
    device.upload_data_to_resource<float>(input_a_buffer, input_a_data_host);
    auto input_b_buffer = device.allocate_resource(tensor_b_bytes_width);
    device.upload_data_to_resource<float>(input_b_buffer, input_b_data_host);

    const auto outputs_mappings = model.get_outputs();
    ASSERT_EQ(outputs_mappings.size(), 1);
    ASSERT_NE(outputs_mappings.find(activation_nodes.back()), std::end(outputs_mappings));
    const auto& output_mapping = outputs_mappings.at(activation_nodes.back());
    ASSERT_EQ(output_mapping.data_type, data_type);
    const std::vector<std::uint64_t> expected_output_size = { 1ull, 1ull, 32ull, 32ull };
    ASSERT_EQ(output_mapping.dims, expected_output_size);

    const auto output_tensor_elemenets_count = accumulate_tensor_dims(output_mapping);
    const auto output_tensor_bytes_width = output_tensor_elemenets_count * sizeof(float);

    auto output_buffer = device.allocate_resource(output_tensor_bytes_width);
    // set resources
    model.set_resource(input_a, input_a_buffer);
    model.set_resource(input_b, input_b_buffer);
    model.set_resource(outputs_mappings.begin()->first, output_buffer);

    // finally execute model
    model.execute(stream);

    // readback data and wait for execution
    const auto data_out = device.readback_data_from_resource<float>(output_buffer);


    // validate conformance
    for (int i = 0; i < data_out.size(); i++)
    {
        // relu activation reference
        const auto matmul_result = 32.0f;
        auto reference = matmul_result;
        for (int j = 0; j < num_activations; ++j)
            reference = activation_desc.params.linear.a * reference + activation_desc.params.linear.b;

        const auto& real_data = data_out[i];
        // for now switch it off so we have the test passing and result is not polluted with fake fail
        ASSERT_FLOAT_EQ(real_data, reference) << "idx: " << i;
    }
}

// we tried INSTANTIATE_TEST_SUITE_P(VariablePorts, ModelTestWithParams, ::testing::Values(1, 5));, but DX objectes caused the app to crash for some reason...

TEST(ModelTest, MatMul_fused_activation_single)
{
    test_fusion_activation_impl(1); // no activation should be fused as it's the last one and the output node
}

TEST(ModelTest, MatMul_fused_activation_two)
{
    test_fusion_activation_impl(2); // one activation should be fused, 
}

TEST(ModelTest, MatMul_fused_activation_five)
{
    test_fusion_activation_impl(5);
}

TEST(ModelTest, MatMul_6_nodes)
{
    // *   *  *
    //  \ /  /
    //   *  *  // matmul, activation
    //    \/
    //     * // mat mul

    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    inference_engine::ModelDescriptor md{};

    const auto data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = data_type;
    auto input_a = md.add_port(input_desc);
    auto input_b = md.add_port(input_desc);
    auto input_c = md.add_port(input_desc);
    // matmul
    auto port_matmul_a = md.add_matmul(inference_engine_matmul_desc_t{ input_a, input_b });
    // activation
    auto port_activation = md.add_activation(inference_engine_activation_desc_t{ input_c, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU });
    // MatMul final
    auto port_matmul_out_final = md.add_matmul(inference_engine_matmul_desc_t{ port_matmul_a, port_activation });

    inference_engine::TensorMapping inputs{};
    inputs[input_a] = inference_engine::Tensor(data_type, { 1, 1, 8, 16 });
    inputs[input_b] = inference_engine::Tensor(data_type, { 1, 2, 16, 32 });
    inputs[input_c] = inference_engine::Tensor(data_type, { 1, 2, 32, 64 });

    auto model = inference_engine::Model(ctx, stream, md, inputs);
    const auto outputs_mappings = model.get_outputs();
    ASSERT_EQ(outputs_mappings.size(), 1);
    ASSERT_NE(outputs_mappings.find(port_matmul_out_final), std::end(outputs_mappings));
    const auto& output_mapping = outputs_mappings.at(port_matmul_out_final);
    ASSERT_EQ(output_mapping.data_type, data_type);
    const std::vector<std::uint64_t> expected_output_size = { 1, 1, 8, 64 };
    ASSERT_EQ(output_mapping.dims, expected_output_size);
}

TEST(ModelTest, ConvPlusAddFusion)
{
    // *           * port, port
    //  \         /
    //   *       *   conv, conv
    //    \     /
    //     \   *     activation
    //      \ /
    //       *       elementwise_add
    //       *       activation at the end, so that we can fuse "inner" nodes

    // we'll fuse nodes on the right side of the graph
    // *        * port, port
    //  \      /
    //   *    /   conv
    //    \  /
    //     * conv fused with activation fused with elementwise_add 
    //     * activation at the end, unchanged


    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);
    const auto data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;

    inference_engine::ModelDescriptor md{};

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = data_type;
    auto input_a = md.add_port(input_desc);
    auto input_b = md.add_port(input_desc);

    // Conv left
    auto port_conv_a = md.add_convolution(inference_engine_convolution_desc_t{ input_a });
    md.set_node_name(port_conv_a, "conv_a");

    // Conv right
    auto port_conv_b = md.add_convolution(inference_engine_convolution_desc_t{ input_b });
    md.set_node_name(port_conv_b, "conv_b");

    // activation
    auto activation = md.add_activation(inference_engine_activation_desc_t{port_conv_b, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU });

    // elementwise_add
    inference_engine_elementwise_desc_t add_desc_final{};
    add_desc_final.type = inference_engine_elementwise_type_t::INFERENCE_ENGINE_ELEMENTWISE_TYPE_ADD;
    add_desc_final.input_a = port_conv_a;
    add_desc_final.input_b = activation;
    auto port_add_final = md.add_elementwise(add_desc_final);

    // activation final
    auto final_activation = md.add_activation(inference_engine_activation_desc_t{port_add_final, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU });
    md.set_node_name(final_activation, "final_activation");

    // define input mappings
    inference_engine::TensorMapping inputs{};
    inputs[input_a] = inference_engine::Tensor(data_type, { 1, 1, 4, 4 });
    inputs[input_b] = inference_engine::Tensor(data_type, { 1, 1, 4, 4 });

    auto model = inference_engine::Model(ctx, stream, md, inputs);
    const auto outputs_mappings = model.get_outputs();
    ASSERT_EQ(outputs_mappings.size(), 1);
    ASSERT_NE(outputs_mappings.find(final_activation), std::end(outputs_mappings));
    const auto& output_mapping = outputs_mappings.at(final_activation);
    ASSERT_EQ(output_mapping.data_type, data_type);
    const std::vector<std::uint64_t> expected_output_size = { 1, 1, 4, 4 };
    ASSERT_EQ(output_mapping.dims, expected_output_size);

    auto input_a_buffer = device.allocate_resource(accumulate_tensor_dims(inputs[input_a]));
    auto input_b_buffer = device.allocate_resource(accumulate_tensor_dims(inputs[input_b]));
    auto output_buffer = device.allocate_resource(accumulate_tensor_dims(output_mapping));

    // set resources
    model.set_resource(input_a, input_a_buffer);
    model.set_resource(input_b, input_a_buffer);
    model.set_resource(final_activation, output_buffer);

    // can execute here (assign resources call execute)
    model.execute(stream);

    // do conformance check etc ..
    // ...
}