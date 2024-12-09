#include "test_gpu_context.h"
#include "inference_engine_model.h"
#include "utils.h"

#include <numeric>
#include <vector>
#include <array>

TEST(ModelTest, Activation_0)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    auto md = inferenceEngineCreateModelDescriptor();

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32;
    auto port_id = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(port_id, INFERENCE_ENGINE_INVALID_NODE_ID);

    inference_engine_activation_desc_t activation_desc{};
    activation_desc.type = inference_engine_activation_type_t::INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    activation_desc.input = port_id;
    auto out_node = inferenceEngineModelDescriptorAddActivation(md, activation_desc);
    ASSERT_NE(out_node, INFERENCE_ENGINE_INVALID_NODE_ID);

    inference_engine_tensor_mapping_t input{};
    input.id = port_id;
    input.tensor.data_type = input_desc.data_type;  
    set_array(input.tensor.dims, 1, 2, 4, 4);
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, &input, 1);
    ASSERT_NE(model, nullptr);

    const auto tensor_elements_count = accumulate_tensor_dims(input.tensor);
    const auto tensor_size_bytes = tensor_elements_count * sizeof(float);
    std::vector<float> input_data_host(tensor_elements_count, 0.0f);

    // randomize data
    std::mt19937 random_generator(42); // static, create it once!
    std::uniform_real_distribution<float> uniform_distribution(-5.0f, 5.0f);
    randomize_linear_container_float(random_generator, uniform_distribution, input_data_host);

    auto input_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_size_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    // copy data to GPU
    auto upload_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_size_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_COPY_SOURCE);
    std::byte* upload_mapped_ptr = nullptr;
    upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
    std::size_t memcopy_offset = 0;
    std::memcpy(upload_mapped_ptr, input_data_host.data(), tensor_size_bytes);
    upload_buffer->Unmap(0, nullptr);
    G_DX12_ENGINE.command_list->CopyResource(input_buffer.Get(), upload_buffer.Get());
    dispatch_resource_barrier(G_DX12_ENGINE.command_list.Get(), { CD3DX12_RESOURCE_BARRIER::Transition(input_buffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) });
    G_DX12_ENGINE.wait_for_execution();
    auto output_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_size_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // set resources
    ASSERT_EQ(inferenceEngineModelSetResource(model, port_id, reinterpret_cast<inference_engine_resource_t>(input_buffer.Get())), true);
    ASSERT_EQ(inferenceEngineModelSetResource(model, out_node, reinterpret_cast<inference_engine_resource_t>(output_buffer.Get())), true);

    // ask model for output size (we know that there has to be 1 output in this test case)
    inference_engine_tensor_mapping_t output_mapping{};
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, &output_mapping, nullptr), true);
    ASSERT_EQ(output_mapping.id, out_node);
    ASSERT_EQ(output_mapping.tensor.data_type, input.tensor.data_type);
    for (auto i = 0; i < 4; i++)
    {
        ASSERT_EQ(output_mapping.tensor.dims[i], input.tensor.dims[i]);
    }
   
    // finally execute model
    ASSERT_EQ(inferenceEngineExecuteModel(model, stream), true);

    // readback data and wait for execution
    auto readback_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_size_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    dispatch_resource_barrier(G_DX12_ENGINE.command_list.Get(), { CD3DX12_RESOURCE_BARRIER::Transition(output_buffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)});
    G_DX12_ENGINE.command_list->CopyResource(readback_buffer.Get(), output_buffer.Get());
    G_DX12_ENGINE.wait_for_execution();

    // copy output data to host
    std::vector<float> data_out(tensor_elements_count);
    std::byte* readback_mapped_ptr = nullptr;
    readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
    std::memcpy(data_out.data(), readback_mapped_ptr, tensor_size_bytes);
    readback_buffer->Unmap(0, nullptr);

    // validate conformance
    for (int i = 0; i < tensor_elements_count; i++)
    {
        // relu activation reference
        const auto reference = std::max(input_data_host[i], 0.0f);
        const auto& real_data = data_out[i];
        // for now switch it off so we have the test passing and result is not polluted with fake fail
        ASSERT_FLOAT_EQ(real_data, reference) << "idx: " << i;
    }

    inferenceEngineDestroyModelDescriptor(md);
    inferenceEngineDestroyModel(model);
    inferenceEngineDestroyContext(ctx);
}

void test_fusion_activation_impl(int num_activations)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    auto md = inferenceEngineCreateModelDescriptor();

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32;
    auto input_a = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(input_a, INFERENCE_ENGINE_INVALID_NODE_ID);
    auto input_b = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(input_b, INFERENCE_ENGINE_INVALID_NODE_ID);

    // MatMul
    inference_engine_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = input_a;
    matmul_desc.input_b = input_b;
    auto port_matmul_out = inferenceEngineModelDescriptorAddMatMul(md, matmul_desc);
    ASSERT_NE(port_matmul_out, INFERENCE_ENGINE_INVALID_NODE_ID);

    // activation nodes
    std::vector<inference_engine_node_id_t> activation_nodes;
    inference_engine_activation_desc_t activation_desc{};    
    activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_LINEAR;
    activation_desc.params.linear.a = 2.0f;
    activation_desc.params.linear.b = 0.5f;
    for (int i = 0; i < num_activations; ++i)
    {
        activation_desc.input = i == 0 ? port_matmul_out : activation_nodes.back();
        auto port_out = inferenceEngineModelDescriptorAddActivation(md, activation_desc);
        ASSERT_NE(port_out, INFERENCE_ENGINE_INVALID_NODE_ID);
        activation_nodes.push_back(port_out);        
    }

    // define input mappings
    std::array<inference_engine_tensor_mapping_t, 2> inputs{};
    // input a
    {
        inputs[0].id = input_a;
        inputs[0].tensor.data_type = input_desc.data_type;
        set_array(inputs[0].tensor.dims, 1, 1, 32, 32);
    }
    // input b
    {
        inputs[1].id = input_b;
        inputs[1].tensor.data_type = input_desc.data_type;
        set_array(inputs[1].tensor.dims, 1, 1, 32, 32);
    }
    const auto tensor_a_elements_count = accumulate_tensor_dims(inputs[0].tensor);
    const auto tensor_a_bytes_width = tensor_a_elements_count * sizeof(float);
    const auto tensor_b_elements_count = accumulate_tensor_dims(inputs[0].tensor);
    const auto tensor_b_bytes_width = tensor_b_elements_count * sizeof(float);
    std::vector<float> input_a_data_host(tensor_a_elements_count, 1.0f);
    std::vector<float> input_b_data_host(tensor_b_bytes_width, 1.0f);

    auto input_a_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_a_bytes_width, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto input_b_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_b_bytes_width, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    // copy data to GPU
    auto upload_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_a_bytes_width + tensor_b_bytes_width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_COPY_SOURCE);
    std::byte* upload_mapped_ptr = nullptr;
    upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
    std::memcpy(upload_mapped_ptr, input_a_data_host.data(), tensor_a_bytes_width);
    std::memcpy(upload_mapped_ptr + tensor_a_bytes_width, input_b_data_host.data(), tensor_b_bytes_width);
    upload_buffer->Unmap(0, nullptr);
    G_DX12_ENGINE.command_list->CopyBufferRegion(input_a_buffer.Get(), 0, upload_buffer.Get(), 0, tensor_a_bytes_width);
    G_DX12_ENGINE.command_list->CopyBufferRegion(input_b_buffer.Get(), 0, upload_buffer.Get(), tensor_a_bytes_width, tensor_b_bytes_width);
    dispatch_resource_barrier(G_DX12_ENGINE.command_list.Get(), { CD3DX12_RESOURCE_BARRIER::Transition(input_a_buffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) });
    dispatch_resource_barrier(G_DX12_ENGINE.command_list.Get(), { CD3DX12_RESOURCE_BARRIER::Transition(input_b_buffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) });
    G_DX12_ENGINE.wait_for_execution();

    // create model
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, inputs.data(), inputs.size());
    ASSERT_NE(model, nullptr);

    // ask model for output size (we know that there has to be 1 output in this test case)
    inference_engine_tensor_mapping_t output_mapping{};
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, &output_mapping, nullptr), true);
    ASSERT_EQ(output_mapping.id, activation_nodes.back()); // we fuse to matmul, but the last actiovation won't be fused as it's the output...
    ASSERT_EQ(output_mapping.tensor.data_type, input_desc.data_type);
    ASSERT_EQ(output_mapping.tensor.dims[0], 1);
    ASSERT_EQ(output_mapping.tensor.dims[1], 1);
    ASSERT_EQ(output_mapping.tensor.dims[2], 32);
    ASSERT_EQ(output_mapping.tensor.dims[3], 32);
    const auto output_tensor_elemenets_count = accumulate_tensor_dims(output_mapping.tensor);
    const auto output_tensor_bytes_width = output_tensor_elemenets_count * sizeof(float);


    auto output_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), output_tensor_bytes_width,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ASSERT_TRUE(inferenceEngineModelSetResource(model, inputs[0].id, reinterpret_cast<inference_engine_resource_t>(input_a_buffer.Get())));
    ASSERT_TRUE(inferenceEngineModelSetResource(model, inputs[1].id, reinterpret_cast<inference_engine_resource_t>(input_b_buffer.Get())));
    ASSERT_TRUE(inferenceEngineModelSetResource(model, output_mapping.id, reinterpret_cast<inference_engine_resource_t>(output_buffer.Get())));

    // can execute here (assign resources call execute)
    ASSERT_TRUE(inferenceEngineExecuteModel(model, stream));

    auto readback_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), output_buffer->GetDesc().Width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    dispatch_resource_barrier(G_DX12_ENGINE.command_list.Get(), { CD3DX12_RESOURCE_BARRIER::Transition(output_buffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE) });
    G_DX12_ENGINE.command_list->CopyResource(readback_buffer.Get(), output_buffer.Get());
    G_DX12_ENGINE.wait_for_execution();

    // copy output data to host
    std::vector<float> data_out(output_tensor_elemenets_count);
    std::byte* readback_mapped_ptr = nullptr;
    readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
    std::memcpy(data_out.data(), readback_mapped_ptr, output_tensor_bytes_width);
    readback_buffer->Unmap(0, nullptr);

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

    inferenceEngineDestroyModelDescriptor(md);
    inferenceEngineDestroyModel(model);
    inferenceEngineDestroyContext(ctx);
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

    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    auto md = inferenceEngineCreateModelDescriptor();

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    auto input_a = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(input_a, INFERENCE_ENGINE_INVALID_NODE_ID);
    auto input_b = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(input_b, INFERENCE_ENGINE_INVALID_NODE_ID);
    auto input_c = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(input_c, INFERENCE_ENGINE_INVALID_NODE_ID);

    // MatMul
    inference_engine_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = input_a;
    matmul_desc.input_b = input_b;
    auto port_matmul_a = inferenceEngineModelDescriptorAddMatMul(md, matmul_desc);
    ASSERT_NE(port_matmul_a, INFERENCE_ENGINE_INVALID_NODE_ID);

    // activation
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.input = input_c;
    activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto port_activation = inferenceEngineModelDescriptorAddActivation(md, activation_desc);
    ASSERT_NE(port_activation, INFERENCE_ENGINE_INVALID_NODE_ID);

    // MatMul final
    inference_engine_matmul_desc_t matmul_desc_final{};
    matmul_desc_final.input_a = port_matmul_a;
    matmul_desc_final.input_b = port_activation;
    auto port_matmul_out_final = inferenceEngineModelDescriptorAddMatMul(md, matmul_desc_final);
    ASSERT_NE(port_matmul_out_final, INFERENCE_ENGINE_INVALID_NODE_ID);

    // define input mappings
    std::array<inference_engine_tensor_mapping_t, 3> inputs{};
    // input a
    {
        inputs[0].id = input_a;
        inputs[0].tensor.data_type = input_desc.data_type;
        set_array(inputs[0].tensor.dims, 1, 1, 8, 16);
    }
    // input b
    {
        inputs[1].id = input_b;
        inputs[1].tensor.data_type = input_desc.data_type;
        set_array(inputs[1].tensor.dims, 1, 1, 16, 32);
    }
    // input c
    {
        inputs[2].id = input_c;
        inputs[2].tensor.data_type = input_desc.data_type;
        set_array(inputs[2].tensor.dims, 1, 1, 32, 64);
    }
    // create model
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, inputs.data(), inputs.size());
    ASSERT_NE(model, nullptr);

    // ask model for output size (we know that there has to be 1 output in this test case)
    inference_engine_tensor_mapping_t output_mapping{};
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, &output_mapping, nullptr), true);
    ASSERT_EQ(output_mapping.id, port_matmul_out_final);
    ASSERT_EQ(output_mapping.tensor.data_type, input_desc.data_type);
    ASSERT_EQ(output_mapping.tensor.dims[0], 1);
    ASSERT_EQ(output_mapping.tensor.dims[1], 1);
    ASSERT_EQ(output_mapping.tensor.dims[2], 8);
    ASSERT_EQ(output_mapping.tensor.dims[3], 64);

    auto input_a_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(inputs[0].tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto input_b_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(inputs[1].tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto input_c_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(inputs[2].tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto output_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(output_mapping.tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    inferenceEngineModelSetResource(model, inputs[0].id, reinterpret_cast<inference_engine_resource_t>(input_a_buffer.Get()));
    inferenceEngineModelSetResource(model, inputs[1].id, reinterpret_cast<inference_engine_resource_t>(input_b_buffer.Get()));
    inferenceEngineModelSetResource(model, inputs[2].id, reinterpret_cast<inference_engine_resource_t>(input_c_buffer.Get()));
    inferenceEngineModelSetResource(model, output_mapping.id, reinterpret_cast<inference_engine_resource_t>(output_buffer.Get()));

    // can execute here (assign resources call execute)
    inferenceEngineExecuteModel(model, stream);
    // do conformance check etc ..
    // ...
    // delete model

    inferenceEngineDestroyModelDescriptor(md);
    inferenceEngineDestroyModel(model);
    inferenceEngineDestroyContext(ctx);
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


    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    auto md = inferenceEngineCreateModelDescriptor();

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    auto input_a = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(input_a, INFERENCE_ENGINE_INVALID_NODE_ID);
    inferenceEngineSetNodeName(md, input_a, "input_a");
    auto input_b = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(input_b, INFERENCE_ENGINE_INVALID_NODE_ID);
    inferenceEngineSetNodeName(md, input_b, "input_b");

    // Conv left
    inference_engine_convolution_desc_t conv_desc{.input = input_a};
    auto port_conv_a = inferenceEngineModelDescriptorAddConvolution(md, conv_desc);
    ASSERT_NE(port_conv_a, INFERENCE_ENGINE_INVALID_NODE_ID);
    inferenceEngineSetNodeName(md, port_conv_a, "conv_a");

    // Conv right
    inference_engine_convolution_desc_t conv_desc1{ .input = input_b };
    auto port_conv_b = inferenceEngineModelDescriptorAddConvolution(md, conv_desc1);
    ASSERT_NE(port_conv_b, INFERENCE_ENGINE_INVALID_NODE_ID);
    inferenceEngineSetNodeName(md, port_conv_b, "conv_b");

    // activation
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.input = port_conv_b;
    activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto activation = inferenceEngineModelDescriptorAddActivation(md, activation_desc);
    ASSERT_NE(activation, INFERENCE_ENGINE_INVALID_NODE_ID);

    // elementwise_add
    inference_engine_elementwise_add_desc_t add_desc_final{};
    add_desc_final.input_a = port_conv_a;
    add_desc_final.input_b = activation;
    auto port_add_final = inferenceEngineModelDescriptorAddElementwiseAdd(md, add_desc_final);
    ASSERT_NE(port_add_final, INFERENCE_ENGINE_INVALID_NODE_ID);

    // activation final
    inference_engine_activation_desc_t activation_desc_final{};
    activation_desc_final.input = port_add_final;
    activation_desc_final.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto final_activation = inferenceEngineModelDescriptorAddActivation(md, activation_desc_final);
    ASSERT_NE(final_activation, INFERENCE_ENGINE_INVALID_NODE_ID);
    inferenceEngineSetNodeName(md, final_activation, "final_activation");

    // define input mappings
    std::array<inference_engine_tensor_mapping_t, 3> inputs{};
    // input a
    {
        inputs[0].id = input_a;
        inputs[0].tensor.data_type = input_desc.data_type;
        set_array(inputs[0].tensor.dims, 1, 1, 4, 4);
    }
    // input b
    {
        inputs[1].id = input_b;
        inputs[1].tensor.data_type = input_desc.data_type;
        set_array(inputs[1].tensor.dims, 1, 1, 4, 4);
    }

    // create model
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, inputs.data(), inputs.size());
    ASSERT_NE(model, nullptr);

    // ask model for output size (we know that there has to be 1 output in this test case)
    inference_engine_tensor_mapping_t output_mapping{};
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, &output_mapping, nullptr), true);
    ASSERT_EQ(output_mapping.id, final_activation);
    ASSERT_EQ(output_mapping.tensor.data_type, input_desc.data_type);
    ASSERT_EQ(output_mapping.tensor.dims[0], 1);
    ASSERT_EQ(output_mapping.tensor.dims[1], 1);
    ASSERT_EQ(output_mapping.tensor.dims[2], 4);
    ASSERT_EQ(output_mapping.tensor.dims[3], 4);

    auto input_a_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(inputs[0].tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto input_b_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(inputs[1].tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto input_c_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(inputs[2].tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto output_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), accumulate_tensor_dims(output_mapping.tensor),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    inferenceEngineModelSetResource(model, inputs[0].id, reinterpret_cast<inference_engine_resource_t>(input_a_buffer.Get()));
    inferenceEngineModelSetResource(model, inputs[1].id, reinterpret_cast<inference_engine_resource_t>(input_b_buffer.Get()));
    inferenceEngineModelSetResource(model, inputs[2].id, reinterpret_cast<inference_engine_resource_t>(input_c_buffer.Get()));
    inferenceEngineModelSetResource(model, output_mapping.id, reinterpret_cast<inference_engine_resource_t>(output_buffer.Get()));

    // can execute here (assign resources call execute)
    inferenceEngineExecuteModel(model, stream);
    // do conformance check etc ..
    // ...
    // delete model

    inferenceEngineDestroyModelDescriptor(md);
    inferenceEngineDestroyModel(model);
    inferenceEngineDestroyContext(ctx);
}