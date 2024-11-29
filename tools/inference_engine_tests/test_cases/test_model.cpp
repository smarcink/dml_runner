#include "test_gpu_context.h"
#include "inference_engine_model.h"
#include "utils.h"

#include <numeric>
#include <vector>

TEST(OperatorTest, Basic_0)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32;
    set_array(input_desc.tensor.dims, 1, 2, 4, 4);
    auto input_node = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_node != nullptr);
    const auto tensor_elements_count = accumulate_tensor_dims(input_desc.tensor);
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
    dispatch_resource_barrier(G_DX12_ENGINE.command_list.Get(), { CD3DX12_RESOURCE_BARRIER::Transition(input_buffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) });
    G_DX12_ENGINE.wait_for_execution();
    auto output_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_size_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // define activation
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.type = inference_engine_activation_type_t::INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    activation_desc.input = input_node;
    auto out_node = inferenceEngineCreateActivation(activation_desc);

    inferenceEngineSetResource(input_node, reinterpret_cast<inference_engine_resource_t>(input_buffer.Get()));
    inferenceEngineSetResource(out_node, reinterpret_cast<inference_engine_resource_t>(output_buffer.Get()));

    // create model with empty config
    auto model_desc = inferenceEngineCreateModelDescriptor(&out_node, 1);
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, model_desc);


    auto exec_result = inferenceEngineExecuteModel(model, stream);
    EXPECT_EQ(exec_result, INFERENCE_ENGINE_RESULT_SUCCESS);

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
        EXPECT_FLOAT_EQ(real_data, reference);
    }

    // ToDo: destroy nodes etc.
    inferenceEngineDestroyContext(ctx);
}

TEST(ModelTest, MatMul_fused_activation)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    set_array(input_desc.tensor.dims, 1, 1, 32, 32);
    // create 2 port with the same description
    // M = 32, K = 32, N = 32
    auto input_a = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_a != nullptr);
    auto input_b = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_b != nullptr);

    // MatMul
    inference_engine_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = input_a;
    matmul_desc.input_b = input_b;
    auto port_matmul_out = inferenceEngineCreateMatMul(matmul_desc);
    EXPECT_TRUE(port_matmul_out != nullptr);

    // activation
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.input = port_matmul_out;
    activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto port_out = inferenceEngineCreateActivation(activation_desc);
    EXPECT_TRUE(port_out != nullptr);

    // create model
    auto model_desc = inferenceEngineCreateModelDescriptor(&port_out, 1);
    const auto model = inferenceEngineCompileModelDescriptor(ctx, stream, model_desc);

    // can execute here (assign resources call execute)
    inferenceEngineExecuteModel(model, stream);
    // do conformance check etc ..
    // ...
    // delete model
    if (model_desc)
    {
        inferenceEngineDestroyModelDescriptor(model_desc);
    }
    destroy_node_if_valid(input_a);
    destroy_node_if_valid(input_b);
    destroy_node_if_valid(port_out);
    destroy_node_if_valid(port_matmul_out);
    inferenceEngineDestroyContext(ctx);
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

    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    set_array(input_desc.tensor.dims, 1, 1, 16, 16);
    // create 2 port with the same description
    // M = 32, K = 16, N = 16
    auto input_a = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_a != nullptr);
    auto input_b = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_b != nullptr);
    auto input_c = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_c != nullptr);

    // MatMul
    inference_engine_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = input_a;
    matmul_desc.input_b = input_b;
    auto port_matmul_out = inferenceEngineCreateMatMul(matmul_desc);
    EXPECT_TRUE(port_matmul_out != nullptr);

    // activation
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.input = input_c;
    activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto port_out = inferenceEngineCreateActivation(activation_desc);
    EXPECT_TRUE(port_out != nullptr);

    // MatMul final
    inference_engine_matmul_desc_t matmul_desc_final{};
    matmul_desc_final.input_a = port_matmul_out;
    matmul_desc_final.input_b = port_out;
    auto port_matmul_out_final = inferenceEngineCreateMatMul(matmul_desc_final);
    EXPECT_TRUE(port_matmul_out_final != nullptr);

    // create model
    auto model_desc = inferenceEngineCreateModelDescriptor(&port_matmul_out_final, 1);
    const auto model = inferenceEngineCompileModelDescriptor(ctx, stream, model_desc);

    // can execute here (assign resources call execute)
    inferenceEngineExecuteModel(model, stream);
    // do conformance check etc ..
    // ...
    // delete model
    if (model_desc)
    {
        inferenceEngineDestroyModelDescriptor(model_desc);
    }
    destroy_node_if_valid(input_a);
    destroy_node_if_valid(input_b);
    destroy_node_if_valid(input_c);
    destroy_node_if_valid(port_out);
    destroy_node_if_valid(port_matmul_out_final);
    destroy_node_if_valid(port_matmul_out);
    inferenceEngineDestroyContext(ctx);
}