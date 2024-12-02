#include "utils.h"
#include "test_gpu_context.h"
#include "inference_engine_model.h"

#include <array>

TEST(OperatorTest, Port_basic_0)
{
    auto md = inferenceEngineCreateModelDescriptor();
    inference_engine_port_desc_t desc{};
    desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    auto node_id = inferenceEngineModelDescriptorAddPort(md, desc);
    ASSERT_NE(node_id, INFERENCE_ENGINE_INVALID_NODE_ID);
    inferenceEngineDestroyModelDescriptor(md);
}


TEST(OperatorTest, Activation_basic_0)
{
    auto md = inferenceEngineCreateModelDescriptor();
    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    auto port_id = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(port_id, INFERENCE_ENGINE_INVALID_NODE_ID);


    inference_engine_activation_desc_t desc{};
    desc.input = port_id;
    desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto node_id = inferenceEngineModelDescriptorAddActivation(md, desc);
    ASSERT_NE(port_id, INFERENCE_ENGINE_INVALID_NODE_ID);
    inferenceEngineDestroyModelDescriptor(md);
}

TEST(OperatorTest, Matmul_basic_0)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    auto md = inferenceEngineCreateModelDescriptor();

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;

    auto port_a = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(port_a, INFERENCE_ENGINE_INVALID_NODE_ID);
    auto port_b = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(port_b, INFERENCE_ENGINE_INVALID_NODE_ID);

    inference_engine_matmul_desc_t desc{};
    desc.input_a = port_a;
    desc.input_b = port_b;

    auto port_out = inferenceEngineModelDescriptorAddMatMul(md, desc);
    ASSERT_NE(port_out, INFERENCE_ENGINE_INVALID_NODE_ID);

    std::array<inference_engine_tensor_mapping_t, 2> inputs{};
    // input a
    {
        inputs[0].id = port_a;
        inputs[0].tensor.data_type = input_desc.data_type;
        set_array(inputs[0].tensor.dims, 1, 1, 16, 32);
    }
    // input b
    {
        inputs[1].id = port_b;
        inputs[1].tensor.data_type = input_desc.data_type;
        set_array(inputs[1].tensor.dims, 1, 1, 32, 64);
    }
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, inputs.data(), inputs.size());
    ASSERT_NE(model, nullptr);
    inferenceEngineDestroyModelDescriptor(md);
    inferenceEngineDestroyContext(ctx);
}

TEST(OperatorTest, Matmul_basic_wrong_2d_sizes)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());

    auto md = inferenceEngineCreateModelDescriptor();
    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;

    auto port_a = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(port_a, INFERENCE_ENGINE_INVALID_NODE_ID);
    auto port_b = inferenceEngineModelDescriptorAddPort(md, input_desc);
    ASSERT_NE(port_b, INFERENCE_ENGINE_INVALID_NODE_ID);

    inference_engine_matmul_desc_t desc{};
    desc.input_a = port_a;
    desc.input_b = port_b;

    auto port_out = inferenceEngineModelDescriptorAddMatMul(md, desc);
    ASSERT_NE(port_out, INFERENCE_ENGINE_INVALID_NODE_ID);

    std::array<inference_engine_tensor_mapping_t, 2> inputs{};
    // input a
    {
        inputs[0].id = port_a;
        inputs[0].tensor.data_type = input_desc.data_type;
        set_array(inputs[0].tensor.dims, 1, 1, 16, 32);
    }
    // input b
    {
        inputs[1].id = port_b;
        inputs[1].tensor.data_type = input_desc.data_type;
        set_array(inputs[1].tensor.dims, 1, 1, 3333, 11);
    }
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, inputs.data(), inputs.size());
    EXPECT_EQ(inferenceEngineGetLastError(), INFERENCE_ENGINE_RESULT_INVALID_ARGUMENT);
    ASSERT_EQ(model, nullptr);
    inferenceEngineDestroyModelDescriptor(md);
    inferenceEngineDestroyContext(ctx);
}
