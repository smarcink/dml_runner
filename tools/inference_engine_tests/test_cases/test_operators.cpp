#include "utils.h"
#include "test_gpu_context.h"
#include "inference_engine_model.hpp"
#include "inference_engine.hpp"

#include <array>

TEST(OperatorTest, Port_model_descriptor_0)
{
    inference_engine::ModelDescriptor md{};
    md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });  
}


TEST(OperatorTest, Activation_model_descriptor_0)
{
    inference_engine::ModelDescriptor md{};
    auto input  = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });
    auto output = md.add_activation(inference_engine_activation_desc_t{input, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU });
}

TEST(OperatorTest, Matmul_model_descriptor_0)
{
    inference_engine::ModelDescriptor md{};
    auto input_a = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });
    auto input_b = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });
    inference_engine_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = input_a;
    matmul_desc.input_b = input_b;
    auto output = md.add_matmul(matmul_desc);
}

TEST(OperatorTest, Matmul_model_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    inference_engine::ModelDescriptor md{};
    auto port_a = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });
    auto port_b = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });
    inference_engine_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = port_a;
    matmul_desc.input_b = port_b;
    auto output = md.add_matmul(matmul_desc);

    inference_engine::TensorMapping input_mappings{};
    input_mappings[port_a] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP16, { 1, 1, 16, 32 });
    input_mappings[port_b] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP16, { 1, 1, 32, 64 });
    auto model = inference_engine::Model(ctx, stream, md, input_mappings);
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
    ASSERT_EQ(model, nullptr);
    inferenceEngineDestroyModelDescriptor(md);
    inferenceEngineDestroyContext(ctx);
}
