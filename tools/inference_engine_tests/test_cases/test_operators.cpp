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
    input_mappings[port_b] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP16, { 1, 1, 3333, 11 });
    // Negative test, we want this to throw!
    ASSERT_THROW(inference_engine::Model(ctx, stream, md, input_mappings), inference_engine::IEexception);
}
