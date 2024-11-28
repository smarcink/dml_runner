#include "utils.h"
#include "test_gpu_context.h"

TEST(OperatorTest, Port_basic_0)
{
    inference_engine_port_desc_t desc{};
    desc.tensor.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    set_array(desc.tensor.dims, 1, 16, 32, 32);
    auto port = inferenceEngineCreatePort(desc);
    EXPECT_TRUE(port != nullptr);
    destroy_node_if_valid(port);
}

TEST(OperatorTest, Matmul_basic_0)
{
    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    set_array(input_desc.tensor.dims, 1, 1, 32, 32);
    // create 2 port with the same description
    // M = 32, K = 32, N = 32
    auto port_a = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(port_a != nullptr);
    auto port_b = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(port_b != nullptr);

    inference_engine_matmul_desc_t desc{};
    desc.input_a = port_a;
    desc.input_b = port_b;

    auto port_out = inferenceEngineCreateMatMul(desc);
    EXPECT_TRUE(port_out != nullptr);
    destroy_node_if_valid(port_a);
    destroy_node_if_valid(port_b);
    destroy_node_if_valid(port_out);
}

TEST(OperatorTest, Activation_basic_0)
{
    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    set_array(input_desc.tensor.dims, 1, 1, 32, 32);
    auto input_port = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_port != nullptr);

    inference_engine_activation_desc_t desc{};
    desc.input = input_port;
    desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto port_out = inferenceEngineCreateActivation(desc);
    EXPECT_TRUE(port_out != nullptr);
    destroy_node_if_valid(input_port);
    destroy_node_if_valid(port_out);
}