#include "utils.h"
#include "test_d3d12_context.h"

TEST(OperatorTest, Port_basic_0)
{
    test_ctx::TestGpuContext gpu_ctx{};

    inference_engine_port_desc_t desc{};
    desc.tensor.data_type = inference_engine_data_type_t::XESS_DATA_TYPE_FP16;
    set_array(desc.tensor.dims, 1, 16, 32, 32);
    auto port = inferenceEngineCreatePort(desc);
    EXPECT_TRUE(port != nullptr);




    destroy_node_if_valid(port);
}

TEST(OperatorTest, Matmul_basic_0)
{
    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::XESS_DATA_TYPE_FP16;
    set_array(input_desc.tensor.dims, 1, 1, 32, 32);
    // create 2 port with the same description
    // M = 32, K = 32, N = 32
    auto tensor_a = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(tensor_a != nullptr);
    auto tensor_b = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(tensor_b != nullptr);

    inference_engine_matmul_desc_t desc{};
    desc.tensor_a = tensor_a;
    desc.tensor_b = tensor_b;

    auto port_out = inferenceEngineCreateMatMul(desc);
    EXPECT_TRUE(port_out != nullptr);
    destroy_node_if_valid(tensor_a);
    destroy_node_if_valid(tensor_b);
    destroy_node_if_valid(port_out);
}

TEST(OperatorTest, Matmul_basic_wrong_2d_sizes)
{
	inference_engine_port_desc_t input_desc{};
	input_desc.tensor.data_type = inference_engine_data_type_t::XESS_DATA_TYPE_FP16;
	set_array(input_desc.tensor.dims, 16, 32);

	inference_engine_port_desc_t input_desc2{};
	input_desc2.tensor.data_type = inference_engine_data_type_t::XESS_DATA_TYPE_FP16;
	set_array(input_desc2.tensor.dims, 3333, 11);

	auto tensor_a = inferenceEngineCreatePort(input_desc);
	EXPECT_TRUE(tensor_a != nullptr);
	auto tensor_b = inferenceEngineCreatePort(input_desc2);
	EXPECT_TRUE(tensor_b != nullptr);

	inference_engine_matmul_desc_t desc{};
	desc.tensor_a = tensor_a;
	desc.tensor_b = tensor_b;

	auto port_out = inferenceEngineCreateMatMul(desc);
	EXPECT_TRUE(port_out == nullptr);
	destroy_node_if_valid(tensor_a);
	destroy_node_if_valid(tensor_b);
}

TEST(OperatorTest, Activation_basic_0)
{
    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::XESS_DATA_TYPE_FP16;
    set_array(input_desc.tensor.dims, 1, 1, 32, 32);
    auto tensor = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(tensor != nullptr);

    inference_engine_activation_desc_t desc{};
    desc.tensor = tensor;
    desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto port_out = inferenceEngineCreateActivation(desc);
    EXPECT_TRUE(port_out != nullptr);
    destroy_node_if_valid(tensor);
    destroy_node_if_valid(port_out);
}