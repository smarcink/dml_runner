#include "inference_engine_model.h"
#include "utils.h"

TEST(ModelTest, MatMul_fused_activation)
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

    // MatMul
    inference_engine_matmul_desc_t matmul_desc{};
    matmul_desc.tensor_a = tensor_a;
    matmul_desc.tensor_b = tensor_b;
    auto port_matmul_out = inferenceEngineCreateMatMul(matmul_desc);
    EXPECT_TRUE(port_matmul_out != nullptr);

    // activation
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.tensor = port_matmul_out;
    activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    auto port_out = inferenceEngineCreateActivation(activation_desc);
    EXPECT_TRUE(port_out != nullptr);

    // create model with empty config
    auto model_desc = inferenceEngineCreateModelDescriptor({}, &port_out, 1);

    inferenceEngineGetPartitions(model_desc);

    if (model_desc)
    {
        inferenceEngineDestroyModelDescriptor(model_desc);
    }
    destroy_node_if_valid(tensor_a);
    destroy_node_if_valid(tensor_b);
    destroy_node_if_valid(port_out);
}
