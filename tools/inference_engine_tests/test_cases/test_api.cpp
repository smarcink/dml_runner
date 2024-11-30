#include "test_gpu_context.h"
#include "inference_engine_model.h"


#include "utils.h"

TEST(ApiTest, GPU_create_context_0)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());
    EXPECT_TRUE(nullptr != ctx);
    inferenceEngineDestroyContext(ctx);
}


TEST(ApiTest, get_outputs_single_out)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());
    EXPECT_TRUE(nullptr != ctx);

    auto md = inferenceEngineCreateModelDescriptor();

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    auto input_node = inferenceEngineModelDescriptorAddPort(md, input_desc);

    inference_engine_activation_desc_t activation_desc{};
    activation_desc.type = inference_engine_activation_type_t::INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    activation_desc.input = input_node;
    auto out_node = inferenceEngineModelDescriptorAddActivation(md, activation_desc);

    inference_engine_tensor_mapping_t input{};
    input.id = input_node;
    input.tensor.data_type = input_desc.data_type;
    set_array(input.tensor.dims, 1, 2, 4, 4);
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, &input, 1);
    ASSERT_NE(model, nullptr);

    std::size_t outputs_counts = 0;
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, nullptr, &outputs_counts), INFERENCE_ENGINE_RESULT_SUCCESS);
    ASSERT_EQ(outputs_counts, 1);
    std::vector<inference_engine_tensor_mapping_t> output_mappings(outputs_counts);
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, output_mappings.data(), &outputs_counts), INFERENCE_ENGINE_RESULT_SUCCESS);
    ASSERT_EQ(output_mappings[0].id, out_node);
    ASSERT_EQ(output_mappings[0].tensor.data_type, input.tensor.data_type);
    for (auto i = 0; i < 4; i++)
    {
        ASSERT_EQ(output_mappings[0].tensor.dims[i], input.tensor.dims[i]);
    }
    inferenceEngineDestroyContext(ctx);
}

TEST(ApiTest, get_outputs_multiple_outs)
{
    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());
    EXPECT_TRUE(nullptr != ctx);

    auto md = inferenceEngineCreateModelDescriptor();

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;
    auto input_node = inferenceEngineModelDescriptorAddPort(md, input_desc);

    std::vector<inference_engine_node_id_t> output_nodes{};
    inference_engine_activation_desc_t activation_desc{};
    activation_desc.type = inference_engine_activation_type_t::INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    activation_desc.input = input_node;
    for (auto i = 0; i < 6; i++)
    {
        auto node = inferenceEngineModelDescriptorAddActivation(md, activation_desc);
        ASSERT_NE(node, INFERENCE_ENGINE_INVALID_NODE_ID);
        output_nodes.push_back(node);
    }
    
    inference_engine_tensor_mapping_t input{};
    input.id = input_node;
    input.tensor.data_type = input_desc.data_type;
    set_array(input.tensor.dims, 1, 1, 1, 64);
    auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, &input, 1);
    ASSERT_NE(model, nullptr);

    std::size_t outputs_counts = 0;
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, nullptr, &outputs_counts), INFERENCE_ENGINE_RESULT_SUCCESS);
    ASSERT_EQ(outputs_counts, output_nodes.size());
    std::vector<inference_engine_tensor_mapping_t> output_mappings(outputs_counts);
    ASSERT_EQ(inferenceEngineModelGetOutputs(model, output_mappings.data(), &outputs_counts), INFERENCE_ENGINE_RESULT_SUCCESS);
    for (auto& on : output_nodes)
    {
        auto it = std::find_if(std::begin(output_mappings), std::end(output_mappings), [&on](const auto& om)
            {
                return on == om.id;
            });
        ASSERT_NE(it, std::end(output_mappings));
    }
    inferenceEngineDestroyContext(ctx);
}
