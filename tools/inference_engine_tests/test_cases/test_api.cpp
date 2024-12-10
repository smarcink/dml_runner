#include "test_gpu_context.h"
#include "inference_engine.hpp"
#include "inference_engine_model.hpp"

#include "utils.h"

#include <thread>

TEST(ApiTest, GPU_create_context_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);
}

TEST(ApiTest, get_outputs_single_out)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    inference_engine::ModelDescriptor md{};
    auto input_node = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });
    auto out_node = md.add_activation(inference_engine_activation_desc_t{ input_node, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, INFERENCE_ENGINE_DATA_TYPE_FP16 });

    inference_engine::TensorMapping input_mappings{};
    input_mappings[input_node] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP16, { 1, 1, 16, 32 });
    auto model = inference_engine::Model(ctx, stream, md, input_mappings);

    auto output_mappings = model.get_outputs();
    ASSERT_EQ(output_mappings.size(), 1);

    const auto& input = input_mappings[input_node];
    const auto& out_mapping = output_mappings[out_node];
    ASSERT_NE(output_mappings.find(out_node), std::end(output_mappings));
    ASSERT_EQ(out_mapping.data_type, input.data_type);
    ASSERT_EQ(out_mapping.dims.size(), input.dims.size());
    ASSERT_EQ(out_mapping.dims, input.dims);
}

TEST(ApiTest, get_outputs_multiple_outs)
{   
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    inference_engine::ModelDescriptor md{};
    auto input_node = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP16 });

    std::vector<inference_engine::NodeID> output_nodes{};
    for (auto i = 0; i < 6; i++)
    {
        auto node = md.add_activation(inference_engine_activation_desc_t{ input_node, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, INFERENCE_ENGINE_DATA_TYPE_FP16 });
        output_nodes.push_back(node);
    }

    inference_engine::TensorMapping input_mappings{};
    input_mappings[input_node] = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP16, { 1, 1, 1, 64 });
    auto model = inference_engine::Model(ctx, stream, md, input_mappings);

    auto output_mappings = model.get_outputs();
    ASSERT_EQ(output_mappings.size(), output_nodes.size());

    for (auto& on : output_nodes)
    {
        ASSERT_NE(output_mappings.find(on), std::end(output_mappings));
    }
}

TEST(ApiTest, empty_model_descriptor)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    inference_engine::ModelDescriptor md{};
    // Negative test, we expect it to throw.
    ASSERT_THROW(inference_engine::Model(ctx, stream, md, {}), inference_engine::IEexception);
}

TEST(ApiTest, invalid_node_connection)
{
    inference_engine::ModelDescriptor md{};
    const auto input = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP32 });
    const auto invalid_node_id = input + 1331;  // very wrong node id
    
    // Point to invalid node id
    const auto activation_desc = inference_engine_activation_desc_t{ invalid_node_id, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, INFERENCE_ENGINE_DATA_TYPE_FP32 };
    // Negative test, we expect it to throw.
    ASSERT_THROW(md.add_activation(activation_desc), inference_engine::IEexception);
}

TEST(ApiTest, invalid_node_connection_node_loop_to_itself)
{
    inference_engine::ModelDescriptor md{};
    const auto input = md.add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP32 });
    const auto invalid_node_id = input + 1;  // this ID would be assigned by next node added to the graph (which would create a loop which is invalid in current state of the inference engine library)

    // Point to invalid node id (loop to itself)
    const auto activation_desc = inference_engine_activation_desc_t{ invalid_node_id, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, INFERENCE_ENGINE_DATA_TYPE_FP32 };
    // Negative test, we expect it to throw.
    ASSERT_THROW(md.add_activation(activation_desc), inference_engine::IEexception);
}