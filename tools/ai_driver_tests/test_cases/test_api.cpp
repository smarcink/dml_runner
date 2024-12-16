#include "test_dx12_context.h"
#include <ai_driver.hpp>
#include <ai_driver_model.hpp>

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

    ai_driver::ModelDescriptor md{};
    auto input_node = md.add(ai_driver::PortDesc(ai_driver::DataType::fp16));
    auto out_node = md.add(ai_driver::ActivationDesc::relu(input_node, ai_driver::DataType::fp16));

    ai_driver::TensorMapping input_mappings{};
    input_mappings[input_node] = ai_driver::Tensor(ai_driver::DataType::fp16, { 1, 1, 16, 32 });
    auto model = ai_driver::Model(ctx, stream, md, input_mappings);

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

    ai_driver::ModelDescriptor md{};
    auto input_node = md.add(ai_driver::PortDesc(ai_driver::DataType::fp16));

    std::vector<ai_driver::NodeID> output_nodes{};
    for (auto i = 0; i < 6; i++)
    {
        auto node = md.add(ai_driver::ActivationDesc::relu(input_node, ai_driver::DataType::fp16));
        output_nodes.push_back(node);
    }

    ai_driver::TensorMapping input_mappings{};
    input_mappings[input_node] = ai_driver::Tensor(ai_driver::DataType::fp16, { 1, 1, 1, 64 });
    auto model = ai_driver::Model(ctx, stream, md, input_mappings);

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

    ai_driver::ModelDescriptor md{};
    // Negative test, we expect it to throw.
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, {}), ai_driver::IEexception);
}

TEST(ApiTest, invalid_node_connection)
{
    ai_driver::ModelDescriptor md{};
    const auto input = md.add(ai_driver::PortDesc(ai_driver::DataType::fp32));
    const auto invalid_node_id = input + 1331;  // very wrong node id
    
    // Point to invalid node id
    const auto activation_desc = ai_driver::ActivationDesc::relu(invalid_node_id, ai_driver::DataType::fp32);
    // Negative test, we expect it to throw.
    ASSERT_THROW(md.add(activation_desc), ai_driver::IEexception);
}

TEST(ApiTest, invalid_node_connection_node_loop_to_itself)
{
    ai_driver::ModelDescriptor md{};
    const auto input = md.add(ai_driver::PortDesc(ai_driver::DataType::fp32));
    const auto invalid_node_id = input + 1;  // this ID would be assigned by next node added to the graph (which would create a loop which is invalid in current state of the inference engine library)

    // Point to invalid node id (loop to itself)
    const auto activation_desc = ai_driver::ActivationDesc::relu(invalid_node_id, ai_driver::DataType::fp32);
    // Negative test, we expect it to throw.
    ASSERT_THROW(md.add(activation_desc), ai_driver::IEexception);
}