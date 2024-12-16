#include "utils.h"
#include "test_dx12_context.h"
#include <ai_driver_model.hpp>
#include <ai_driver_operators.hpp>
#include <ai_driver.hpp>

#include <array>

TEST(OperatorTest, Port_model_descriptor_0)
{
    ai_driver::ModelDescriptor md{};
    md.add(ai_driver::PortDesc(ai_driver::DataType::fp16));
}

TEST(OperatorTest, Port_mismatch_data_type)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const auto data_type_desc = ai_driver::DataType::fp32;
    const auto data_type_mapping = ai_driver::DataType::fp16;

    ai_driver::ModelDescriptor md{};
    auto port = md.add(ai_driver::PortDesc(data_type_desc));

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port] = ai_driver::Tensor(data_type_mapping, { 1, 1, 32, 32 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}

TEST(OperatorTest, Constant_port_model_descriptor_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const ai_driver::Tensor resource_tensor(ai_driver::DataType::fp32, {32, 8, 3, 3});
    auto resource = device.allocate_resource(1024);

    ai_driver::ModelDescriptor md{};
    md.add(ai_driver::ConstantPortDesc(resource_tensor, resource));
}

TEST(OperatorTest, Activation_model_descriptor_0)
{
    ai_driver::ModelDescriptor md{};
    auto input  = md.add(ai_driver::PortDesc(ai_driver::DataType::fp16));
    auto output = md.add(ai_driver::ActivationDesc::relu(input, ai_driver::DataType::fp16));
}

TEST(OperatorTest, Matmul_model_descriptor_0)
{
    const auto data_type = ai_driver::DataType::fp16;
    ai_driver::ModelDescriptor md{};
    auto input_a = md.add(ai_driver::PortDesc(data_type));
    auto input_b = md.add(ai_driver::PortDesc(data_type));
    auto output = md.add(ai_driver::MatMulDesc(input_a, input_b, data_type));
}

TEST(OperatorTest, Matmul_model_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const auto data_type = ai_driver::DataType::fp16;
    ai_driver::ModelDescriptor md{};
    auto port_a = md.add(ai_driver::PortDesc(data_type));
    auto port_b = md.add(ai_driver::PortDesc(data_type));
    auto output = md.add(ai_driver::MatMulDesc(port_a, port_b, data_type));

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(data_type, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(data_type, { 1, 1, 32, 64 });
    auto model = ai_driver::Model(ctx, stream, md, input_mappings);
}

TEST(OperatorTest, Matmul_basic_wrong_2d_sizes)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const auto data_type = ai_driver::DataType::fp16;
    ai_driver::ModelDescriptor md{};
    auto port_a = md.add(ai_driver::PortDesc(data_type));
    auto port_b = md.add(ai_driver::PortDesc(data_type));
    auto output = md.add(ai_driver::MatMulDesc(port_a, port_b, data_type));

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(data_type, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(data_type, { 1, 1, 3333, 11 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}

TEST(OperatorTest, Elementwise_add_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const auto data_type_desc = ai_driver::DataType::fp32;
    const auto data_type_mapping = ai_driver::DataType::fp32;
    ai_driver::ModelDescriptor md{};
    auto port_a = md.add(ai_driver::PortDesc(data_type_desc));
    auto port_b = md.add(ai_driver::PortDesc(data_type_desc));
    auto output = md.add(ai_driver::ElementwiseDesc::add(port_a, port_b, data_type_desc));

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(data_type_mapping, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(data_type_mapping, { 1, 1, 3333, 11 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}

TEST(OperatorTest, Elementwise_add_0_input_data_type_mismatch)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    ai_driver::ModelDescriptor md{};
    auto port_a = md.add(ai_driver::PortDesc(ai_driver::DataType::fp32));
    auto port_b = md.add(ai_driver::PortDesc(ai_driver::DataType::fp16));
    auto output = md.add(ai_driver::ElementwiseDesc::add(port_a, port_b, ai_driver::DataType::fp32));

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(ai_driver::DataType::fp32, { 1, 1, 32, 32 });
    input_mappings[port_b] = ai_driver::Tensor(ai_driver::DataType::fp16, { 1, 1, 32, 32 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}


TEST(OperatorTest, Elementwise_add_0_wrong_sizes)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const auto data_type = ai_driver::DataType::fp32;
    ai_driver::ModelDescriptor md{};
    auto port_a = md.add(ai_driver::PortDesc(data_type));
    auto port_b = md.add(ai_driver::PortDesc(data_type));
    auto output = md.add(ai_driver::ElementwiseDesc::add(port_a, port_b, data_type));

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(data_type, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(data_type, { 1, 1, 3333, 11 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}
