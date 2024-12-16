#include "utils.h"
#include "test_dx12_context.h"
#include <ai_driver_model.hpp>
#include <ai_driver_operators.hpp>
#include <ai_driver.hpp>

#include <array>

TEST(OperatorTest, Port_model_descriptor_0)
{
    ai_driver::ModelDescriptor md{};
    md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });  
}

TEST(OperatorTest, Port_mismatch_data_type)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    const auto data_type_desc = AI_DRIVER_DATA_TYPE_FP32;
    const auto data_type_mapping = AI_DRIVER_DATA_TYPE_FP16;

    ai_driver::ModelDescriptor md{};
    auto port = md.add_port(ai_driver_port_desc_t{ data_type_desc });

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

    auto resource = device.allocate_resource(1024);

    ai_driver::ModelDescriptor md{};
    md.add(ai_driver::ConstantPortDesc(ai_driver::DataType::fp32, resource));
}

TEST(OperatorTest, Activation_model_descriptor_0)
{
    ai_driver::ModelDescriptor md{};
    auto input  = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    auto output = md.add_activation(ai_driver_activation_desc_t{input, AI_DRIVER_ACTIVATION_TYPE_RELU });
}

TEST(OperatorTest, Matmul_model_descriptor_0)
{
    ai_driver::ModelDescriptor md{};
    auto input_a = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    auto input_b = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    ai_driver_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = input_a;
    matmul_desc.input_b = input_b;
    auto output = md.add_matmul(matmul_desc);
}

TEST(OperatorTest, Matmul_model_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    ai_driver::ModelDescriptor md{};
    auto port_a = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    auto port_b = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    ai_driver_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = port_a;
    matmul_desc.input_b = port_b;
    auto output = md.add_matmul(matmul_desc);

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 32, 64 });
    auto model = ai_driver::Model(ctx, stream, md, input_mappings);
}

TEST(OperatorTest, Matmul_basic_wrong_2d_sizes)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    ai_driver::ModelDescriptor md{};
    auto port_a = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    auto port_b = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    ai_driver_matmul_desc_t matmul_desc{};
    matmul_desc.input_a = port_a;
    matmul_desc.input_b = port_b;
    auto output = md.add_matmul(matmul_desc);

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 3333, 11 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}

TEST(OperatorTest, Elementwise_add_0)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    ai_driver::ModelDescriptor md{};
    auto port_a = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP32 });
    auto port_b = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP32 });
    ai_driver_elementwise_desc_t elem_add_desc{};
    elem_add_desc.input_a = port_a;
    elem_add_desc.input_b = port_b;
    elem_add_desc.type = ai_driver_elementwise_type_t::AI_DRIVER_ELEMENTWISE_TYPE_ADD;
    auto output = md.add_elementwise(elem_add_desc);

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 3333, 11 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}

TEST(OperatorTest, Elementwise_add_0_input_data_type_mismatch)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    ai_driver::ModelDescriptor md{};
    auto port_a = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP32 });
    auto port_b = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP16 });
    ai_driver_elementwise_desc_t elem_add_desc{};
    elem_add_desc.input_a = port_a;
    elem_add_desc.input_b = port_b;
    elem_add_desc.type = ai_driver_elementwise_type_t::AI_DRIVER_ELEMENTWISE_TYPE_ADD;
    auto output = md.add_elementwise(elem_add_desc);

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP32, { 1, 1, 32, 32 });
    input_mappings[port_b] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 32, 32 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}


TEST(OperatorTest, Elementwise_add_0_wrong_sizes)
{
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);

    ai_driver::ModelDescriptor md{};
    auto port_a = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP32 });
    auto port_b = md.add_port(ai_driver_port_desc_t{ AI_DRIVER_DATA_TYPE_FP32 });
    ai_driver_elementwise_desc_t elem_add_desc{};
    elem_add_desc.input_a = port_a;
    elem_add_desc.input_b = port_b;
    elem_add_desc.type = ai_driver_elementwise_type_t::AI_DRIVER_ELEMENTWISE_TYPE_ADD;
    auto output = md.add_elementwise(elem_add_desc);

    ai_driver::TensorMapping input_mappings{};
    input_mappings[port_a] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 16, 32 });
    input_mappings[port_b] = ai_driver::Tensor(AI_DRIVER_DATA_TYPE_FP16, { 1, 1, 3333, 11 });
    // Negative test, we want this to throw!
    ASSERT_THROW(ai_driver::Model(ctx, stream, md, input_mappings), ai_driver::IEexception);
}
