#include "utils.h"
#include "test_gpu_context.h"
#include "inference_engine_model.h"
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

class ResourceDX12 : public inference_engine::Resource
{
public:
    ResourceDX12(ComPtr<ID3D12Resource> resource)
        : rsc_(resource)
    {
    }

    ID3D12Resource* get() { return rsc_.Get(); }

private:
    ComPtr<ID3D12Resource> rsc_;
};

class StreamDX12 : public inference_engine::Stream<StreamDX12>
{
public:
    StreamDX12(ComPtr<ID3D12GraphicsCommandList> cmd_list)
        : cmd_list_(cmd_list)
    {}

    void disaptch_resource_barrier(std::vector<ResourceDX12*> rscs_list)
    {
        std::vector<CD3DX12_RESOURCE_BARRIER> barriers(rscs_list.size());
        for (auto i = 0; i < barriers.size(); i++)
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::UAV(rscs_list[i]->get()));
        }
        cmd_list_->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }
private:
    ComPtr<ID3D12GraphicsCommandList> cmd_list_ = nullptr;
};

class DeviceDX12 : public inference_engine::Device<DeviceDX12>
{
public:
    DeviceDX12(ComPtr<ID3D12Device> device)
        : device_(device)
    {}

    ResourceDX12 allocate_resource(std::size_t size)
    {
        return ResourceDX12(create_buffer(device_.Get(), size, D3D12_HEAP_TYPE_DEFAULT, 
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    }

private:
    ComPtr<ID3D12Device> device_ = nullptr;
};

TEST(OperatorTest, Matmul_model_0)
{

    
    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    inference_engine::Context<DeviceDX12, StreamDX12, ResourceDX12> ctx(device);

 
    //    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    //auto stream = reinterpret_cast<inference_engine_stream_t>(G_DX12_ENGINE.command_list.Get());
    //auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());




    //auto md = inferenceEngineCreateModelDescriptor();

    //inference_engine_port_desc_t input_desc{};
    //input_desc.data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;

    //auto port_a = inferenceEngineModelDescriptorAddPort(md, input_desc);
    //ASSERT_NE(port_a, INFERENCE_ENGINE_INVALID_NODE_ID);
    //auto port_b = inferenceEngineModelDescriptorAddPort(md, input_desc);
    //ASSERT_NE(port_b, INFERENCE_ENGINE_INVALID_NODE_ID);

    //inference_engine_matmul_desc_t desc{};
    //desc.input_a = port_a;
    //desc.input_b = port_b;

    //auto port_out = inferenceEngineModelDescriptorAddMatMul(md, desc);
    //ASSERT_NE(port_out, INFERENCE_ENGINE_INVALID_NODE_ID);

    //std::array<inference_engine_tensor_mapping_t, 2> inputs{};
    //// input a
    //{
    //    inputs[0].id = port_a;
    //    inputs[0].tensor.data_type = input_desc.data_type;
    //    set_array(inputs[0].tensor.dims, 1, 1, 16, 32);
    //}
    //// input b
    //{
    //    inputs[1].id = port_b;
    //    inputs[1].tensor.data_type = input_desc.data_type;
    //    set_array(inputs[1].tensor.dims, 1, 1, 32, 64);
    //}
    //auto model = inferenceEngineCompileModelDescriptor(ctx, stream, md, inputs.data(), inputs.size());
    //ASSERT_NE(model, nullptr);
    //inferenceEngineDestroyModelDescriptor(md);
    //inferenceEngineDestroyContext(ctx);
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
