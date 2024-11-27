#include "test_gpu_context.h"
#include "inference_engine_model.h"
#include "utils.h"

#include <numeric>

inline std::size_t accumulate_tensor_dims(const inference_engine_tensor_t& tensor)
{
    std::size_t ret = 1;
    for (int i = 0; i < INFERENCE_ENGINE_MAX_TENSOR_DIMS; i++)
    {
        const auto& d = tensor.dims[i];
        if (d != 0)
        {
            ret *= d;
        }
    }
    return ret;
}

TEST(OperatorTest, Basic_0)
{

    inference_engine_port_desc_t input_desc{};
    input_desc.tensor.data_type = inference_engine_data_type_t::XESS_DATA_TYPE_FP32;
    set_array(input_desc.tensor.dims, 1, 16, 32, 32);
    auto input_node = inferenceEngineCreatePort(input_desc);
    EXPECT_TRUE(input_node != nullptr);
    auto tensor_size = accumulate_tensor_dims(input_desc.tensor) * sizeof(float);

    inference_engine_activation_desc_t activation_desc{};
    activation_desc.type = inference_engine_activation_type_t::INFERENCE_ENGINE_ACTIVATION_TYPE_RELU;
    activation_desc.tensor = input_node;
    auto out_node = inferenceEngineCreateActivation(activation_desc);

    // create model with empty config
    auto model_desc = inferenceEngineCreateModelDescriptor(&out_node, 1);

    auto device = reinterpret_cast<inference_engine_device_t>(G_DX12_ENGINE.d3d12_device.Get());
    auto ctx = inferenceEngineCreateContext(device, fill_with_dx12_callbacks());
    
    auto input_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    inferenceEngineSetResource(input_node, reinterpret_cast<inference_engine_resource_t>(input_buffer.Get()));
    auto output_buffer = create_buffer(G_DX12_ENGINE.d3d12_device.Get(), tensor_size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    inferenceEngineSetResource(out_node, reinterpret_cast<inference_engine_resource_t>(output_buffer.Get()));

    //test_ctx::TestGpuContext gpu_ctx{};
    //auto input_resource = gpu_ctx.allocate_resource(accumulate_tensor_dims(input_desc.tensor) * sizeof(float));
    const auto model = inferenceEngineCompileModelDescriptor(ctx, model_desc);

    

    // ToDo: destroy nodes;
    //gpu_ctx.destroy_resource(input_resource);
}

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
    auto model_desc = inferenceEngineCreateModelDescriptor(&port_out, 1);


    const auto model = inferenceEngineCompileModelDescriptor(nullptr, model_desc);
    // can execute here (assign resources call execute)
    inference_engine_stream_t stream{};
    inferenceEngineExecuteModel(model, stream);
    // do conformance check etc ..
    // ...
    // delete model
    if (model_desc)
    {
        inferenceEngineDestroyModelDescriptor(model_desc);
    }
    destroy_node_if_valid(tensor_a);
    destroy_node_if_valid(tensor_b);
    destroy_node_if_valid(port_out);
}
