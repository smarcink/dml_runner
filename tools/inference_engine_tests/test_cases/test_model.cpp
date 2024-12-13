#include "test_dx12_context.h"
#include "test_ocl_context.h"
#include "inference_engine.hpp"
#include "inference_engine_model.hpp"
#include "utils.h"
#include <tuple>

#include <numeric>
#include <vector>
#include <array>
#include <format>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl.hpp>

inline dnnl::graph::logical_tensor::data_type to_onednn_data_type(inference_engine_data_type_t dt)
{
    dnnl::graph::logical_tensor::data_type data_type = dnnl::graph::logical_tensor::data_type::undef;
    switch (dt)
    {
    case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32:
    {
        data_type = dnnl::graph::logical_tensor::data_type::f32;
        break;
    }
    case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16:
    {
        data_type = dnnl::graph::logical_tensor::data_type::f16;
        break;
    }
    default:
        assert(!"add more data types support to to_onednn_logical_tensor() function");
    }
    return data_type;
}

inline dnnl::graph::logical_tensor to_onednn_logical_tensor(std::size_t onednn_logical_tensor_id, const inference_engine::Tensor& tensor)
{
    dnnl::graph::logical_tensor::dims dims{};
    for (const auto& d : tensor.dims)
    {
        dims.push_back(static_cast<dnnl::graph::logical_tensor::dim>(d));
    }

    return dnnl::graph::logical_tensor(onednn_logical_tensor_id, to_onednn_data_type(tensor.data_type), dims, dnnl::graph::logical_tensor::layout_type::strided);
}

inline dnnl::graph::tensor onednn_get_or_allocate_graph_mem(const dnnl::graph::logical_tensor& lt, std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>>& data_buffer, const dnnl::engine& eng)
{
    const auto id = lt.get_id();
    const auto mem_size = lt.get_mem_size();
    if (data_buffer.find(id) == std::end(data_buffer))
    {
        data_buffer[id] = std::vector<std::uint8_t>(mem_size);
    }
    return dnnl::graph::tensor{ lt, eng, data_buffer[id].data() };
}

class Onednn
{
public:
    Onednn()
        : engine_(dnnl::engine::kind::cpu, 0)
        , graph_(dnnl::engine::kind::cpu)
    {
        // disable cache due to it not working correctly (ToDo: double check after some OneDNN rebase).
        dnnl::graph::set_compiled_partition_cache_capacity(0);
    }

    std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>> get_results(const std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>>& input_host_data)
    {
        dnnl::stream stream(engine_);

        graph_.finalize();
        auto partitions = graph_.get_partitions();

        // copy input host data, this will be extended with output and intermidate data
        std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>> host_data = input_host_data;

        std::vector<dnnl::graph::compiled_partition> cps;
        cps.reserve(partitions.size());
        for (auto& p : partitions)
        {
            std::vector<dnnl::graph::logical_tensor> onednn_inputs = p.get_input_ports();
            std::vector<dnnl::graph::logical_tensor> onednn_outputs = p.get_output_ports();

            // Update input logical tensors with concrete shape and layout
            for (auto& input : onednn_inputs) 
            {
                const auto id = input.get_id();
                // If the tensor is an output of another partition, use the cached logical tensor
                if (id_to_queried_logical_tensors_.find(id) != id_to_queried_logical_tensors_.end())
                {
                    input = id_to_queried_logical_tensors_[id];
                }
                else
                {
                    // ToDo:
                    //input = onednn_input;
                }
            }

            // Update output logical tensors with concrete shape and layout
            for (auto& output : onednn_outputs)
            {
                const auto id = output.get_id();
                output = dnnl::graph::logical_tensor{ id, output.get_data_type(), DNNL_GRAPH_UNKNOWN_NDIMS, dnnl::graph::logical_tensor::layout_type::strided };
            }

            // compile partition
            cps.push_back(p.compile(onednn_inputs, onednn_outputs, engine_));

            // Update output logical tensors with queried one
            for (auto& output : onednn_outputs) 
            {
                const auto id = output.get_id();
                output = cps.back().query_logical_tensor(id);
                id_to_queried_logical_tensors_[id] = output;
            }

            // Allocate memory for the partition, and bind the data buffers with
            // input and output logical tensors
            std::vector<dnnl::graph::tensor> inputs_ts;
            for (auto i = 0; i < onednn_inputs.size(); i++)
            {
                inputs_ts.push_back(onednn_get_or_allocate_graph_mem(onednn_inputs[i], host_data, engine_));
            }
            std::vector<dnnl::graph::tensor> outputs_ts;
            for (auto i = 0; i < onednn_outputs.size(); i++)
            {
                outputs_ts.push_back(onednn_get_or_allocate_graph_mem(onednn_outputs[i], host_data, engine_));
            }
            cps.back().execute(stream, inputs_ts, outputs_ts);
        }
        stream.wait();

        std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>> return_data{};
        for (const auto& op : partitions.back().get_output_ports())
        {
            const auto id = op.get_id();
            return_data[id] = host_data.at(id);
        }

        return return_data;
    }

    void add_op_to_graph(const dnnl::graph::op& op)
    {
        graph_.add_op(op);
    }

    dnnl::graph::logical_tensor& get_logical_tensor(std::size_t id)
    {
        return id_to_queried_logical_tensors_.at(id);
    }

    const dnnl::graph::logical_tensor& get_logical_tensor(std::size_t id) const
    {
        return id_to_queried_logical_tensors_.at(id);
    }

    const dnnl::graph::logical_tensor& add_logical_tensor(std::size_t id, inference_engine_data_type_t dt)
    {
        id_to_queried_logical_tensors_[id] = dnnl::graph::logical_tensor(id, to_onednn_data_type(dt));
        return get_logical_tensor(id);
    }

    void set_tensor(std::size_t id, const inference_engine::Tensor& tensor)
    {
        id_to_queried_logical_tensors_.at(id) = to_onednn_logical_tensor(id, tensor);
    }

private:
    dnnl::engine engine_;
    dnnl::graph::graph graph_;
    std::unordered_map<std::size_t, dnnl::graph::logical_tensor> id_to_queried_logical_tensors_;
};


template<typename ContextT>
class ModelTest
{
protected:
    ModelTest()
        : device_()
        , ctx_(device_)
    {
    }

    ~ModelTest() = default;

protected:
    void run_test()
    {
        auto stream = device_.create_stream();
        auto model = inference_engine::Model(ctx_, stream, md_, inputs_);

        // allocate and upload inputs
        for (auto& [id, resource] : node_id_to_resource_)
        {
            device_.upload_data_to_resource<std::uint8_t>(stream, resource, input_node_id_to_data_[id]);
        }

        // allocate resources for the model outputs
        auto outputs = model.get_outputs();
        for (const auto& [id, tensor] : outputs)
        {
            node_id_to_resource_[id] = device_.allocate_resource(tensor.bytes_width());
        }

        // set resources to the model
        for (auto& [id, resource] : node_id_to_resource_)
        {
            model.set_resource(id, resource);
        }

        // execute model
        model.execute(stream);

        // readback outputs
        for (const auto& [id, tensor] : outputs)
        {
            output_node_id_to_data_[id] = device_.readback_data_from_resource<std::uint8_t>(stream, node_id_to_resource_[id]);
        }
        // compute references
        references_id_to_data_ = onednn_.get_results(input_node_id_to_data_);
    }

    const std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>>& get_output_data()
    {
        return output_node_id_to_data_;
    }

    const std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>>& get_reference_data()
    {
        return references_id_to_data_;
    }


    inference_engine::NodeID add_port(const inference_engine_port_desc_t& desc)
    {
        const auto ret = md_.add_port(desc);
        onednn_.add_logical_tensor(ret, desc.data_type);
        return ret;
    }

    inference_engine::NodeID add_activation(const inference_engine_activation_desc_t& desc)
    {
        const auto ret = md_.add_activation(desc);

        auto onednn_in = onednn_.get_logical_tensor(desc.input);


        switch (desc.type)
        {
        case inference_engine_activation_type_t::INFERENCE_ENGINE_ACTIVATION_TYPE_RELU:
        {
            auto onednn_out = onednn_.add_logical_tensor(ret, desc.out_data_type);
            auto onednn_activ = dnnl::graph::op(ret, dnnl::graph::op::kind::ReLU, { onednn_in }, { onednn_out });
            onednn_.add_op_to_graph(onednn_activ);
            break;
        }
        case inference_engine_activation_type_t::INFERENCE_ENGINE_ACTIVATION_TYPE_LINEAR:
        {
            auto onednn_in_a = onednn_.add_logical_tensor(onednn_unique_lt_id_, desc.out_data_type);
            onednn_.set_tensor(onednn_in_a.get_id(), inference_engine::Tensor(desc.out_data_type, { 1, 1, 1, 1 }));
            input_node_id_to_data_[onednn_unique_lt_id_].resize(4);
            auto in_a = reinterpret_cast<float*>(input_node_id_to_data_[onednn_unique_lt_id_].data());
            *in_a = desc.params.linear.a;
            onednn_unique_lt_id_++;
            auto onednn_out_temp = onednn_.add_logical_tensor(onednn_unique_lt_id_, desc.out_data_type);
            onednn_unique_lt_id_++;
            auto mult = dnnl::graph::op(onednn_unique_op_id_, dnnl::graph::op::kind::Multiply, { onednn_in, onednn_in_a }, { onednn_out_temp });
            onednn_unique_op_id_++;
            onednn_.add_op_to_graph(mult);

            auto onednn_in_b = onednn_.add_logical_tensor(onednn_unique_lt_id_, desc.out_data_type);
            onednn_.set_tensor(onednn_in_b.get_id(), inference_engine::Tensor(desc.out_data_type, { 1, 1, 1, 1 }));
            input_node_id_to_data_[onednn_unique_lt_id_].resize(4);
            auto in_b = reinterpret_cast<float*>(input_node_id_to_data_[onednn_unique_lt_id_].data());
            *in_b = desc.params.linear.b;
            onednn_unique_lt_id_++;
            auto onednn_out = onednn_.add_logical_tensor(ret, desc.out_data_type);
            auto add = dnnl::graph::op(ret, dnnl::graph::op::kind::Add, { onednn_out_temp, onednn_in_b }, { onednn_out });
            onednn_.add_op_to_graph(add);
            break;
        }
        }    
        
        return ret;
    }

    inference_engine::NodeID add_matmul(const inference_engine_matmul_desc_t& desc)
    {
        const auto ret = md_.add_matmul(desc);

        auto onednn_in_a = onednn_.get_logical_tensor(desc.input_a);
        auto onednn_in_b = onednn_.get_logical_tensor(desc.input_b);
        auto onednn_out = onednn_.add_logical_tensor(ret, desc.out_data_type);
        dnnl::graph::op onednn_matmul(ret, dnnl::graph::op::kind::MatMul, { onednn_in_a, onednn_in_b }, { onednn_out });
        onednn_.add_op_to_graph(onednn_matmul);
        return ret;
    }

    void add_input_mapping_and_data(inference_engine::NodeID port_id, const inference_engine::Tensor& tensor, const std::vector<std::uint8_t>& data)
    {
        inputs_[port_id] = tensor;
        input_node_id_to_data_[port_id] = data;

        onednn_.set_tensor(port_id, tensor);

        node_id_to_resource_[port_id] = device_.allocate_resource(tensor.bytes_width());
    }    

protected:
    ContextT::DeviceT device_;
    ContextT ctx_;

    inference_engine::ModelDescriptor md_{};
    inference_engine::TensorMapping inputs_{};
    std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>> input_node_id_to_data_;
    std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>> output_node_id_to_data_;
    std::unordered_map<inference_engine::NodeID, std::vector<std::uint8_t>> references_id_to_data_;
    std::unordered_map<inference_engine::NodeID, typename ContextT::ResourceT> node_id_to_resource_;

    Onednn onednn_;
    std::size_t onednn_unique_lt_id_ = 10000;
    std::size_t onednn_unique_op_id_ = 10000;
};

template<typename T>
class ModelTestGeneric : public ModelTest<T>, public testing::Test {};  // undefined context
class ModelTestGenericOCL : public ModelTest<ContextOCL>, public testing::Test {};  // OCL context
class ModelTestGenericDX12 : public ModelTest<ContextDX12>, public testing::Test {};  // DX12 context

using TestContextTypes = testing::Types<ContextOCL, ContextDX12>;
TYPED_TEST_SUITE(ModelTestGeneric, TestContextTypes);

// This test will run for each context (currently OCL and DX12).
TYPED_TEST(ModelTestGeneric, Activation_0)
{
    auto port_id = this->add_port(inference_engine_port_desc_t{ INFERENCE_ENGINE_DATA_TYPE_FP32 });
    auto out_node = this->add_activation(inference_engine_activation_desc_t{ port_id, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, INFERENCE_ENGINE_DATA_TYPE_FP32 });

    const auto input_tensor = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 2, 4, 4 });
    auto input_data = randomize_linear_container_float(input_tensor, -5.0f, 5.0f);
    this->add_input_mapping_and_data(port_id, input_tensor, input_data);

    this->run_test();
    const auto outputs = this->get_output_data();
    const auto outputs_reference = this->get_reference_data();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs_reference.size(), 1);
    ASSERT_EQ(outputs.at(out_node).size(), outputs_reference.at(out_node).size());
    const auto* typed_data_out = reinterpret_cast<const float*>(outputs.at(out_node).data());
    const auto* typed_data_out_ref = reinterpret_cast<const float*>(outputs_reference.at(out_node).data());

    // validate conformance
    for (int i = 0; i < outputs.at(out_node).size() / sizeof(float); i++)
    {
        const auto& real_data = typed_data_out[i];
        const auto& reference = typed_data_out_ref[i];
        ASSERT_FLOAT_EQ(real_data, reference) << "idx: " << i;
    }
}


class ModelTestMatMulMultipleActivations : public ModelTestGenericOCL
{
public:
    void build_model_and_run(const std::int32_t& num_activations)
    {
        const auto data_type = INFERENCE_ENGINE_DATA_TYPE_FP32;
        auto input_a = add_port(inference_engine_port_desc_t{ data_type });
        auto input_b = add_port(inference_engine_port_desc_t{ data_type });
        auto port_matmul_out = add_matmul(inference_engine_matmul_desc_t{ input_a, input_b, data_type });

        std::vector<inference_engine::NodeID> activation_nodes;
        inference_engine_activation_desc_t activation_desc{};
        activation_desc.type = INFERENCE_ENGINE_ACTIVATION_TYPE_LINEAR;
        activation_desc.params.linear.a = 2.0f;
        activation_desc.params.linear.b = 0.5f;
        for (int i = 0; i < num_activations; ++i)
        {
            activation_desc.input = i == 0 ? port_matmul_out : activation_nodes.back();
            activation_nodes.push_back(add_activation(activation_desc));
        }

        const auto input_a_tensor = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 1, 32, 32 });
        auto input_a_data = randomize_linear_container_float(input_a_tensor, -1.0f, 1.0f);
        add_input_mapping_and_data(input_a, input_a_tensor, input_a_data);

        const auto input_b_tensor = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 1, 32, 32 });
        auto input_b_data = randomize_linear_container_float(input_b_tensor, -1.0f, 1.0f);
        add_input_mapping_and_data(input_b, input_b_tensor, input_b_data);

        run_test();
        const auto out_node = activation_nodes.back();
        const auto out_ref_node = activation_nodes.back();
        const auto outputs = get_output_data();
        const auto outputs_reference = get_reference_data();
        ASSERT_EQ(outputs.size(), 1);
        ASSERT_NE(outputs.find(out_node), std::end(outputs));
        ASSERT_EQ(outputs_reference.size(), 1);
        ASSERT_EQ(outputs.at(out_node).size(), outputs_reference.at(out_ref_node).size());
        const auto* typed_data_out = reinterpret_cast<const float*>(outputs.at(out_node).data());
        const auto* typed_data_out_ref = reinterpret_cast<const float*>(outputs_reference.at(out_ref_node).data());

        // validate conformance
        for (int i = 0; i < outputs.at(out_node).size() / sizeof(float); i++)
        {
            const auto& real_data = typed_data_out[i];
            const auto& reference = typed_data_out_ref[i];
            ASSERT_NEAR(real_data, reference, 0.0001f) << "idx: " << i;
        }
    }
};

TEST_F(ModelTestMatMulMultipleActivations, fused_single)
{
    build_model_and_run(1);
}

TEST_F(ModelTestMatMulMultipleActivations, fused_two)
{
    build_model_and_run(2);
}

TEST_F(ModelTestMatMulMultipleActivations, fused_five)
{
    build_model_and_run(5);
}

TEST_F(ModelTestGenericOCL, MatMul_6_nodes)
{
    // *   *  *
    //  \ /  /
    //   *  *  // matmul, activation
    //    \/
    //     * // mat mul

    const auto data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32;
    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = data_type;
    // inputs
    auto input_a = add_port(input_desc);
    auto input_b = add_port(input_desc);
    auto input_c = add_port(input_desc);
    // matmul
    auto port_matmul_a = add_matmul(inference_engine_matmul_desc_t{ input_a, input_b, data_type });
    // activation
    auto port_activation = add_activation(inference_engine_activation_desc_t{ input_c, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, data_type });
    // MatMul final
    auto port_matmul_out_final = add_matmul(inference_engine_matmul_desc_t{ port_matmul_a, port_activation, data_type });

    const auto input_a_tensor = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 1, 8, 16 });
    auto input_a_data = randomize_linear_container_float(input_a_tensor, -1.0f, 1.0f);
    add_input_mapping_and_data(input_a, input_a_tensor, input_a_data);

    const auto input_b_tensor = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 1, 16, 32 });
    auto input_b_data = randomize_linear_container_float(input_b_tensor, -1.0f, 1.0f);
    add_input_mapping_and_data(input_b, input_b_tensor, input_b_data);

    const auto input_c_tensor = inference_engine::Tensor(INFERENCE_ENGINE_DATA_TYPE_FP32, { 1, 1, 32, 64 });
    auto input_c_data = randomize_linear_container_float(input_c_tensor, -1.0f, 1.0f);
    add_input_mapping_and_data(input_c, input_c_tensor, input_c_data);

    run_test();
    const auto out_node = port_matmul_out_final;
    const auto out_ref_node = port_matmul_out_final;
    const auto outputs = get_output_data();
    const auto outputs_reference = get_reference_data();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(outputs.find(out_node), std::end(outputs));
    ASSERT_EQ(outputs_reference.size(), 1);
    ASSERT_EQ(outputs.at(out_node).size(), outputs_reference.at(out_ref_node).size());
    const auto* typed_data_out = reinterpret_cast<const float*>(outputs.at(out_node).data());
    const auto* typed_data_out_ref = reinterpret_cast<const float*>(outputs_reference.at(out_ref_node).data());

    // validate conformance
    for (int i = 0; i < outputs.at(out_node).size() / sizeof(float); i++)
    {
        const auto& real_data = typed_data_out[i];
        const auto& reference = typed_data_out_ref[i];
        ASSERT_NEAR(real_data, reference, 0.0001f) << "idx: " << i;
    }
}

//ToDo refactor this test to use ModelTestGeneric test suite once we have fully fledged convolution primitive
TEST(ModelTest, ConvPlusAddFusion)
{
    // *           * port, port
    //  \         /
    //   *       *   conv, conv
    //    \     /
    //     \   *     activation
    //      \ /
    //       *       elementwise_add
    //       *       activation at the end, so that we can fuse "inner" nodes

    // we'll fuse nodes on the right side of the graph
    // *        * port, port
    //  \      /
    //   *    /   conv
    //    \  /
    //     * conv fused with activation fused with elementwise_add 
    //     * activation at the end, unchanged


    DeviceDX12 device(G_DX12_ENGINE.d3d12_device);
    StreamDX12 stream(G_DX12_ENGINE.command_list);
    ContextDX12 ctx(device);
    const auto data_type = inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16;

    inference_engine::ModelDescriptor md{};

    inference_engine_port_desc_t input_desc{};
    input_desc.data_type = data_type;
    auto input_a = md.add_port(input_desc);
    auto input_b = md.add_port(input_desc);

    // Conv left
    auto port_conv_a = md.add_convolution(inference_engine_convolution_desc_t{ input_a, data_type }, "conv_a");

    // Conv right
    auto port_conv_b = md.add_convolution(inference_engine_convolution_desc_t{ input_b, data_type }, "conv_b");

    // activation
    auto activation = md.add_activation(inference_engine_activation_desc_t{port_conv_b, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, data_type });

    // elementwise_add
    inference_engine_elementwise_desc_t add_desc_final{};
    add_desc_final.type = inference_engine_elementwise_type_t::INFERENCE_ENGINE_ELEMENTWISE_TYPE_ADD;
    add_desc_final.input_a = port_conv_a;
    add_desc_final.input_b = activation;
    add_desc_final.out_data_type = data_type;
    auto port_add_final = md.add_elementwise(add_desc_final, "elementwise");

    // activation final
    auto final_activation = md.add_activation(inference_engine_activation_desc_t{port_add_final, INFERENCE_ENGINE_ACTIVATION_TYPE_RELU, data_type }, "final_activation");

    // define input mappings
    inference_engine::TensorMapping inputs{};
    inputs[input_a] = inference_engine::Tensor(data_type, { 1, 1, 4, 4 });
    inputs[input_b] = inference_engine::Tensor(data_type, { 1, 1, 4, 4 });

    auto model = inference_engine::Model(ctx, stream, md, inputs);
    const auto outputs_mappings = model.get_outputs();
    ASSERT_EQ(outputs_mappings.size(), 1);
    ASSERT_NE(outputs_mappings.find(final_activation), std::end(outputs_mappings));
    const auto& output_mapping = outputs_mappings.at(final_activation);
    ASSERT_EQ(output_mapping.data_type, data_type);
    const std::vector<std::uint64_t> expected_output_size = { 1, 1, 4, 4 };
    ASSERT_EQ(output_mapping.dims, expected_output_size);

    auto input_a_buffer = device.allocate_resource(accumulate_tensor_dims(inputs[input_a]));
    auto input_b_buffer = device.allocate_resource(accumulate_tensor_dims(inputs[input_b]));
    auto output_buffer = device.allocate_resource(accumulate_tensor_dims(output_mapping));

    // set resources
    model.set_resource(input_a, input_a_buffer);
    model.set_resource(input_b, input_a_buffer);
    model.set_resource(final_activation, output_buffer);

    // can execute here (assign resources call execute)
    model.execute(stream);

    // do conformance check etc ..
    // ...
}