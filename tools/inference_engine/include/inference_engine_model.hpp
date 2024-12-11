#pragma once
#include "inference_engine_model.h"
#include "inference_engine.hpp"
#include "inference_engine_operators.hpp"

#include <utility>
#include <string>

/*
    Header only CPP API for inference engine.
*/
namespace inference_engine
{
    class ModelDescriptor
    {
    public:
        ModelDescriptor()
            : handle_(inferenceEngineCreateModelDescriptor())
        {
            if (!handle_)
            {
                throw IEexception("Could not create model descriptor!");
            }
        }
        ModelDescriptor(const ModelDescriptor&& rhs) = delete;
        ModelDescriptor& operator=(const ModelDescriptor&& rhs) = delete;
        ModelDescriptor(ModelDescriptor&& rhs) noexcept
        {
            std::swap(handle_, rhs.handle_);
        } 
        ModelDescriptor& operator=(ModelDescriptor&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(handle_, rhs.handle_);
            }
            return *this;
        }
        ~ModelDescriptor()
        {
            if (handle_)
            {
                inferenceEngineDestroyModelDescriptor(handle_);
            }        
        }

        inference_engine_model_descriptor_t get() { return handle_; }
        const inference_engine_model_descriptor_t get() const { return handle_; }


    public:
        void set_node_name(NodeID id, std::string_view name)
        {
            auto res = inferenceEngineSetNodeName(handle_, id, name.data());
            if (!res)
            {
                throw IEexception("Cant set node name for a node. Probably node does not exist?");
            }
        }
        NodeID add_port(const inference_engine_port_desc_t& desc)
        {
            return add_node(desc, inferenceEngineModelDescriptorAddPort);
        }
        NodeID add_matmul(const inference_engine_matmul_desc_t& desc)
        {
            return add_node(desc, inferenceEngineModelDescriptorAddMatMul);
        }
        NodeID add_activation(const inference_engine_activation_desc_t& desc)
        {
            return add_node(desc, inferenceEngineModelDescriptorAddActivation);
        }
        NodeID add_elementwise(const inference_engine_elementwise_desc_t& desc)
        {
            return add_node(desc, inferenceEngineModelDescriptorAddElementwise);
        }
        NodeID add_convolution(const inference_engine_convolution_desc_t& desc)
        {
            return add_node(desc, inferenceEngineModelDescriptorAddConvolution);
        }
    private:
        template<typename TDesc, typename TFunc>
        NodeID add_node(const TDesc& desc, TFunc tfunc)
        {
            const auto ret = tfunc(handle_, desc);
            if (ret == INVALID_NODE_ID)
            {
                throw IEexception("Could not add node to the model descriptor");
            }
            return ret;
        }

    private:
        inference_engine_model_descriptor_t handle_ = nullptr;
    };

    class Model
    {
    public:
        template<typename ContextT, typename StreamT>
        Model(ContextT& ctx, StreamT& stream, const ModelDescriptor& md, const TensorMapping& input_mappings)
        {
            std::vector<inference_engine_tensor_mapping_t> c_mapping;
            c_mapping.reserve(input_mappings.size());
            for (const auto& [id, tensor] : input_mappings)
            {
                c_mapping.push_back({ id, tensor });
            }
            handle_ = inferenceEngineCompileModelDescriptor(ctx.get(), stream.get(), md.get(), c_mapping.data(), c_mapping.size());
            if (!handle_)
            {
                throw IEexception("Could not compile model!");
            }
        }

        Model(const Model&& rhs) = delete;
        Model& operator=(const Model&& rhs) = delete;
        Model(Model&& rhs) noexcept
        {
            std::swap(handle_, rhs.handle_);
        }
        Model& operator=(Model&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(handle_, rhs.handle_);
            }
            return *this;
        }
        ~Model()
        {
            if (handle_)
            {
                inferenceEngineDestroyModel(handle_);
            }   
        }

        inference_engine_model_t get() { return handle_; }

        TensorMapping get_outputs() const
        {
            assert(handle_);
            std::size_t outputs_counts = 0;
            auto result = inferenceEngineModelGetOutputs(handle_, nullptr, &outputs_counts);
            if (!result)
            {
                throw IEexception("Could not get number of outputs from the model!");
            }
            if (outputs_counts == 0)
            {
                throw IEexception("Model has 0 outputs, something gone wrong or model descriptor was not valid!");
            }
            std::vector<inference_engine_tensor_mapping_t> output_mappings(outputs_counts);
            result = inferenceEngineModelGetOutputs(handle_, output_mappings.data(), &outputs_counts);
            if (!result)
            {
                throw IEexception("Could not get output mappings from the model!");
            }
            TensorMapping ret;
            for (auto i = 0; i < outputs_counts; i++)
            {
                ret[output_mappings[i].id] = Tensor(output_mappings[i].tensor);
            }
            return ret;
        }

        template<typename ResourceT>
        void set_resource(NodeID node_id, const ResourceT& rsc)
        {
            assert(handle_);
            auto result = inferenceEngineModelSetResource(handle_, node_id, rsc.get());
            if (!result)
            {
                throw IEexception("Could not set resource!");
            }
        }

        template<typename StreamT>
        void execute(StreamT& stream)
        {
            auto result = inferenceEngineExecuteModel(handle_, stream.get());
            if (!result)
            {
                throw IEexception("Could not execute model successfully!");
            }
        }

    private:
        Model(inference_engine_model_t handle)
            : handle_(handle)
        {
        }

    private:
        inference_engine_model_t handle_;
    };

}  // namespace inference_engine