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
    class Model
    {
        friend class ModelDescriptor;
        Model(const Model&& rhs) = delete;
        Model& operator=(const Model&& rhs) = delete;
        Model(Model&& rhs)
        {
            std::swap(handle_, rhs.handle_);
        }
        Model& operator=(Model&& rhs)
        {
            if (this != &rhs)
            {
                std::swap(handle_, rhs.handle_);
            }
            return *this;
        }
        ~Model()
        {
            inferenceEngineDestroyModel(handle_);
        }

        inference_engine_model_t get() { return handle_; }

        std::vector<TensorMapping> get_outputs() const
        {
            std::size_t outputs_counts = 0;
            auto result = inferenceEngineModelGetOutputs(handle_, nullptr, &outputs_counts);
            if (!result)
            {
                throw IEexception("Could not get number of outputs from the model!");
            }
            if (outputs_counts == 0)
            {
                return {};
            }
            std::vector<inference_engine_tensor_mapping_t> output_mappings(outputs_counts);
            result = inferenceEngineModelGetOutputs(handle_, output_mappings.data(), &outputs_counts);
            if (!result)
            {
                throw IEexception("Could not get output mappings from the model!");
            }
            std::vector<TensorMapping> ret(output_mappings.size());
            for (auto i = 0; i < ret.size(); i++)
            {
                ret[i] = TensorMapping{ output_mappings[i].id, output_mappings[i].tensor };
            }
            return ret;
        }

    private:
        Model(inference_engine_model_t handle)
            : handle_(handle)
        {
        }

    private:
        inference_engine_model_t handle_;
    };

    class ModelDescriptor
    {
    public:
        ModelDescriptor()
            : handle_(inferenceEngineCreateModelDescriptor())
        {
        }
        ModelDescriptor(const ModelDescriptor&& rhs) = delete;
        ModelDescriptor& operator=(const ModelDescriptor&& rhs) = delete;
        ModelDescriptor(ModelDescriptor&& rhs)
        {
            std::swap(handle_, rhs.handle_);
        }
        ModelDescriptor& operator=(ModelDescriptor&& rhs)
        {
            if (this != &rhs)
            {
                std::swap(handle_, rhs.handle_);
            }
            return *this;
        }
        ~ModelDescriptor()
        {
            inferenceEngineDestroyModelDescriptor(handle_);
        }

        inference_engine_model_descriptor_t get() { return handle_; }

        template<typename ContextT, typename StreamT>
        Model compile_model(ContextT& ctx, StreamT& stream, const std::vector<TensorMapping>& input_mappings) const
        {
            return Model(inferenceEngineCompileModelDescriptor(ctx.get(), stream.get(), handle_, nullptr, 0));
        }

    public:
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

}  // namespace inference_engine