#pragma once
#include "ai_driver_model.h"
#include "ai_driver.hpp"
#include "ai_driver_operators.hpp"

#include <utility>
#include <string>

/*
    Header only CPP API for inference engine.
*/
namespace ai_driver
{
    class ModelDescriptor
    {
    public:
        ModelDescriptor()
            : handle_(aiDriverCreateModelDescriptor())
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
                aiDriverDestroyModelDescriptor(handle_);
            }        
        }

        ai_driver_model_descriptor_t get() { return handle_; }
        const ai_driver_model_descriptor_t get() const { return handle_; }


    public:
        NodeID add_port(const ai_driver_port_desc_t& desc, const char* name = "")
        {
            return add_node(desc, aiDriverModelDescriptorAddPortNamed, name);
        }
        template<typename T>
        NodeID add(const ConstantPortDesc<T>& desc, const char* name = "")
        {
            return add_node(desc, aiDriverModelDescriptorAddConstantPortNamed, name);
        }
        NodeID add_matmul(const ai_driver_matmul_desc_t& desc, const char* name = "")
        {
            return add_node(desc, aiDriverModelDescriptorAddMatMulNamed, name);
        }
        NodeID add_activation(const ai_driver_activation_desc_t& desc, const char* name = "")
        {
            return add_node(desc, aiDriverModelDescriptorAddActivationNamed, name);
        }
        NodeID add_elementwise(const ai_driver_elementwise_desc_t& desc, const char* name = "")
        {
            return add_node(desc, aiDriverModelDescriptorAddElementwiseNamed, name);
        }
        NodeID add_convolution(const ai_driver_convolution_desc_t& desc, const char* name = "")
        {
            return add_node(desc, aiDriverModelDescriptorAddConvolutionNamed, name);
        }
    private:
        template<typename TDesc, typename TFunc>
        NodeID add_node(const TDesc& desc, TFunc tfunc, const char* name)
        {
            const auto ret = tfunc(handle_, desc, name);
            if (ret == INVALID_NODE_ID)
            {
                throw IEexception("Could not add node to the model descriptor");
            }
            return ret;
        }

    private:
        ai_driver_model_descriptor_t handle_ = nullptr;
    };

    class Model
    {
    public:
        template<typename ContextT, typename StreamT>
        Model(ContextT& ctx, StreamT& stream, const ModelDescriptor& md, const TensorMapping& input_mappings)
        {
            std::vector<ai_driver_tensor_mapping_t> c_mapping;
            c_mapping.reserve(input_mappings.size());
            for (const auto& [id, tensor] : input_mappings)
            {
                c_mapping.push_back({ id, tensor });
            }
            handle_ = aiDriverCompileModelDescriptor(ctx.get(), stream.get(), md.get(), c_mapping.data(), c_mapping.size());
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
                aiDriverDestroyModel(handle_);
            }   
        }

        ai_driver_model_t get() { return handle_; }

        TensorMapping get_outputs() const
        {
            assert(handle_);
            std::size_t outputs_counts = 0;
            auto result = aiDriverModelGetOutputs(handle_, nullptr, &outputs_counts);
            if (!result)
            {
                throw IEexception("Could not get number of outputs from the model!");
            }
            if (outputs_counts == 0)
            {
                throw IEexception("Model has 0 outputs, something gone wrong or model descriptor was not valid!");
            }
            std::vector<ai_driver_tensor_mapping_t> output_mappings(outputs_counts);
            result = aiDriverModelGetOutputs(handle_, output_mappings.data(), &outputs_counts);
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
        void set_resource(NodeID node_id, ResourceT& rsc)
        {
            assert(handle_);
            auto result = aiDriverModelSetResource(handle_, node_id, rsc.get());
            if (!result)
            {
                throw IEexception("Could not set resource!");
            }
        }

        template<typename StreamT>
        void execute(StreamT& stream)
        {
            auto result = aiDriverExecuteModel(handle_, stream.get());
            if (!result)
            {
                throw IEexception("Could not execute model successfully!");
            }
        }

    private:
        Model(ai_driver_model_t handle)
            : handle_(handle)
        {
        }

    private:
        ai_driver_model_t handle_;
    };

}  // namespace ai_driver