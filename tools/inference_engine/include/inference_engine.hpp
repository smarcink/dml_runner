#pragma once

#include "inference_engine.h"
#include "inference_engine_model.h"
#include "inference_engine_operators.h"
#include "inference_engine_tensor.hpp"
#include <cstdint>
#include <utility>


/*
    Header only CPP API for inference engine.
    It is wrapper of C API which is ABI stable.
*/
namespace inference_engine
{
using NodeID = std::size_t;
constexpr static inline NodeID INVALID_NODE_ID = INFERENCE_ENGINE_INVALID_NODE_ID;

class IEexception : public std::exception
{
public:
    IEexception(const char* msg)
    {
        what_message_ += std::string(msg);
    }

    const char* what() const override
    {
        return what_message_.c_str();
    }
private:
    std::string what_message_ = "[IE Exception]";
};

struct TensorMapping
{
    NodeID id;
    Tensor tensor;
};



template<typename Impl>
class Stream
{
public:

protected:
    Stream()
        : handle_(reinterpret_cast<inference_engine_stream_t>(this))
    {
    }

    template<typename ResourceT>
    void disaptch_resource_barrier(std::vector<ResourceT*> rscs_list)
    {
        Impl& derived = static_cast<Impl&>(*this);
        return derived.disaptch_resource_barrier(rscs_list);
    }

    inference_engine_stream_t get() { return handle_; }
private:
    inference_engine_stream_t handle_;
};

class Resource
{
protected:
    Resource()
    {
    }
public:
    virtual ~Resource() = default;
};

template<typename Impl>
class Device
{
public:
    template<typename ResourceT>
    ResourceT allocate_resource(std::size_t size)
    {
        Impl& derived = static_cast<Impl&>(*this);
        return derived.allocate_resource(size);
    }
    virtual ~Device() = default;
    inference_engine_device_t get() { return handle_; }

protected:
    Device()
        : handle_(reinterpret_cast<inference_engine_device_t>(this))
    {
    }
protected:
    inference_engine_device_t handle_ = nullptr;
};

template<typename DeviceT, typename StreamT, typename ResourceT>
class Context
{
public:
    Context(DeviceT& device)
    {
        inference_engine_context_callbacks_t cbs{};
        cbs.fn_gpu_device_allocate_resource = &allocate_resource;

        cbs.fn_gpu_stream_resource_barrier = &disaptch_resource_barrier;

        handle_ = inferenceEngineCreateContext(device.get(), cbs);

        if (!handle_)
        {
            throw IEexception("Can't create context.");
        }
    }

    static inference_engine_resource_t allocate_resource(inference_engine_device_t handle, std::size_t size)
    {
        auto typed_impl = reinterpret_cast<Device<DeviceT>*>(handle);
        auto typed_resource = new ResourceT(typed_impl->allocate_resource<ResourceT>(size));
        return reinterpret_cast<inference_engine_resource_t>(typed_resource);
    }

    static void disaptch_resource_barrier(inference_engine_stream_t handle, inference_engine_resource_t* resource_list, std::size_t resource_count)
    {
        auto typed_impl = reinterpret_cast<StreamT*>(handle);
        std::vector<ResourceT*> rscs(resource_count);
        for (auto i = 0; i < resource_count; i++)
        {
            rscs[i] = reinterpret_cast<ResourceT*>(resource_list[i]);
        }
        typed_impl->disaptch_resource_barrier(rscs);
    }

    inference_engine_context_handle_t get() { return handle_; }
private:
    inference_engine_context_handle_t handle_;
};


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

}