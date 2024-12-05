#pragma once

#include "inference_engine.h"
#include "inference_engine_model.h"
#include "inference_engine_operators.h"

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

class Context
{
public:

    inference_engine_context_handle_t get() { return handle_; }
private:
    inference_engine_context_handle_t handle_;
};

class Stream
{
public:

    inference_engine_stream_t get() { return handle_; }
private:
    inference_engine_stream_t handle_;
};

class Resource
{
public:

    inference_engine_resource_t get() { return handle_; }
private:
    inference_engine_resource_t handle_;
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

    Model compile_model(Context& ctx, Stream& stream) const
    {
        return Model(inferenceEngineCompileModelDescriptor(ctx.get(), stream.get(), handle_, nullptr, 0));
    }

private:
    inference_engine_model_descriptor_t handle_ = nullptr;
};

}