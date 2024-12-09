#pragma once

#include "inference_engine.h"
#include "inference_engine_operators.hpp"
#include "inference_engine_tensor.hpp"
#include <utility>
#include <string>
#include <unordered_map>

/*
    Header only CPP API for inference engine.
    It is wrapper of C API which is ABI stable.
*/
namespace inference_engine
{


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

using TensorMapping = std::unordered_map<NodeID, Tensor>;

template<typename Impl>
class Stream
{
public:
    inference_engine_stream_t get() { return handle_; }

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

template<typename DeviceT, typename StreamT, typename ResourceT, typename KernelT>
class Context
{
public:
    Context(DeviceT& device)
        : device_(device)
    {
        inference_engine_context_callbacks_t cbs{};
        cbs.fn_gpu_device_allocate_resource = &allocate_resource;

        cbs.fn_gpu_stream_resource_barrier = &disaptch_resource_barrier;

        handle_ = inferenceEngineCreateContext(device_.get(), cbs);

        if (!handle_)
        {
            throw IEexception("Can't create context.");
        }
    }

    static inference_engine_kernel_t create_kernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
    {
        //auto typed_impl = reinterpret_cast<Device<DeviceT>*>(handle);
        //auto typed_resource = new ResourceT(typed_impl->allocate_resource<ResourceT>(size));
        //return reinterpret_cast<inference_engine_resource_t>(typed_resource);
        return nullptr;
    }

    static inference_engine_resource_t allocate_resource(inference_engine_device_t handle, std::size_t size)
    {
        auto typed_impl = reinterpret_cast<DeviceT*>(handle);
        auto typed_resource = new ResourceT(typed_impl->allocate_resource(size));
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
    DeviceT& device_{};
};

}