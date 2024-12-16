#pragma once

#include "ai_driver.h"
#include "ai_driver_operators.hpp"
#include "ai_driver_tensor.hpp"
#include <utility>
#include <string>
#include <unordered_map>

/*
    Header only CPP API for inference engine.
    It is wrapper of C API which is ABI stable.
*/
namespace ai_driver
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



template <typename TImpl, typename THandle>
struct crtp_handle
{
    TImpl& underlying() { return static_cast<TImpl&>(*this); }
    TImpl const& underlying() const { return static_cast<TImpl const&>(*this); }

    const THandle get() const { return reinterpret_cast<const THandle>(&underlying()); }
    THandle get() { return reinterpret_cast<THandle>(&underlying()); }

    static TImpl* from_handle(THandle& handle) { return reinterpret_cast<TImpl*>(handle); }
};

template<typename Impl>
class Resource : public crtp_handle<Impl, ai_driver_resource_t>
{
protected:
    Resource() = default;
};

template<typename Impl>
class Kernel : public crtp_handle<Impl, ai_driver_kernel_t>
{
protected:
    Kernel() = default;

    template<typename ResourceT>
    void set_arg(std::uint32_t idx, ResourceT* rsc, std::size_t offset = 0)
    {
        return this->underlying().set_arg(idx, rsc, offset);
    }
    template<typename T>
    void set_arg(std::uint32_t idx, T u32)
    {
        return this->underlying().set_arg(idx, u32);
    }
};

template<typename Impl>
class Stream : public crtp_handle<Impl, ai_driver_stream_t>
{
protected:
    Stream() = default;

    template<typename KernelT>
    void disaptch_kernel(KernelT& kernel, std::uint32_t gws[3], std::uint32_t lws[3])
    {
        return this->underlying().dispatch_kernel(kernel, gws, lws);
    }

    template<typename ResourceT>
    void disaptch_resource_barrier(std::vector<ResourceT*> rscs_list)
    {
        return this->underlying().disaptch_resource_barrier(rscs_list);
    }
};


template<typename Impl>
class Device : public crtp_handle<Impl, ai_driver_device_t>
{
public:
    template<typename ResourceT>
    ResourceT allocate_resource(std::size_t size)
    {
        return this->underlying().allocate_resource(size);
    }

    template<typename KernelT>
    KernelT create_kernel(const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, ai_driver_kernel_language_t language)
    {
        return this->underlying().create_kernel(kernel_name, kernel_code, kernel_code_size, build_options, language);
    }

protected:
    Device() = default;
};

template<typename DeviceTT, typename StreamTT, typename ResourceTT, typename KernelTT>
class Context
{
public:
    using DeviceT = DeviceTT;
    using StreamT = StreamTT;
    using ResourceT = ResourceTT;
    using KernelT = KernelTT;
public:
    Context(DeviceT& device)
        : device_(device)
    {
        ai_driver_context_callbacks_t cbs{};
        cbs.fn_gpu_device_allocate_resource = &allocate_resource;

        cbs.fn_gpu_device_create_kernel = &create_kernel;
        cbs.fn_gpu_kernel_destroy = &destroy_kernel;
        cbs.fn_gpu_kernel_set_arg_resource = &kernel_set_arg_resource;
        cbs.fn_gpu_kernel_set_arg_uint32 = &kernel_set_arg_u32;
        cbs.fn_gpu_kernel_set_arg_float = &kernel_set_arg_f32;

        cbs.fn_gpu_stream_resource_barrier = &disaptch_resource_barrier;
        cbs.fn_gpu_stream_execute_kernel = &disaptch_kernel;

        handle_ = aiDriverCreateContext(device_.get(), cbs);

        if (!handle_)
        {
            throw IEexception("Can't create context.");
        }
    }

    ~Context()
    {
        if (handle_)
        {
            aiDriverDestroyContext(handle_);
        }
    }

    static ai_driver_kernel_t create_kernel(ai_driver_device_t handle, const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, ai_driver_kernel_language_t language)
    {
        auto typed_impl = DeviceT::from_handle(handle);
        auto typed_kernel = new KernelT(typed_impl->create_kernel(kernel_name, kernel_code, kernel_code_size,
            build_options, language));
        return typed_kernel->get();
    }

    static void destroy_kernel(ai_driver_kernel_t kernel)
    {
        auto typed_kernel = KernelT::from_handle(kernel);
        delete typed_kernel;
    }

    static void kernel_set_arg_resource(ai_driver_kernel_t kernel, uint32_t index, ai_driver_resource_t resource, size_t offset)
    {
        auto typed_kernel = KernelT::from_handle(kernel);
        auto typed_rsc = ResourceT::from_handle(resource);
        typed_kernel->set_arg(index, typed_rsc, offset);
    }

    static void kernel_set_arg_u32(ai_driver_kernel_t kernel, uint32_t index, uint32_t u32)
    {
        auto typed_kernel = KernelT::from_handle(kernel);
        typed_kernel->set_arg(index, u32);
    }

    static void kernel_set_arg_f32(ai_driver_kernel_t kernel, uint32_t index, float f32)
    {
        auto typed_kernel = KernelT::from_handle(kernel);
        typed_kernel->set_arg(index, f32);
    }

    static ai_driver_resource_t allocate_resource(ai_driver_device_t handle, std::size_t size)
    {
        auto typed_impl = DeviceT::from_handle(handle);
        auto typed_resource = new ResourceT(typed_impl->allocate_resource(size));
        return typed_resource->get();
    }

    static void disaptch_resource_barrier(ai_driver_stream_t stream, ai_driver_resource_t* resource_list, std::size_t resource_count)
    {
        auto typed_stream = StreamT::from_handle(stream);
        std::vector<ResourceT*> rscs(resource_count);
        for (auto i = 0; i < resource_count; i++)
        {
            rscs[i] = ResourceT::from_handle(resource_list[i]);
        }
        typed_stream->disaptch_resource_barrier(rscs);
    }

    static void disaptch_kernel(ai_driver_stream_t stream, ai_driver_kernel_t kernel, uint32_t gws[3], uint32_t lws[3])
    {
        auto typed_stream = StreamT::from_handle(stream);
        auto typed_kernel = KernelT::from_handle(kernel);
        typed_stream->dispatch_kernel(*typed_kernel, gws, lws);
    }

    ai_driver_context_handle_t get() { return handle_; }
private:
    ai_driver_context_handle_t handle_;
    DeviceT& device_{};
};

}