#pragma once
#include <inference_engine.h>

#include <cstdint>
#include <memory>
#include <iostream>
#include <cassert>
#include <vector>
#include <array>
#include <span>
#include <format>

namespace inference_engine
{

    inline inference_engine_context_callbacks_t G_GPU_CBCS = {};
    class GpuKernel;

    class GpuResource
    {
    public:
        using Ptr = std::shared_ptr<GpuResource>;  // resources in general are shared, so we use shared_ptr
    public:
        GpuResource() = default;
        GpuResource(inference_engine_resource_t r) 
            : handle_(r)
            , is_owner_(false)
        {
            std::cout << "GpuResource C-tor non-owning handle" << std::endl;
        }
        GpuResource(inference_engine_device_t device, std::size_t size)
            : handle_(G_GPU_CBCS.fn_gpu_device_allocate_resource(device, size))
            , is_owner_(true)
        {
            std::cout << "GpuResource C-tor owning handle, size: " << size << std::endl;
        }

        GpuResource(const GpuResource&& rhs) = delete;
        GpuResource& operator=(const GpuResource&& rhs) = delete;
        GpuResource(GpuResource&& rhs) noexcept
        {
            std::swap(handle_, rhs.handle_);
        }
        GpuResource& operator=(GpuResource&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(this->handle_, rhs.handle_);
            }
            return *this;
        }

        ~GpuResource()
        {
            std::cout << "~GpuResource(), ToDo: deallocate resource if is_owner_ is true" << std::endl;
        }
        inference_engine_resource_t get() { return handle_; }
        const inference_engine_resource_t get() const { return handle_; }

    protected:
        inference_engine_resource_t handle_ = nullptr;
        bool is_owner_ = false;
    };


    class GpuStream
    {
    public:
        using Ptr = std::unique_ptr<GpuResource>;
    public:
        GpuStream(inference_engine_stream_t stream)
            : handle_(stream)
        {
            std::cout << "GpuStream C-tor" << std::endl;
        }

        GpuStream(const GpuStream&& rhs) = delete;
        GpuStream& operator=(const GpuStream&& rhs) = delete;
        GpuStream(GpuStream&& rhs) noexcept
        {
            std::swap(handle_, rhs.handle_);
        }
        GpuStream& operator=(GpuStream&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(this->handle_, rhs.handle_);
            }
            return *this;
        }

        ~GpuStream()
        {
            std::cout << "~GpuStream(), ToDo: deallocate stream if is owner" << std::endl;
        }
        inference_engine_stream_t get() { return handle_; }

        void dispatch_resource_barrier(GpuResource& resource);  //ToDo: this in future will need to support list of resources/events
        void dispatch_kernel(const GpuKernel& kernel, uint32_t gws[3], uint32_t lws[3]);
    protected:
        inference_engine_stream_t handle_;
    };

    class GpuKernel
    {
    public:
        using Ptr = std::unique_ptr<GpuKernel>;
    public:
        GpuKernel() = default;
        GpuKernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, std::size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
            : handle_(G_GPU_CBCS.fn_gpu_device_create_kernel(device, kernel_name, kernel_code, kernel_code_size, build_options, language))
        {
            std::cout << "Created GpuKernel" << std::endl;
            assert(handle_ != nullptr);
        }

        GpuKernel(const GpuKernel&& rhs) = delete;
        GpuKernel& operator=(const GpuKernel&& rhs) = delete;
        GpuKernel(GpuKernel&& rhs) noexcept
        {
            std::swap(handle_, rhs.handle_);
        }
        GpuKernel& operator=(GpuKernel&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(this->handle_, rhs.handle_);
            }
            return *this;
        }

        ~GpuKernel()
        {
            std::cout << "Deleted GpuKernel" << std::endl;
            if (handle_)
            {
                G_GPU_CBCS.fn_gpu_kernel_destroy(handle_);
            }
        }

        inference_engine_kernel_t get() { return handle_; }
        inference_engine_kernel_t get() const { return handle_; }

        void set_arg(std::uint32_t idx, const GpuResource& rsc, std::size_t offset = 0)
        {
            assert(handle_ != nullptr);
            auto rsc_handle = rsc.get();
            assert(rsc_handle != nullptr);
            G_GPU_CBCS.fn_gpu_kernel_set_arg_resource(handle_, idx, rsc_handle, offset);
        }

        void set_arg(std::uint32_t idx, std::uint32_t u32)
        {
            assert(handle_ != nullptr);
            G_GPU_CBCS.fn_gpu_kernel_set_arg_uint32(handle_, idx, u32);
        }

        void set_arg(std::uint32_t idx, float f32)
        {
            assert(handle_ != nullptr);
            G_GPU_CBCS.fn_gpu_kernel_set_arg_float(handle_, idx, f32);
        }

    private:
        inference_engine_kernel_t handle_ = nullptr;
    };

    class GpuContext
    {
    public:
        GpuContext(inference_engine_device_t device, inference_engine_context_callbacks_t callbacks)
            : device_(device)
        {
            if (G_GPU_CBCS.fn_gpu_device_allocate_resource)
            {
                assert(!"Error! For now single callback per device supported.");
            }
            G_GPU_CBCS = callbacks;
            throw_no_set_callback(G_GPU_CBCS.fn_gpu_device_create_kernel, "fn_gpu_device_create_kernel");
            throw_no_set_callback(G_GPU_CBCS.fn_gpu_device_allocate_resource, "fn_gpu_device_allocate_resource");

            throw_no_set_callback(G_GPU_CBCS.fn_gpu_kernel_set_arg_resource, "fn_gpu_kernel_set_arg_resource");
            throw_no_set_callback(G_GPU_CBCS.fn_gpu_kernel_set_arg_uint32, "fn_gpu_kernel_set_arg_uint32");
            throw_no_set_callback(G_GPU_CBCS.fn_gpu_kernel_set_arg_float, "fn_gpu_kernel_set_arg_float");
            throw_no_set_callback(G_GPU_CBCS.fn_gpu_kernel_destroy, "fn_gpu_kernel_destroy");

            throw_no_set_callback(G_GPU_CBCS.fn_gpu_stream_execute_kernel, "fn_gpu_stream_execute_kernel");
            throw_no_set_callback(G_GPU_CBCS.fn_gpu_stream_resource_barrier, "fn_gpu_stream_resource_barrier");


            std::cout << "Created GpuContext" << std::endl;
        }

        GpuContext(const GpuContext&& rhs) = delete;
        GpuContext& operator=(const GpuContext&& rhs) = delete;
        GpuContext(GpuContext&& rhs) noexcept
        {
            std::swap(device_, rhs.device_);
        }
        GpuContext& operator=(GpuContext&& rhs) noexcept
        {
            if (this != &rhs) 
            {
                std::swap(this->device_, rhs.device_);
            }
            return *this;
        }

        ~GpuContext()
        {
            std::cout << "Deleted GpuContext" << std::endl;
            G_GPU_CBCS = {};
        }

        GpuKernel::Ptr create_kernel(const char* kernel_name,
            const void* kernel_code, std::size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
        {
            return std::make_unique<GpuKernel>(device_, kernel_name, kernel_code, kernel_code_size, build_options, language);
        }

        GpuResource allocate_resource(std::size_t size)
        {
            return GpuResource(device_, size);
        }

    private:
        template<typename T>
        void throw_no_set_callback(T& cbc, std::string_view msg)
        {
            if (!cbc)
            {
                throw std::runtime_error(std::format("No {} callback set!", msg));
            }         
        }

    private:
        inference_engine_device_t device_ = nullptr;
    };

} // namespace inference_engine