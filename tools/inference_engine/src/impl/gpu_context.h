#pragma once
#include <inference_engine.h>

#include <cstdint>
#include <memory>
#include <iostream>
#include <cassert>

namespace inference_engine
{

    inline inference_engine_context_callbacks_t G_GPU_CBCS = {};

    struct GpuEvent
    {
    };

    class GpuResource
    {
    public:
        using Ptr = std::unique_ptr<GpuResource>;
    public:
        GpuResource(inference_engine_device_t device, std::size_t size)
            : handle_(G_GPU_CBCS.fn_gpu_device_allocate_resource(device, size))
        {
            std::cout << "GpuResource C-tor" << std::endl;
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
            std::cout << "~GpuResource(), ToDo: deallocate resource" << std::endl;
        }
        inference_engine_resource_t get() { return handle_; }

    protected:
        inference_engine_resource_t handle_;
    };


    class GpuKernel
    {
    public:
        using Ptr = std::unique_ptr<GpuKernel>;
    public:
        GpuKernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, std::size_t kernel_code_size, const char* build_options)
            : handle_(G_GPU_CBCS.fn_gpu_device_create_kernel(device, kernel_name, kernel_code, kernel_code_size, build_options))
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
            G_GPU_CBCS.fn_gpu_kernel_destroy(handle_);
        }

        void set_arg(std::uint32_t idx, GpuResource& rsc)
        {
            assert(handle_ != nullptr);
            auto rsc_handle = rsc.get();
            assert(rsc_handle != nullptr);
            G_GPU_CBCS.fn_gpu_kernel_set_arg_resource(handle_, idx, rsc_handle);
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
            const void* kernel_code, std::size_t kernel_code_size, const char* build_options)
        {
            return std::make_unique<GpuKernel>(device_, kernel_name, kernel_code, kernel_code_size, build_options);
        }

        GpuResource allocate_resource(std::size_t size)
        {
            return GpuResource(device_, size);
        }

    private:
        inference_engine_device_t device_ = nullptr;
    };

} // namespace inference_engine