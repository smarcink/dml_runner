#pragma once
#include "context.h"
#include <inference_engine.h>

#include <cstdint>
#include <memory>
#include <iostream>
#include <cassert>

namespace inference_engine
{

    inline inference_engine_context_callbacks_t G_GPU_CBCS = {};

    class GpuKernel
    {
    public:
        using Ptr = std::unique_ptr<GpuKernel>;
    public:
        GpuKernel(inference_engine_device_t device, const char* kernel_name, const void* kernel_code, std::size_t kernel_code_size, const char* build_options)
            : kernel_(G_GPU_CBCS.fn_gpu_device_create_kernel(device, kernel_name, kernel_code, kernel_code_size, build_options))
        {
            std::cout << "Created GpuKernel" << std::endl;
            assert(kernel_ != nullptr);
        }

        GpuKernel(const GpuKernel&& rhs) = delete;
        GpuKernel& operator=(const GpuKernel&& rhs) = delete;
        GpuKernel(GpuKernel&& rhs) noexcept
        {
            std::swap(kernel_, rhs.kernel_);
        }
        GpuKernel& operator=(GpuKernel&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(this->kernel_, rhs.kernel_);
            }
            return *this;
        }

        ~GpuKernel()
        {
            std::cout << "Deleted GpuKernel" << std::endl;
            G_GPU_CBCS.fn_gpu_kernel_destroy(kernel_);
        }

    private:
        inference_engine_kernel_t kernel_ = nullptr;
    };

    class GpuContext : public IContext
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

    private:
        inference_engine_device_t device_ = nullptr;
    };

} // namespace inference_engine