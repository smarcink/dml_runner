#pragma once

#include <inference_engine.hpp>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>

#include <span>
#include <iostream>
#include <format>

class ResourceOCL : public inference_engine::Resource<ResourceOCL>
{
public:
    ResourceOCL() = default;
    ResourceOCL(cl::Buffer buffer)
        : mem_(buffer)
    {
    }

    cl::Buffer get_buffer()
    {
        assert(mem_() != nullptr);
        return mem_;
    }

private:
    cl::Buffer mem_;
};


class KernelOCL : public inference_engine::Kernel<KernelOCL>
{
public:
    KernelOCL(cl::Kernel kernel, std::string_view name)
        : kernel_(kernel)
        , name_(name)
    {

    }

    void set_arg(std::uint32_t idx, ResourceOCL* rsc, std::size_t offset = 0)
    {
        assert(offset == 0);
        kernel_.setArg(idx, rsc->get_buffer());
    }

    void set_arg(std::uint32_t idx, std::uint32_t u32)
    {
        kernel_.setArg(idx, u32);
    }

    void set_arg(std::uint32_t idx, float f32)
    {
        kernel_.setArg(idx, f32);
    }

    void execute(cl::CommandQueue queue, std::uint32_t gws[3], std::uint32_t lws[3])
    {
        cl::NDRange cl_gws(gws[0], gws[1], gws[2]);
        cl::NDRange cl_lws(lws[0], lws[1], lws[2]);

        queue.enqueueNDRangeKernel(kernel_, cl::NullRange, cl_gws, cl_lws, nullptr, nullptr/* &profiling_events[i]*/);
    }

private:
    std::string name_{};
    cl::Kernel kernel_;
};


class StreamOCL : public inference_engine::Stream<StreamOCL>
{
public:
    StreamOCL(cl::CommandQueue cmd_queue)
        : cmd_queue_(cmd_queue)
    {}

    void disaptch_resource_barrier(std::vector<ResourceOCL*> rscs_list)
    {
        // not perfect, but OK for now.
        cmd_queue_.enqueueBarrierWithWaitList();
    }

    void dispatch_kernel(KernelOCL& kernel, std::uint32_t gws[3], std::uint32_t lws[3])
    {
        kernel.execute(cmd_queue_, gws, lws);
    }

    template<typename T>
    void upload_data_to_resource(ResourceOCL& dst, std::span<const T> data)
    {
        cmd_queue_.enqueueWriteBuffer(dst.get_buffer(), true, 0, data.size_bytes(), data.data());
    }

    template<typename T>
    std::vector<T> readback_data_from_resource(ResourceOCL& src)
    {
        std::vector<T> ret{};
        const auto size = src.get_buffer().getInfo< CL_MEM_SIZE>();
        ret.resize(size / sizeof(T));
        cmd_queue_.enqueueReadBuffer(src.get_buffer(), true, 0, size, ret.data());
        return ret;
    }


private:
    cl::CommandQueue cmd_queue_;
};

class DeviceOCL : public inference_engine::Device<DeviceOCL>
{
public:
    DeviceOCL()
    {
        // get all platforms (drivers),
        std::vector<cl::Platform> all_platforms;
        cl::Platform::get(&all_platforms);

        if (all_platforms.size() == 0) 
        {
            std::cout << " No platforms found. Check OpenCL installation!\n";
            return;
        }
        
        cl::Platform default_platform{};
        for (auto& p : all_platforms)
        {
            std::cout << "Found platform: " << p.getInfo<CL_PLATFORM_NAME>() << ", vendor: " << p.getInfo<CL_PLATFORM_VENDOR>() << "\n";
            if (p.getInfo<CL_PLATFORM_VENDOR>().find("Intel") != std::string::npos)
            {
                default_platform = p;
                break;
            }
        }

        std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() <<", vendor: " << default_platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";

        // get default GPU of the default platform
        std::vector<cl::Device> all_devices;
        default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
        if (all_devices.size() == 0) 
        {
            std::cout << " No devices found. Check OpenCL installation!\n";
            return;
        }

        for (auto& d : all_devices)
        {
            std::cout << "Found device: " << d.getInfo<CL_DEVICE_NAME>() << " max supported version: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << "\n";
        }

        device_ = all_devices[0];
        std::cout << "Using device: " << device_.getInfo<CL_DEVICE_NAME>() << "\n";
        context_ = cl::Context(device_);
    }

    StreamOCL create_stream(bool profiling_enabled = false) const
    {
        cl_int err;
        cl_queue_properties props = profiling_enabled ? CL_QUEUE_PROFILING_ENABLE : 0u;
        auto ret = StreamOCL(cl::CommandQueue(context_, device_, props, &err));
        if (err != CL_SUCCESS)
        {
            throw std::runtime_error("Cant create OCL stream!");
        }
        return ret;
    }

    KernelOCL create_kernel(const char* kernel_name, const void* kernel_code, size_t kernel_code_size, const char* build_options, inference_engine_kernel_language_t language)
    {
        cl::Program::Sources sources{ { reinterpret_cast<const char*>(kernel_code), kernel_code_size}};
        cl::Program program(context_, sources);
        if (program.build({ device_ }, build_options) != CL_SUCCESS) 
        {
            throw std::runtime_error(std::format("Error building: {}", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_)));
        }
        std::vector<cl::Kernel> kernels;
        program.createKernels(&kernels);
        assert(kernels.size() == 1); // only single kernel for now
        return KernelOCL(kernels.at(0), kernel_name);
    }

    ResourceOCL allocate_resource(std::size_t size)
    {
        return ResourceOCL(cl::Buffer(context_, CL_MEM_READ_WRITE, size));
    }

    template<typename T>
    void upload_data_to_resource(StreamOCL& stream, ResourceOCL& dst, std::span<const T> data)
    {
        stream.upload_data_to_resource(dst, data);
    }

    template<typename T>
    std::vector<T> readback_data_from_resource(StreamOCL& stream, ResourceOCL& src)
    {
        return stream.readback_data_from_resource<T>(src);
    }

private:
    cl::Device device_;
    cl::Context context_;
};

using ContextOCL = inference_engine::Context<DeviceOCL, StreamOCL, ResourceOCL, KernelOCL>;
