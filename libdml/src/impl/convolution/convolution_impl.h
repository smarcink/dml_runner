#pragma once
#include <dml_convolution.hpp>

#include <array>

namespace libdml
{
    /*
    *   Stateless interface of convolution implementation;
    */

    class ConvolutionImplementation
    {
    public:
        ConvolutionImplementation(const DeviceInfo& device_info, const ConvolutionDescriptor& desc)
            : device_info_(device_info)
            , desc_(desc)
        {
        }
        virtual ~ConvolutionImplementation() = default;

    protected:
        DeviceInfo device_info_;
        ConvolutionDescriptor desc_;
    };


    class ConvolutionExampleImplementation_0 : public ConvolutionImplementation
    {
    public:
        using ConvolutionImplementation::ConvolutionImplementation;

        static bool is_supported_descriptor(const DeviceInfo& device_info, const ConvolutionDescriptor& desc)
        {
            return false;
        }
    };

    class ConvolutionExampleImplementation_1 : public ConvolutionImplementation
    {
    public:
        using ConvolutionImplementation::ConvolutionImplementation;

        static bool is_supported_descriptor(const DeviceInfo& device_info, const ConvolutionDescriptor& desc)
        {
            return true;
        }
    };


}