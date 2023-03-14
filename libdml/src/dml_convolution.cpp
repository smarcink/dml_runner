#include "../include/dml_convolution.hpp"
#include "impl/convolution/convolution_impl.h"

#include <vector>
#include <cassert>

void libdml::ConvolutionPrimitive::ImplDeleter::operator()(ConvolutionImplementation* impl) const
{
    delete impl;
}


libdml::ConvolutionPrimitive::ConvolutionPrimitive(ConvolutionImplementation* impl)
    : impl_(std::move(std::unique_ptr<ConvolutionImplementation, ConvolutionPrimitive::ImplDeleter>(impl)))
{

}

libdml::ConvolutionPrimitive::~ConvolutionPrimitive() = default;


libdml::ConvolutionExecutionParams libdml::ConvolutionPrimitive::get_execution_map(ErrorCode* error_code) const
{
    if (!impl_)
    {
        if (error_code)
        {
            *error_code = ErrorCode::eGeneralError;
        }
        return {};
    }

    return impl_->get_execution_map();
}

std::vector<libdml::ConvolutionPrimitive> libdml::get_convolution_implementation_list(const DeviceInfo& device_info, const ConvolutionDescriptor& desc)
{
    std::vector<libdml::ConvolutionPrimitive> ret{};

    if (ConvolutionExampleImplementation_0::is_supported_descriptor(device_info, desc))
    {
        ret.push_back(new ConvolutionExampleImplementation_0(device_info, desc));
    }

    if (ConvolutionExampleImplementation_1::is_supported_descriptor(device_info, desc))
    {
        ret.push_back(new ConvolutionExampleImplementation_1(device_info, desc));
    }

    return ret;
}

