#pragma once
#include <dml_convolution.hpp>

#include <array>
#include <initializer_list>
#include <algorithm>

namespace libdml
{
    /*
    *   Stateless interface of convolution implementation;
    */

    namespace conv_helpers
    {
        inline bool is_supported_platform(HwPlatform platform, std::initializer_list<HwPlatform> supported_platforms)
        {
            return std::any_of(supported_platforms.begin(), supported_platforms.end(), [&platform](HwPlatform p) { return p == platform; });
        }
    }

    class ConvolutionImplementation
    {
    public:
        ConvolutionImplementation(const DeviceInfo& device_info, const ConvolutionDescriptor& desc)
            : device_info_(device_info)
            , desc_(desc)
        {
        }
        virtual ~ConvolutionImplementation() = default;

        virtual ConvolutionExecutionParams get_execution_map() const = 0;

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

        ConvolutionExecutionParams get_execution_map() const override
        {
            ConvolutionExecutionParams ret{};
            ret.params_map[CONVOLUTION_EXEC_PARAM_TYPE_INPUT] = ExecParamInfo{ 0 };
            ret.params_map[CONVOLUTION_EXEC_PARAM_TYPE_OUTPUT] = ExecParamInfo{ 1 };
            ret.params_map[CONVOLUTION_EXEC_PARAM_TYPE_WEIGHTS] = ExecParamInfo{ 2 };
            return ret;
        }
    };

    class ConvolutionExampleImplementation_1 : public ConvolutionImplementation
    {
    public:
        using ConvolutionImplementation::ConvolutionImplementation;

        static bool is_supported_descriptor(const DeviceInfo& device_info, const ConvolutionDescriptor& desc)
        {
            if (!conv_helpers::is_supported_platform(device_info.platform, {HwPlatform::eDG2, HwPlatform::eTGL, HwPlatform::eADL}))
            {
                return false;
            }
            return true;
        }

        ConvolutionExecutionParams get_execution_map() const override
        {
            ConvolutionExecutionParams ret{};
            ret.params_map[CONVOLUTION_EXEC_PARAM_TYPE_INPUT] = ExecParamInfo{ 0 };
            ret.params_map[CONVOLUTION_EXEC_PARAM_TYPE_OUTPUT] = ExecParamInfo{ 1 };
            ret.params_map[CONVOLUTION_EXEC_PARAM_TYPE_WEIGHTS] = ExecParamInfo{ 2 };
            if (desc_.tensor_bias.has_value())
            {
                ret.params_map[CONVOLUTION_EXEC_PARAM_TYPE_BIAS] = ExecParamInfo{ 3 };
            }
            return ret;
        }
    };


}