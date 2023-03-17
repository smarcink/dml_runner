#pragma once
#include "dml_types.hpp"

#include <vector>
#include <optional>
#include <memory>
#include <map>

namespace libdml
{
enum class ConvolutionDirection
{
    eForward = 0,
    eBackward
};

enum ConvolutionExecParamsType
{
    CONVOLUTION_EXEC_PARAM_TYPE_UNDEFINED = 0,
    CONVOLUTION_EXEC_PARAM_TYPE_INPUT,
    CONVOLUTION_EXEC_PARAM_TYPE_OUTPUT,
    CONVOLUTION_EXEC_PARAM_TYPE_WEIGHTS,
    CONVOLUTION_EXEC_PARAM_TYPE_BIAS,
    CONVOLUTION_EXEC_PARAM_TYPE_SCALARS  //ToDo: can we assume that scalar are always packed together? Or should be seperated list of scalaras?
};

/*
*   Library will look for kernel which supports given paramters in ConvolutionDescriptor struct.
*   For Tensor memebers it's possible to specify DataLayout = DataLayout::eAny so kernel know that runtime supports data reordering
*       If Tensors DataLayout will be concrete then its assumed data reorder is not possible.
*/
struct ConvolutionDescriptor
{
    Tensor tensor_input;
    Tensor tensor_output;
    Tensor tensor_weights;
    std::optional<Tensor> tensor_bias = std::nullopt;

    TensorDims strides = { 0, 0 };
    TensorDims dilations = { 0, 0 };
    TensorDims start_padding = { 0, 0 };
    TensorDims end_padding = { 0, 0 };

    std::uint32_t group_count = 0;

    DataType datatype_accumulator = DataType::eFp32;
    ConvolutionDirection direction = ConvolutionDirection::eForward;

    std::optional<Activation> fused_activation = std::nullopt;
};

/*
* 
*/

struct ConvolutionPrefferedLayout
{
    std::optional<DataLayout> input_layout = std::nullopt;
    std::optional<DataLayout> weights_layout = std::nullopt;
    std::optional<DataLayout> bias_layout = std::nullopt;
};

struct ExecParamInfo
{
    std::uint32_t index;
};

struct ConvolutionExecutionParams
{
    std::map<ConvolutionExecParamsType, ExecParamInfo> params_map;
    std::vector<std::byte> scalars_buffer;
};

class ConvolutionImplementation;
class ConvolutionPrimitive
{
public:
    ConvolutionPrimitive(ConvolutionImplementation* impl);
    ConvolutionPrimitive(const ConvolutionPrimitive& rhs) = delete;
    ConvolutionPrimitive& operator=(const ConvolutionPrimitive& rhs) = delete;

    ConvolutionPrimitive(ConvolutionPrimitive&& rhs) noexcept
        : impl_(std::move(rhs.impl_))
    {
    }

    ConvolutionPrimitive& operator=(ConvolutionPrimitive&& rhs) noexcept
    {
        if (this != &rhs)
        {
            impl_ = std::move(rhs.impl_);
        }
        return *this;
    }

    ~ConvolutionPrimitive();

    ImplementationInfo get_info() const;
    ConvolutionPrefferedLayout query_preffered_layouts() const;
    KernelInfo get_kernel_info(const ConvolutionPrefferedLayout& layouts) const;

    ConvolutionExecutionParams get_execution_map(ErrorCode* error_code) const;

    struct ImplDeleter
    {
        void operator()(ConvolutionImplementation*) const;
    };
private:
    std::unique_ptr<ConvolutionImplementation, ImplDeleter> impl_;
};

/*
*   Call get_convolution_implementation_list function to get list of convolution supported on given device and with given paramteres.
*   If size of vector is 0 (it's empty) then no convolution was supported for given case.
*/
std::vector<ConvolutionPrimitive> get_convolution_implementation_list(const DeviceInfo& device_info, const ConvolutionDescriptor& desc);

} // namespace libdml