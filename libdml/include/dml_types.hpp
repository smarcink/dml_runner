#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <array>

namespace libdml
{
enum class ErrorCode
{
    eSuccess = 0,
    eGeneralError = 1,

    // ..
    // ..
};

enum class HwPlatform
{
    eUndefined = 0,
    eSKL,
    eTGL,
    eADL,
    eDG1,
    eDG2,
    // ..
    // ..
    eCount
};

enum class KernelLanguage
{
    eUndefined = 0,
    eOpenCL,
    eCM,
    // ..
    // ..
    eCount
};

enum class KernelGrfCount
{
    e128,
    e256,
    // ... add more for variable grf support
};

enum class ExecutionParamType
{
    eUndefined = 0,
    eGlobalResource,
    eScalar
};

enum class DataLayout
{
    eUndefined = 0,
    eAny = 1,
    eNCHW,
    eNHWC,

    //..
    //..
    eCount
};

enum class DataType
{
    eUndefined = 0,
    eFp32,
    eFp16,
    eUint32,
    eInt32,
    eUint8,
    eInt8,
    
    //..
    //..
    eCount
};


// not a en enum class, so name it differently
enum TensorDimension4D
{
    TENSOR_DIMENSION_4D_N = 0,
    TENSOR_DIMENSION_4D_C = 1,
    TENSOR_DIMENSION_4D_H = 2,
    TENSOR_DIMENSION_4D_W = 3
};

// not a en enum class, so name it differently
enum TensorDimension3D
{
    TENSOR_DIMENSION_3D_N = 0,
    TENSOR_DIMENSION_3D_C = 1,
    TENSOR_DIMENSION_3D_W = 2
};

// not a en enum class, so name it differently
enum TensorDimension2D
{
    TENSOR_DIMENSION_2D_H = 0,
    TENSOR_DIMENSION_2D_W = 1
};

enum class ActivationType
{
    eUndefined = 0,
    eRelu = 1,

    //..
    //..
    eCount
};

using TensorDims = std::vector<std::int32_t>;

struct Tensor
{
    TensorDims dims;
    DataLayout data_layout;
    DataType   data_type;
};

struct Activation
{
    ActivationType type;
    //ToDo: support more activation and add params here as union {};
};

struct DeviceInfo
{
    HwPlatform platform;
    std::uint32_t eu_count;
};

struct ImplementationPriority
{
    std::uint32_t value;
};

struct ImplementationInfo
{
    ImplementationPriority priority;
    std::string name;
};

struct Jit
{
    std::string opt;
    std::string value;
};

struct KernelInfo
{
    KernelLanguage language;
    std::string code;
    std::vector<Jit> jits;

    std::array<std::uint32_t, 3> lws;
    KernelGrfCount grf_count;

};

}  // namespace libdml