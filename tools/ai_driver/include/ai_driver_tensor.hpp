#pragma once
#include <ai_driver_tensor.h>

#include <vector>
#include <cassert>

namespace ai_driver
{

    enum class DataType
    {
        fp32 = AI_DRIVER_DATA_TYPE_FP32,
        fp16 = AI_DRIVER_DATA_TYPE_FP16,

        unknown = AI_DRIVER_DATA_TYPE_UNKNOWN
    };

    struct Tensor
    {
        DataType data_type = DataType::unknown;
        std::vector<std::uint64_t> dims;
        std::vector<std::uint64_t> strides;

        Tensor() = default;
        Tensor(DataType dt, std::vector<std::uint64_t>&& dimensions)
            : data_type(dt)
            , dims(std::move(dimensions))
        {
            // ToDo: Decide if should we allow for default strides?
            // if Yes: NCHW strides calculation should be default
            strides.resize(dims.size());
        }
        Tensor(const ai_driver_tensor_t& tensor_desc)
            : data_type(static_cast<DataType>(tensor_desc.data_type))
        {
            for (int i = 0; i < AI_DRIVER_MAX_TENSOR_DIMS && tensor_desc.dims.v[i] != 0; ++i)
            {
                dims.push_back(tensor_desc.dims.v[i]);
                strides.push_back(tensor_desc.strides.v[i]);
            }
        }

        bool operator==(const Tensor& other) const
        {
            return data_type == other.data_type && dims == other.dims && strides == other.strides;
        }

        operator ai_driver_tensor_t() const
        {
            ai_driver_tensor_t ret{};
            ret.data_type = static_cast<ai_driver_data_type_t>(data_type);
            for (auto i = 0; i < dims.size(); i++)
            {
                ret.dims.v[i] = dims[i];
                ret.strides.v[i] = strides[i];
            }
            return ret;
        }

        std::size_t bytes_width() const
        {
            std::size_t size = 1;
            for (const auto& d : dims)
            {
                size *= d;
            }
            switch (data_type)
            {
            case DataType::fp32:
                return size * sizeof(float);
            case DataType::fp16:
                return size * sizeof(std::uint16_t);
            default:
                assert(!"unsupported");
            }
            return 1;
        }
    };
}  // namespace ai_driver