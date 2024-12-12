#pragma once
#include <inference_engine_tensor.h>

#include <vector>
#include <cassert>

namespace inference_engine
{
    struct Tensor
    {
        inference_engine_data_type_t data_type = INFERENCE_ENGINE_DATA_TYPE_UNKNOWN;
        std::vector<std::uint64_t> dims;
        std::vector<std::uint64_t> strides;

        Tensor() = default;
        Tensor(inference_engine_data_type_t dt, std::vector<std::uint64_t>&& dimensions)
            : data_type(dt)
            , dims(std::move(dimensions))
        {
            // ToDo: Decide if should we allow for default strides?
            // if Yes: NCHW strides calculation should be default
            strides.resize(dims.size());
        }
        Tensor(const inference_engine_tensor_t& tensor_desc)
            : data_type(tensor_desc.data_type)
        {
            for (int i = 0; i < INFERENCE_ENGINE_MAX_TENSOR_DIMS && tensor_desc.dims[i] != 0; ++i)
            {
                dims.push_back(tensor_desc.dims[i]);
                strides.push_back(tensor_desc.strides[i]);
            }
        }

        bool operator==(const Tensor& other) const
        {
            return data_type == other.data_type && dims == other.dims && strides == other.strides;
        }

        operator inference_engine_tensor_t() const
        {
            inference_engine_tensor_t ret{};
            ret.data_type = data_type;
            for (auto i = 0; i < dims.size(); i++)
            {
                ret.dims[i] = dims[i];
                ret.strides[i] = strides[i];
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
            case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP32:
                return size * sizeof(float);
            case inference_engine_data_type_t::INFERENCE_ENGINE_DATA_TYPE_FP16:
                return size * sizeof(std::uint16_t);
            default:
                assert(!"unsupported");
            }
            return 1;
        }
    };

    struct IdToTensor
    {
        std::size_t id;
        Tensor tensor;
    };

}  // namespace inference_engine