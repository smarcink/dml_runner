#pragma once
#include <inference_engine_tensor.h>

#include <vector>
#include <cassert>

namespace inference_engine
{
    enum class Layout
    {
        NCHW,
        NHWC,        
        UNKNOWN
    };

    struct Tensor
    {
        inference_engine_data_type_t data_type = INFERENCE_ENGINE_DATA_TYPE_UNKNOWN;
        std::vector<std::uint64_t> dims;    // specified in NCHW layout as DirectML
        std::vector<std::uint64_t> strides; // specified in NCHW layout as DirectML

        Tensor() = default;
        Tensor(inference_engine_data_type_t dt, std::vector<std::uint64_t>&& dimensions, Layout layout = Layout::NCHW)
            : data_type(dt)
            , dims(std::move(dimensions))
        {
            strides.resize(dims.size());
            calculate_strides(layout);
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

    protected:
        void calculate_strides(Layout layout)
        {
            if (layout == Layout::NCHW && dims.size() > 1)
            {
                strides[dims.size() - 1] = 1;
                for (size_t i = dims.size() - 2; i < dims.size(); --i)
                    strides[i] = strides[i + 1] * dims[i + 1];
            }
            else if (layout == Layout::NHWC && dims.size() == 4)
            {
                // dims are also given in NCHW order, as DML
                strides[0] = dims[1] * dims[2] * dims[3]; // n stride = H * W * C
                strides[1] = 1;                           // c stride = 1
                strides[2] = dims[3] * dims[1];           // h stride = W * C
                strides[3] = dims[1];                     // w stride = C
            }
            else
            {
                assert(!"unsupported layout");
            }
        }
    };

    struct IdToTensor
    {
        std::size_t id;
        Tensor tensor;
    };

}  // namespace inference_engine