#pragma once 
#include <inference_engine.hpp>
#include <inference_engine_operators.hpp>
#include <inference_engine_tensor.hpp>
#include <gtest/gtest.h>
#include <random>

template <typename T, typename... Args>
inline void set_array(T* array, Args&&... args)
{
	((*(array++) = std::forward<Args>(args)), ...);
}


inline std::size_t accumulate_tensor_dims(const inference_engine_tensor_t& tensor)
{
    std::size_t ret = 1;
    for (int i = 0; i < INFERENCE_ENGINE_MAX_TENSOR_DIMS; i++)
    {
        const auto& d = tensor.dims[i];
        if (d != 0)
        {
            ret *= d;
        }
    }
    return ret;
}

inline std::size_t accumulate_tensor_dims(const inference_engine::Tensor& tensor)
{
    std::size_t ret = 1;
    for (int i = 0; i < tensor.dims.size(); i++)
    {
        const auto& d = tensor.dims[i];
        if (d != 0)
        {
            ret *= d;
        }
    }
    return ret;
}

inline void randomize_linear_container_float(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<float> container)
{
    for (auto i = 0; i < container.size(); i++)
    {
        container[i] = dist(gen);
    }
}