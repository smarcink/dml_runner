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

inline void randomize_linear_container_float(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::uint8_t> container)
{
    using Dt = float;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = static_cast<Dt>(dist(gen));
    }
}

inline std::vector<std::uint8_t> randomize_linear_container_float(const inference_engine::Tensor& tensor, float random_min, float random_max)
{
    const auto tensor_elements_count = accumulate_tensor_dims(tensor);
    const auto tensor_size_bytes = tensor_elements_count * sizeof(float);
    std::vector<std::uint8_t> data(tensor_size_bytes);

    // randomize data
    std::mt19937 random_generator(42); // static, create it once!
    std::uniform_real_distribution<float> uniform_distribution(random_min, random_max);
    randomize_linear_container_float(random_generator, uniform_distribution, data);
    return data;
}