#pragma once 
#include <inference_engine.h>
#include <inference_engine_operators.h>

#include <gtest/gtest.h>

template <typename T, typename U>
inline void set_array(T* array, U x)
{
    *array = x;
}

template <typename T, typename U, typename... V>
inline void set_array(T* array, U x, V... y)
{
    *array = x;
    set_array(array + 1, y...);
}

inline void destroy_node_if_valid(inference_engine_node_t n)
{
    if (n)
    {
        inferenceEngineDestroyNode(n);
    }
}