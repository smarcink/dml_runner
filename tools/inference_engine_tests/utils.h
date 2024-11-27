#pragma once 
#include <inference_engine.h>
#include <inference_engine_operators.h>

#include <gtest/gtest.h>

template <typename T, typename... Args>
inline void set_array(T* array, Args&&... args)
{
	((*(array++) = std::forward<Args>(args)), ...);
}

inline void destroy_node_if_valid(inference_engine_node_t n)
{
    if (n)
    {
        inferenceEngineDestroyNode(n);
    }
}
