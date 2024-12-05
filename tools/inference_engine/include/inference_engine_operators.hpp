#pragma once
#include "inference_engine_operators.h"

#include <cstddef>

/*
    Header only CPP API for inference engine.
*/
namespace inference_engine
{
    using NodeID = std::size_t;
    constexpr static inline NodeID INVALID_NODE_ID = INFERENCE_ENGINE_INVALID_NODE_ID;
}  // namespace inference_engine