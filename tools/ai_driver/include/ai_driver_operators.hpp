#pragma once
#include "ai_driver_operators.h"

#include <cstddef>

/*
    Header only CPP API for inference engine.
*/
namespace ai_driver
{
    using NodeID = std::size_t;
    constexpr static inline NodeID INVALID_NODE_ID = AI_DRIVER_INVALID_NODE_ID;
}  // namespace ai_driver