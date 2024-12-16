#pragma once
#include "ai_driver_operators.h"
#include "ai_driver.hpp"
#include "ai_driver_tensor.hpp"
#include <cstddef>

/*
    Header only CPP API for inference engine.
*/
namespace ai_driver
{
    using NodeID = std::size_t;
    constexpr static inline NodeID INVALID_NODE_ID = AI_DRIVER_INVALID_NODE_ID;

    template<typename ResourceT>
    class ConstantPortDesc : public ai_driver_constant_port_desc_t
    {
    public:       
        ConstantPortDesc(const Tensor& tensor, ResourceT& resource)
            :ai_driver_constant_port_desc_t{ tensor, resource.get() }
        {

        }
    };
}  // namespace ai_driver