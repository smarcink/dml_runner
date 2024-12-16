#pragma once
#include "ai_driver_operators.h"
#include "ai_driver.hpp"
#include <cstddef>

/*
    Header only CPP API for inference engine.
*/
namespace ai_driver
{
    using NodeID = std::size_t;
    constexpr static inline NodeID INVALID_NODE_ID = AI_DRIVER_INVALID_NODE_ID;

    enum class DataType
    {
        fp32 = AI_DRIVER_DATA_TYPE_FP32,
        fp16 = AI_DRIVER_DATA_TYPE_FP16,

        unknown = AI_DRIVER_DATA_TYPE_UNKNOWN
    };

    template<typename ResourceT>
    class ConstantPortDesc : public ai_driver_constant_port_desc_t
    {
    public:       
        ConstantPortDesc(DataType dt, ResourceT& resource)
            :ai_driver_constant_port_desc_t{ static_cast<ai_driver_data_type_t>(dt), resource.get() }
        {

        }
    };
}  // namespace ai_driver