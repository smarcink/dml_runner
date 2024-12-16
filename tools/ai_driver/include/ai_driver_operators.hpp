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

    struct PortDesc : public ai_driver_port_desc_t
    {
        PortDesc(DataType data_type)
            : ai_driver_port_desc_t({static_cast<ai_driver_data_type_t>(data_type) })
        {

        }
    };

    template<typename ResourceT>
    struct ConstantPortDesc : public ai_driver_constant_port_desc_t
    { 
        ConstantPortDesc(const Tensor& tensor, ResourceT& resource)
            :ai_driver_constant_port_desc_t{ tensor, resource.get() }
        {

        }
    };

    struct ActivationDesc : public ai_driver_activation_desc_t
    {
        static ActivationDesc relu(NodeID input, DataType out_data_type)
        {
            ActivationDesc desc = set_common_params(input, AI_DRIVER_ACTIVATION_TYPE_RELU, out_data_type);
            return desc;
        }

        static ActivationDesc linear(NodeID input, float alpha, float beta, DataType out_data_type)
        {
            ActivationDesc desc = set_common_params(input, AI_DRIVER_ACTIVATION_TYPE_LINEAR, out_data_type);
            desc.params.linear.a = alpha;
            desc.params.linear.b = beta;
            return desc;
        }

    private:
        static ActivationDesc set_common_params(NodeID input, ai_driver_activation_type_t type, DataType out_data_type)
        {
            ActivationDesc desc;
            desc.input = input;
            desc.type = type;
            desc.out_data_type = static_cast<ai_driver_data_type_t>(out_data_type);
            return desc;
        }
    };

    struct MatMulDesc : public ai_driver_matmul_desc_t
    {
        MatMulDesc(NodeID input_a, NodeID input_b, DataType out_data_type)
            : ai_driver_matmul_desc_t({ input_a, input_b, static_cast<ai_driver_data_type_t>(out_data_type) })
        {

        }
    };

    struct ElementwiseDesc : public ai_driver_elementwise_desc_t
    {
        static ElementwiseDesc add(NodeID input_a, NodeID input_b, DataType out_data_type)
        {
            ElementwiseDesc desc = set_common_params(input_a, input_b, AI_DRIVER_ELEMENTWISE_TYPE_ADD, out_data_type);
            return desc;
        }

    private:
        static ElementwiseDesc set_common_params(NodeID input_a, NodeID input_b, ai_driver_elementwise_type_t type, DataType out_data_type)
        {
            ElementwiseDesc desc;
            desc.input_a = input_a;
            desc.input_b = input_b;
            desc.type = type;
            desc.out_data_type = static_cast<ai_driver_data_type_t>(out_data_type);
            return desc;
        }
    };

    struct ConvolutionDesc : public ai_driver_convolution_desc_t
    {
        ConvolutionDesc(NodeID input, DataType out_data_type)
            : ai_driver_convolution_desc_t({ input, static_cast<ai_driver_data_type_t>(out_data_type) })
        {

        }
    };

}  // namespace ai_driver