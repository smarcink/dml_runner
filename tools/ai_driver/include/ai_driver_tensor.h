#ifndef AI_DRIVER_TENSOR_H
#define AI_DRIVER_TENSOR_H

#define AI_DRIVER_MAX_TENSOR_DIMS 8
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef enum _ai_driver_data_type_t
    {
        AI_DRIVER_DATA_TYPE_FP32 = 0,
        AI_DRIVER_DATA_TYPE_FP16,


        AI_DRIVER_DATA_TYPE_UNKNOWN = -1000,
    } ai_driver_data_type_t;

	typedef struct _ai_driver_tensor_t
	{
		ai_driver_data_type_t data_type;
		uint64_t dims[AI_DRIVER_MAX_TENSOR_DIMS];
		uint64_t strides[AI_DRIVER_MAX_TENSOR_DIMS];
	} ai_driver_tensor_t;

#ifdef __cplusplus
}
#endif

#endif  // AI_DRIVER_TENSOR_H