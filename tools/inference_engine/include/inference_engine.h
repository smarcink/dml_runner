#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#ifdef inference_engine_EXPORTS
#define INFERENCE_ENGINE_API __declspec(dllexport)
#else
#define INFERENCE_ENGINE_API __declspec(dllimport)
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _inference_engine_context_handle_t* inference_engine_context_handle_t;

typedef enum _inference_engine_result_t
{
    XESS_RESULT_SUCCESS = 0,


    XESS_RESULT_ERROR_UNKNOWN = -1000,
} inference_engine_result_t;

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineCreateContext(inference_engine_context_handle_t hContext);

#ifdef __cplusplus
}
#endif

#endif  // INFERENCE_ENGINE_H