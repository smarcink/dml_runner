#ifndef INFERENCE_ENGINE_ERROR_H
#define INFERENCE_ENGINE_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    INFERENCE_ENGINE_RESULT_SUCCESS = 0,
    INFERENCE_ENGINE_RESULT_INVALID_ARGUMENT,       // possibly extend to give more context...
    INFERENCE_ENGINE_RESULT_BAD_ALLOC,
    INFERENCE_ENGINE_RESULT_OTHER,
    INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN = -1000,
} inference_engine_result_t;

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_ENGINE_ERROR_H