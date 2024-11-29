#pragma once

#include "error.h"

#ifdef _DEBUG
#include <iostream>
#endif

static inference_engine_result_t g_last_error = INFERENCE_ENGINE_RESULT_SUCCESS;

namespace inference_engine
{

    void set_last_error(inference_engine_result_t err) {
        g_last_error = err;
    }

    inference_engine_result_t get_last_error() {
        return g_last_error;
    }

    const char* to_string(inference_engine_result_t result) {
        switch (result) {
        case INFERENCE_ENGINE_RESULT_SUCCESS:
            return "INFERENCE_ENGINE_RESULT_SUCCESS";
        case INFERENCE_ENGINE_RESULT_INVALID_ARGUMENT:
            return "INFERENCE_ENGINE_RESULT_INVALID_ARGUMENT";
        case INFERENCE_ENGINE_RESULT_BAD_ALLOC:
            return "INFERENCE_ENGINE_RESULT_BAD_ALLOC";
        case INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN:
            return "INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN";
        }
        return "UNKNOWN_RESULT";
    }

    inference_engine_exception::inference_engine_exception(inference_engine_result_t val) : val_(val)
    {
#ifdef _DEBUG
        std::cerr << "inference_engine_exception: " << to_string(val_) << '\n';
#endif
    }

} // namespace inference_engine