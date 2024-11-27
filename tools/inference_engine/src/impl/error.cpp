#pragma once

#include "inference_engine_error.h"

static inference_engine_result_t g_last_error = INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN;

namespace inference_engine
{

	void set_last_error(inference_engine_result_t err) {
		g_last_error = err;
	}

	inference_engine_result_t get_last_error() {
		return g_last_error;
	}

} // namespace inference_engine