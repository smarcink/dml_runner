#pragma once

#include "inference_engine_error.h"

namespace inference_engine
{
	void set_last_error(inference_engine_result_t err);
	inference_engine_result_t get_last_error();
	const char* to_string(inference_engine_result_t t);

	struct inference_engine_exception {
		inference_engine_result_t val_;

		inference_engine_exception(inference_engine_result_t val);
	};
}