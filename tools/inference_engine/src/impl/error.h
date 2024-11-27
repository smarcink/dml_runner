#pragma once

namespace inference_engine
{
	void set_last_error(inference_engine_result_t err);
	inference_engine_result_t get_last_error();

}