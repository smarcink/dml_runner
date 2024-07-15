/*========================== begin_copyright_notice ============================

INTEL CONFIDENTIAL

Copyright (C) 2023 Intel Corporation

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>
#include <cm/cmtl.h>

#if !CM_HAS_LSC
#error [Error_device_no_lsc] Kernel designed to use lsc. Current device does not support lsc.
#endif

#if(CM_GENX >= 1280)
#define DPAS_EXEC_SIZE 16
#else
#define DPAS_EXEC_SIZE 8
#endif

#define WEIGHT_TYPE_SIZE sizeof(INPUT_TYPE)
#define DPAS_DEPTH 8
#define DPAS_LOAD_SIZE (DPAS_DEPTH * WEIGHT_TYPE_SIZE)
#define DPAS_STORE_SIZE (DPAS_EXEC_SIZE * WEIGHT_TYPE_SIZE)
#define DPAS_STORE_BLOCK (DPAS_EXEC_SIZE * DPAS_LOAD_SIZE)
#define MAX_STORE_SIZE (DPAS_EXEC_SIZE * DPAS_DEPTH)
#define MAX_STORE_BYTES 128

extern "C" _GENX_MAIN_ void weights_reorder(SurfaceIndex surface_input [[type("buffer_t")]], SurfaceIndex surface_output [[type("buffer_t")]])
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t oc = thread_id_0 * DPAS_EXEC_SIZE;
		
#if INPUT_LAYOUT == LAYOUT_OIYX && OUTPUT_LAYOUT == LAYOUT_IO_i8_o8_i2 && K_SIZE == 3
	const uint32_t ic_chunks_per_hw_thread = DPAS_DEPTH/(WEIGHT_TYPE_SIZE * WEIGHT_TYPE_SIZE);
	const uint32_t ic_per_hw_thread = (DPAS_LOAD_SIZE * ic_chunks_per_hw_thread);
	const uint32_t data_load_size = ic_per_hw_thread/WEIGHT_TYPE_SIZE;
	
    const uint32_t ic = thread_id_1 * ic_per_hw_thread;	
    uint32_t input_offset = (oc * IC + ic) * K_SIZE * K_SIZE * sizeof(INPUT_TYPE);
    uint32_t output_offset = WEI_OFFSET + ((oc * IC) + (ic * DPAS_EXEC_SIZE)) * K_SIZE * K_SIZE * sizeof(INPUT_TYPE);
	
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_0;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_1;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_2;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_3;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_4;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_5;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_6;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_7;
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input_typed_8;
	#pragma unroll
    for(int i = 0; i < DPAS_EXEC_SIZE; i++)
    {
		uint32_t load_offset = input_offset + i * IC * K_SIZE * K_SIZE * sizeof(INPUT_TYPE);
		
		vector<INPUT_TYPE, ic_per_hw_thread * K_SIZE * K_SIZE> load_line;	
		load_line.select<ic_per_hw_thread,1>(0 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 0 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(1 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 1 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(2 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 2 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(3 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 3 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(4 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 4 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(5 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 5 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(6 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 6 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(7 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 7 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);
		load_line.select<ic_per_hw_thread,1>(8 * ic_per_hw_thread).format<uint32_t>() 	= cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, load_offset + 8 * ic_per_hw_thread * WEIGHT_TYPE_SIZE);	

		data_input_typed_0.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(0);
		data_input_typed_1.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(1);
		data_input_typed_2.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(2);
		data_input_typed_3.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(3);
		data_input_typed_4.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(4);
		data_input_typed_5.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(5);
		data_input_typed_6.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(6);
		data_input_typed_7.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(7);
		data_input_typed_8.row(i) = load_line.select<ic_per_hw_thread, K_SIZE * K_SIZE>(8);
    }

	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_0 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_1 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_2 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_3 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_4 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_5 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_6 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_7 = 0;
	vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out_8 = 0;
    #pragma unroll
    for(int i = 0; i < ic_chunks_per_hw_thread; i++)
    {
		#pragma unroll
        for(int j = 0; j < DPAS_DEPTH; j++)
        {
			data_out_0.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_0.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_1.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_1.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_2.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_2.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_3.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_3.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_4.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_4.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_5.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_5.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_6.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_6.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_7.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_7.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
			data_out_8.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input_typed_8.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
        }
		
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (0 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_0.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (1 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_1.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (2 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_2.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (3 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_3.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (4 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_4.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (5 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_5.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());	
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (6 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_6.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (7 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_7.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + (8 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_8.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		
		#if DPAS_EXEC_SIZE == 16	
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (0 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_0.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (1 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_1.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (2 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_2.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (3 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_3.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (4 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_4.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (5 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_5.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());	
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (6 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_6.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (7 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_7.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK + (8 * MAX_STORE_SIZE * sizeof(uint32_t)), data_out_8.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
		#endif	
        output_offset += DPAS_EXEC_SIZE * K_SIZE * K_SIZE * DPAS_LOAD_SIZE * sizeof(OUTPUT_TYPE);
    }
#elif INPUT_LAYOUT == LAYOUT_OIYX && OUTPUT_LAYOUT == LAYOUT_IO_i8_o8_i2 && K_SIZE == 1
	const uint32_t ic_chunks_per_hw_thread = DPAS_DEPTH/WEIGHT_TYPE_SIZE;
	const uint32_t ic_per_hw_thread = (DPAS_LOAD_SIZE * ic_chunks_per_hw_thread);
	const uint32_t data_load_size = ic_per_hw_thread/WEIGHT_TYPE_SIZE;
		
	const uint32_t ic = thread_id_1 * ic_per_hw_thread;
	uint32_t input_offset = (oc * IC + ic) * sizeof(INPUT_TYPE);
	uint32_t output_offset = WEI_OFFSET + (oc * DPAS_LOAD_SIZE + ic * OC) * sizeof(INPUT_TYPE); 
	
	matrix<INPUT_TYPE, DPAS_EXEC_SIZE, ic_per_hw_thread> data_input;
    #pragma unroll
    for(int i = 0; i < DPAS_EXEC_SIZE; i++)
    {
		data_input.row(i).format<uint32_t>() = cm_load<uint32_t, data_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, input_offset);
        input_offset += IC * sizeof(INPUT_TYPE);
    }

    vector<OUTPUT_TYPE, DPAS_STORE_BLOCK> data_out = 0;
    #pragma unroll
    for(int i = 0; i < ic_chunks_per_hw_thread; i++)
    {
        #pragma unroll
        for(int j = 0; j < DPAS_DEPTH; j++)
        {
            data_out.select<DPAS_STORE_SIZE, 1>(j * DPAS_STORE_SIZE) = data_input.select<DPAS_EXEC_SIZE, 1, WEIGHT_TYPE_SIZE, 1>(0, WEIGHT_TYPE_SIZE * j + i * DPAS_LOAD_SIZE);
        }

        cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset, data_out.select<MAX_STORE_BYTES, 1>(0).format<uint32_t>());
		#if DPAS_EXEC_SIZE == 16		
			cm_store<uint32_t, MAX_STORE_BYTES/WEIGHT_TYPE_SIZE>(surface_output, output_offset + DPAS_STORE_BLOCK, data_out.select<MAX_STORE_BYTES, 1>(MAX_STORE_BYTES).format<uint32_t>());
		#endif
        output_offset += OC * DPAS_LOAD_SIZE * sizeof(OUTPUT_TYPE);
    }
#else
#error Not supported layouts.
#endif
}
