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

#define DPAS_DEPTH 8 
#if(CM_GENX >= 1280)
#define DPAS_EXEC_SIZE 16
#else
#define DPAS_EXEC_SIZE 8
#endif

#if OUTPUT_LAYOUT == LAYOUT_OYXI_o8
#define SIMD_SIZE 8
#elif OUTPUT_LAYOUT == LAYOUT_OYXI_o16
#define SIMD_SIZE 16
#endif

#define WEIGHT_TYPE_SIZE sizeof(INPUT_TYPE)
#define WEIGHTS_IC_OFSET (K_SIZE * K_SIZE * WEIGHT_TYPE_SIZE)
#define WEIGHTS_OC_OFSET (IC * WEIGHTS_IC_OFSET)

static const uint32_t weights_init_linear_offsets[] = {
													0 * WEIGHT_TYPE_SIZE,
													1 * WEIGHT_TYPE_SIZE,
													2 * WEIGHT_TYPE_SIZE,
													3 * WEIGHT_TYPE_SIZE,
													4 * WEIGHT_TYPE_SIZE,
													5 * WEIGHT_TYPE_SIZE,
													6 * WEIGHT_TYPE_SIZE,
													7 * WEIGHT_TYPE_SIZE,
													8 * WEIGHT_TYPE_SIZE,
													9 * WEIGHT_TYPE_SIZE,
													10 * WEIGHT_TYPE_SIZE,
													11 * WEIGHT_TYPE_SIZE,
													12 * WEIGHT_TYPE_SIZE,
													13 * WEIGHT_TYPE_SIZE,
													14 * WEIGHT_TYPE_SIZE,
													15 * WEIGHT_TYPE_SIZE
													};

static const uint32_t weights_init_offsets[] = {
                                                0 * WEIGHTS_OC_OFSET, 0 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                1 * WEIGHTS_OC_OFSET, 1 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                2 * WEIGHTS_OC_OFSET, 2 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                3 * WEIGHTS_OC_OFSET, 3 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                4 * WEIGHTS_OC_OFSET, 4 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                5 * WEIGHTS_OC_OFSET, 5 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                6 * WEIGHTS_OC_OFSET, 6 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                7 * WEIGHTS_OC_OFSET, 7 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                };



extern "C" _GENX_MAIN_ void weights_reorder(SurfaceIndex surface_input [[type("buffer_t")]], SurfaceIndex surface_output [[type("buffer_t")]])
{
    const uint thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
    
#if INPUT_LAYOUT == LAYOUT_OIYX && OUTPUT_LAYOUT == LAYOUT_IO_i8_o8_i2
	const uint32_t ic_chunk_size = DPAS_DEPTH * (sizeof(uint32_t)/ sizeof(INPUT_TYPE));
	const uint32_t ic_chunks_per_hw_thread = 8;
	const uint32_t ic_per_hw_thread = (ic_chunk_size * ic_chunks_per_hw_thread);
	const uint32_t ic_per_hw_thread_packed = (ic_per_hw_thread * sizeof(INPUT_TYPE)) / sizeof(uint32_t);
    const uint32_t int_block = (sizeof(uint32_t) / sizeof(OUTPUT_TYPE));
    const uint32_t dpas_input_channels = DPAS_DEPTH * int_block;
    
    const uint32_t oc = thread_id_0 * DPAS_EXEC_SIZE;
    const uint32_t ic = thread_id_1 * ic_per_hw_thread;
    
    const uint chunks_count = DPAS_EXEC_SIZE;
    // load
    matrix<uint32_t, chunks_count, ic_per_hw_thread_packed> data_input_typed;
    uint32_t input_offset = (oc * IC + ic) * sizeof(INPUT_TYPE);
    #pragma unroll
    for(int i = 0; i < chunks_count; i++)
    {
        data_input_typed.row(i) = cm_load<uint32_t, ic_per_hw_thread_packed, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, input_offset);
        input_offset += IC * sizeof(INPUT_TYPE);
    }
    matrix_ref<INPUT_TYPE, chunks_count, ic_per_hw_thread> data_input = data_input_typed.format<INPUT_TYPE, chunks_count, ic_per_hw_thread>();
    
    uint32_t output_offset = (oc * dpas_input_channels + ic * OC) * sizeof(INPUT_TYPE);  
    vector<OUTPUT_TYPE, DPAS_EXEC_SIZE * dpas_input_channels> data_out;
    #pragma unroll
    for(int i = 0; i < ic_chunks_per_hw_thread; i++)
    {
        #pragma unroll
        for(int j = 0; j < DPAS_EXEC_SIZE; j++)
        {
            data_out.select<dpas_input_channels, 1>(j * dpas_input_channels) = data_input.select<DPAS_EXEC_SIZE, 1, int_block, 1>(0, int_block * j + i * dpas_input_channels);
        }
        const uint32_t packed_size = (DPAS_EXEC_SIZE * dpas_input_channels)/2;
        cm_store<uint32_t, packed_size>(surface_output, output_offset, data_out.format<uint32_t>());
        output_offset += OC * dpas_input_channels * sizeof(OUTPUT_TYPE);
    }
#elif INPUT_LAYOUT == LAYOUT_OIYX && (OUTPUT_LAYOUT == LAYOUT_OYXI_o8 || OUTPUT_LAYOUT == LAYOUT_OYXI_o16)
	
	const uint32_t oc = thread_id_0 * SIMD_SIZE;
    const uint32_t ic = thread_id_1;
    const uint32_t kh = thread_id_2;
	
	const uint32_t chunks_count = SIMD_SIZE;
    matrix<INPUT_TYPE, SIMD_SIZE, K_SIZE> data_input;
	
	vector<uint32_t, 8> offsets(weights_init_linear_offsets);
	offsets += oc * WEIGHTS_OC_OFSET + ic * WEIGHTS_IC_OFSET + kh * K_SIZE * WEIGHT_TYPE_SIZE;
	#pragma unroll
	for(int i = 0; i < chunks_count; i++)
	{
		vector<INPUT_TYPE, 8> data_load = cm_load<INPUT_TYPE, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface_input, offsets);
		data_input.select<1, 1, K_SIZE, 1>(i, 0) = data_load.select<K_SIZE, 1>();
		offsets += K_SIZE * K_SIZE * IC * WEIGHT_TYPE_SIZE;
	}
	
	uint32_t ouput_offset = (oc * K_SIZE * K_SIZE * IC + ic * SIMD_SIZE + kh * K_SIZE * IC * SIMD_SIZE) * WEIGHT_TYPE_SIZE;
	#pragma unroll
	for(int kw = 0; kw < K_SIZE; kw++)
	{
		vector<OUTPUT_TYPE, SIMD_SIZE> data_out = data_input.select<SIMD_SIZE, 1, 1, 1>(0, kw);
		cm_store<uint32_t, SIMD_SIZE/2>(surface_output, ouput_offset, data_out.format<uint32_t>());
		ouput_offset += IC * SIMD_SIZE * WEIGHT_TYPE_SIZE;
	}
#else
#error Not supported layouts.
#endif
}
