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

#if !CM_HAS_DPAS
#error [Error_device_no_dpas] Kernel designed to use dpas. Current device does not support dpas.
#endif

#if !CM_HAS_LSC
#error [Error_device_no_lsc] Kernel designed to use lsc. Current device does not support lsc.
#endif

#if BLOCK_W > 8
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_w in range: <1; 7>;
#endif

#if BLOCK_OC != 8 && BLOCK_OC != 16 && BLOCK_OC != 32 && BLOCK_OC != 40 && BLOCK_OC != 64 && BLOCK_OC != 80
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_oc which is equal to 8 or 16 or 32 or 40 or 64 or 80;
#endif

#define DPAS_DEPTH 8 
#if(CM_GENX >= 1280)
#define EXEC_SIZE 16
#else
#define EXEC_SIZE 8
#endif
#define OUTPUT_CHANNEL_MULTIPLIER (EXEC_SIZE/DPAS_DEPTH)
#define BLOCK_H 1
#define DT_OUT half
#define DT_IN half
#define DT_IN_SIZE 2 
#define DT_WEIGHTS half
// accu on DG2 have to be float for half dt inputs
#define DT_ACCU float

#define DPAS_INPUT_CHANNELS (DPAS_DEPTH * sizeof(DT_IN))
#define DPAS_OUTPUT_CHANNELS EXEC_SIZE
#define DPAS_RC BLOCK_W

// currently it is fixed with 1, can be tuned for larger input channel sizes for any future case
#define SLICE_IC 1  

#define CONV_LOOP_COUNT ((INPUT_CHANNELS/DPAS_INPUT_CHANNELS) / SLICE_IC)

#define WEIGHTS_REG_SIZE (DPAS_INPUT_CHANNELS * DPAS_OUTPUT_CHANNELS)

#define INPUT_NCHW_PLANE_SIZE (INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN))
#define OUTPUT_NCHW_PLANE_SIZE (OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT))

#define INPUT_NHWC_PLANE_SIZE (INPUT_CHANNELS * sizeof(DT_IN))
#define OUTPUT_NHWC_PLANE_SIZE (OUTPUT_CHANNELS * sizeof(DT_OUT))

#define NCHW 0
#define NHWC 1

#define LOAD_3x3_BLOCK_SIZE (BLOCK_W + 2)
#define LOAD_3x3_BLOCK_START 0
#define LOAD_3x3_BLOCK_END 9

#if(INPUT_LAYOUT == NHWC)
#define OUTPUT_DPAS_OFFSET DPAS_OUTPUT_CHANNELS * sizeof(DT_OUT)
#else
#define OUTPUT_DPAS_OFFSET (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT)
#endif

static const uint32_t init_linear_offsets_16[] = { 0, 2, 4, 6,  8, 10, 12, 14,  16, 18, 20, 22,  24, 26, 28, 30 };

template<uint32_t LOAD_W>
_GENX_ inline vector<DT_IN, LOAD_3x3_BLOCK_SIZE * DPAS_INPUT_CHANNELS> load_3x3_input(SurfaceIndex surface [[type("buffer_t")]], int input_offset, int w_chunk_id)
{
#if(INPUT_LAYOUT == NHWC)
	const uint32_t LOAD_W_WIDTH = DPAS_INPUT_CHANNELS;
#else
    const uint32_t LOAD_W_WIDTH = LOAD_W * STRIDE_W;
#endif
    const uint32_t LOAD_W_BYTES_WIDTH = LOAD_W_WIDTH * sizeof(DT_IN);
    const uint32_t LOAD_W_DWORDS = LOAD_W_BYTES_WIDTH / sizeof(uint32_t);
    vector<DT_IN, LOAD_3x3_BLOCK_SIZE * DPAS_INPUT_CHANNELS> data_out;
	vector<uint32_t, LOAD_W_WIDTH> load_offsets(init_linear_offsets_16);
	const int current_kw = w_chunk_id * BLOCK_W * STRIDE_W;
	const float left_pad = (current_kw == LOAD_3x3_BLOCK_START) ? 0.0f : 1.0f;
	const float right_pad = ((current_kw + LOAD_3x3_BLOCK_END) > INPUT_WIDTH) ? 0.0f : 1.0f;
#if(INPUT_LAYOUT == NHWC)
	load_offsets += input_offset - INPUT_CHANNELS * sizeof(DT_IN);
    #pragma unroll
    for(int i = 0; i < LOAD_W + 2; i++)
    {
		vector<half, LOAD_W_WIDTH> load_chunk = cm_load<half, VectorSize::N1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, load_offsets);
		if( i == LOAD_3x3_BLOCK_START )
		{
			load_chunk *= left_pad;
		}
		if( i == LOAD_3x3_BLOCK_END )
		{
			load_chunk *= right_pad;
		}
        data_out.select<DPAS_INPUT_CHANNELS, 1>(i * DPAS_INPUT_CHANNELS) = load_chunk.select<DPAS_INPUT_CHANNELS, 1>();
        load_offsets += INPUT_NHWC_PLANE_SIZE;
    }
#else
	load_offsets += input_offset - INPUT_PAD * sizeof(DT_IN);
	#pragma unroll
    for(int i = 0; i < DPAS_INPUT_CHANNELS; i++)
    {
        vector<half, LOAD_W_WIDTH> load_chunk = cm_load<half, VectorSize::N1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, load_offsets);
		load_chunk[LOAD_3x3_BLOCK_START] *= left_pad;
		load_chunk[LOAD_3x3_BLOCK_END] *= right_pad;
        data_out.select<LOAD_3x3_BLOCK_SIZE, DPAS_INPUT_CHANNELS>(i) = load_chunk.select<LOAD_3x3_BLOCK_SIZE, STRIDE_W>();
        load_offsets += INPUT_NCHW_PLANE_SIZE;
    }
#endif
	return data_out;
}

template<uint32_t LOAD_W>
_GENX_ inline vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> load_1x1_input(SurfaceIndex surface [[type("buffer_t")]], uint byte_offset)
{
#if(INPUT_LAYOUT == NHWC)
	const uint32_t LOAD_W_WIDTH = DPAS_INPUT_CHANNELS;
#else
    const uint32_t LOAD_W_WIDTH = LOAD_W * STRIDE_W;
#endif
    const uint32_t LOAD_W_BYTES_WIDTH = LOAD_W_WIDTH * sizeof(DT_IN);
    const uint32_t LOAD_W_DWORDS = LOAD_W_BYTES_WIDTH / sizeof(uint32_t);
    vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> data_out;
	vector<uint32_t, LOAD_W_DWORDS> load_chunk;
#if(INPUT_LAYOUT == NHWC)
    #pragma unroll
    for(int i = 0; i < LOAD_W; i++)
    {
        load_chunk = cm_load<uint32_t, LOAD_W_DWORDS, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
        data_out.select<DPAS_INPUT_CHANNELS, 1>(i * DPAS_INPUT_CHANNELS) = load_chunk.format<half>().select<DPAS_INPUT_CHANNELS, 1>();
        byte_offset += INPUT_NHWC_PLANE_SIZE;
    }
#else
	#pragma unroll
    for(int i = 0; i < DPAS_INPUT_CHANNELS; i++)
    {
        load_chunk = cm_load<uint32_t, LOAD_W_DWORDS, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
        data_out.select<BLOCK_W, DPAS_INPUT_CHANNELS>(i) = load_chunk.format<half>().select<BLOCK_W, STRIDE_W>();
        byte_offset += INPUT_NCHW_PLANE_SIZE;
    }
#endif
    return data_out;
}

_GENX_ inline vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> load_filter_nchw_data(SurfaceIndex surface [[type("buffer_t")]], uint32_t byte_offset)
{
#if WEIGHTS_IN_OPTIMAL_FORMAT
    vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> data_out;
    //vector_ref<uint32_t, 64> data_load_view = data_out.select<128,1>(0).format<uint32_t>();
    data_out.select<128,1>(0).format<uint32_t>() = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
	#if EXEC_SIZE == 16
		data_out.select<128,1>(128).format<uint32_t>() = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset + 256);	
	#endif
    return data_out;
#else
	#error Kernel only supports reordered weight layouts.
#endif
}

_GENX_ inline vector<DT_OUT, BLOCK_OC * OUTPUT_CHANNEL_MULTIPLIER> load_bias(SurfaceIndex surface [[type("buffer_t")]], uint32_t byte_offset)
{
	vector<DT_OUT, BLOCK_OC * OUTPUT_CHANNEL_MULTIPLIER> data_out;
#if BLOCK_OC == 40
	data_out.select<32 * OUTPUT_CHANNEL_MULTIPLIER,1>(OUTPUT_CHANNEL_MULTIPLIER * 0 ).format<uint32_t>() = cm_load<uint32_t, 16 * OUTPUT_CHANNEL_MULTIPLIER, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
	data_out.select<8  * OUTPUT_CHANNEL_MULTIPLIER,1>(OUTPUT_CHANNEL_MULTIPLIER * 32).format<uint32_t>() = cm_load<uint32_t,  4 * OUTPUT_CHANNEL_MULTIPLIER, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset + 64 * OUTPUT_CHANNEL_MULTIPLIER);
#endif
#if BLOCK_OC == 80
	data_out.select<64 * OUTPUT_CHANNEL_MULTIPLIER,1>(0  * OUTPUT_CHANNEL_MULTIPLIER).format<uint32_t>() = cm_load<uint32_t, 32 * OUTPUT_CHANNEL_MULTIPLIER, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
	data_out.select<16 * OUTPUT_CHANNEL_MULTIPLIER,1>(64 * OUTPUT_CHANNEL_MULTIPLIER).format<uint32_t>() = cm_load<uint32_t, 8  * OUTPUT_CHANNEL_MULTIPLIER, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset + 128 * OUTPUT_CHANNEL_MULTIPLIER);
#endif
#if BLOCK_OC != 40 && BLOCK_OC != 80
	data_out.format<uint32_t>() = cm_load<uint32_t, BLOCK_OC/2 * OUTPUT_CHANNEL_MULTIPLIER, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
#endif
	return data_out;
}

template<uint32_t STORE_W>
_GENX_ inline void store_output(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, BLOCK_W * DPAS_OUTPUT_CHANNELS> grf_chunk, uint32_t byte_offset)
{
	uint32_t offsets = byte_offset;
#if(INPUT_LAYOUT == NHWC)
	#pragma unroll
	for(int i = 0; i < STORE_W; i++)
    {
        vector<DT_OUT, DPAS_OUTPUT_CHANNELS> grf_chunk_store = grf_chunk.select<DPAS_OUTPUT_CHANNELS, 1>(i * DPAS_OUTPUT_CHANNELS);                  
        cm_store<U32, 8, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store.format<U32>());
        offsets += OUTPUT_NHWC_PLANE_SIZE;
    }
#else
    #pragma unroll
    for(int i = 0; i < DPAS_OUTPUT_CHANNELS; i++)
    {
        vector<DT_OUT, STORE_W> grf_chunk_store = grf_chunk.select<STORE_W, DPAS_OUTPUT_CHANNELS>(i);                  
        cm_store<U32, 4, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store.format<U32>());
        offsets += OUTPUT_NCHW_PLANE_SIZE;
    }
#endif
}

extern "C" _GENX_MAIN_ void conv_nchw_dpas_fp16(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_weights [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
#if USE_BIAS
	,SurfaceIndex surface_bias [[type("buffer_t")]]
#endif
)
{
    const uint32_t thg_0 = (cm_group_id(0) * cm_local_size(0) + cm_local_id(0));
    const uint w_chunk_id = thg_0 / SLICE_IC;
    const uint slice_ic_id = thg_0 % SLICE_IC;
    const uint h_chunk_id = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint thread_id_2 = (cm_group_id(2) * cm_local_size(2) + cm_local_id(2));
    
    const uint THREADS_FOR_OC = (OUTPUT_CHANNELS / BLOCK_OC) / OUTPUT_CHANNEL_MULTIPLIER;
    const uint batch_id = (thread_id_2 / THREADS_FOR_OC);
    const uint oc_chunk_id = (thread_id_2 % THREADS_FOR_OC) * (BLOCK_OC / DPAS_DEPTH);
    
#if(INPUT_LAYOUT == NHWC)
	const uint32_t input_row_offset_size = BLOCK_H * STRIDE_H * INPUT_WIDTH * INPUT_CHANNELS;
	const uint32_t input_dpas_ic_offset_size = DPAS_INPUT_CHANNELS;
	const uint32_t input_batch_offset = batch_id * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
	const uint32_t input_w_chunk_offset = w_chunk_id * BLOCK_W * STRIDE_W * INPUT_CHANNELS;
	const uint32_t input_h_chunk_offset = h_chunk_id * BLOCK_H * STRIDE_H * INPUT_WIDTH * INPUT_CHANNELS;
#else
	const uint32_t input_row_offset_size = BLOCK_H * STRIDE_H * INPUT_WIDTH;
	const uint32_t input_dpas_ic_offset_size = INPUT_HEIGHT * DPAS_INPUT_CHANNELS * INPUT_WIDTH;	
	const uint32_t input_batch_offset = batch_id * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
	const uint32_t input_w_chunk_offset = w_chunk_id * BLOCK_W * STRIDE_W;
	const uint32_t input_h_chunk_offset = h_chunk_id * input_row_offset_size;
#endif
	const uint32_t input_slice_ic_chunk_offset = slice_ic_id * CONV_LOOP_COUNT * input_dpas_ic_offset_size;
	uint32_t input_offset = (input_batch_offset + input_slice_ic_chunk_offset + input_h_chunk_offset + input_w_chunk_offset) * sizeof(DT_IN);

#if WEIGHTS_IN_OPTIMAL_FORMAT
	#if KERNEL_SIZE == 1
		const uint32_t weights_oc_chunk_offset = EXEC_SIZE * DPAS_INPUT_CHANNELS * sizeof(DT_WEIGHTS);
		const uint32_t weights_ic_offset_size = OUTPUT_CHANNELS * DPAS_INPUT_CHANNELS * sizeof(DT_WEIGHTS);
	#elif KERNEL_SIZE == 3
		const uint32_t weights_oc_chunk_offset = DPAS_OUTPUT_CHANNELS * INPUT_CHANNELS * sizeof(DT_WEIGHTS) * KERNEL_SIZE * KERNEL_SIZE;
		const uint32_t weights_ic_offset_size =  DPAS_INPUT_CHANNELS * EXEC_SIZE * sizeof(DT_WEIGHTS) * KERNEL_SIZE * KERNEL_SIZE;
	#else
		#error unsupported Kernel Size
	#endif
#else
    #error Kernel only supports reordered weight layouts.
#endif

    uint32_t weights_offset_0 = WEI_OFFSET + oc_chunk_id * weights_oc_chunk_offset + (slice_ic_id * CONV_LOOP_COUNT * weights_ic_offset_size);
    uint32_t weights_offset_1 = weights_offset_0 + weights_oc_chunk_offset;
    uint32_t weights_offset_2 = weights_offset_1 + weights_oc_chunk_offset;
    uint32_t weights_offset_3 = weights_offset_2 + weights_oc_chunk_offset;
	uint32_t weights_offset_4 = weights_offset_3 + weights_oc_chunk_offset;
    uint32_t weights_offset_5 = weights_offset_4 + weights_oc_chunk_offset;
	uint32_t weights_offset_6 = weights_offset_5 + weights_oc_chunk_offset;
    uint32_t weights_offset_7 = weights_offset_6 + weights_oc_chunk_offset;
    uint32_t weights_offset_8 = weights_offset_7 + weights_oc_chunk_offset;
    uint32_t weights_offset_9 = weights_offset_8 + weights_oc_chunk_offset;
	
    const uint ACCU_REG_SIZE = BLOCK_W * DPAS_OUTPUT_CHANNELS;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_0 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_1 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_2 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_3 = 0;
	vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_4 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_5 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_6 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_7 = 0;
	vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_8 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_9 = 0;
	
    // todo debug performance with pragma unroll
    //#pragma unroll
    for(int i = 0; i < CONV_LOOP_COUNT; i++)
    {
		#if KERNEL_SIZE == 1
			vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> input_row_0 = load_1x1_input<BLOCK_W>(surface_input, input_offset);
			
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_0 = load_filter_nchw_data(surface_weights, weights_offset_0);
			#if BLOCK_OC >= 16
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_1 = load_filter_nchw_data(surface_weights, weights_offset_1);
			#endif  
			#if BLOCK_OC >= 32
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_2 = load_filter_nchw_data(surface_weights, weights_offset_2);
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_3 = load_filter_nchw_data(surface_weights, weights_offset_3);
			#endif
			#if BLOCK_OC >= 40
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_4 = load_filter_nchw_data(surface_weights, weights_offset_4);
			#endif
			#if BLOCK_OC >= 64
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_5 = load_filter_nchw_data(surface_weights, weights_offset_5);
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_6 = load_filter_nchw_data(surface_weights, weights_offset_6);
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_7 = load_filter_nchw_data(surface_weights, weights_offset_7);
			#endif
			#if BLOCK_OC == 80
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_8 = load_filter_nchw_data(surface_weights, weights_offset_8);
			vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_9 = load_filter_nchw_data(surface_weights, weights_offset_9);
			#endif
	
			accu_row_0_oc_0 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_0, weights_0.format<uint32_t>(), input_row_0.format<uint32_t>());
			#if BLOCK_OC >= 16
			accu_row_0_oc_1 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_1, weights_1.format<uint32_t>(), input_row_0.format<uint32_t>());
			#endif
			#if BLOCK_OC >= 32
			accu_row_0_oc_2 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_2, weights_2.format<uint32_t>(), input_row_0.format<uint32_t>());
			accu_row_0_oc_3 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_3, weights_3.format<uint32_t>(), input_row_0.format<uint32_t>());
			#endif
			#if BLOCK_OC >= 40
			accu_row_0_oc_4 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_4, weights_4.format<uint32_t>(), input_row_0.format<uint32_t>());
			#endif
			#if BLOCK_OC >= 64
			accu_row_0_oc_5 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_5, weights_5.format<uint32_t>(), input_row_0.format<uint32_t>());
			accu_row_0_oc_6 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_6, weights_6.format<uint32_t>(), input_row_0.format<uint32_t>());
			accu_row_0_oc_7 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_7, weights_7.format<uint32_t>(), input_row_0.format<uint32_t>());
			#endif
			#if BLOCK_OC == 80
			accu_row_0_oc_8 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_8, weights_8.format<uint32_t>(), input_row_0.format<uint32_t>());
			accu_row_0_oc_9 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_9, weights_9.format<uint32_t>(), input_row_0.format<uint32_t>());
			#endif
		#elif KERNEL_SIZE == 3
			#pragma unroll
			for(int kh = -INPUT_PAD; kh < KERNEL_SIZE-INPUT_PAD; kh++)
			{
				int input_load_offset_kh = input_offset + (kh * input_row_offset_size * sizeof(DT_IN));
				if(h_chunk_id + kh < 0 || h_chunk_id + kh  >= INPUT_HEIGHT) { continue; };
				vector<DT_IN, (LOAD_3x3_BLOCK_SIZE) * DPAS_INPUT_CHANNELS> input_row_0 = load_3x3_input<LOAD_3x3_BLOCK_SIZE>(surface_input, input_load_offset_kh, w_chunk_id);

				#pragma unroll
				for(int kw = 0; kw < KERNEL_SIZE; kw++)
				{
					uint32_t kernel_index = ((kh + INPUT_PAD) * KERNEL_SIZE + kw) * sizeof(DT_WEIGHTS);
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_0 = load_filter_nchw_data(surface_weights, weights_offset_0 + (kernel_index * WEIGHTS_REG_SIZE));
					
					#if BLOCK_OC >= 16
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_1 = load_filter_nchw_data(surface_weights, weights_offset_1 + (kernel_index * WEIGHTS_REG_SIZE));
					#endif  
					#if BLOCK_OC >= 32
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_2 = load_filter_nchw_data(surface_weights, weights_offset_2 + kernel_index * WEIGHTS_REG_SIZE);
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_3 = load_filter_nchw_data(surface_weights, weights_offset_3 + kernel_index * WEIGHTS_REG_SIZE);
					#endif
					#if BLOCK_OC >= 40
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_4 = load_filter_nchw_data(surface_weights, weights_offset_4 + kernel_index * WEIGHTS_REG_SIZE);
					#endif
					#if BLOCK_OC >= 64
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_5 = load_filter_nchw_data(surface_weights, weights_offset_5 + kernel_index * WEIGHTS_REG_SIZE);
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_6 = load_filter_nchw_data(surface_weights, weights_offset_6 + kernel_index * WEIGHTS_REG_SIZE);
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_7 = load_filter_nchw_data(surface_weights, weights_offset_7 + kernel_index * WEIGHTS_REG_SIZE);
					#endif
					#if BLOCK_OC == 80
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_8 = load_filter_nchw_data(surface_weights, weights_offset_8 + kernel_index * WEIGHTS_REG_SIZE);
					vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_9 = load_filter_nchw_data(surface_weights, weights_offset_9 + kernel_index * WEIGHTS_REG_SIZE);
					#endif

					accu_row_0_oc_0 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_0, weights_0.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					#if BLOCK_OC >= 16
					accu_row_0_oc_1 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_1, weights_1.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					#endif  
					#if BLOCK_OC >= 32
					accu_row_0_oc_2 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_2, weights_2.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					accu_row_0_oc_3 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_3, weights_3.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					#endif
					#if BLOCK_OC >= 40
					accu_row_0_oc_4 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_4, weights_4.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					#endif
					#if BLOCK_OC >= 64
					accu_row_0_oc_5 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_5, weights_5.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					accu_row_0_oc_6 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_6, weights_6.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					accu_row_0_oc_7 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_7, weights_7.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					#endif
					#if BLOCK_OC == 80
					accu_row_0_oc_8 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_8, weights_8.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					accu_row_0_oc_9 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_9, weights_9.format<uint32_t>(), input_row_0.select<128,1>(kw * DPAS_INPUT_CHANNELS).format<uint32_t>());
					#endif
				}
			}
		#else
			#error unsupported Kernel Size
		#endif
        input_offset += (input_dpas_ic_offset_size * sizeof(DT_IN));
        weights_offset_0 += weights_ic_offset_size;
        weights_offset_1 += weights_ic_offset_size;
        weights_offset_2 += weights_ic_offset_size;
        weights_offset_3 += weights_ic_offset_size;
        weights_offset_4 += weights_ic_offset_size;
        weights_offset_5 += weights_ic_offset_size;
        weights_offset_6 += weights_ic_offset_size;
        weights_offset_7 += weights_ic_offset_size;
		weights_offset_8 += weights_ic_offset_size;
		weights_offset_9 += weights_ic_offset_size;
    }

    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_0 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_0);
#if BLOCK_OC >= 16
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_1 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_1);
#endif

#if BLOCK_OC >= 32
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_2 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_2);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_3 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_3);
#endif
#if BLOCK_OC >= 40
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_4 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_4);
#endif
#if BLOCK_OC >= 64
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_5 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_5);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_6 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_6);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_7 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_7);
#endif
#if BLOCK_OC == 80
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_8 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_8);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_9 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_9);
#endif

#if USE_BIAS
	vector<DT_OUT, BLOCK_OC * OUTPUT_CHANNEL_MULTIPLIER> bias = load_bias(surface_bias, oc_chunk_id * EXEC_SIZE * sizeof(DT_OUT));
	#pragma unroll
	for(int bw = 0; bw < BLOCK_W; bw++)
	{
		output_row_0_oc_0.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(0 * OUTPUT_CHANNEL_MULTIPLIER);
#if BLOCK_OC >= 16
		output_row_0_oc_1.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(8 * OUTPUT_CHANNEL_MULTIPLIER);
#endif
#if BLOCK_OC >= 32
		output_row_0_oc_2.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(16 * OUTPUT_CHANNEL_MULTIPLIER);
		output_row_0_oc_3.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(24 * OUTPUT_CHANNEL_MULTIPLIER);
#endif
#if BLOCK_OC >= 40
		output_row_0_oc_4.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(32 * OUTPUT_CHANNEL_MULTIPLIER);
#endif
#if BLOCK_OC >= 64
		output_row_0_oc_5.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(40 * OUTPUT_CHANNEL_MULTIPLIER);
		output_row_0_oc_6.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(48 * OUTPUT_CHANNEL_MULTIPLIER);
		output_row_0_oc_7.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(56 * OUTPUT_CHANNEL_MULTIPLIER);
#endif
#if BLOCK_OC >= 80
		output_row_0_oc_8.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(64 * OUTPUT_CHANNEL_MULTIPLIER);
		output_row_0_oc_9.select<EXEC_SIZE, 1>(bw * EXEC_SIZE) += bias.select<EXEC_SIZE, 1>(72 * OUTPUT_CHANNEL_MULTIPLIER);
#endif
	}
#endif 

#if USE_RELU
    output_row_0_oc_0 = cm_max<DT_OUT>(output_row_0_oc_0, 0);
	#if BLOCK_OC >= 16
		output_row_0_oc_1 = cm_max<DT_OUT>(output_row_0_oc_1, 0);
	#endif
	#if BLOCK_OC >= 32
		output_row_0_oc_2 = cm_max<DT_OUT>(output_row_0_oc_2, 0);
		output_row_0_oc_3 = cm_max<DT_OUT>(output_row_0_oc_3, 0);
	#endif
	#if BLOCK_OC >= 40
		output_row_0_oc_4 = cm_max<DT_OUT>(output_row_0_oc_4, 0);
	#endif
	#if BLOCK_OC >= 64
		output_row_0_oc_5 = cm_max<DT_OUT>(output_row_0_oc_5, 0);
		output_row_0_oc_6 = cm_max<DT_OUT>(output_row_0_oc_6, 0);
		output_row_0_oc_7 = cm_max<DT_OUT>(output_row_0_oc_7, 0);
	#endif
	#if BLOCK_OC == 80
		output_row_0_oc_8 = cm_max<DT_OUT>(output_row_0_oc_8, 0);
		output_row_0_oc_9 = cm_max<DT_OUT>(output_row_0_oc_9, 0);
	#endif
#endif

#if(INPUT_LAYOUT == NHWC)
    const uint output_oc_chunk_offset = oc_chunk_id * DPAS_OUTPUT_CHANNELS;
    const uint output_w_chunk_offset = w_chunk_id * BLOCK_W * OUTPUT_CHANNELS;
    const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * OUTPUT_WIDTH * OUTPUT_CHANNELS;
#else
    const uint output_oc_chunk_offset = oc_chunk_id * DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH;
    const uint output_w_chunk_offset = w_chunk_id * BLOCK_W;
    const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * OUTPUT_WIDTH;
#endif
	const uint output_batch_offset = batch_id * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS;
	uint32_t output_offset = (output_batch_offset + output_oc_chunk_offset + output_h_chunk_offset + output_w_chunk_offset) * sizeof(DT_OUT);
	
    store_output<BLOCK_W>(surface_output, output_row_0_oc_0, output_offset);
    
#if BLOCK_OC >= 16
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_1, output_offset); 
#endif

#if BLOCK_OC >= 32
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_2, output_offset); 
    
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_3, output_offset); 
#endif

#if BLOCK_OC >= 40
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_4, output_offset); 
#endif

#if BLOCK_OC >= 64
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_5, output_offset);
	
	output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_6, output_offset); 
    
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_7, output_offset);
#endif

#if BLOCK_OC == 80
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_8, output_offset);
    
    output_offset += OUTPUT_DPAS_OFFSET;
    store_output<BLOCK_W>(surface_output, output_row_0_oc_9, output_offset); 
#endif
}