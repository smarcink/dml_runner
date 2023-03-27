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
#error [Error_device_not_supported] Kernel is not designed for this architecutre.
#endif

#if BLOCK_W > 8
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_w in range: <1; 7>;
#endif

#if BLOCK_OC != 8 && BLOCK_OC != 16 && BLOCK_OC != 32
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_oc which is equal to 8 or 16 or 32;
#endif

#if INPUT_CHANNELS >= 16
#error [Error_input_channel_count] This kernel was designed to work with small input channels, which are not fitting dpas scenarios.
#endif

#define BLOCK_H 1
#define WIDTH_LEFTOVER (OUTPUT_WIDTH % BLOCK_W)
#define HAS_LEFTOVER (WIDTH_LEFTOVER != 0)
#define LEFTOVER_COVERS_FULL_WIDTH (OUTPUT_WIDTH == WIDTH_LEFTOVER)

#define DT_OUT half
#define DT_IN half
#define DT_IN_SIZE 2 
#define DT_WEIGHTS half

#if 1
#define DT_ACCU DT_IN 
#else
#define DT_ACCU float 
#define NEED_OUTPUT_CAST 1
#endif

#define OUTPUT_CHANNELS_PER_REG 8


#define DWORD_SIZE 4
#define INPUT_WIDTH_ALIGNED_TO_DWORD ((INPUT_WIDTH * DT_IN_SIZE) % DWORD_SIZE == 0)

#define INPUT_NCHW_PLANE_SIZE (INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN))
#define OUTPUT_NCHW_PLANE_SIZE (OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT))
#define OUTPUT_ELEMENT_SIZE (sizeof(DT_OUT))

#define INPUT_REG_W (BLOCK_W + KERNEL_SIZE - 1)
#define INPUT_REG_SIZE (INPUT_REG_W* INPUT_CHANNELS)
#define INPUT_ELEMENT_SIZE (sizeof(DT_IN))

// (kernel width) * input channels * block oc
#define WEIGHTS_REG_SIZE (KERNEL_SIZE * INPUT_CHANNELS * BLOCK_OC)
#define WEIGHT_ELEMENT_SIZE (sizeof(DT_WEIGHTS))
#define WEIGHTS_OC_OFFSET (KERNEL_SIZE * KERNEL_SIZE * INPUT_CHANNELS * WEIGHT_ELEMENT_SIZE)


#define ACCU_REG_SIZE (BLOCK_OC * BLOCK_W)

#define LINEAR_LOAD_SIZE 16
#define INPUT_ELEMENT_SIZE_I32 int32_t(INPUT_ELEMENT_SIZE)
static const int32_t init_linear_offsets[] = {  (0 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (1 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32, 
											    (2 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (3 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (4 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (5 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (6 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (7 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
												(8 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32, 
											    (9 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (10 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (11 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (12 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (13 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (14 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32,
											    (15 - INPUT_PAD) * INPUT_ELEMENT_SIZE_I32
											  };

static const uint32_t output_linear_init_offsets[] = {
                                                0 * OUTPUT_ELEMENT_SIZE,
                                                1 * OUTPUT_ELEMENT_SIZE,
                                                2 * OUTPUT_ELEMENT_SIZE,
                                                3 * OUTPUT_ELEMENT_SIZE,
                                                4 * OUTPUT_ELEMENT_SIZE,
                                                5 * OUTPUT_ELEMENT_SIZE,
                                                6 * OUTPUT_ELEMENT_SIZE,
                                                7 * OUTPUT_ELEMENT_SIZE, 
												8 * OUTPUT_ELEMENT_SIZE, 
											    9 * OUTPUT_ELEMENT_SIZE,
											    10 * OUTPUT_ELEMENT_SIZE,
											    11 * OUTPUT_ELEMENT_SIZE,
											    12 * OUTPUT_ELEMENT_SIZE,
											    13 * OUTPUT_ELEMENT_SIZE,
											    14 * OUTPUT_ELEMENT_SIZE,
											    15 * OUTPUT_ELEMENT_SIZE
                                            };

static const uint32_t output_scattered_init_offsets[] = {
                                                0 * OUTPUT_NCHW_PLANE_SIZE,
                                                1 * OUTPUT_NCHW_PLANE_SIZE,
                                                2 * OUTPUT_NCHW_PLANE_SIZE,
                                                3 * OUTPUT_NCHW_PLANE_SIZE,
                                                4 * OUTPUT_NCHW_PLANE_SIZE,
                                                5 * OUTPUT_NCHW_PLANE_SIZE,
                                                6 * OUTPUT_NCHW_PLANE_SIZE,
                                                7 * OUTPUT_NCHW_PLANE_SIZE, 
												8 * OUTPUT_NCHW_PLANE_SIZE, 
											    9 * OUTPUT_NCHW_PLANE_SIZE,
											    10 * OUTPUT_NCHW_PLANE_SIZE,
											    11 * OUTPUT_NCHW_PLANE_SIZE,
											    12 * OUTPUT_NCHW_PLANE_SIZE,
											    13 * OUTPUT_NCHW_PLANE_SIZE,
											    14 * OUTPUT_NCHW_PLANE_SIZE,
											    15 * OUTPUT_NCHW_PLANE_SIZE
                                            };

static const uint32_t weights_init_offsets[] = {
                                                0 * WEIGHTS_OC_OFFSET,
                                                1 * WEIGHTS_OC_OFFSET,
                                                2 * WEIGHTS_OC_OFFSET,
                                                3 * WEIGHTS_OC_OFFSET,
                                                4 * WEIGHTS_OC_OFFSET,
                                                5 * WEIGHTS_OC_OFFSET,
                                                6 * WEIGHTS_OC_OFFSET,
                                                7 * WEIGHTS_OC_OFFSET, 
												8 * WEIGHTS_OC_OFFSET, 
											    9 * WEIGHTS_OC_OFFSET,
											    10 * WEIGHTS_OC_OFFSET,
											    11 * WEIGHTS_OC_OFFSET,
											    12 * WEIGHTS_OC_OFFSET,
											    13 * WEIGHTS_OC_OFFSET,
											    14 * WEIGHTS_OC_OFFSET,
											    15 * WEIGHTS_OC_OFFSET
                                                };

static const uint32_t predicate_init_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

_GENX_ inline vector<DT_IN, INPUT_REG_SIZE> load_input_nchw(SurfaceIndex surface [[type("buffer_t")]], uint32_t w_offset, int32_t h_offset, uint32_t batch_base_offset_bytes)
{
	const int32_t h_offset_pad = h_offset - int32_t(INPUT_PAD);
	vector<DT_IN, INPUT_REG_SIZE> ret(0.0f);
	vector<uint8_t, LINEAR_LOAD_SIZE> predicate(0);
	vector<int32_t, LINEAR_LOAD_SIZE> offsets(init_linear_offsets);
	offsets += w_offset * INPUT_ELEMENT_SIZE; 	// offset by X
	
	// update predicate mask for left and right padding
	predicate.merge(1, offsets >= 0 & offsets < (INPUT_WIDTH * INPUT_ELEMENT_SIZE));
	// update predicate mask for top and bottom padding
	vector<int32_t, LINEAR_LOAD_SIZE> is_not_in_height_range(h_offset_pad < 0 |  h_offset_pad > (INPUT_HEIGHT - 1));
	predicate.merge(0, is_not_in_height_range);
	
	offsets += h_offset_pad * INPUT_WIDTH * INPUT_ELEMENT_SIZE;
	vector<uint32_t, LINEAR_LOAD_SIZE> offsets_u32 = offsets;
	
#if 0
	 //if(w_offset == 0 && h_offset == 0)
	 {
		printf("%d, %d \n", w_offset, h_offset);
		 for(int i = 0; i < LINEAR_LOAD_SIZE; i++)
		 {
			printf("%d, ", predicate[i]); 
		 }
		printf("\n"); 
				 for(int i = 0; i < LINEAR_LOAD_SIZE; i++)
		 {
			printf("%d, ", offsets[i]); 
		 }
		printf("\n"); 
	 }
#endif	


	#pragma unroll
	for(int i = 0; i < INPUT_CHANNELS; i++)
	{
		vector<DT_IN, LINEAR_LOAD_SIZE> load_chunk = cm_load<half, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32);	
		ret.select<INPUT_REG_W, 1>(i * INPUT_REG_W).merge(load_chunk.select<INPUT_REG_W, 1>(), predicate.select<INPUT_REG_W, 1>());
		//ret.select<INPUT_REG_W, 1>(i * INPUT_REG_W) = offsets_u32.select<INPUT_REG_W, 1>();
		offsets_u32 += (INPUT_WIDTH * INPUT_HEIGHT * INPUT_ELEMENT_SIZE);
	}
	
#if 0
	 //if(w_offset == 0 && h_offset == 0)
	 {
		 for(int i = 0; i < INPUT_CHANNELS; i++)
		 {
			 for(int w = 0; w < INPUT_REG_W; w++)
			 {
				 printf("%f, ", (float)ret[w + i * INPUT_REG_W]);
			 }
			printf("\n"); 
		 }
		printf("\n"); 
	 }
#endif
	
	return ret;
}


_GENX_ inline vector<DT_IN, BLOCK_OC * INPUT_CHANNELS> load_weights(SurfaceIndex surface [[type("buffer_t")]], uint32_t kw, uint32_t kh, uint32_t oc_chunk)
{
	/*
		This function requires weights to be in optimal format:  OYXI_o8  or OYXI_o16  (depending on the oc_block)  (oc_block == simd size of the "mad" instructions)
	*/
	vector<DT_IN, BLOCK_OC * INPUT_CHANNELS> ret;
	uint32_t offset = (oc_chunk * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw) * INPUT_CHANNELS * BLOCK_OC * WEIGHT_ELEMENT_SIZE;

	#pragma unroll
	for(int i = 0; i < INPUT_CHANNELS; i++)
	{
		vector_ref<uint32_t, BLOCK_OC/2> typed_view = ret.select<BLOCK_OC, 1>(i * BLOCK_OC).format<uint32_t>();
		typed_view = cm_load<uint32_t, BLOCK_OC/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, offset);	
		offset += BLOCK_OC * WEIGHT_ELEMENT_SIZE;
	}

	return ret;
}


_GENX_ inline void store_output_wc8_as_nchw(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, ACCU_REG_SIZE> grf_chunk, uint32_t byte_offset, uint32_t w_chunk_id)
{    

	if constexpr(BLOCK_W == 8 || BLOCK_W == 16)
	{
		vector<uint32_t, BLOCK_W> offsets(output_linear_init_offsets);
		offsets += byte_offset;
		#pragma unroll
		for(int i = 0; i < BLOCK_OC; i++)
		{
			// pick data to store
			vector<DT_OUT, BLOCK_W> grf_chunk_store = grf_chunk.select<BLOCK_W, BLOCK_OC>(i);                  
			cm_store<half, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store/*, predicate*/);
			offsets += OUTPUT_NCHW_PLANE_SIZE;
		}		
	}
	else
	{
		vector<uint32_t, BLOCK_OC> offsets(output_scattered_init_offsets);
		offsets += byte_offset;
		#pragma unroll
		for(int i = 0; i < BLOCK_W; i++)
		{
			// pick data to store
			vector<DT_OUT, BLOCK_OC> grf_chunk_store = grf_chunk.select<BLOCK_OC, 1>(i * BLOCK_OC);                  
			cm_store<half, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store/*, predicate*/);
			offsets += OUTPUT_ELEMENT_SIZE;
		}
	}
}

extern "C" _GENX_MAIN_ void convolution_nchw_nondpas(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_weights [[type("buffer_t")]],
#if USE_BIAS
	SurfaceIndex surface_bias [[type("buffer_t")]],
#endif
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint32_t thg_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thg_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thg_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);   

	const uint32_t w_chunk_id = thg_0;
	const uint32_t h_chunk_id = thg_1;
	const uint32_t batch_id = 0;
	const uint32_t oc_chunk_id = thg_2;

    const uint32_t input_batch_offset = batch_id * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(DT_IN);
    const uint32_t input_w_chunk_offset = w_chunk_id * BLOCK_W;
    const uint32_t input_h_chunk_offset = h_chunk_id * BLOCK_H;
	
	vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0(0);
	
	#pragma unroll
	for(int kh = 0; kh < KERNEL_SIZE; kh++)
	{
		vector<DT_IN, INPUT_REG_SIZE> input_0 = load_input_nchw(surface_input, input_w_chunk_offset, input_h_chunk_offset + kh, input_batch_offset);
		#pragma unroll
		for(int kw = 0; kw < KERNEL_SIZE; kw++)
		{
			vector<DT_WEIGHTS, BLOCK_OC * INPUT_CHANNELS> weights_chunk_oc_ic = load_weights(surface_weights, kw, kh, oc_chunk_id);
			#pragma unroll
			for(int i = 0; i < INPUT_CHANNELS; i++)
			{
				vector_ref<DT_IN, BLOCK_W> input_chunk = input_0.select<BLOCK_W, 1>(kw + i * INPUT_REG_W);
				vector_ref<DT_IN, BLOCK_OC> weights_chunk_ic = weights_chunk_oc_ic.select<BLOCK_OC, 1>(i * BLOCK_OC);
				#pragma unroll
				for(int bw = 0; bw < BLOCK_W; bw++)
				{
					// as long as accumulator, input and weights are the same data type this will compile into single mad instruction
					accu_row_0.select<BLOCK_OC, 1>(bw * BLOCK_OC) += input_chunk.select<1, 1>(bw).replicate<BLOCK_OC>() * weights_chunk_ic;
				}
			}		
		}		
	}

	// if the DT_OUT == DT_ACCU then compiler will not do anything here
	// but if data types are different then this cast accumulator to output type
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0);
	
	const uint output_batch_offset = batch_id * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS;
    const uint output_oc_chunk_offset = oc_chunk_id * BLOCK_OC * OUTPUT_HEIGHT * OUTPUT_WIDTH;
    const uint output_w_chunk_offset = w_chunk_id * BLOCK_W;
    const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * OUTPUT_WIDTH;
    uint32_t output_offset = (output_batch_offset + output_oc_chunk_offset + output_h_chunk_offset + output_w_chunk_offset) * sizeof(DT_OUT);
	
	store_output_wc8_as_nchw(surface_output, output_row_0, output_offset, w_chunk_id);
}
