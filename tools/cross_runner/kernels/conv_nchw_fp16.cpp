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

#define UNIT_VAL_ONE 1.0f
#define UNIT_VAL_ZERO 0.0f

#define FLOAT32 0
#define FLOAT16 1

#define BLOCK_H 1
#define BLOCK_BATCH 1
#define BLOCK_OC 16
#define BLOCK_W 16

#if !CM_HAS_LSC
#error [Error_device_no_lsc] Kernel designed to use lsc. Current device does not support lsc.
#endif

#if BLOCK_W != 16
#error [Error_kernel_config_unsupported_block_w] Kernel designed to work with block_w=16;
#endif

#if BLOCK_OC != 16
#error [Error_kernel_config_unsupported_block_w] Kernel designed to work with block_oc=16;
#endif

#define DT_ACCU float

#if DT == FLOAT32
#define DT_OUT float
#define DT_IN float
#define DT_WEIGHTS float
#else
#define DT_OUT half
#define DT_IN half
#define DT_WEIGHTS half
#endif


#define OUTPUT_ELEMENT_SIZE (sizeof(DT_OUT))

#define INPUT_REG_W (BLOCK_W * STRIDE_W + KERNEL_SIZE - 1)
#define INPUT_REG_SIZE (INPUT_REG_W)
#define INPUT_ELEMENT_SIZE (sizeof(DT_IN))
#define WEIGHT_ELEMENT_SIZE (sizeof(DT_WEIGHTS))
#define MAX_ELEMENT_SIZE (sizeof(float))

#define ACCU_REG_SIZE (BLOCK_OC * BLOCK_W)

#define INPUT_ELEMENT_SIZE_I32 int32_t(INPUT_ELEMENT_SIZE)

#define ACTIVATION_LOGISTIC 	  41
#define ACTIVATION_RELU		  	  32
#define ACTIVATION_HYPERBOLIC_TAN 33
#define ACTIVATION_LEAKY_RELU 	  39
#define MATH_E 2.718281828459045235360287471352f

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
											
_GENX_ inline DT_ACCU activation_function(uint32_t activation_type, DT_ACCU input, DT_ACCU m, DT_ACCU n)
{
	if(activation_type == ACTIVATION_LOGISTIC)
	{
		DT_ACCU e_pow_x = cm_pow<DT_ACCU>((DT_ACCU)MATH_E, input);
		return e_pow_x/(e_pow_x + UNIT_VAL_ONE);
	}
	else if(activation_type == ACTIVATION_RELU)
	{
		return (input >= UNIT_VAL_ZERO) ? input : UNIT_VAL_ZERO;
	}
	else if(activation_type == ACTIVATION_LEAKY_RELU)
	{
		return (input >= UNIT_VAL_ZERO) ? input : m * input;
	}
	else if(activation_type == ACTIVATION_HYPERBOLIC_TAN)
	{
		return cmtl::cm_tanh(input);
	}
	else
	{
		return input;
	}
}

_GENX_ inline vector<DT_ACCU, BLOCK_OC> load_bias(SurfaceIndex surface [[type("buffer_t")]], uint32_t oc_chunk, uint32_t output_channels)
{
	vector<uint32_t, BLOCK_OC> load_offsets(output_linear_init_offsets);
	load_offsets += oc_chunk * BLOCK_OC * OUTPUT_ELEMENT_SIZE;
		
	return cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, load_offsets);
}

_GENX_ inline vector<DT_ACCU, INPUT_REG_SIZE> load_input_nchw(SurfaceIndex surface [[type("buffer_t")]], uint32_t input_width, uint32_t input_height, uint32_t input_pad, uint32_t w_offset, int32_t h_offset, uint32_t batch_base_offset_bytes)
{
	const uint32_t LINEAR_LOAD_SIZE = INPUT_REG_W + (16 - (INPUT_REG_W % 16));
	const int32_t h_offset_pad = h_offset - int32_t(input_pad);
	vector<DT_ACCU, INPUT_REG_SIZE> ret(0.0f);
	vector<DT_ACCU, LINEAR_LOAD_SIZE> load_chunk_accu_dt(0.0f);
	vector<uint8_t, LINEAR_LOAD_SIZE> predicate(0);
	vector<int32_t, LINEAR_LOAD_SIZE> offsets;
	
	#pragma unroll
	for(int i = 0; i < LINEAR_LOAD_SIZE; i++)
	{
		offsets[i] = (i - int32_t(input_pad)) * INPUT_ELEMENT_SIZE_I32;
	}
	
	offsets += w_offset * INPUT_ELEMENT_SIZE; 	// offset by X
	
	// update predicate mask for left and right padding
	predicate.merge(1, offsets >= 0 & offsets < (input_width * INPUT_ELEMENT_SIZE));
	// update predicate mask for top and bottom padding
	vector<int32_t, LINEAR_LOAD_SIZE> is_not_in_height_range(h_offset_pad < 0 |  h_offset_pad > (input_height - 1));
	predicate.merge(0, is_not_in_height_range);
	
	offsets += h_offset_pad * input_width * INPUT_ELEMENT_SIZE;
	vector<uint32_t, LINEAR_LOAD_SIZE> offsets_u32 = offsets;
	offsets_u32 += batch_base_offset_bytes;

	load_chunk_accu_dt.select<16,1>(0)  = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(0));	
#if INPUT_REG_W > 16
	load_chunk_accu_dt.select<16,1>(16) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(16));
#endif
#if INPUT_REG_W > 32
	load_chunk_accu_dt.select<16,1>(32) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(32));
#endif
#if INPUT_REG_W > 48
	load_chunk_accu_dt.select<16,1>(48) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(48));
#endif
#if INPUT_REG_W > 64
	load_chunk_accu_dt.select<16,1>(64) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(64));
#endif
#if INPUT_REG_W > 80
	load_chunk_accu_dt.select<16,1>(80) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(80));
#endif
#if INPUT_REG_W > 96
	load_chunk_accu_dt.select<16,1>(96) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(96));
#endif
#if INPUT_REG_W > 112
	load_chunk_accu_dt.select<16,1>(112) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(112));
#endif
#if INPUT_REG_W > 128
	load_chunk_accu_dt.select<16,1>(128) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(128));
#endif
	ret.select<INPUT_REG_W, 1>(0).merge(load_chunk_accu_dt.select<INPUT_REG_W, 1>(), predicate.select<INPUT_REG_W, 1>());
	return ret;
}

_GENX_ inline vector<DT_ACCU, INPUT_REG_SIZE> load_input_nhwc(SurfaceIndex surface [[type("buffer_t")]], uint32_t input_width, uint32_t input_height, uint32_t input_pad, uint32_t w_offset, int32_t h_offset, uint32_t batch_base_offset_bytes, uint32_t input_channels)
{
	const uint32_t LINEAR_LOAD_SIZE = INPUT_REG_W + (16 - (INPUT_REG_W % 16));
	const int32_t h_offset_pad = h_offset - int32_t(input_pad);
	vector<DT_ACCU, INPUT_REG_SIZE> ret(0.0f);
	vector<DT_ACCU, LINEAR_LOAD_SIZE> load_chunk_accu_dt(0.0f);
	vector<uint8_t, LINEAR_LOAD_SIZE> predicate(0);
	vector<int32_t, LINEAR_LOAD_SIZE> offsets;

	#pragma unroll
	for(int i = 0; i < LINEAR_LOAD_SIZE; i++)
	{
		offsets[i] = (i - int32_t(input_pad)) * INPUT_ELEMENT_SIZE_I32;
	}

	// update predicate mask for left and right padding
	predicate.merge(1, offsets + (w_offset * INPUT_ELEMENT_SIZE) >= 0 & offsets + (w_offset * INPUT_ELEMENT_SIZE) < (input_width * INPUT_ELEMENT_SIZE));
	
	// update predicate mask for top and bottom padding
	vector<int32_t, LINEAR_LOAD_SIZE> is_not_in_height_range(h_offset_pad < 0 |  h_offset_pad > (input_height - 1));
	predicate.merge(0, is_not_in_height_range);
	
	offsets *= input_channels;
	offsets += w_offset * input_channels * INPUT_ELEMENT_SIZE;
	offsets += h_offset_pad * input_width * input_channels * INPUT_ELEMENT_SIZE;
	
	
	vector<uint32_t, LINEAR_LOAD_SIZE> offsets_u32 = offsets;
	offsets_u32 += batch_base_offset_bytes;
	
	load_chunk_accu_dt.select<16,1>(0)  = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(0));
#if INPUT_REG_W > 16
	load_chunk_accu_dt.select<16,1>(16) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(16));
#endif
#if INPUT_REG_W > 32
	load_chunk_accu_dt.select<16,1>(32) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(32));
#endif
#if INPUT_REG_W > 48
	load_chunk_accu_dt.select<16,1>(48) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(48));
#endif
#if INPUT_REG_W > 64
	load_chunk_accu_dt.select<16,1>(64) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(64));
#endif
#if INPUT_REG_W > 80
	load_chunk_accu_dt.select<16,1>(80) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(80));
#endif
#if INPUT_REG_W > 96
	load_chunk_accu_dt.select<16,1>(96) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(96));
#endif
#if INPUT_REG_W > 112
	load_chunk_accu_dt.select<16,1>(112) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(112));
#endif
#if INPUT_REG_W > 128
	load_chunk_accu_dt.select<16,1>(128) = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32.select<16,1>(128));
#endif
	ret.select<INPUT_REG_W, 1>(0).merge(load_chunk_accu_dt.select<INPUT_REG_W, 1>(), predicate.select<INPUT_REG_W, 1>());
	return ret;
}

_GENX_ inline vector<DT_ACCU, BLOCK_OC> load_weights(SurfaceIndex surface [[type("buffer_t")]], uint32_t kw, uint32_t kh, uint32_t oc_chunk, uint32_t input_ch_index, uint32_t input_channels)
{
	//This function requires weights to be in optimal format:  OYXI_o8  or OYXI_o16  (depending on the oc_block)  (oc_block == simd size of the "mad" instructions)
	
	vector<DT_ACCU, BLOCK_OC> ret;
	uint32_t offset = (oc_chunk * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw) * input_channels * BLOCK_OC * WEIGHT_ELEMENT_SIZE + input_ch_index * BLOCK_OC * WEIGHT_ELEMENT_SIZE;
	const uint32_t BLOCK_SC = MAX_ELEMENT_SIZE/WEIGHT_ELEMENT_SIZE;
	
	vector<uint32_t, BLOCK_OC/BLOCK_SC> typed_load = cm_load<uint32_t, BLOCK_OC/BLOCK_SC, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, offset);	
	vector_ref<DT_WEIGHTS, BLOCK_OC> load_data = typed_load.format<DT_WEIGHTS>();
	ret.select<BLOCK_OC, 1>(0) = vector<DT_ACCU, BLOCK_OC>(load_data);
	return ret;
}

_GENX_ inline void store_output_wc8_as_nchw(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, ACCU_REG_SIZE> grf_chunk, uint32_t output_width, uint32_t output_height, uint32_t byte_offset, uint32_t output_channels)
{   
	const uint32_t output_nchw_plane_size = (output_width * output_height * sizeof(DT_OUT));
	const uint32_t BLOCK_SC = MAX_ELEMENT_SIZE/OUTPUT_ELEMENT_SIZE;
	const uint32_t pixel_offset = byte_offset / OUTPUT_ELEMENT_SIZE;
	const bool b_oc_unalign_byte_offset = ((output_channels - (pixel_offset % output_channels)) < BLOCK_OC);
	const bool b_ow_unalign_byte_offset = ((BLOCK_W + (pixel_offset % output_width)) > output_width);
	
#if DT == FLOAT32
	if( output_width == 1 && b_oc_unalign_byte_offset == false)
	{
		// Corner Case 1: This IF Statement handles incoming 1-D write blocks with channel sizes aligned to BLOCK_OC
		// For Example: --input_shape=1,24,1,1 --filter_shape=6,24,1,1
		cm_store<uint32_t, BLOCK_OC>(surface, byte_offset, grf_chunk.select<BLOCK_OC, 1>(0).format<uint32_t>());
	}
	else if (output_width == 1 && b_oc_unalign_byte_offset == true)
	{
		// Corner Case 2: This IF Statement handles incoming write blocks with output channels not aligned to BLOCK_OC and are less than BLOCK_OC
		// For Example: --filter_shape=12,48,1,1 where output channel = 12 % BLOCK_OC (i.e., 8) = 4, which is unaligned and less than BLOCK_OC
		for(int i = 0; i < output_channels % BLOCK_OC; i++)
		{
			vector<DT_OUT, 1> grf_chunk_store = grf_chunk.select<1, 1>(i);                  
			cm_store<uint32_t, 1>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
			byte_offset += output_width * output_height * OUTPUT_ELEMENT_SIZE;
		}
	}
	else if (b_ow_unalign_byte_offset == true)
	{
		// Corner Case 3: This only works for BLOCK_W=2
		#pragma unroll
		for(int i = 0; i < BLOCK_OC; i++)
		{
			uint32_t byte_offset_local = byte_offset;
			for(int j = 0; j < output_width % BLOCK_W; j++)
			{			
				vector<DT_OUT, 1> grf_chunk_store = grf_chunk.select<1, 1>(i + j * BLOCK_OC);                  
				cm_store<uint32_t, 1>(surface, byte_offset_local, grf_chunk_store.format<uint32_t>());
				byte_offset_local += OUTPUT_ELEMENT_SIZE;
			}
			byte_offset += output_width * output_height * OUTPUT_ELEMENT_SIZE;
		}
	}
	else
	{
		// Default Case BLOCK_W x OB_BLOCK: Most of the incoming write blocks hit this case.
		#pragma unroll
		for(int i = 0; i < BLOCK_OC; i++)
		{
			vector<DT_OUT, BLOCK_W> grf_chunk_store = grf_chunk.select<BLOCK_W, BLOCK_OC>(i);                  
			cm_store<uint32_t, BLOCK_W/BLOCK_SC>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
			byte_offset += output_width * output_height * OUTPUT_ELEMENT_SIZE;
		}
	}
#else
	vector<uint32_t, BLOCK_W> store_offsets(output_linear_init_offsets);
	vector<ushort, BLOCK_W> predicate(1);
	
	if (b_ow_unalign_byte_offset == true)
	{
		for(int i = output_width % BLOCK_W; i < BLOCK_W; i++)
		{
			predicate.select<1, 1>(i) = 0;
		}
	}
	store_offsets += byte_offset;
		
	#pragma unroll
	for(int i = 0; i < BLOCK_OC; i++)
	{
		cm_store<DT_OUT, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, store_offsets, grf_chunk.select<BLOCK_W, BLOCK_OC>(i), predicate);
		store_offsets += output_width * output_height * OUTPUT_ELEMENT_SIZE;
	}
#endif
}

_GENX_ inline void store_output_wc8_as_nhwc(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, ACCU_REG_SIZE> grf_chunk, uint32_t output_width, uint32_t output_channels, uint32_t oc_chunk_offset, uint32_t output_w_chunk_offset, uint32_t byte_offset)
{
	//NCHW->NHWC
	const uint32_t BLOCK_SC = MAX_ELEMENT_SIZE / OUTPUT_ELEMENT_SIZE;
	const uint32_t pixel_offset = (byte_offset/output_channels) / OUTPUT_ELEMENT_SIZE;
	const uint32_t pixel_offset_per_row = output_width - (pixel_offset % output_width);
	
	const bool unaligned_byte_offset = (pixel_offset_per_row < BLOCK_W) ? true : false;
	const uint32_t store_width = (unaligned_byte_offset == false) ? BLOCK_W : pixel_offset_per_row;
	const uint32_t oc_block_width = output_channels - oc_chunk_offset;
	
#if DT == FLOAT32
	if(output_channels == 1 && unaligned_byte_offset == false)
	{
		vector<DT_OUT, BLOCK_OC> grf_chunk_store = grf_chunk.select<BLOCK_OC, BLOCK_W>(0);
		cm_store<uint32_t, BLOCK_OC / BLOCK_SC>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
	}
	else if(output_channels == 1 && unaligned_byte_offset == true)
	{
		for(int i = 0; i < pixel_offset_per_row; i++)
		{
			vector<DT_OUT, 1> grf_chunk_store = grf_chunk.select<1, 1>(i * BLOCK_W);
			cm_store<uint32_t, 1>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
			byte_offset += output_channels * OUTPUT_ELEMENT_SIZE;
		}
	}
	else
	{
		if(oc_block_width < BLOCK_OC)
		{
			for(int i = 0; i < store_width; i++)
			{
				for(int j = 0; j < oc_block_width; j++)
				{
					vector<DT_OUT, 1> grf_chunk_store = grf_chunk.select<1, 1>(i * BLOCK_W + j);
					cm_store<uint32_t, 1>(surface, byte_offset + j * OUTPUT_ELEMENT_SIZE, grf_chunk_store.format<uint32_t>());
				}
				byte_offset += output_channels * OUTPUT_ELEMENT_SIZE;
			}
		}
		else
		{
			for(int i = 0; i < store_width; i++)
			{
				vector<DT_OUT, BLOCK_OC> grf_chunk_store = grf_chunk.select<BLOCK_OC, 1>(i * BLOCK_W);
				cm_store<uint32_t, BLOCK_OC  / BLOCK_SC>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
				byte_offset += output_channels * OUTPUT_ELEMENT_SIZE;
			}
		}
	}
#else
	vector<uint32_t, BLOCK_OC> store_offsets(output_linear_init_offsets);
	vector<ushort, BLOCK_OC> predicate(1);
	
	if(oc_block_width < BLOCK_OC)
	{
		for(int i = oc_block_width % BLOCK_OC; i < BLOCK_OC; i++)
		{
			predicate.select<1, 1>(i) = 0;
		}
	}
	
	store_offsets += byte_offset;
	
	for(int i = 0; i < store_width; i++)
	{
		cm_store<DT_OUT, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, store_offsets, grf_chunk.select<BLOCK_OC, 1>(i * BLOCK_W), predicate);
		store_offsets += output_channels * OUTPUT_ELEMENT_SIZE;
	}
#endif
}

extern "C" _GENX_MAIN_ void convolution_nchw_nondpas(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_weights [[type("buffer_t")]],
#if USE_BIAS
	SurfaceIndex surface_bias [[type("buffer_t")]],
#endif
	SurfaceIndex surface_constants [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
	vector<uint32_t, 16> constants = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_constants, 0);
	const uint32_t input_height = constants[0];
	const uint32_t input_width = constants[1];
	const uint32_t input_pad = constants[2];
	const uint32_t output_channels = constants[3];
	const uint32_t output_height = constants[4];
	const uint32_t output_width = constants[5];
	const uint32_t stride_h = constants[6];
	const uint32_t activation_type = constants[9];
	const uint32_t activation_alpha = constants[10];
	const uint32_t activation_beta = constants[11];
	const uint32_t output_layout_is_nhwc = constants[12];
	const uint32_t input_layout_is_nhwc = constants[13];
	const uint32_t input_channels = constants[14];
	
	const uint32_t thg_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thg_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thg_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);

	const uint32_t w_chunk_id = thg_0;
	const uint32_t h_chunk_id = thg_1;
	const uint32_t thread_per_full_oc = (output_channels + BLOCK_OC - 1)/ BLOCK_OC;
	const uint32_t batch_id = thg_2 / thread_per_full_oc;
	const uint32_t oc_chunk_id = thg_2 % thread_per_full_oc;
    const uint32_t input_batch_offset = batch_id * BLOCK_BATCH * input_width * input_height * input_channels * sizeof(DT_IN);
    const uint32_t input_w_chunk_offset = w_chunk_id * BLOCK_W * STRIDE_W;
    const uint32_t input_h_chunk_offset = h_chunk_id * BLOCK_H * stride_h;

	matrix<DT_ACCU, BLOCK_BATCH, ACCU_REG_SIZE> accu_row_0(0.0f);

	if (input_channels == 1 || KERNEL_SIZE == 1)
	{
		if(input_layout_is_nhwc)
		{
			for(int i = 0; i < input_channels; i++)
			{		
				#pragma unroll
				for(int kh = 0; kh < KERNEL_SIZE; kh++)
				{
					matrix<DT_ACCU, BLOCK_BATCH, INPUT_REG_SIZE> input_0;
					
					#pragma unroll
					for(int b = 0; b < BLOCK_BATCH; b++)
					{
						input_0.row(b) = load_input_nhwc(surface_input, input_width, input_height, input_pad, input_w_chunk_offset, input_h_chunk_offset + kh, input_batch_offset + b * input_width * input_height * input_channels * sizeof(DT_IN) + (i * INPUT_ELEMENT_SIZE), input_channels);
					}
					
					#pragma unroll
					for(int kw = 0; kw < KERNEL_SIZE; kw++)
					{				
						matrix_ref<DT_ACCU, BLOCK_BATCH, BLOCK_W * STRIDE_W> input_chunk_0 = input_0.select<BLOCK_BATCH, 1, BLOCK_W * STRIDE_W, 1>(0, kw);
						vector<DT_ACCU, BLOCK_OC> weights_chunk_ic = load_weights(surface_weights, kw, kh, oc_chunk_id, i, input_channels);
				
						#pragma unroll
						for(int b = 0; b < BLOCK_BATCH; b++)
						{
							#pragma unroll
							for(int bw = 0; bw < BLOCK_W; bw++)
							{
								// as long as accumulator, input and weights are the same data type this will compile into single mad instruction				
								accu_row_0.select<1, 1, BLOCK_OC, 1>(b, bw * BLOCK_OC) += input_chunk_0.select<1, 1, 1, 1>(b, bw * STRIDE_W).replicate<BLOCK_OC>() * weights_chunk_ic;
							}
						}
					}		
				}		
			}
		}
		else
		{
			for(int i = 0; i < input_channels; i++)
			{		
				#pragma unroll
				for(int kh = 0; kh < KERNEL_SIZE; kh++)
				{
					matrix<DT_ACCU, BLOCK_BATCH, INPUT_REG_SIZE> input_0;
					
					#pragma unroll
					for(int b = 0; b < BLOCK_BATCH; b++)
					{
						input_0.row(b) = load_input_nchw(surface_input, input_width, input_height, input_pad, input_w_chunk_offset, input_h_chunk_offset + kh, input_batch_offset + b * input_width * input_height * input_channels * sizeof(DT_IN) + (i * input_width * input_height * INPUT_ELEMENT_SIZE));
					}
					
					#pragma unroll
					for(int kw = 0; kw < KERNEL_SIZE; kw++)
					{				
						matrix_ref<DT_ACCU, BLOCK_BATCH, BLOCK_W * STRIDE_W> input_chunk_0 = input_0.select<BLOCK_BATCH, 1, BLOCK_W * STRIDE_W, 1>(0, kw);
						vector<DT_ACCU, BLOCK_OC> weights_chunk_ic = load_weights(surface_weights, kw, kh, oc_chunk_id, i, input_channels);
				
						#pragma unroll
						for(int b = 0; b < BLOCK_BATCH; b++)
						{
							#pragma unroll
							for(int bw = 0; bw < BLOCK_W; bw++)
							{
								// as long as accumulator, input and weights are the same data type this will compile into single mad instruction				
								accu_row_0.select<1, 1, BLOCK_OC, 1>(b, bw * BLOCK_OC) += input_chunk_0.select<1, 1, 1, 1>(b, bw * STRIDE_W).replicate<BLOCK_OC>() * weights_chunk_ic;
							}
						}
					}		
				}		
			}
		}
	}
	else
	{
		for(int i = 0; i < input_channels; i++)
		{		
			#pragma unroll
			for(int kh = 0; kh < KERNEL_SIZE; kh++)
			{
				matrix<DT_ACCU, BLOCK_BATCH, INPUT_REG_SIZE> input_0;
				
				#pragma unroll
				for(int b = 0; b < BLOCK_BATCH; b++)
				{
					if(input_layout_is_nhwc)
					{
						input_0.row(b) = load_input_nhwc(surface_input, input_width, input_height, input_pad, input_w_chunk_offset, input_h_chunk_offset + kh, input_batch_offset + b * input_width * input_height * input_channels * sizeof(DT_IN) + (i * INPUT_ELEMENT_SIZE), input_channels);
					}
					else
					{
						input_0.row(b) = load_input_nchw(surface_input, input_width, input_height, input_pad, input_w_chunk_offset, input_h_chunk_offset + kh, input_batch_offset + b * input_width * input_height * input_channels * sizeof(DT_IN) + (i * input_width * input_height * INPUT_ELEMENT_SIZE));
					}	
				}
				
				#pragma unroll
				for(int kw = 0; kw < KERNEL_SIZE; kw++)
				{				
					matrix_ref<DT_ACCU, BLOCK_BATCH, BLOCK_W * STRIDE_W> input_chunk_0 = input_0.select<BLOCK_BATCH, 1, BLOCK_W * STRIDE_W, 1>(0, kw);
					vector<DT_ACCU, BLOCK_OC> weights_chunk_ic = load_weights(surface_weights, kw, kh, oc_chunk_id, i, input_channels);
			
					#pragma unroll
					for(int b = 0; b < BLOCK_BATCH; b++)
					{
						#pragma unroll
						for(int bw = 0; bw < BLOCK_W; bw++)
						{
							// as long as accumulator, input and weights are the same data type this will compile into single mad instruction				
							accu_row_0.select<1, 1, BLOCK_OC, 1>(b, bw * BLOCK_OC) += input_chunk_0.select<1, 1, 1, 1>(b, bw * STRIDE_W).replicate<BLOCK_OC>() * weights_chunk_ic;
						}
					}
				}		
			}		
		}
	}
#if USE_BIAS
	vector<DT_ACCU, BLOCK_OC> bias = load_bias(surface_bias, oc_chunk_id, output_channels);
	#pragma unroll
	for(int bw = 0; bw < BLOCK_W; bw++)
	{
		accu_row_0.select<1, 1, BLOCK_OC, 1>(0, bw * BLOCK_OC) += bias;
	}
#endif
	
	#pragma unroll
	for(int b = 0; b < BLOCK_BATCH; b++)
	{
		#pragma unroll
		for(int bw = 0; bw < ACCU_REG_SIZE; bw++)
		{
            DT_ACCU activation_input = accu_row_0[b][bw];
			accu_row_0.select<1, 1, 1, 1>(b, bw) = activation_function(activation_type, activation_input, (float)activation_alpha, (float)activation_beta);
        }
    }

	// if the DT_OUT == DT_ACCU then compiler will not do anything here
	// but if data types are different then this cast accumulator to output type
    //vector<DT_OUT, ACCU_REG_SIZE> output_row_0 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0);
    matrix<DT_OUT, BLOCK_BATCH, ACCU_REG_SIZE> output_row_0 = matrix<DT_OUT, BLOCK_BATCH, ACCU_REG_SIZE>(accu_row_0);

	const uint output_batch_offset = batch_id * BLOCK_BATCH * output_height * output_width * output_channels;
	if (output_layout_is_nhwc)
	{
		// offset_nhwc(n, h, w, c) = n * HWC + h * WC + w * C + c
		const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * output_width * output_channels;
		const uint output_w_chunk_offset = w_chunk_id * BLOCK_W * output_channels;
		const uint output_oc_chunk_offset = oc_chunk_id * BLOCK_OC;
		uint32_t output_offset = (output_batch_offset + output_h_chunk_offset + output_w_chunk_offset + output_oc_chunk_offset) * sizeof(DT_OUT);
		store_output_wc8_as_nhwc(surface_output, output_row_0.row(0), output_width, output_channels, output_oc_chunk_offset, output_w_chunk_offset, output_offset);
	}
	else
	{
		// offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
		const uint output_oc_chunk_offset = oc_chunk_id * BLOCK_OC * output_height * output_width;
		const uint output_w_chunk_offset = w_chunk_id * BLOCK_W;
		const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * output_width;
		uint32_t output_offset = (output_batch_offset + output_oc_chunk_offset + output_h_chunk_offset + output_w_chunk_offset) * sizeof(DT_OUT);
		store_output_wc8_as_nchw(surface_output, output_row_0.row(0), output_width, output_height, output_offset, output_channels);
	}
}
