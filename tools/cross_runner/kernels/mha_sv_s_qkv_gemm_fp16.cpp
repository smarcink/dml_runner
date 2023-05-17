#include <cm/cm.h>
#include <cm/cmtl.h>


#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif


#define INPUT_B_OFFSET ((SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE)* sizeof(DT))
static const int32_t init_linear_offsets[] = {  0  * INPUT_B_OFFSET,
											    1  * INPUT_B_OFFSET, 
											    2  * INPUT_B_OFFSET,
											    3  * INPUT_B_OFFSET,
											    4  * INPUT_B_OFFSET,
											    5  * INPUT_B_OFFSET,
											    6  * INPUT_B_OFFSET,
											    7  * INPUT_B_OFFSET,
												8  * INPUT_B_OFFSET, 
											    9  * INPUT_B_OFFSET,
											    10 * INPUT_B_OFFSET,
											    11 * INPUT_B_OFFSET,
											    12 * INPUT_B_OFFSET,
											    13 * INPUT_B_OFFSET,
											    14 * INPUT_B_OFFSET,
											    15 * INPUT_B_OFFSET,
											  };

_GENX_ inline uint32_t get_input_b_base_offset(uint32_t thread_id_0, uint32_t thread_id_1, uint32_t thread_id_2, uint32_t batch_thread_offset, uint32_t head_thread_offset, uint32_t k_slice_thread_offset)
{    
	return ( batch_thread_offset * SIZE_SEQ_LEN * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE
			+ head_thread_offset * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE  // offset batch + channels
			+ thread_id_1 * TILE_N
			+ k_slice_thread_offset * TILE_K * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE 
			+ 2 * SIZE_HEAD_SIZE) * sizeof(DT); // offset k 
}


extern "C" _GENX_MAIN_ void mha_sv_s_qka_gemm(
	SurfaceIndex surface_input_s [[type("buffer_t")]],
	SurfaceIndex surface_input_qkv [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t batch_thread_offset = cm_group_id(2) / SIZE_NUM_HEADS;
	const uint32_t head_thread_offset = cm_group_id(2) % SIZE_NUM_HEADS;
	
	matrix<DT_ACCU, TILE_M, TILE_N> accu(0.0f);
	
	const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);

	for(int k_chunk = 0; k_chunk < SIZE_K/ TILE_K; k_chunk++)
	{
		const uint32_t input_a_base_offset = 
					(batch_thread_offset * SIZE_C * SIZE_M * SIZE_K
					+ head_thread_offset * SIZE_M * SIZE_K
					+ k_chunk * TILE_K
					+ thread_id_0 * TILE_M * SIZE_K) * sizeof(DT);
	
		const uint32_t input_b_base_offset = get_input_b_base_offset(thread_id_0, thread_id_1, thread_id_2, batch_thread_offset, head_thread_offset, k_chunk);	
	    #pragma unroll
		for(int m = 0; m < TILE_M; m++)
		{
			const uint32_t input_a_offset = input_a_base_offset + m * SIZE_K * sizeof(DT);
			vector<uint32_t, TILE_K/2> input_a_packed = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_s, input_a_offset);
			#pragma unroll
			for(int k = 0; k < TILE_K; k++)
			{	
				const uint32_t input_b_offset = input_b_base_offset + k * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE * sizeof(DT);
				vector<uint32_t, TILE_N/2> packed_row;
#if TILE_N == 40
				packed_row.select<16, 1>() = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset);
				packed_row.select<4, 1>(16) = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset + 16 * sizeof(uint32_t));
#elif TILE_N == 80		
				packed_row.select<32, 1>() = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset);
				packed_row.select<8, 1>(32) = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset + 32 * sizeof(uint32_t));
#else
				packed_row = cm_load<uint32_t, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset);
#endif

				accu.row(m) += packed_row.format<DT>() * input_a_packed.format<DT>().select<1, 1>(k).replicate<TILE_N>();
			}
		}	
	}
	
	const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = (batch_thread_offset * SIZE_NUM_HEADS * SIZE_M * SIZE_N
				+ head_thread_offset * SIZE_M * SIZE_N
				+ thread_id_0 * TILE_M * SIZE_N
				+ thread_id_1 * TILE_N)* sizeof(DT);
	
	matrix<DT, TILE_M, TILE_N> accu_out = accu;  // if DT_ACCU == DT then compiler removes this line
	accu_out *= DT(SCALE);
	// store results
	#pragma unroll
    for(int m = 0; m < TILE_M; m++)
    {
		vector_ref<uint32_t, output_store_size> accu_0_packed = accu_out.select<1, 1, TILE_N, 1>(m, 0).format<uint32_t>();
#if TILE_N == 40
		cm_store<uint32_t, 16, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed.select<16, 1>());
		cm_store<uint32_t, 4, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset + 16 * sizeof(uint32_t), accu_0_packed.select<4, 1>(16));
#elif TILE_N == 80	
		cm_store<uint32_t, 32, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed.select<32, 1>());
		cm_store<uint32_t, 8, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset + 32 * sizeof(uint32_t), accu_0_packed.select<8, 1>(32));	
#else
		cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
#endif
		output_offset += SIZE_N * sizeof(DT);
	}
}
