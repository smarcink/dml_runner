#include <cm/cm.h>
#include <cm/cmtl.h>

#define K_PER_THREAD SIZE_K

#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif

_GENX_ inline uint32_t get_input_b_base_offset(uint32_t thread_id_0, uint32_t thread_id_1, uint32_t thread_id_2, uint32_t batch_thread_offset, uint32_t head_thread_offset)
{    
	return ( batch_thread_offset * SIZE_SEQ_LEN * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE
			+ head_thread_offset * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE  // offset batch + channels
			+ thread_id_1 * TILE_N
			+ SIZE_HEAD_SIZE) * sizeof(DT); // offset k 
}

extern "C" _GENX_MAIN_ void mha_sv_s_kv_gemm_fp16(
	SurfaceIndex surface_input_s [[type("buffer_t")]],
	SurfaceIndex surface_input_kv [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t batch_thread_offset = cm_group_id(2) / SIZE_NUM_HEADS;
	const uint32_t head_thread_offset = cm_group_id(2) % SIZE_NUM_HEADS;
	
	const uint32_t input_a_base_offset = 
			(batch_thread_offset * SIZE_M * SIZE_K * SIZE_C
			+ head_thread_offset * SIZE_M  * SIZE_K
			+ thread_id_0 * TILE_M * SIZE_K) * sizeof(DT);
	const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
	
	
    matrix<DT, TILE_M, TILE_K> input_a;   
    matrix_ref<uint32_t, TILE_M, TILE_K/2> input_a_packed = input_a.format<uint32_t, TILE_M, TILE_K/2>();
	
	#pragma unroll
	for(int m = 0; m < TILE_M; m++)
	{
		const uint32_t input_a_offset = input_a_base_offset + m * SIZE_K * sizeof(DT);
        input_a_packed.row(m) = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_s, input_a_offset);
	}
	
	
    matrix<DT_ACCU, TILE_M, TILE_N> accu(0.0f);
	
	const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
	const uint32_t input_b_base_offset = get_input_b_base_offset(thread_id_0, thread_id_1, thread_id_2, batch_thread_offset, head_thread_offset);;
	
    #pragma unroll
    for(uint32_t k = 0; k < K_PER_THREAD; k++)
    {
		const uint32_t input_b_offset = input_b_base_offset + k * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE * sizeof(DT);
		vector<uint32_t, TILE_N/2> packed_row;
#if TILE_N == 40
		packed_row.select<16, 1>() = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, input_b_offset);
		packed_row.select<4, 1>(16) = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, input_b_offset + 16 * sizeof(uint32_t));
#elif TILE_N == 80		
		packed_row.select<32, 1>() = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, input_b_offset);
		packed_row.select<8, 1>(32) = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, input_b_offset + 32 * sizeof(uint32_t));
#else
		packed_row = cm_load<uint32_t, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_kv, input_b_offset);
#endif

		#pragma unroll
		for(int m = 0; m < TILE_M; m++)
		{
			accu.row(m) += packed_row.format<DT>() * input_a.row(m).select<1, 1>(k).replicate<TILE_N>();
		}
    }

	const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = 
				cm_group_id(2) * SIZE_M * SIZE_N * sizeof(DT)
				+ thread_id_0 * TILE_M * SIZE_N * sizeof(DT)
				+ (thread_id_1 * TILE_N * sizeof(DT));
				
	//printf("[%d, %d]: %d\n", batch_channels_thread_offset, output_offset, batch_channels_thread_offset * SIZE_M * SIZE_N * sizeof(DT));
				
	matrix<DT, TILE_M, TILE_N> accu_out = accu;  // if DT_ACCU == DT then compiler removes this line
	accu_out *= DT(SCALE);
	#pragma unroll
    for(uint32_t i = 0; i < TILE_M; i++)
    {
        vector_ref<uint32_t, output_store_size> accu_0_packed = accu_out.select<1, 1, TILE_N, 1>(i, 0).format<uint32_t>();
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
