#include <cm/cm.h>
#include <cm/cmtl.h>


#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif

// optimization, which gives up to ~20% perf gain, can be forced to off during debug (or change to lower size like 2)
#define CAN_PRECACHE_TILE_M_CHUNK (TILE_M > 4 && TILE_M % 4 == 0)
#if CAN_PRECACHE_TILE_M_CHUNK
#define PRECACHE_TILE_M_SIZE 4
#else
#define PRECACHE_TILE_M_SIZE 1
#endif

#define SLM_KN_SHARING 1

_GENX_ inline uint32_t get_input_b_base_offset(uint32_t thread_id_0, uint32_t thread_id_1, uint32_t thread_id_2, uint32_t batch_thread_offset, uint32_t head_thread_offset, uint32_t k_slice_thread_offset)
{    
	// input is 5d qkv  input so use SEQ_LEN/HEADSIZE  etc. variables for offset calculations
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
	
	for(int batch_thread_offset = 0; batch_thread_offset < SIZE_BATCH; batch_thread_offset++)
	{
	for(int head_thread_offset = 0; head_thread_offset < SIZE_NUM_HEADS; head_thread_offset++)
	{
	
	matrix<DT_ACCU, TILE_M, TILE_N> accu(0.0f);
	
	const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);

#if SLM_KN_SHARING
	cm_slm_init(TILE_K * TILE_N * sizeof(DT));
#endif
	
	for(int k_chunk = 0; k_chunk < SIZE_K/ TILE_K; k_chunk++)
	{	
		// input is 4d regular gemm input so use SIZE_M/N?/ etc. for offset calculations
		const uint32_t input_a_base_offset = 
					(batch_thread_offset * SIZE_C * SIZE_M * SIZE_K
					+ head_thread_offset * SIZE_M * SIZE_K
					+ k_chunk * TILE_K
					+ thread_id_0 * TILE_M * SIZE_K) * sizeof(DT);

		const uint32_t input_b_base_offset = get_input_b_base_offset(thread_id_0, thread_id_1, thread_id_2, batch_thread_offset, head_thread_offset, k_chunk);	
		
#if TILE_N == 40
	const uint32_t S0 = 16;
	const uint32_t S1 = 4;
#elif TILE_N == 80	
	const uint32_t S0 = 32;
	const uint32_t S1 = 8;
#endif
		
#if SLM_KN_SHARING



		const uint32_t input_b_offset = input_b_base_offset + cm_local_id(0) * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE * sizeof(DT);
		vector<uint32_t, TILE_N/2> packed_row;
		packed_row.select<S0, 1>() = cm_load<uint32_t, S0, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset);
		packed_row.select<S1, 1>(S0) = cm_load<uint32_t, S1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset + S0 * sizeof(uint32_t));
		
		const uint32_t slm_write_base_offset = cm_local_id(0) * TILE_N * sizeof(DT);
		cm_store_slm<uint32_t, S0>(slm_write_base_offset, packed_row.select<S0, 1>());
		cm_store_slm<uint32_t, S1>(slm_write_base_offset + S0 * sizeof(uint32_t), packed_row.select<S1, 1>(S0));
		cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
		cm_barrier();
#endif	
				
	    #pragma unroll
		for(int m = 0; m < TILE_M/PRECACHE_TILE_M_SIZE; m++)
		{
			matrix<DT, PRECACHE_TILE_M_SIZE, TILE_K> input_a;
			for(int m_chunk = 0; m_chunk < PRECACHE_TILE_M_SIZE; m_chunk++)
			{
				const uint32_t input_a_offset = input_a_base_offset + (m * PRECACHE_TILE_M_SIZE + m_chunk) * SIZE_K * sizeof(DT);
				input_a.row(m_chunk).format<uint32_t>() = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_s, input_a_offset);
			}
			
			#pragma unroll
			for(int k = 0; k < TILE_K; k++)
			{	
#if SLM_KN_SHARING
				const uint32_t slm_read_base_offset = k * TILE_N * sizeof(DT);
				vector<uint32_t, TILE_N/2> packed_row;
				packed_row.select<S0, 1>() = cm_load_slm<uint32_t, S0>(slm_read_base_offset);
				packed_row.select<S1, 1>(S0) = cm_load_slm<uint32_t, S1>(slm_read_base_offset + S0 * sizeof(uint32_t));
#else
				const uint32_t input_b_offset = input_b_base_offset + k * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE * sizeof(DT);
				vector<uint32_t, TILE_N/2> packed_row;
#if TILE_N == 40 || TILE_N == 80
				packed_row.select<S0, 1>() = cm_load<uint32_t, S0, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset);
				packed_row.select<S1, 1>(S0) = cm_load<uint32_t, S1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset + S0 * sizeof(uint32_t));
#else
				packed_row = cm_load<uint32_t, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offset);
#endif
#endif  // SLM_KN_SHARING
				
				#pragma unroll
				for(int m_chunk = 0; m_chunk < PRECACHE_TILE_M_SIZE; m_chunk++)
				{
					accu.row(m * PRECACHE_TILE_M_SIZE + m_chunk) += packed_row.format<DT>() * input_a.row(m_chunk).format<DT>().select<1, 1>(k).replicate<TILE_N>();
				}
			}
		}	
	}
	
	const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = (batch_thread_offset * SIZE_NUM_HEADS * SIZE_SEQ_LEN * SIZE_HEAD_SIZE
				+ head_thread_offset * SIZE_HEAD_SIZE
				+ thread_id_0 * TILE_M * SIZE_NUM_HEADS * SIZE_HEAD_SIZE
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
		output_offset += SIZE_HEAD_SIZE * SIZE_NUM_HEADS * sizeof(DT);
	}
	}//batch
	}//head
}
