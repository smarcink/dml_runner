#include <cm/cm.h>
#include <cm/cmtl.h>


#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif



extern "C" _GENX_MAIN_ void mha_qk_and_sv_q_kv_gemm(
	SurfaceIndex surface_input_s [[type("buffer_t")]],   // Q or S input
	SurfaceIndex surface_input_q_kv [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t batch_thread_offset = cm_group_id(2) / SIZE_NUM_HEADS;
	const uint32_t head_thread_offset = cm_group_id(2) % SIZE_NUM_HEADS;
	
	const uint32_t input_a_base_offset = 
			(batch_thread_offset * SIZE_SEQ_LEN * SIZE_NUM_HEADS * SIZE_HEAD_SIZE
			+ head_thread_offset * SIZE_HEAD_SIZE
			//+ k_chunk * TILE_K
			+ thread_id_0 * SIZE_NUM_HEADS * SIZE_HEAD_SIZE) * sizeof(DT);
	const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
	
	#pragma unroll
	for(int m = 0; m < TILE_M; m++)
	{
		const uint32_t input_a_offset = input_a_base_offset + m * SIZE_NUM_HEADS * SIZE_HEAD_SIZE * sizeof(DT);
		vector<uint32_t, TILE_K/2> input_a_packed = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_s, input_a_offset);
	}
	
}
