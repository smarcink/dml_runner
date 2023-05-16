#include <cm/cm.h>
#include <cm/cmtl.h>


#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif

#define K_PER_THREAD TILE_K

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
			+ thread_id_1 * TILE_N * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE // offset n
			+ k_slice_thread_offset * K_PER_THREAD
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
	const uint32_t k_slice_thread_offset = cm_local_id(2);
	
	const uint32_t batch_thread_offset = cm_group_id(2) / SIZE_NUM_HEADS;
	const uint32_t head_thread_offset = cm_group_id(2) % SIZE_NUM_HEADS;
	
	matrix<DT_ACCU, TILE_M, TILE_N> accu(0.0f);
	
	vector<DT, TILE_K * 2> input_b;
	vector_ref<uint32_t, TILE_K> input_b_packed = input_b.format<uint32_t>();
	
	const uint32_t load_simd_size = 16;
	const uint32_t packed_eles = sizeof(uint32_t) / sizeof(DT);
	const uint32_t load_eles = load_simd_size * packed_eles;
	const uint32_t ks = 2;
	const uint32_t ksp = ks/packed_eles;
	vector<uint32_t, 16> input_b_offsets(init_linear_offsets);
    input_b_offsets += get_input_b_base_offset(thread_id_0, thread_id_1, thread_id_2, batch_thread_offset, head_thread_offset, k_slice_thread_offset);
	#pragma unroll
	for(int j = 0; j < TILE_K / load_simd_size; j++)
	{
		input_b_packed.select<load_simd_size * ksp, 1>(j * load_simd_size * ksp) = cm_load<uint32_t, details::lsc_vector_size<ksp>(), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_b_offsets);  
		input_b_offsets += load_simd_size * INPUT_B_OFFSET;
	}
	
	
	const uint32_t input_a_base_offset = 
						(batch_thread_offset * SIZE_NUM_HEADS * SIZE_M * SIZE_K
						+ head_thread_offset * SIZE_M * SIZE_K
						+ thread_id_0 * TILE_M * SIZE_K) * sizeof(DT);
	const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    #pragma unroll
    for(int m = 0; m < TILE_M; m++)
    {
        const uint32_t input_a_offset = input_a_base_offset + m * SIZE_K * sizeof(DT);
        vector<uint32_t, TILE_K/2> input_a_packed = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_a_offset);
		vector_ref<DT, TILE_K> input_a = input_a_packed.format<DT>();
		#pragma unroll
		for(int n = 0; n < TILE_N; n++)
		{
			vector<DT, TILE_K> temp = input_a * input_b.select<TILE_K, 2>(n);
			accu[m][n] = cm_sum<DT>(temp);
		}
		
    }
	
	// store results
    for(int m = 0; m < TILE_M; m++)
    {
		for(int n = 0; n < TILE_N; n++)
		{
			printf("%f, ", (float)accu[m][n]);
		}
	}
	
	
}
