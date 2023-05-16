#include <cm/cm.h>
#include <cm/cmtl.h>

#define K_PER_THREAD (SIZE_K / SLICE_K)
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

_GENX_ inline uint32_t get_input_a_base_offset(uint32_t thread_id_0, uint32_t thread_id_1, uint32_t thread_id_2, uint32_t batch_thread_offset, uint32_t head_thread_offset, uint32_t k_slice_thread_offset)
{    
	return (batch_thread_offset * SIZE_SEQ_LEN * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE
			+ head_thread_offset * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE  // offset batch + channels
			+ thread_id_0 * TILE_M * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE // offset m
			+ k_slice_thread_offset * K_PER_THREAD) * sizeof(DT); // offset k 
}

_GENX_ inline uint32_t get_input_b_base_offset(uint32_t thread_id_0, uint32_t thread_id_1, uint32_t thread_id_2, uint32_t batch_thread_offset, uint32_t head_thread_offset, uint32_t k_slice_thread_offset)
{    
	return ( batch_thread_offset * SIZE_SEQ_LEN * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE
			+ head_thread_offset * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE  // offset batch + channels
			+ thread_id_1 * TILE_N * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE // offset n
			+ k_slice_thread_offset * K_PER_THREAD
			+ SIZE_HEAD_SIZE) * sizeof(DT); // offset k 
}


extern "C" _GENX_MAIN_ void mha_qk_qkv_gemm(
	SurfaceIndex surface_input_qkv [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);

	const uint32_t batch_thread_offset = cm_group_id(2) / SIZE_NUM_HEADS;
	const uint32_t head_thread_offset = cm_group_id(2) % SIZE_NUM_HEADS;
	
	const uint32_t k_slice_thread_offset = cm_local_id(2);
	
    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
        
    const uint32_t input_a_base_offset = get_input_a_base_offset(thread_id_0, thread_id_1, thread_id_2, batch_thread_offset, head_thread_offset, k_slice_thread_offset);
 

    matrix<DT, TILE_M, TILE_K> input;   
    matrix_ref<uint32_t, TILE_M, TILE_K/2> input_packed = input.format<uint32_t, TILE_M, TILE_K/2>();
 
    #pragma unroll
    for(int i = 0; i < TILE_M; i++)
    {
        const uint32_t input_a_offset = input_a_base_offset + i * SIZE_NUM_HEADS * SIZE_STACKED_TENSORS * SIZE_HEAD_SIZE * sizeof(DT);

#if TILE_K == 40
        input_packed.row(i).select<16, 1>() = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_a_offset);
        input_packed.row(i).select<4, 1>(16) = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_a_offset + 16 * sizeof(uint32_t));
#elif TILE_K == 80		
        input_packed.row(i).select<32, 1>() = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_a_offset);
        input_packed.row(i).select<8, 1>(32) = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_a_offset + 32 * sizeof(uint32_t));
#else
        input_packed.row(i) = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, input_a_offset);
#endif
    }

    matrix<DT_ACCU, TILE_M, TILE_N> accu(0.0f);
	
	const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
	const uint32_t input_b_base_offset = get_input_b_base_offset(thread_id_0, thread_id_1, thread_id_2, batch_thread_offset, head_thread_offset, k_slice_thread_offset);
	uint32_t input_b_offset = input_b_base_offset;
	
	vector<uint32_t, 16> base_b_offsets(init_linear_offsets);
    base_b_offsets += input_b_base_offset;
	
	const uint32_t load_simd_size = 16;
	const uint32_t packed_eles = sizeof(uint32_t) / sizeof(DT);
	const uint32_t load_eles = load_simd_size * packed_eles;  // load elements when VectorSize == 1
	const uint32_t ks = 8;   //ToDo:  this can be a reason of spills, it can be decreased to: {2 or 4}, but it can affect performance
	const uint32_t ksp = ks/packed_eles;
    //#pragma unroll
    for(uint32_t k_chunk = 0; k_chunk < K_PER_THREAD/ks; k_chunk++)
    {

		vector<uint32_t, load_simd_size> offsets(base_b_offsets);
		vector<uint32_t, TILE_N * ksp> input_b_packed;
		vector_ref<DT, TILE_N * ks> input_b_line = input_b_packed.format<DT>();
		#pragma unroll
		for(int j = 0; j < TILE_N / load_simd_size; j++)
		{
			input_b_packed.select<load_simd_size * ksp, 1>(j * load_simd_size * ksp) = cm_load<uint32_t, details::lsc_vector_size<ksp>(), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_qkv, offsets);  
			offsets += load_simd_size * INPUT_B_OFFSET;
		}

		#pragma unroll
		for(int k = 0; k < ksp; k++)
		{
			vector<DT, TILE_N * packed_eles> input_b_ksp_chunk;
			#pragma unroll
			for(int j = 0; j < TILE_N / load_simd_size; j++)
			{
				input_b_ksp_chunk.select<load_eles, 1>(j * load_eles) = input_b_line.select<load_eles, 1>(k * load_eles + j * ksp * load_eles);
			}
			for(int i = 0; i < packed_eles; i++)
			{
				vector<DT, TILE_N> input_b = input_b_ksp_chunk.select<TILE_N, packed_eles>(i);
				#pragma unroll
				for(uint32_t j = 0; j < TILE_M; j++)
				{
					accu.select<1, 1, TILE_N, 1>(j, 0) += input_b * input.select<1, 1, 1, 1>(j, k_chunk * ks + k * packed_eles + i).replicate<TILE_N>();
				}
			}
			
		}
		base_b_offsets += ksp * sizeof(uint32_t);
    }

#if SLICE_K > 1
	const uint32_t TILE_N_PACKED = TILE_N / (sizeof(uint32_t)/sizeof(DT_ACCU));
    cm_slm_init(TILE_M * TILE_N * sizeof(DT_ACCU) * (LWS_SIZE_Z - 1));

    if(cm_local_id(2) > 0)
    {
        #pragma unroll
        for(uint32_t i = 0; i < TILE_M; i++)
        {
            vector_ref<DT_ACCU, TILE_N> data_to_store_slm = accu.row(i);
            vector_ref<uint32_t, TILE_N_PACKED> data_to_store_slm_typed = data_to_store_slm.format<uint32_t>();
            cm_store_slm<uint32_t, TILE_N_PACKED>(i * TILE_N * sizeof(DT_ACCU) + (k_slice_thread_offset - 1) * TILE_M * TILE_N * sizeof(DT_ACCU), data_to_store_slm_typed);
            //cm_store_slm<uint32_t, TILE_N/2>(i * TILE_N * sizeof(DT), data_to_store_slm_typed);
            //cm_store_slm<uint32_t, TILE_N/2>(0, data_to_store_slm_typed);
        }
    }
   
    cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
    cm_barrier();
    if(cm_local_id(2) > 0)
    {
        return;
    }
    
    #pragma unroll
    for(uint32_t i = 0; i < TILE_M; i++)
    {
        vector<DT_ACCU, TILE_N> data_to_load_slm;
        vector_ref<uint32_t, TILE_N_PACKED> data_to_load_slm_typed = data_to_load_slm.format<uint32_t>();
        #pragma unroll
        for(int j = 0; j < (LWS_SIZE_Z - 1); j++)
        {
            data_to_load_slm_typed = cm_load_slm<uint32_t, TILE_N_PACKED>(i * TILE_N * sizeof(DT_ACCU) + j * TILE_M * TILE_N * sizeof(DT_ACCU));
            accu.row(i) += data_to_load_slm;
        }
    }
#endif  
 
    //printf("%d, %d, %d\n", thread_id_0, thread_id_1, thread_id_2);
    
    const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = 
				cm_group_id(2) * SIZE_M * SIZE_N * sizeof(DT)
				+ thread_id_0 * TILE_M * SIZE_N * sizeof(DT)
				+ (thread_id_1 * TILE_N * sizeof(DT));
				
	//printf("[%d, %d]: %d\n", batch_channels_thread_offset, output_offset, batch_channels_thread_offset * SIZE_M * SIZE_N * sizeof(DT));
				
	matrix<DT, TILE_M, TILE_N> accu_out = accu;  // if DT_ACCU == DT then compiler removes this line
	accu_out *= DT(SCALE);
    for(uint32_t i = 0; i < TILE_M; i++)
    {
        vector_ref<uint32_t, output_store_size> accu_0_packed = accu_out.select<1, 1, TILE_N, 1>(i, 0).format<uint32_t>();
        cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
        output_offset += SIZE_N * sizeof(DT);
    }

}
