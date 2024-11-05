#include <cm/cm.h>
#include <cm/cmtl.h>


#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif


extern "C" _GENX_MAIN_ void gemm_nchw_fp16_stateless(
    uint64_t surface_input_a [[type("svmptr_t half")]],
    uint64_t surface_input_b [[type("svmptr_t half")]],
    uint64_t surface_output [[type("svmptr_t half")]]
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
        
    const uint32_t input_a_base_offset = thread_id_0 * TILE_M * SIZE_K * sizeof(DT);
    
	matrix<DT_ACCU, TILE_M, TILE_N> accu(0.0f);
    uint32_t input_b_offset = thread_id_1 * TILE_N * sizeof(DT);
	
    matrix<DT, TILE_M, TILE_K> input_a;   
    matrix_ref<uint32_t, TILE_M, TILE_K/2> input_a_packed = input_a.format<uint32_t, TILE_M, TILE_K/2>();
    
	for(uint32_t i = 0; i < SIZE_K / TILE_K; i++)
	{
		#pragma unroll
		for(int m = 0; m < TILE_M; m++)
		{
			const uint32_t input_a_offset = input_a_base_offset + (m * SIZE_K + i * TILE_K) * sizeof(DT);
			input_a_packed.row(m) = cm_ptr_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)surface_input_a, input_a_offset);

		}
	   

		

		#pragma unroll
		for(uint32_t k = 0; k < TILE_K; k++)
		{
			vector<uint32_t, input_b_load_size> input_b_packed =  cm_ptr_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)surface_input_b, input_b_offset);
			
			vector_ref<DT, TILE_N> input_b = input_b_packed.format<DT>();        
			input_b_offset += SIZE_N * sizeof(DT);

		
			#pragma unroll
			for(uint32_t j = 0; j < TILE_M; j++)
			{
#if ACCU_IS_FP32
				vector<DT_ACCU, TILE_N> input_b_fp32 = vector<DT_ACCU, TILE_N>(input_b);
				vector<DT_ACCU, TILE_N> input_a_fp32 = vector<DT_ACCU, TILE_N>(input_a.select<1, 1, 1, 1>(j, k).replicate<TILE_N>());
				accu.select<1, 1, TILE_N, 1>(j, 0) += input_b_fp32 * input_a_fp32;
#else
				accu.select<1, 1, TILE_N, 1>(j, 0) += input_b * input_a.select<1, 1, 1, 1>(j, k).replicate<TILE_N>();
#endif
			}

		}
    }
    const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = (thread_id_0 * TILE_M * SIZE_N + thread_id_1 * TILE_N) * sizeof(DT);
				
		
	matrix<DT, TILE_M, TILE_N> accu_out = accu;  // if DT_ACCU == DT then compiler removes this line
	accu_out *= DT(SCALE);
	
	#pragma unroll
    for(uint32_t i = 0; i < TILE_M; i++)
    {
        vector_ref<uint32_t, output_store_size> accu_0_packed = accu_out.select<1, 1, TILE_N, 1>(i, 0).format<uint32_t>();
		cm_ptr_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>((uint32_t*)surface_output, output_offset, accu_0_packed);

        output_offset += SIZE_N * sizeof(DT);
    }

}
