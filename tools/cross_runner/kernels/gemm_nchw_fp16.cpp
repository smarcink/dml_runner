#include <cm/cm.h>
#include <cm/cmtl.h>

#define K_PER_THREAD (SIZE_K / SLICE_K)
#if ACCU_IS_FP32
#define DT_ACCU float
#else
#define DT_ACCU DT
#endif
extern "C" _GENX_MAIN_ void gemm_nchw_fp16(
	SurfaceIndex surface_input_a [[type("buffer_t")]],
	SurfaceIndex surface_input_b [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
#if USE_C
	, SurfaceIndex surface_input_c [[type("buffer_t")]]
#endif
)
{
    const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	const uint32_t batch_channels_thread_offset = cm_group_id(2);
	const uint32_t k_slice_thread_offset = cm_local_id(2);
	
    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
        
    const uint32_t input_a_base_offset = 
						batch_channels_thread_offset * SIZE_M * SIZE_K * sizeof(DT)
						+ thread_id_0 * TILE_M * SIZE_K * sizeof(DT)
						+ (k_slice_thread_offset * K_PER_THREAD * sizeof(DT));
    
    matrix<DT, TILE_M, TILE_K> input;   
    matrix_ref<uint32_t, TILE_M, TILE_K/2> input_packed = input.format<uint32_t, TILE_M, TILE_K/2>();
    
    #pragma unroll
    for(int i = 0; i < TILE_M; i++)
    {
        const uint32_t input_a_offset = input_a_base_offset + (i * SIZE_K * sizeof(DT));
#if TILE_K == 40
        input_packed.row(i).select<16, 1>() = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);
        input_packed.row(i).select<4, 1>(16) = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset + 16 * sizeof(uint32_t));
#elif TILE_K == 80		
        input_packed.row(i).select<32, 1>() = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);
        input_packed.row(i).select<8, 1>(32) = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset + 32 * sizeof(uint32_t));
#else
        input_packed.row(i) = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);
#endif
    }
    
#if 0
	for(int i = 0; i < TILE_K; i++)
	{
		printf("%f, ", (float)input[0][i]);
	}
    printf("\n");
#endif

	
    matrix<DT_ACCU, TILE_M, TILE_N> accu(0.0f);
    uint32_t input_b_offset = 
					batch_channels_thread_offset * SIZE_K * SIZE_N * sizeof(DT)
					+ thread_id_1 * TILE_N * sizeof(DT)
					+ (k_slice_thread_offset * K_PER_THREAD * SIZE_N * sizeof(DT));
    //#pragma unroll
    for(uint32_t i = 0; i < K_PER_THREAD; i++)
    {
        vector<uint32_t, input_b_load_size> input_b_packed = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);  
        vector_ref<DT, TILE_N> input_b = input_b_packed.format<DT>();        
        input_b_offset += SIZE_N * sizeof(DT);

	
        #pragma unroll
        for(uint32_t j = 0; j < TILE_M; j++)
        {
            accu.select<1, 1, TILE_N, 1>(j, 0) += input_b * input.select<1, 1, 1, 1>(j, i).replicate<TILE_N>();
        }
#if 0
		printf("\n ACCU: \n");
		for(int i = 0; i < TILE_N; i++)
		{
			printf("%f, ", (float)accu[0][i]);
		}
		printf("\n");
		for(int i = 0; i < TILE_N; i++)
		{
			printf("%f, ", (float)input_b[i]);
		}
		printf("\n");
#endif	
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
				batch_channels_thread_offset * SIZE_M * SIZE_N * sizeof(DT)
				+ thread_id_0 * TILE_M * SIZE_N * sizeof(DT)
				+ (thread_id_1 * TILE_N * sizeof(DT));
				
	//printf("[%d, %d]: %d\n", batch_channels_thread_offset, output_offset, batch_channels_thread_offset * SIZE_M * SIZE_N * sizeof(DT));
				
	matrix<DT, TILE_M, TILE_N> accu_out = accu;  // if DT_ACCU == DT then compiler removes this line
    for(uint32_t i = 0; i < TILE_M; i++)
    {
        vector_ref<uint32_t, output_store_size> accu_0_packed = accu_out.select<1, 1, TILE_N, 1>(i, 0).format<uint32_t>();
        cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
        output_offset += SIZE_N * sizeof(DT);
    }

}
