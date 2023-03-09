#include <cm/cm.h>
#include <cm/cmtl.h>

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

    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
        
    const uint32_t input_a_base_offset = thread_id_0 * TILE_M * SIZE_K * sizeof(DT);
    
    matrix<DT, TILE_M, TILE_K> input;   
    matrix_ref<uint32_t, TILE_M, TILE_K/2> input_packed = input.format<uint32_t, TILE_M, TILE_K/2>();
    
    #pragma unroll
    for(int i = 0; i < TILE_M; i++)
    {
        const uint32_t input_a_offset = input_a_base_offset + (i * SIZE_K * sizeof(DT));
        input_packed.row(i) = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);
    }
    
    
    matrix<DT, TILE_M, TILE_N> accu(0);
    uint32_t input_b_offset = thread_id_1 * TILE_N * sizeof(DT);
    #pragma unroll
    for(uint32_t i = 0; i < SIZE_K; i++)
    {
        vector<uint32_t, input_b_load_size> input_b_packed = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);  
        vector_ref<DT, TILE_N> input_b = input_b_packed.format<DT>();        
        input_b_offset += SIZE_N * sizeof(DT);
        
        #pragma unroll
        for(uint32_t j = 0; j < TILE_M; j++)
        {
            accu.select<1, 1, TILE_N, 1>(j, 0) += input_b * input.select<1, 1, 1, 1>(j, i).replicate<TILE_N>();
        }
    }

    const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = (thread_id_0 * TILE_M * SIZE_N * sizeof(DT)) + (thread_id_1 * TILE_N * sizeof(DT));
    for(uint32_t i = 0; i < TILE_M; i++)
    {
        vector_ref<uint32_t, output_store_size> accu_0_packed = accu.select<1, 1, TILE_N, 1>(i, 0).format<uint32_t>();
        cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
        output_offset += SIZE_N * sizeof(DT);
    }

}
