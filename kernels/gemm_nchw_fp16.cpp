#include <cm/cm.h>
#include <cm/cmtl.h>

#define TILE_K 16
#define TILE_N 16

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


    
    vector<DT, TILE_K> accu_0(0);
    
    const uint32_t input_a_load_size = (TILE_K * sizeof(DT)) / sizeof(uint32_t);
    const uint32_t input_a_offset = 0;
    vector<uint32_t, input_a_load_size> input_a_packed = cm_load<uint32_t, input_a_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, input_a_offset);
    vector_ref<DT, TILE_K> input_a = input_a_packed.format<DT>();
    
    const uint32_t input_b_load_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t input_b_offset = 0;
    #pragma unroll
    for(uint32_t i = 0; i < SIZE_K; i++)
    {
        vector<uint32_t, input_b_load_size> input_b_packed = cm_load<uint32_t, input_b_load_size, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_b_offset);  
        vector_ref<DT, TILE_N> input_b = input_b_packed.format<DT>();        
        input_b_offset += SIZE_N * sizeof(DT);
        
        accu_0 += input_b * input_a.select<1, 1>(i).replicate<TILE_N>();
    }

    const uint32_t output_store_size = (TILE_N * sizeof(DT)) / sizeof(uint32_t);
    uint32_t output_offset = 0;
    vector_ref<uint32_t, output_store_size> accu_0_packed = accu_0.format<uint32_t>();
    cm_store<uint32_t, output_store_size, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, output_offset, accu_0_packed);
}
