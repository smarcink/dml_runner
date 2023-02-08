#include <cm/cm.h>
#include <cm/cmtl.h>

#define USE_BIAS 0

#define EXEC_SIZE 8
#define DPAS_DEPTH 8 

#define DT_OUT half
#define DT_IN half
#define DT_WEIGHTS half
// accu on DG2 have to be float for half dt inputs
#define DT_ACCU float 

#define INPUT_WIDTH 8
#define INPUT_HEIGHT 1
#define INPUT_CHANNELS 16

#define OUTPUT_WIDTH 8
#define OUTPUT_HEIGHT 1
#define OUTPUT_CHANNELS 8

#define DPAS_INPUT_CHANNELS (DPAS_DEPTH * sizeof(DT_IN))
#define DPAS_OUTPUT_CHANNELS EXEC_SIZE


#define WEIGHTS_REG_SIZE (DPAS_INPUT_CHANNELS * DPAS_OUTPUT_CHANNELS)

static const uint32_t store_init_offsets_for_halfs[8] = { 0, 16, 32, 48, 64, 80, 96, 112 };

template<typename DT, unsigned VS, CacheHint CH_L1, CacheHint CH_L3>
_GENX_ inline vector<DT, VS> lsc_load(SurfaceIndex surface [[type("buffer_t")]], const uint byte_offset)
{
    return cm_load<DT, VS, DataSize::Default, CH_L1, CH_L3>(surface, byte_offset);
}

template<typename DT, unsigned VS, CacheHint CH_L1, CacheHint CH_L3>
_GENX_ inline vector<DT, VS> lsc_store(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT, VS> grf_chunk, const uint byte_offset)
{
    cm_store<DT, VS, DataSize::Default, CH_L1, CH_L3>(surface, byte_offset, grf_chunk);   
}

template<uint32_t LOAD_W>
_GENX_ inline vector<DT_IN, LOAD_W * DPAS_INPUT_CHANNELS> load_input_nchw_and_reorder_to_wc16(SurfaceIndex surface [[type("buffer_t")]], uint byte_offset)
{
    const uint32_t LOAD_W_BYTES_WIDTH = LOAD_W * sizeof(DT_IN);
    const uint32_t LOAD_W_DWORDS = LOAD_W_BYTES_WIDTH / sizeof(uint32_t);
    
    vector<DT_IN, LOAD_W * DPAS_INPUT_CHANNELS> data_out;
    #pragma unroll
    for(int i = 0; i < DPAS_INPUT_CHANNELS; i++)
    {
        // load data
        vector<uint32_t, LOAD_W_DWORDS> load_data_dword = lsc_load<uint32_t, LOAD_W_DWORDS, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
        vector<DT_IN, LOAD_W> load_data_dt_view = load_data_dword.format<DT_IN>();  
        // reshuffle data into return registers
        data_out.select<LOAD_W, DPAS_INPUT_CHANNELS>(i) = load_data_dt_view;    
        byte_offset += INPUT_WIDTH * sizeof(DT_IN);
    }
    return data_out;
}

_GENX_ inline vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> load_filter_nchw_data(SurfaceIndex surface [[type("buffer_t")]], uint byte_offset)
{
    const uint32_t LOAD_SIZE = WEIGHTS_REG_SIZE * sizeof(DT_WEIGHTS);
    const uint32_t LOAD_SIZE_DWORDS = LOAD_SIZE / sizeof(uint32_t);
    
    vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> data_out;
    vector_ref<uint32_t, LOAD_SIZE_DWORDS> load_data_dword_view = data_out.format<uint32_t>();
    load_data_dword_view = lsc_load<uint32_t, LOAD_SIZE_DWORDS, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);   
    return data_out;
}

template<uint32_t STORE_W>
_GENX_ inline void store_output_wc8_as_nchw(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, STORE_W * DPAS_OUTPUT_CHANNELS> grf_chunk, uint byte_offset)
{    
    #pragma unroll
    for(int i = 0; i < DPAS_OUTPUT_CHANNELS; i++)
    {
        // pick data to store
        vector<DT_OUT, STORE_W> grf_chunk_store = grf_chunk.select<STORE_W, DPAS_OUTPUT_CHANNELS>(i);
        // store with non-transposed msg
        cm_store<uint32_t, 4, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
        byte_offset += OUTPUT_WIDTH * sizeof(DT_OUT);
    }
}

extern "C" _GENX_MAIN_ void convolution_nchw_1x1(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_weights [[type("buffer_t")]],
#if USE_BIAS
	SurfaceIndex surface_bias [[type("buffer_t")]],
#endif
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint global_id_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint global_id_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint global_id_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
    
    const uint LOAD_W_SIZE = 8;
    
    vector<DT_IN, LOAD_W_SIZE * INPUT_CHANNELS> input_row_0 = load_input_nchw_and_reorder_to_wc16<LOAD_W_SIZE>(surface_input, 0);
    vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_0 = load_filter_nchw_data(surface_weights, 0);
    
    const uint ACCU_REG_SIZE = LOAD_W_SIZE * DPAS_OUTPUT_CHANNELS;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0(0.0f); 
    accu_row_0 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(accu_row_0, weights_0.format<uint32_t>(), input_row_0.format<uint32_t>());
    
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0);
    store_output_wc8_as_nchw<8>(surface_output, output_row_0, 0);  
}
