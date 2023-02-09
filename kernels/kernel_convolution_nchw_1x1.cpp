#include <cm/cm.h>
#include <cm/cmtl.h>

#if !CM_HAS_DPAS
#error [Error_device_no_dpas] Kernel designed to use dpas. Current device does not support dpas.
#endif

#if !CM_HAS_LSC
#error [Error_device_no_lsc] Kernel designed to use lsc. Current device does not support lsc.
#endif

#if BLOCK_H != 1 && BLOCK_H != 4
#error [Error_kernel_config_bad_block_h] Kernel designed to work with block_h = {1, 4}.
#endif

#define WIDTH_LEFTOVER 1

#define EXEC_SIZE 8
#define DPAS_DEPTH 8 

#define DT_OUT half
#define DT_IN half
#define DT_WEIGHTS half
// accu on DG2 have to be float for half dt inputs
#define DT_ACCU float 

#define DPAS_INPUT_CHANNELS (DPAS_DEPTH * sizeof(DT_IN))
#define DPAS_OUTPUT_CHANNELS EXEC_SIZE

#define CONV_LOOP_COUNT (INPUT_CHANNELS/DPAS_INPUT_CHANNELS)

#define WEIGHTS_REG_SIZE (DPAS_INPUT_CHANNELS * DPAS_OUTPUT_CHANNELS)

static const uint32_t store_init_offsets_for_halfs[8] = { 0, 16, 32, 48, 64, 80, 96, 112 };


template<uint32_t LOAD_W>
_GENX_ inline vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> load_input_nchw_and_reorder_to_wc16(SurfaceIndex surface [[type("buffer_t")]], uint byte_offset)
{
    const uint32_t LOAD_W_BYTES_WIDTH = LOAD_W * sizeof(DT_IN);
    const uint32_t LOAD_W_DWORDS = LOAD_W_BYTES_WIDTH / sizeof(uint32_t);
    
    vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> data_out;
    // non transposed scattered reads
    vector<uint32_t, DPAS_INPUT_CHANNELS> offsets = 
    {
        0 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        1 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        2 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        3 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        4 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        5 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        6 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        7 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        8 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        9 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        10 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        11 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        12 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        13 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        14 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
        15 * INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN),
    };
    offsets += byte_offset;
    #pragma unroll
    for(int i = 0; i < LOAD_W; i++)
    {
        // pick registers
        vector_ref<DT_IN, DPAS_INPUT_CHANNELS> grf_chunk_store = data_out.select<DPAS_INPUT_CHANNELS, 1>(i * DPAS_INPUT_CHANNELS);
        // read with non-transposed msg
        grf_chunk_store = cm_load<half, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets);
        offsets += sizeof(DT_IN);  // move by one element
    } 
    #pragma unroll
    for(int i = LOAD_W; i < BLOCK_W; i++)
    {
        vector_ref<DT_IN, DPAS_INPUT_CHANNELS> grf_chunk_store = data_out.select<DPAS_INPUT_CHANNELS, 1>(i * DPAS_INPUT_CHANNELS);
        grf_chunk_store = DT_IN(0);
    }

    return data_out;
}

_GENX_ inline vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> load_filter_nchw_data(SurfaceIndex surface [[type("buffer_t")]], uint byte_offset)
{
    const uint32_t LOAD_SIZE = WEIGHTS_REG_SIZE * sizeof(DT_WEIGHTS);
    const uint32_t LOAD_SIZE_DWORDS = LOAD_SIZE / sizeof(uint32_t);
    
    vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> data_out;
    vector_ref<uint32_t, LOAD_SIZE_DWORDS> load_data_dword_view = data_out.format<uint32_t>(); 
    load_data_dword_view = cm_load<uint32_t, LOAD_SIZE_DWORDS, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);   
    return data_out;
}

template<uint32_t STORE_W>
_GENX_ inline void store_output_wc8_as_nchw(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, BLOCK_W * DPAS_OUTPUT_CHANNELS> grf_chunk, uint byte_offset)
{    
    // non transposed scattered writes
    vector<uint32_t, DPAS_OUTPUT_CHANNELS> offsets = 
    {
        0 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
        1 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
        2 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
        3 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
        4 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
        5 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
        6 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
        7 * OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT),
    };
    offsets += byte_offset;
    #pragma unroll
    for(int i = 0; i < STORE_W; i++)
    {
        // pick data to store
        vector<DT_OUT, DPAS_OUTPUT_CHANNELS> grf_chunk_store = grf_chunk.select<DPAS_OUTPUT_CHANNELS, 1>(i * DPAS_OUTPUT_CHANNELS);
        // store with non-transposed msg
        cm_store<half, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store);
        offsets += sizeof(DT_OUT);  // move by one element
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
    const uint h_chunk_id = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint oc_chunk_id = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
         
    const uint32_t input_row_offset_size = INPUT_WIDTH * sizeof(DT_IN);
    const uint32_t input_dpas_ic_offset_size = INPUT_HEIGHT * DPAS_INPUT_CHANNELS * input_row_offset_size;
    
    const uint input_h_chunk_offset = h_chunk_id * BLOCK_H * input_row_offset_size;
    uint32_t input_offset = input_h_chunk_offset;
        
    uint32_t weights_offset = 0;
    const uint weights_nchw_dpas_ic_offset_size = DPAS_INPUT_CHANNELS * sizeof(DT_WEIGHTS);
    
    vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> input_row_0 = load_input_nchw_and_reorder_to_wc16<8>(surface_input, input_offset);
#if BLOCK_H == 4
    vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> input_row_1 = load_input_nchw_and_reorder_to_wc16<BLOCK_W>(surface_input, input_offset + input_row_offset_size);
    vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> input_row_2 = load_input_nchw_and_reorder_to_wc16<BLOCK_W>(surface_input, input_offset + 2 * input_row_offset_size);
    vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> input_row_3 = load_input_nchw_and_reorder_to_wc16<BLOCK_W>(surface_input, input_offset + 3 * input_row_offset_size);
#endif
    vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_0 = load_filter_nchw_data(surface_weights, weights_offset);
    
    const uint ACCU_REG_SIZE = BLOCK_W * DPAS_OUTPUT_CHANNELS;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_1;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_2;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_3;
    accu_row_0 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8, float, uint, uint, 64>(0, weights_0.format<uint32_t>(), input_row_0.format<uint32_t>());
#if BLOCK_H == 4
    accu_row_1 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8, float, uint, uint, 64>(0, weights_0.format<uint32_t>(), input_row_1.format<uint32_t>());
    accu_row_2 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8, float, uint, uint, 64>(0, weights_0.format<uint32_t>(), input_row_2.format<uint32_t>());
    accu_row_3 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8, float, uint, uint, 64>(0, weights_0.format<uint32_t>(), input_row_3.format<uint32_t>());
#endif    
    #pragma unroll
    for(int i = 1; i < CONV_LOOP_COUNT; i++)
    {
        input_offset += input_dpas_ic_offset_size;
        weights_offset += weights_nchw_dpas_ic_offset_size;
        
        input_row_0 = load_input_nchw_and_reorder_to_wc16<8>(surface_input, input_offset);
#if BLOCK_H == 4
        input_row_1 = load_input_nchw_and_reorder_to_wc16<BLOCK_W>(surface_input, input_offset + input_row_offset_size);
        input_row_2 = load_input_nchw_and_reorder_to_wc16<BLOCK_W>(surface_input, input_offset + 2 * input_row_offset_size);
        input_row_3 = load_input_nchw_and_reorder_to_wc16<BLOCK_W>(surface_input, input_offset + 3 * input_row_offset_size);
#endif
        weights_0 = load_filter_nchw_data(surface_weights, weights_offset);
        accu_row_0 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(accu_row_0, weights_0.format<uint32_t>(), input_row_0.format<uint32_t>());
#if BLOCK_H == 4
        accu_row_1 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(accu_row_1, weights_0.format<uint32_t>(), input_row_1.format<uint32_t>());
        accu_row_2 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(accu_row_2, weights_0.format<uint32_t>(), input_row_2.format<uint32_t>());
        accu_row_3 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(accu_row_3, weights_0.format<uint32_t>(), input_row_3.format<uint32_t>());
#endif
    }
    
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0);
#if BLOCK_H == 4
    vector<DT_OUT, ACCU_REG_SIZE> output_row_1 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_1);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_2 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_2);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_3 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_3);
#endif
#if USE_BIAS
#error ToDo: add support for use_bias case here.
#endif 
    
    const uint output_row_offset_size = OUTPUT_WIDTH * sizeof(DT_OUT);
    const uint output_oc_chunk_offset = oc_chunk_id * DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * output_row_offset_size;
    const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * output_row_offset_size;
    uint32_t output_offset = output_oc_chunk_offset + output_h_chunk_offset;
    store_output_wc8_as_nchw<8>(surface_output, output_row_0, output_offset);  
#if BLOCK_H == 4
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_1, output_offset + output_row_offset_size);  
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_2, output_offset + 2 * output_row_offset_size);  
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_3, output_offset + 3 * output_row_offset_size);  
#endif
}
