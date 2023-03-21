/*========================== begin_copyright_notice ============================

INTEL CONFIDENTIAL

Copyright (C) 2023 Intel Corporation

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>
#include <cm/cmtl.h>

#if !CM_HAS_LSC
#error [Error_device_no_lsc] Kernel designed to use lsc. Current device does not support lsc.
#endif

#define DPAS_DEPTH 8 
#if(CM_GENX >= 1280)
#define EXEC_SIZE 16
#else
#define EXEC_SIZE 8
#endif

#if IC_CHUNK_SIZE != 16 && IC_CHUNK_SIZE != 32 && IC_CHUNK_SIZE != 64 && IC_CHUNK_SIZE != 128 
#error [Error param] Not tested ic chunk size. Probably this case works, but should be validated first.
#endif

#if OC_CHUNK_SIZE != 8
#error [Error param] Not tested oc chunk size. Probably this case works, but should be validated first.
#endif

#define WEIGHTS_IC_OFSET sizeof(INPUT_TYPE)
#define WEIGHTS_OC_OFSET (IC * WEIGHTS_IC_OFSET)

#define IC_PER_HW_THREAD (IC_CHUNK_SIZE * IC_CHUNKS_PER_HW_THREAD)
#define IC_PER_HW_THREAD_PACKED ((IC_PER_HW_THREAD * sizeof(INPUT_TYPE)) / sizeof(uint32_t))

static const uint32_t weights_init_offsets[] = {
                                                0 * WEIGHTS_OC_OFSET, 0 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                1 * WEIGHTS_OC_OFSET, 1 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                2 * WEIGHTS_OC_OFSET, 2 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                3 * WEIGHTS_OC_OFSET, 3 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                4 * WEIGHTS_OC_OFSET, 4 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                5 * WEIGHTS_OC_OFSET, 5 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                6 * WEIGHTS_OC_OFSET, 6 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                7 * WEIGHTS_OC_OFSET, 7 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                };



extern "C" _GENX_MAIN_ void weights_reorder(SurfaceIndex surface_input [[type("buffer_t")]], SurfaceIndex surface_output [[type("buffer_t")]])
{
    const uint thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
    
    
    const uint32_t int_block = (sizeof(uint32_t) / sizeof(OUTPUT_TYPE));
    const uint32_t dpas_input_channels = DPAS_DEPTH * int_block;
    
    const uint32_t oc = thread_id_0 * EXEC_SIZE;
    const uint32_t ic = thread_id_1 * IC_PER_HW_THREAD;
    
    const uint chunks_count = EXEC_SIZE;
    // load
    matrix<uint32_t, chunks_count, IC_PER_HW_THREAD_PACKED> data_input_typed;
    uint32_t input_offset = (oc * IC + ic) * sizeof(INPUT_TYPE);
    #pragma unroll
    for(int i = 0; i < chunks_count; i++)
    {
        data_input_typed.row(i) = cm_load<uint32_t, IC_PER_HW_THREAD_PACKED, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, input_offset);
        input_offset += IC * sizeof(INPUT_TYPE);
    }
    matrix_ref<INPUT_TYPE, chunks_count, IC_PER_HW_THREAD> data_input = data_input_typed.format<INPUT_TYPE, chunks_count, IC_PER_HW_THREAD>();
    
    uint32_t output_offset = (oc * dpas_input_channels + ic * OC) * sizeof(INPUT_TYPE);  
    vector<OUTPUT_TYPE, EXEC_SIZE * dpas_input_channels> data_out;
    #pragma unroll
    for(int i = 0; i < IC_PER_HW_THREAD/dpas_input_channels; i++)
    {
        #pragma unroll
        for(int j = 0; j < EXEC_SIZE; j++)
        {
            data_out.select<dpas_input_channels, 1>(j * dpas_input_channels) = data_input.select<EXEC_SIZE, 1, int_block, 1>(0, int_block * j + i * dpas_input_channels);
        }
        const uint32_t packed_size = (EXEC_SIZE * dpas_input_channels)/2;
        cm_store<uint32_t, packed_size>(surface_output, output_offset, data_out.format<uint32_t>());
        output_offset += OC * dpas_input_channels * sizeof(OUTPUT_TYPE);
    }
}
