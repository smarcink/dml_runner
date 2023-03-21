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

#define DPAS_OUTPUT_CHANNELS EXEC_SIZE
#define DPAS_INPUT_CHANNELS 16
#define WEIGHTS_IC_OFSET sizeof(INPUT_TYPE)
#define WEIGHTS_OC_OFSET (IC * WEIGHTS_IC_OFSET)

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
    
    const uint32_t oc = thread_id_0;
    const uint32_t ic = thread_id_1;

    const uint32_t int_block = (sizeof(uint32_t) / sizeof(OUTPUT_TYPE));
    const uint32_t dpas_input_channels = DPAS_DEPTH * int_block;
    
    // load
    vector<uint32_t, 16> input_offsets(weights_init_offsets);
    input_offsets += (ic * int_block * WEIGHTS_IC_OFSET) + oc * EXEC_SIZE * WEIGHTS_OC_OFSET;
    vector<INPUT_TYPE, 16> data_load = cm_load<half, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface_input, input_offsets); 

    // store
    const uint32_t ic_tile_id = ic / DPAS_DEPTH;
    const uint32_t ic_chunk_id = ic % DPAS_DEPTH;
    
    const uint32_t ic_tile_offset = ic_tile_id * OC * dpas_input_channels;  // move by full OC and ic chunk size (which is dpas depth * int block)
    const uint32_t ic_chunk_offset = ic_chunk_id * EXEC_SIZE * int_block;   // chunk within tile (dpas depth number of ic * int block)
    const uint32_t oc_offset = oc * EXEC_SIZE * DPAS_DEPTH * int_block;      // move by size needed for a single dpas (dg2: 64 uints)
    const uint32_t output_offset = (ic_tile_offset + ic_chunk_offset + oc_offset) * sizeof(OUTPUT_TYPE);
    cm_store<uint32_t, 8>(surface_output, output_offset, data_load.format<uint32_t>());
}
