/*========================== begin_copyright_notice ============================

INTEL CONFIDENTIAL

Copyright (C) 2021-2023 Intel Corporation

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.

============================= end_copyright_notice ===========================*/

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_intel_subgroups_short)
#pragma OPENCL EXTENSION  cl_intel_subgroups_short : enable
#endif

    //const uint WEI_OFFSET,
    //const uint IC,
    //const uint OC,
    //const uint OC_BLOCK,
    //const uint K_SIZE,
    //const uint WEIGHT_IC_BLOCK,
    //const uint INT_BLOCK,
    //const uint DPAS_EXEC_SIZE

int align_to(const int value, const unsigned alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

inline uint get_src_offset(
    uint o,
    uint i,
    uint h,
    uint w,
    uint IC,
    uint K_SIZE)
{
    uint index = (o*IC*K_SIZE*K_SIZE) +
                (i*K_SIZE*K_SIZE) + (h*K_SIZE) + w;

    return index;
}

inline uint get_dst_offset(
    uint o,
    uint i,
    uint h,
    uint w,
    uint OC,
    uint OC_BLOCK,
    uint K_SIZE,
    uint WEIGHT_IC_BLOCK,
    uint INT_BLOCK,
    uint DPAS_EXEC_SIZE)
{
    unsigned alignedOC_BLOCK = align_to(OC, OC_BLOCK);
    // // Memory layout: (IC_CHUNK)_Y_X_(OC_TILES)_DPAS_DEPTH_(DPAS_EXEC_SIZE)_(INT_BLOCK)
    unsigned ic_chunk_offset = (i / WEIGHT_IC_BLOCK) * K_SIZE * K_SIZE * alignedOC_BLOCK * WEIGHT_IC_BLOCK;
    unsigned y_offset = h * K_SIZE * alignedOC_BLOCK * WEIGHT_IC_BLOCK;
    unsigned x_offset = w * WEIGHT_IC_BLOCK * alignedOC_BLOCK;

    unsigned ic_offset = i % INT_BLOCK + ((i / INT_BLOCK) % (WEIGHT_IC_BLOCK / INT_BLOCK)) * DPAS_EXEC_SIZE * INT_BLOCK;
    unsigned oc_offset = (o % DPAS_EXEC_SIZE) * INT_BLOCK + (o / DPAS_EXEC_SIZE) * WEIGHT_IC_BLOCK * DPAS_EXEC_SIZE;

    return ic_chunk_offset + y_offset + x_offset + ic_offset + oc_offset;
}

//__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void Reorder_weights_cm_1x1(const __global INPUT_TYPE* input, __global OUTPUT_TYPE* output)
{
    // weigtht dimension is OC x IC x K_SIZE x K_SIZE
   // adjust the offset
    output += WEI_OFFSET;
    uint oc = get_global_id(0);
    uint ic = get_global_id(1);

    if (oc >= OC || ic >= IC)
        return;

    for (uint h = 0; h < K_SIZE; h++) {
        for (uint w = 0; w < K_SIZE; w++) {
            int srcoff = get_src_offset(oc, ic, h, w, IC, K_SIZE);
            int dstoff = get_dst_offset(oc, ic, h, w, OC, OC_BLOCK, K_SIZE, WEIGHT_IC_BLOCK, INT_BLOCK, DPAS_EXEC_SIZE);
            output[dstoff] = input[srcoff];
            //printf ("oc %d, ic %d, srcoffset %d, dstoffset %d\n", oc, ic, srcoff, dstoffset);
        }
    }
}
