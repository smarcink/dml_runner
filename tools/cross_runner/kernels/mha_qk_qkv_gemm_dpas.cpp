#include <cm/cm.h>
#include <cm/cmtl.h>

#if !defined(EMPTY)

#define SIZE_OF_HF16_BYTE 2

#define WG_TILE_M 64
#define WG_TILE_N 64

#define SG_TILE_M 8
#define SG_TILE_N 32

#define SG_TILE_NUM_ROWS 8
#define HALF half
#define FLOAT float
#define DIM_X 0
#define DIM_Y 1
#define DIM_Z 2
#define BASE_OUTPUT_OFFSET 0

_GENX_ inline void myDPAS8(matrix_ref<HALF, 8, 16> matA,
                            matrix_ref<HALF, 8, 16> matB,
                            matrix_ref<FLOAT, 8, 8> result)
{
	result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<FLOAT>(), matB.format<U32>(), matA.format<U32>());
}

#endif

extern "C" _GENX_MAIN_ void
mha_qk_qkv_gemm_dpas(SurfaceIndex INMTXa[[type("buffer_t half")]], // 0 input qkv surface
         SurfaceIndex OUTMTX[[type("buffer_t half")]]       // 1 output qxk surface
) {
#if !defined(EMPTY)
   
	//A matrix format: [K/16][M][16k]
	//A tile: 32Mx16K
	vector<HALF, 128> readA1;//M=0..7,K=0..15
	matrix_ref<HALF, 8, 16> readA1_m = readA1.format<HALF, 8, 16>();
			
	//B matrix format: [K/16][N/8][8K][8N][2K]
	//B tile: 32Nx16K
	matrix<HALF, 8, 16> readB1;//N=0..7,K=0..15
	matrix<HALF, 8, 16> readB2;//N=8..15,K=0..15
	matrix<HALF, 8, 16> readB3;//N=16..23,K=0..15
	matrix<HALF, 8, 16> readB4;//N=24..32,K=0..15
	
	matrix_ref<HALF, 8, 16> readB1_m = readB1.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB2_m = readB2.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB3_m = readB3.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB4_m = readB4.format<HALF, 8, 16>();
	
	matrix<FLOAT, 8, 8> result11;
    matrix<FLOAT, 8, 8> result12;
    matrix<FLOAT, 8, 8> result13;
    matrix<FLOAT, 8, 8> result14;

    matrix_ref<FLOAT, 8, 8> result11ref = result11;
    matrix_ref<FLOAT, 8, 8> result12ref = result12;
    matrix_ref<FLOAT, 8, 8> result13ref = result13;
    matrix_ref<FLOAT, 8, 8> result14ref = result14;

    uint gidY = cm_group_id(DIM_Y);
    uint gidX = cm_group_id(DIM_X);
	uint gidZ = cm_group_id(DIM_Z);
    uint tidX = cm_local_id(DIM_X);
    uint tidY = cm_local_id(DIM_Y);
	uint tidZ = cm_local_id(DIM_Z);
	
	//input surface control variables
	const unsigned input_head_count_stride_qkv = SIZE_HEAD_SIZE * 3;
	const unsigned input_sequence_stride_qkv = SIZE_NUM_HEADS * input_head_count_stride_qkv;
	const unsigned input_batch_stride_qkv = SIZE_SEQ_LEN * input_sequence_stride_qkv;
	const unsigned input_surface_tile_base_offset_q = tidX * SG_TILE_M + gidX * WG_TILE_M;
	const unsigned input_surface_tile_base_offset_k = tidY * SG_TILE_N + gidY * WG_TILE_N;
	const unsigned input_cacheline_stride = input_sequence_stride_qkv * SIZE_OF_HF16_BYTE;
	const unsigned input_cacheline_stride_rows = SG_TILE_NUM_ROWS * input_cacheline_stride;
	
	//output surface control variables
	const unsigned output_head_count_stride = SIZE_SEQ_LEN * SIZE_SEQ_LEN;
	const unsigned output_batch_stride = SIZE_NUM_HEADS * output_head_count_stride;
	const unsigned output_tile_base_offset = ((tidX * SG_TILE_M + gidX * WG_TILE_M) * SIZE_SEQ_LEN) + (tidY * SG_TILE_N + gidY * WG_TILE_N);
	//const int sg_tile_base_offset = tidX * SG_TILE_M * SIZE_SEQ_LEN + tidY * SG_TILE_N;
	
	const unsigned batch_count = gidZ/SIZE_NUM_HEADS;
	const unsigned head_count = gidZ % SIZE_NUM_HEADS;
	const unsigned input_surface_batch_start_offset = input_batch_stride_qkv * batch_count;
	const unsigned output_surface_batch_start_offset = output_batch_stride * batch_count;
    
	const unsigned head_start_offset = output_head_count_stride * head_count;
	const unsigned output_surface_tile_start_offset = BASE_OUTPUT_OFFSET + output_surface_batch_start_offset + head_start_offset + output_tile_base_offset; // + sg_tile_base_offset;
	const unsigned input_surface_tile_start_offset_q = head_count * input_head_count_stride_qkv + batch_count * input_surface_batch_start_offset;
	const unsigned input_surface_tile_start_offset_k = head_count * input_head_count_stride_qkv + batch_count * input_surface_batch_start_offset + SIZE_HEAD_SIZE;
	
	//init the accumulators
	result11.select_all() = 0;
	result12.select_all() = 0;
	result13.select_all() = 0;
	result14.select_all() = 0;
	
	for (int head_size_step = 0; head_size_step < SIZE_HEAD_SIZE; head_size_step += 16) //iterates to process the entire K for A and B.
	{
		const bool is_head_size_step_multiple_16 = (head_size_step + 16 <= SIZE_HEAD_SIZE) ? true : false;
		const unsigned input_surface_tile_start_q = (input_surface_tile_start_offset_q + head_size_step + input_surface_tile_base_offset_q * input_sequence_stride_qkv) * SIZE_OF_HF16_BYTE;;
		const unsigned input_surface_tile_start_k = (input_surface_tile_start_offset_k + head_size_step + input_surface_tile_base_offset_k * input_sequence_stride_qkv) * SIZE_OF_HF16_BYTE;;
		readA1.select_all() = 0.0;
		readB1.select_all() = 0.0;
		readB2.select_all() = 0.0;
		readB3.select_all() = 0.0;
		readB4.select_all() = 0.0;

		#pragma unroll
		for (int row = 0; row < SG_TILE_NUM_ROWS; row++)
		{
			const unsigned row_offset_in_bytes = row * SG_TILE_NUM_ROWS * SIZE_OF_HF16_BYTE;
			const unsigned rowX2 = row * 2;
			const unsigned input_surface_strride = row * input_cacheline_stride;
			const unsigned input_surface_q_CL1 = input_surface_tile_start_q + input_surface_strride;
			const unsigned input_surface_k_CL1 = input_surface_tile_start_k + input_surface_strride;
			const unsigned input_surface_k_CL2 = input_surface_k_CL1 + input_cacheline_stride_rows;
			const unsigned input_surface_k_CL3 = input_surface_k_CL2 + input_cacheline_stride_rows;
			const unsigned input_surface_k_CL4 = input_surface_k_CL3 + input_cacheline_stride_rows;
			
			// 8M x 16K (one complete row per iteration)
			if(is_head_size_step_multiple_16)
			{
				readA1.select<16,1>(row_offset_in_bytes).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_q_CL1);
				readB1.select<8,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL1);
				readB2.select<8,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL2);
				readB3.select<8,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL3);
				readB4.select<8,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL4);
			}
			else
			{
				readA1.select<8,1>(row_offset_in_bytes).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_q_CL1);
				readB1.select<4,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL1);
				readB2.select<4,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL2);
				readB3.select<4,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL3);
				readB4.select<4,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(INMTXa, input_surface_k_CL4);
			}
		}
		
		myDPAS8(readA1_m, readB1_m, result11ref);
		myDPAS8(readA1_m, readB2_m, result12ref);
		myDPAS8(readA1_m, readB3_m, result13ref);
		myDPAS8(readA1_m, readB4_m, result14ref);
	}
	
	vector<HALF, 32> result_hf16_CL1 = 0.0;
	result11 *= HALF(ALPHA);
	result12 *= HALF(ALPHA);
	result13 *= HALF(ALPHA);
	result14 *= HALF(ALPHA);
	
	#pragma unroll
	for(int j = 0; j < SG_TILE_NUM_ROWS; j++)
	{
		const unsigned write_index_base = j * SIZE_SEQ_LEN;
		const unsigned write_index_0 = (output_surface_tile_start_offset + write_index_base) * SIZE_OF_HF16_BYTE;
		
		result_hf16_CL1.select<8, 1>(0)  = result11ref.select<1, 1, 8, 1>(j, 0);
		result_hf16_CL1.select<8, 1>(8)  = result12ref.select<1, 1, 8, 1>(j, 0);
		result_hf16_CL1.select<8, 1>(16) = result13ref.select<1, 1, 8, 1>(j, 0);
		result_hf16_CL1.select<8, 1>(24) = result14ref.select<1, 1, 8, 1>(j, 0);
		
		cm_store<U32, 16, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(OUTMTX, write_index_0, result_hf16_CL1.format<U32>());
	}
#endif // !defined(EMPTY)
}
