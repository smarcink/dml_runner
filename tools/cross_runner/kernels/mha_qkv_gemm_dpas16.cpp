#if !defined(EMPTY)

#include <cm/cm.h>
#include <cm/cmtl.h>

#define MATH_E 2.718281828459045235360287471352f
#define FLOAT_MAX 3.402823466e+38f

#define DT half
#define DT_ACCU float
#define SIZE_OF_FP16_BYTE 2

#define CONTIGUOUS_K_SIZE 16

_GENX_ inline void myDPAS16(matrix_ref<half, 8, 16> matA, matrix_ref<half, 8, 32> matB, matrix_ref<float, 8, 16> result)
{
	result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<float>(), matB.format<U32>(), matA.format<U32>());
}

#endif

extern "C" _GENX_MAIN_ void
mha_qk_qkv_gemm_dpas16(
	uint64_t INMTXa[[type("svmptr_t half")]],  // 0 input qkv surface
	uint64_t OUTMTX[[type("svmptr_t half")]]   // 1 output qxk surface
) {
#if !defined(EMPTY)

    const uint32_t global_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t thread_b = global_z / NUM_HEADS;
	const uint32_t thread_h = global_z % NUM_HEADS;
	const uint32_t thread_seq = global_y * SEQ_LEN_BLOCK;

	const uint64_t q_base = INMTXa + SIZE_OF_FP16_BYTE * HEAD_SIZE * 3 * (thread_h + NUM_HEADS * MAX_SEQ * thread_b);
	const uint64_t k_base = q_base + SIZE_OF_FP16_BYTE * HEAD_SIZE;
	const uint64_t v_base = k_base + SIZE_OF_FP16_BYTE * HEAD_SIZE;
	const uint64_t out_base = OUTMTX + SIZE_OF_FP16_BYTE * HEAD_SIZE * (thread_h + NUM_HEADS * SEQUENCE_LENGTH * thread_b);

	vector<uint, 8> read_Q_msg;
	read_Q_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	read_Q_msg(1) = SEQUENCE_LENGTH - 1; // surface height in elements - 1
	read_Q_msg(2) = (HEAD_SIZE * 3 * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	read_Q_msg(3) = 0; // startX
	read_Q_msg(4) = thread_seq; // startY

	vector<uint, 8> read_K_msg;
	read_K_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	read_K_msg(1) = KV_SEQUENCE_LENGTH - 1; // surface height in elements - 1
	read_K_msg(2) = (HEAD_SIZE * 3 * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	read_K_msg(3) = 0; // startX
	read_K_msg(4) = 0; // startY

	vector<uint, 8> read_V_msg;
	read_V_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	read_V_msg(1) = KV_SEQUENCE_LENGTH - 1; // surface height in elements - 1
	read_V_msg(2) = (HEAD_SIZE * 3 * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	read_V_msg(3) = 0; // startX
	read_V_msg(4) = 0; // startY

	vector<uint, 8> write_msg;
	write_msg(0) = (HEAD_SIZE * SIZE_OF_FP16_BYTE) - 1; // surface width in bytes - 1
	write_msg(1) = SEQUENCE_LENGTH - 1; // surface height in elements - 1
	write_msg(2) = (HEAD_SIZE * NUM_HEADS * SIZE_OF_FP16_BYTE) - 1; // surface pitch in bytes - 1
	write_msg(3) = 0; // startX
	write_msg(4) = thread_seq; // startY

	matrix<DT, SEQ_LEN_BLOCK, HEAD_SIZE + HEAD_SIZE % CONTIGUOUS_K_SIZE > input_q;
	matrix<DT, CONTIGUOUS_K_SIZE / 2, CONTIGUOUS_K_SIZE * 2> input_k;
	matrix<DT, CONTIGUOUS_K_SIZE / 2, CONTIGUOUS_K_SIZE * 2> input_v;
	matrix<DT_ACCU, SEQ_LEN_BLOCK, CONTIGUOUS_K_SIZE> acc_s;
	input_q = 0;
	input_k = 0;
	input_v = 0;


	matrix<DT_ACCU, SEQ_LEN_BLOCK, CONTIGUOUS_K_SIZE> p;
	matrix<DT, SEQ_LEN_BLOCK, CONTIGUOUS_K_SIZE> p_half;
	vector<DT_ACCU, SEQ_LEN_BLOCK> m_prev = (0 - FLOAT_MAX);  // m --> max
	vector<DT_ACCU, SEQ_LEN_BLOCK> m_cur;                     // m --> max
	vector<DT_ACCU, SEQ_LEN_BLOCK> f = 0;                     // f --> exp(m_prev - m_cur); 
	vector<DT_ACCU, SEQ_LEN_BLOCK> l_prev = 0;                // l --> sum of exp(Xi-m)
	vector<DT_ACCU, SEQ_LEN_BLOCK> l_cur;                     // l --> sum of exp(Xi-m)
	matrix<DT_ACCU, SEQ_LEN_BLOCK, HEAD_SIZE> acc;
	acc = 0;

	// Load Q
#pragma unroll
	for (int j = 0; j < (HEAD_SIZE + CONTIGUOUS_K_SIZE - 1) / CONTIGUOUS_K_SIZE; ++j)
	{
		input_q.select<SEQ_LEN_BLOCK, 1, CONTIGUOUS_K_SIZE, 1>(0, j * CONTIGUOUS_K_SIZE).format<DT>() = cm_load<DT, CONTIGUOUS_K_SIZE, SEQ_LEN_BLOCK, 1, false, false, CacheHint::Cached, CacheHint::Cached>((DT*)q_base, read_Q_msg(0), read_Q_msg(1), read_Q_msg(2), read_Q_msg(3), read_Q_msg(4));
		read_Q_msg(3) += CONTIGUOUS_K_SIZE;
	}
	
#if HEAD_SIZE % CONTIGUOUS_K_SIZE != 0
	input_q.select<SEQ_LEN_BLOCK, 1, HEAD_SIZE % CONTIGUOUS_K_SIZE, 1>(0, HEAD_SIZE).format<DT>() = 0;
#endif
	
	// Main loop of K/V seq_len. On each iteration, CONTIGUOUS_K_SIZE lines are processed.
	for (int j = 0; j < KV_SEQUENCE_LENGTH; j += CONTIGUOUS_K_SIZE) {
		// Perform first matmul: Q*K^T
		acc_s = 0;
#pragma unroll
		for (int block = 0; block < (HEAD_SIZE + CONTIGUOUS_K_SIZE - 1) / CONTIGUOUS_K_SIZE; block++)
		{
			input_k.format<uint32_t>() = cm_load<uint32_t, 8, CONTIGUOUS_K_SIZE, 1, true, false, CacheHint::Cached, CacheHint::Cached>((uint32_t*)k_base, read_K_msg(0), read_K_msg(1), read_K_msg(2), read_K_msg(3) / 2, read_K_msg(4));
			read_K_msg(3) += CONTIGUOUS_K_SIZE;
			myDPAS16(input_q.select<8, 1, CONTIGUOUS_K_SIZE, 1>(0, block * CONTIGUOUS_K_SIZE), input_k, acc_s.select<8,1,CONTIGUOUS_K_SIZE,1>(0,0));
#if SEQ_LEN_BLOCK == 16
			myDPAS16(input_q.select<8, 1, CONTIGUOUS_K_SIZE, 1>(8, block * CONTIGUOUS_K_SIZE), input_k, acc_s.select<8,1,CONTIGUOUS_K_SIZE,1>(8,0));
#endif
		}
		read_K_msg(3) = 0;
		read_K_msg(4) += CONTIGUOUS_K_SIZE;
		acc_s *= ALPHA;

		// If past_seq_len(0)+1 is not a multiple of CONTIGUOUS_K_SIZE=16, additional masking is required
		// vector<ushort,CONTIGUOUS_K_SIZE> mask = {0,1,...,CONTIGUOUS_K_SIZE-1};
		cm_vector(mask, ushort, CONTIGUOUS_K_SIZE, 0, 1);
		mask = (j + mask >= KV_SEQUENCE_LENGTH);

		constexpr float float_min = (0 - FLOAT_MAX);
#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			acc_s.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>().merge(float_min, mask);
			m_cur(i) = cm_reduced_max<float>(acc_s.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>());
		}

		m_cur.merge(m_prev, m_prev > m_cur);

		f = cm_pow((DT_ACCU)MATH_E, (m_prev - m_cur));
		l_prev *= f;

#pragma unroll
		for (int i = 0; i < SEQ_LEN_BLOCK; ++i)
		{
			p.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>() = cm_pow((DT_ACCU)MATH_E, (acc_s.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>() - m_cur(i)));
			// Masking for p
			p.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>().merge(0, mask);
			l_cur(i) = l_prev(i) + cm_sum<float>(p.select<1, 1, CONTIGUOUS_K_SIZE, 1>(i, 0).format<DT_ACCU>());
			acc.select<1, 1, HEAD_SIZE, 1>(i, 0).format<DT_ACCU>() *= f(i);
		}

		// Conversion to datatype suitable for DPAS
		p_half = p;

		// Second matmul: P*V
#pragma unroll
		for (int block = 0; block < (HEAD_SIZE + CONTIGUOUS_K_SIZE - 1) / CONTIGUOUS_K_SIZE; block++)
		{
			input_v.format<DT>() = cm_load<DT, CONTIGUOUS_K_SIZE, CONTIGUOUS_K_SIZE, 1, false, true, CacheHint::Cached, CacheHint::Cached>((DT*)v_base, read_V_msg(0), read_V_msg(1), read_V_msg(2), read_V_msg(3), read_V_msg(4));
			if((HEAD_SIZE - read_V_msg(3)) < CONTIGUOUS_K_SIZE)
			{
				input_v.select<CONTIGUOUS_K_SIZE / 2, 1, CONTIGUOUS_K_SIZE, 1>(0, CONTIGUOUS_K_SIZE) = 0;
			}
			read_V_msg(3) += CONTIGUOUS_K_SIZE;
			myDPAS16(p_half.select<8, 1, CONTIGUOUS_K_SIZE, 1>(0,0), input_v, acc.select<8, 1, CONTIGUOUS_K_SIZE, 1>(0, block * CONTIGUOUS_K_SIZE));
#if SEQ_LEN_BLOCK == 16
			myDPAS16(p_half.select<8, 1, CONTIGUOUS_K_SIZE, 1>(8,0), input_v, acc.select<8, 1, CONTIGUOUS_K_SIZE, 1>(8, block * CONTIGUOUS_K_SIZE));
#endif
		}
		read_V_msg(3) = 0;
		read_V_msg(4) += CONTIGUOUS_K_SIZE;

		m_prev = m_cur;
		l_prev = l_cur;
	}

	matrix<DT_ACCU, SEQ_LEN_BLOCK, HEAD_SIZE> acc_out = acc;
#pragma unroll
	for (int j = 0; j < SEQ_LEN_BLOCK; ++j)
	{
		acc_out.select<1, 1, HEAD_SIZE, 1>(j, 0) /= l_prev(j);
	}
#pragma unroll
	for (int block = 0; block < (HEAD_SIZE + CONTIGUOUS_K_SIZE - 1) / CONTIGUOUS_K_SIZE; ++block)
	{
		matrix<DT, SEQ_LEN_BLOCK, CONTIGUOUS_K_SIZE> out_tile = 0;
		out_tile = acc_out.select<SEQ_LEN_BLOCK, 1, CONTIGUOUS_K_SIZE, 1>(0, block * CONTIGUOUS_K_SIZE);
		cm_store<DT, CONTIGUOUS_K_SIZE, SEQ_LEN_BLOCK, CacheHint::WriteBack, CacheHint::WriteBack>((DT*)out_base, write_msg(0), write_msg(1), write_msg(2), write_msg(3), write_msg(4), out_tile.format<DT>());
		write_msg(3) += CONTIGUOUS_K_SIZE;
	}
#endif // !defined(EMPTY)
}
