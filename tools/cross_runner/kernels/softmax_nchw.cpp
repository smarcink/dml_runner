#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define DT_ACCU float 

#define MATH_E 2.7182818f

#if INOUT_WIDTH == 77
#define NON_ALIGNED_WIDTH 1
#define ALIGNED_WIDTH 80
#define ALIGNED_WIDTH_LOAD_SIZE 16
#endif

#define ITEMNUM_PER_HW_PACKED ((ITEMNUM_PER_HW * sizeof(DT))/sizeof(uint32_t))

static const int32_t init_linear_offsets[] = {  0  * sizeof(DT),
											    1  * sizeof(DT), 
											    2  * sizeof(DT),
											    3  * sizeof(DT),
											    4  * sizeof(DT),
											    5  * sizeof(DT),
											    6  * sizeof(DT),
											    7  * sizeof(DT),
												8  * sizeof(DT), 
											    9  * sizeof(DT),
											    10 * sizeof(DT),
											    11 * sizeof(DT),
											    12 * sizeof(DT),
											    13 * sizeof(DT),
											    14 * sizeof(DT),
											    15 * sizeof(DT),
											  };

extern "C" _GENX_MAIN_ void softmax_nchw(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]])
{
    const uint32_t global_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t in_out_offset = (global_x * ITEMNUM_PER_HW + global_y * INOUT_WIDTH + global_z * (INOUT_WIDTH * INOUT_HEIGHT)) * sizeof(DT);
	
#if NON_ALIGNED_WIDTH
	vector<ushort, ALIGNED_WIDTH> predicate(1);
	predicate.select<ALIGNED_WIDTH-INOUT_WIDTH, 1>(INOUT_WIDTH) = 0;

	vector<uint32_t, ALIGNED_WIDTH_LOAD_SIZE> load_offsets(init_linear_offsets);
	load_offsets += in_out_offset;
	vector<DT, ALIGNED_WIDTH> my_data_load;
	#pragma unroll
	for(int i = 0; i < (ALIGNED_WIDTH / ALIGNED_WIDTH_LOAD_SIZE); i++)
	{
		my_data_load.select<ALIGNED_WIDTH_LOAD_SIZE, 1>(i * ALIGNED_WIDTH_LOAD_SIZE) = cm_load<DT, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface_input, load_offsets, predicate.select<ALIGNED_WIDTH_LOAD_SIZE, 1>(i * ALIGNED_WIDTH_LOAD_SIZE));
		load_offsets += ALIGNED_WIDTH_LOAD_SIZE * sizeof(DT);
	}
	vector_ref<DT, ITEMNUM_PER_HW> my_data = my_data_load.select<ITEMNUM_PER_HW, 1>();
#else
	vector<uint32_t, ITEMNUM_PER_HW_PACKED> in_data_packed = cm_load<uint32_t, ITEMNUM_PER_HW_PACKED, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, in_out_offset);
	vector_ref<DT, ITEMNUM_PER_HW> my_data = in_data_packed.format<DT>();
#endif

	
	// do the local (hw) reduce 
	my_data = cm_pow(MATH_E, my_data);
	vector<DT_ACCU, 1> my_sum = cm_sum<DT_ACCU>(my_data);
	
#if LWS_SIZE_X > 1
	// store to slm to share partially reduced sums
	cm_slm_init(LWS_SIZE_X * sizeof(DT_ACCU));
	cm_store_slm<DT_ACCU, 1>(global_x * sizeof(DT_ACCU), my_sum);
	cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
    cm_barrier();
	
	// read from slm and further reduce
	vector<DT_ACCU, LWS_SIZE_X> all_threads_sums = cm_load_slm<DT_ACCU, LWS_SIZE_X>(0);
	my_sum = cm_sum<DT_ACCU>(all_threads_sums);
#endif
	
	// do the division in full preicison
	vector<DT_ACCU, ITEMNUM_PER_HW> my_data_accu = my_data;
	my_data_accu = my_data_accu / my_sum[0];
	
	// cast back to inout data type
	my_data = my_data_accu;
	
	// for(int i = 0; i < ALIGNED_WIDTH; i++)
	// {
		// printf("[%d] %f\n", i, (float)my_data[i]);
	// }
	
	
	// store results
#if NON_ALIGNED_WIDTH
	vector<uint32_t, ALIGNED_WIDTH_LOAD_SIZE> store_offsets(init_linear_offsets);
	store_offsets += in_out_offset;
	
	#pragma unroll
	for(int i = 0; i < (ALIGNED_WIDTH / ALIGNED_WIDTH_LOAD_SIZE); i++)
	{
		cm_store<DT, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, store_offsets, my_data.select<ALIGNED_WIDTH_LOAD_SIZE, 1>(i * ALIGNED_WIDTH_LOAD_SIZE), predicate.select<ALIGNED_WIDTH_LOAD_SIZE, 1>(i * ALIGNED_WIDTH_LOAD_SIZE));
		store_offsets += ALIGNED_WIDTH_LOAD_SIZE * sizeof(DT);
	}
#else
	vector_ref<uint32_t, ITEMNUM_PER_HW_PACKED> out_data_packed = in_data_packed;
    cm_store<uint32_t, ITEMNUM_PER_HW_PACKED, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, in_out_offset, out_data_packed);
#endif
}
