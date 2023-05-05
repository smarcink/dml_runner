#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half

#define ITEMNUM_PER_HW_PACKED ((ITEMNUM_PER_HW * sizeof(DT))/sizeof(uint32_t))
#define MATH_E 2.7182818f

extern "C" _GENX_MAIN_ void softmax_nchw(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]])
{
    const uint32_t global_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t in_out_offset = (global_x * ITEMNUM_PER_HW + global_y * INOUT_WIDTH + global_z * (INOUT_WIDTH * INOUT_HEIGHT)) * sizeof(DT);
	
	vector<uint32_t, ITEMNUM_PER_HW_PACKED> in_data_packed = cm_load<uint32_t, ITEMNUM_PER_HW_PACKED, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, in_out_offset);
	vector_ref<DT, ITEMNUM_PER_HW> my_data = in_data_packed.format<DT>();
	my_data = cm_pow(MATH_E, my_data);
	vector<float, 1> my_sum = cm_sum<float>(my_data);
	
#if LWS_SIZE_X > 1
	cm_slm_init(LWS_SIZE_X * sizeof(float));
	//vector<uint32_t, 1> offset(global_x);
	cm_store_slm<float, 1>(global_x * sizeof(float), my_sum);
	cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
    cm_barrier();
#endif
	
	vector<float, LWS_SIZE_X> all_threads_sums = cm_load_slm<float, LWS_SIZE_X>(0);
	my_sum = cm_sum<float>(all_threads_sums);
	my_data = my_data / my_sum[0];
	
	vector<uint32_t, ITEMNUM_PER_HW_PACKED> out_data_packed = in_data_packed;
    cm_store<uint32_t, ITEMNUM_PER_HW_PACKED, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, in_out_offset, out_data_packed);
}
