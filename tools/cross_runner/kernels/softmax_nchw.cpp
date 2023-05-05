#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define ITEMNUM 64
#define ITEMNUM_PACKED ((ITEMNUM * sizeof(DT))/sizeof(uint32_t))
#define MATH_E 2.7182818f

extern "C" _GENX_MAIN_ void softmax_nchw(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]])
{
    const uint32_t global_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	
	vector<uint32_t, ITEMNUM_PACKED> in_data_packed = cm_load<uint32_t, ITEMNUM_PACKED, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, 0);
	vector_ref<DT, ITEMNUM> my_data = in_data_packed.format<DT>();
	my_data = cm_pow(MATH_E, my_data);
	DT my_sum = cm_sum<DT>(my_data);
	
	my_data = my_data / my_sum;
	
	vector<uint32_t, ITEMNUM_PACKED> out_data_packed = in_data_packed;
    cm_store<uint32_t, ITEMNUM_PACKED, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, 0, out_data_packed);
}
