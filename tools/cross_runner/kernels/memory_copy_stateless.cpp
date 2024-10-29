#include <cm/cm.h>
#include <cm/cmtl.h>

#define DT half
#define ITEMNUM_PER_HW_PACKED ((ITEMS_PER_HW * sizeof(DT))/sizeof(uint32_t))

extern "C" _GENX_MAIN_ void memory_copy(
    uint64_t surface_input [[type("svmptr_t half")]],
    uint64_t surface_output [[type("svmptr_t half")]]
)
{
    const uint32_t thread_id = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t in_out_offset = thread_id * ITEMS_PER_HW * sizeof(DT);
	vector<uint32_t, ITEMNUM_PER_HW_PACKED> data_packed  = cm_ptr_load<uint32_t, ITEMNUM_PER_HW_PACKED, DataSize::Default, CacheHint::Cached, CacheHint::Cached>((uint32_t*)surface_input, in_out_offset);
	cm_ptr_store<uint32_t, ITEMNUM_PER_HW_PACKED, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>((uint32_t*)surface_output, in_out_offset, data_packed);
}