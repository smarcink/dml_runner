#include <cm/cm.h>
#include <cm/cmtl.h>



extern "C" _GENX_MAIN_ void mvn_nchw(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]])
{
    const uint32_t global_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t global_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t global_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
}
