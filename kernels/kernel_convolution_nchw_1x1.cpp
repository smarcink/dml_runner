#include <cm/cm.h>
#include <cm/cmtl.h>

#define USE_BIAS 0

extern "C" _GENX_MAIN_ void convolution_nchw_1x1(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_weights [[type("buffer_t")]],
#if USE_BIAS
	SurfaceIndex surface_weights [[type("buffer_t")]],
#endif
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint global_id_x = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint global_id_y = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint global_id_z = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
    
    vector<uint, 4> data;
    data[0] = global_id_x;
    data[1] = global_id_y;
    data[2] = global_id_z;
    cm_store<uint, 4>(surface_output, 0, data);
}
