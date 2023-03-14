#include <cm/cm.h>
#include <cm/cmtl.h>

#define DATASET_SIZE (INOUT_WIDTH * INOUT_HEIGHT)
#define DT half

#define ITEMNUM 128
#define ITEMNUM_PACKED ((ITEMNUM * sizeof(DT))/sizeof(uint32_t))

extern "C" _GENX_MAIN_ void mvn_nchw(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
#if USE_BIAS
	, SurfaceIndex surface_bias [[type("buffer_t")]]
#endif
#if USE_SCALE
	, SurfaceIndex surface_scale [[type("buffer_t")]]
#endif
)
{
    const uint32_t batch = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t channel = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t dataset_chunk = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
    
    const uint32_t dataset_id = batch * INOUT_CHANNELS + channel;
    const uint32_t inout_offset = dataset_id * DATASET_SIZE * sizeof(DT) + dataset_chunk * ITEMNUM* sizeof(DT);

    vector<uint32_t, ITEMNUM_PACKED> in_data_packed = cm_load<uint32_t, ITEMNUM_PACKED, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, inout_offset);
    vector_ref<DT, ITEMNUM> in_data = in_data_packed.format<DT>();
    DT mean = cm_sum<DT>(in_data) / DATASET_SIZE;
    
    // variance
    vector<DT, ITEMNUM> variance_data = in_data;
    variance_data -= mean;
    variance_data += variance_data * variance_data;
    DT variance = cm_sum<DT>(variance_data) / DATASET_SIZE;
    
    vector<DT, ITEMNUM> numerator = in_data - mean;
    vector<DT, ITEMNUM> denominator = cm_sqrt(variance + (DT)EPSILON);
    
    vector<DT, ITEMNUM> data_out =  numerator / denominator; 
    
#if USE_SCALE
    vector<DT,1> scale;
    read(surface_scale, 0, channel, scale);
    data_out *= scale.replicate<ITEMNUM>();
#endif
 
#if USE_BIAS
    vector<DT,1> bias;
    read(surface_bias, 0, channel, bias);
    data_out += bias.replicate<ITEMNUM>();
#endif
 
    vector_ref<uint32_t, ITEMNUM_PACKED> data_out_packed = data_out.format<uint32_t>();    
    cm_store<uint32_t, ITEMNUM_PACKED, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, inout_offset, data_out_packed);
}
