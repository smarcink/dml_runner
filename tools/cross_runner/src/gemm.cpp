#include "gemm.h"
#include "dnnl_utils.h"

std::vector<std::byte> dnnl_gemm_op::gemm(const bindings_t& bindings, opts_t opts)
{
    using namespace dnnl_utils;
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    static dnnl::stream stream(engine);
    const auto engine_kind = engine.get_kind();
    stream.wait();  // just to be sure we can freely upload the input data   


    dnnl::memory input_memory = [&](const auto& binding)
    {
        const auto dims = to_dnnl_dims(binding.shape);
        const auto dt = to_dnnl_data_type(binding.dt);
        const auto ft = to_dnnl_format(binding.layout);
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_a);

    dnnl::memory weights_memory = [&](const auto& binding)
    {
        const auto dims = to_dnnl_dims(binding.shape);
        const auto dt = to_dnnl_data_type(binding.dt);
        const auto ft = to_dnnl_format(binding.layout);
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_b);
   
    dnnl::memory bias_memory = [&](const auto& binding)
    {
        if (!binding.data)
        {
            return dnnl::memory{};
        }
        const auto dims = to_dnnl_dims(binding.shape);
        const auto dt = to_dnnl_data_type(binding.dt);
        const auto ft = to_dnnl_format(binding.layout);
        auto ret = dnnl::memory({ dims, dt, ft }, engine);
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_c);

    dnnl::memory output_memory = [&]()
    {
        return create_dnnl_memory(binding_t{ nullptr, opts.out_dt, opts.out_layout, opts.output_shape }, engine);
    }();

    dnnl::matmul::primitive_desc matmul_desc(engine,
        input_memory.get_desc(),
        weights_memory.get_desc(),
        bias_memory ? bias_memory.get_desc() : dnnl::memory::desc{},
        output_memory.get_desc()
    );
    const auto guery_impl_str = matmul_desc.impl_info_str();
    std::cout << "ref query impl: " << guery_impl_str << std::endl;

    auto matmul = dnnl::matmul(matmul_desc);
    std::unordered_map<int, dnnl::memory> args;
    args.insert({ DNNL_ARG_SRC, input_memory });
    args.insert({ DNNL_ARG_WEIGHTS, weights_memory });
    args.insert({ DNNL_ARG_BIAS, bias_memory });
    args.insert({ DNNL_ARG_DST, output_memory });
    matmul.execute(stream, args);
    stream.wait();

    auto* out_dnnl_data = output_memory.map_data<uint8_t>();
    assert(out_dnnl_data != nullptr && "[dnnl][gemm] Couldnt map output memory!");

    const auto om_desc = output_memory.get_desc();
    const auto om_dims = om_desc.get_dims();
    const auto copy_size = dimensions_product(om_dims) * dnnl::memory::data_type_size(om_desc.get_data_type());
    std::vector<std::byte> ret(copy_size);
    std::memcpy(ret.data(), out_dnnl_data, copy_size);
    output_memory.unmap_data(out_dnnl_data);
    return ret;
}