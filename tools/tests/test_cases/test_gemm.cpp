#include <tuple>

#include <gtest/gtest.h>

#include <gemm.h>

#include "utils.h"


class NodeDispatcherBase
{
public:
    virtual bool run()
    {
        auto node = create_dispatcher_impl();

        // wait for any potential uploads
        g_dx12_engine.wait_for_execution();

        // bind descriptor heap
        const auto descriptors_count = node->get_total_descriptor_count();
        auto descriptor_heap = create_descriptor_heap(g_dx12_engine.d3d12_device.Get(), descriptors_count);
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        g_dx12_engine.command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        // initalize
        node->initialize(g_dx12_engine.command_list.Get(), descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        g_dx12_engine.wait_for_execution();
        
        // Bind and execute node
        g_dx12_engine.command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
        node->execute(g_dx12_engine.command_list.Get());
        g_dx12_engine.wait_for_execution();

        // finally validate conformance
        const auto conformance_result = node->validate_conformance(g_dx12_engine.command_queue.Get(), g_dx12_engine.command_allocator.Get(), g_dx12_engine.command_list.Get(), false);
        
        // we expect perfect match
        // comaprision have to be done vs dnnl!
        // vs HLSL there can be differences
        const auto perfect_match = conformance_result.biggest_difference == 0.0f;
        if (!perfect_match && conformance_result.passed)
        {
            std::cout << "Conformance has passed, but it wasn't perfect match. Was the tested validate vs dnnl?" << std::endl;
        }
        return perfect_match;
    }

protected:
    inline static Dx12Engine g_dx12_engine = Dx12Engine(); // create single engine to be reused across tests!
    virtual std::unique_ptr<NodeDispatcher> create_dispatcher_impl() = 0;
};



// The fixture for testing class Foo.
class DnnlPluginNext_GEMM_Params : public NodeDispatcherBase, public testing::TestWithParam<std::tuple<
    std::int32_t, // batch
    std::int32_t, // M
    std::int32_t, // K
    std::int32_t, // N
    DataType
    >>
{
public:
    static std::string params_to_str(const testing::TestParamInfo<DnnlPluginNext_GEMM_Params::ParamType>& info)
    {
        const auto& params = info.param;

        const auto batch = std::get<TUPLE_ID_BATCH>(params);
        const auto M = std::get<TUPLE_ID_M>(params);
        const auto K = std::get<TUPLE_ID_K>(params);
        const auto N = std::get<TUPLE_ID_N>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("batch_{}__M_{}__K_{}__N_{}__datatype_{}", batch, M, K, N, get_data_type_str(dt));
        return fmt;
    }

protected:
    enum TupleID
    {
        TUPLE_ID_BATCH = 0,
        TUPLE_ID_M     = 1,
        TUPLE_ID_K,
        TUPLE_ID_N,
        TUPLE_ID_DT,
    };


protected:
    DnnlPluginNext_GEMM_Params() {
        // You can do set-up work for each test here.
    }

    ~DnnlPluginNext_GEMM_Params() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    void set_use_c_tensor() { use_c_tensor_ = true; }
    void set_alpha_value(float v) { alpha_ = v; }
    void set_beta_value(float v) { beta_ = v; }

protected:
    std::unique_ptr<NodeDispatcher> create_dispatcher_impl() override
    {
        const auto params = GetParam();
        const auto batch = std::get<TUPLE_ID_BATCH>(params);
        const auto M = std::get<TUPLE_ID_M>(params);
        const auto K = std::get<TUPLE_ID_K>(params);
        const auto N = std::get<TUPLE_ID_N>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        GemmBaseDispatcher::create_params_t opts{};

        opts.shape_a = TensorShape(batch, 1, M, K);
        opts.shape_b = TensorShape(batch, 1, K, N);
        opts.shape_c = use_c_tensor_ ? TensorShape(batch, 1, M, N) : TensorShape();
        opts.allow_fp16_computations = dt == DataType::eFp16;
        opts.type = GemmType::GemmType_AB;
        opts.use_dnnl_for_reference_calculations = true; // we expect perfect match!(set to false to compare with DML backend)
        opts.layout = DataLayout::eNCHW;
        opts.dt = dt;
        opts.alpha = alpha_;
        opts.beta = beta_;
        opts.b_managed = false;
        opts.c_managed = false;
        opts.b_transposed = false;
        auto node = std::make_unique<GemmUmdD3d12Dispatcher>(std::move(opts),
            g_dx12_engine.intel_extension_d3d12,
            g_dx12_engine.d3d12_device.Get(),
            g_dx12_engine.dml_device.Get(),
            g_dx12_engine.dml_command_recorder.Get(),
            g_dx12_engine.command_list.Get());
        return node;
    }

private:
    bool use_c_tensor_ = false;
    float alpha_ = 1.0f;
    float beta_ = 1.0f;
};


TEST_P(DnnlPluginNext_GEMM_Params, Basic)
{
    EXPECT_TRUE(run());
}

TEST_P(DnnlPluginNext_GEMM_Params, WithAlpha)
{
    set_alpha_value(0.1337f);
    EXPECT_TRUE(run());
}


TEST_P(DnnlPluginNext_GEMM_Params, WithCtensor)
{
    set_use_c_tensor();
    EXPECT_TRUE(run());
}

TEST_P(DnnlPluginNext_GEMM_Params, WithCtensorAndAlpha)
{
    set_use_c_tensor();
    set_alpha_value(0.25f);
    EXPECT_TRUE(run());
}

TEST_P(DnnlPluginNext_GEMM_Params, WithCtensorAndAlphaAndBeta)
{
    set_use_c_tensor();
    set_alpha_value(1.5f);
    set_beta_value(0.25f);
    EXPECT_TRUE(run());
}

INSTANTIATE_TEST_SUITE_P(
    DimensionsPowerOfTwo, DnnlPluginNext_GEMM_Params,
    testing::Combine(
        testing::Values(1, 16),
        testing::Values(128),
        testing::Values(256),
        testing::Values(64),
        testing::Values(DataType::eFp32, DataType::eFp16)),
        [](const testing::TestParamInfo<DnnlPluginNext_GEMM_Params::ParamType>& info) {
            return DnnlPluginNext_GEMM_Params::params_to_str(info);
        });

INSTANTIATE_TEST_SUITE_P(
    DimensionsNonPowerOfTwo, DnnlPluginNext_GEMM_Params,
    testing::Combine(
        testing::Values(13),
        testing::Values(69),
        testing::Values(7),
        testing::Values(125),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_GEMM_Params::ParamType>& info) {
        return DnnlPluginNext_GEMM_Params::params_to_str(info);
    });


INSTANTIATE_TEST_SUITE_P(
    BigDimensions_0, DnnlPluginNext_GEMM_Params, 
    testing::Combine(
        testing::Values(2), 
        testing::Values(1024), 
        testing::Values(4096), 
        testing::Values(512), 
        testing::Values(DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_GEMM_Params::ParamType>& info) {
        return DnnlPluginNext_GEMM_Params::params_to_str(info);
    });

INSTANTIATE_TEST_SUITE_P(
    BigDimensions_1, DnnlPluginNext_GEMM_Params, 
    testing::Combine(
        testing::Values(2), 
        testing::Values(128), 
        testing::Values(77), 
        testing::Values(4096), 
        testing::Values(DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_GEMM_Params::ParamType>& info) {
        return DnnlPluginNext_GEMM_Params::params_to_str(info);
    });


