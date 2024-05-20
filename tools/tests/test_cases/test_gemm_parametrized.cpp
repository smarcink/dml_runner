#include <tuple>

#include <gtest/gtest.h>

#include <gemm.h>

#include "utils.h"
#include "layers_utils.h"

#include "test_gemm_base.h"

class DnnlPluginNext_GEMM_Params : public GemmBaseTestDispatcher, public testing::TestWithParam<std::tuple<
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
    void set_b_managed() { use_b_managed_ = true; }
    void set_alpha_value(float v) { alpha_ = v; }
    void set_beta_value(float v) { beta_ = v; }
    void set_activation_setting(ActivationSettings act) { activation_ = std::move(act); };
    void set_a_transposed() { use_a_transposed_ = true; }
    void set_b_transposed() { use_b_transposed_ = true; }

protected:
    GemmBaseDispatcher::create_params_t get_params() override
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
        opts.b_managed = use_b_managed_;
        opts.c_managed = false;
        opts.a_transposed = use_a_transposed_;
        opts.b_transposed = use_b_transposed_;
        opts.activation = activation_;
        return opts;
    }

private:
    bool use_c_tensor_ = false;
    bool use_b_managed_ = false;
    float alpha_ = 1.0f;
    float beta_ = 1.0f;
    ActivationSettings activation_ = {};
    bool use_a_transposed_ = false;
    bool use_b_transposed_ = false;
};


TEST_P(DnnlPluginNext_GEMM_Params, Basic)
{
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, BasicWithActivation)
{
    run();
    set_activation_setting(ActivationSettings{ ActivationType::eSigmoid });
}

TEST_P(DnnlPluginNext_GEMM_Params, WithAlpha)
{
    set_alpha_value(0.1337f);
    run();
}


TEST_P(DnnlPluginNext_GEMM_Params, WithCtensor)
{
    set_use_c_tensor();
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, WithCtensorAndAlpha)
{
    set_use_c_tensor();
    set_alpha_value(0.25f);
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, WithCtensorAndAlphaAndBeta)
{
    set_use_c_tensor();
    set_alpha_value(1.5f);
    set_beta_value(0.25f);
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, WithCtensorAndAlphaAndBetaAndActivation)
{
    set_use_c_tensor();
    set_alpha_value(1.5f);
    set_beta_value(0.25f);
    set_activation_setting(ActivationSettings{ ActivationType::eTanh });
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, WithBmanaged)
{
    set_b_managed();
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, WithATransposed)
{
    set_a_transposed();
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, WithBTransposed)
{
    set_b_transposed();
    run();
}

TEST_P(DnnlPluginNext_GEMM_Params, WithAandBTransposed)
{
    set_a_transposed();
    set_b_transposed();
    run();
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




class DnnlPluginNext_GEMM_Activations : public GemmBaseTestDispatcher, public testing::TestWithParam<std::tuple<
    ActivationSettings,
    DataType
    >>
{
public:
    static std::string params_to_str(const testing::TestParamInfo<DnnlPluginNext_GEMM_Activations::ParamType>& info)
    {
        const auto& params = info.param;

        const auto act = std::get<TUPLE_ID_ACTIVATION>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("activation_{}__datatype_{}", get_activation_type_str(act.type), get_data_type_str(dt));
        return fmt;
    }

protected:
    enum TupleID
    {
        TUPLE_ID_ACTIVATION = 0,
        TUPLE_ID_DT,
    };


protected:
    DnnlPluginNext_GEMM_Activations() {
        // You can do set-up work for each test here.
    }

    ~DnnlPluginNext_GEMM_Activations() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    void set_use_c_tensor() { use_c_tensor_ = true; }
    void set_b_managed() { use_b_managed_ = true; }
    void set_alpha_value(float v) { alpha_ = v; }
    void set_beta_value(float v) { beta_ = v; }
    void set_batch(std::uint32_t v) { batch_ = v; }
    void set_M(std::uint32_t v) { M_ = v; }
    void set_K(std::uint32_t v) { K_ = v; }
    void set_N(std::uint32_t v) { N_ = v; }

protected:
    GemmBaseDispatcher::create_params_t get_params() override
    {
        const auto params = GetParam();
        const auto act = std::get<TUPLE_ID_ACTIVATION>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        GemmBaseDispatcher::create_params_t opts{};

        opts.shape_a = TensorShape(batch_, 1, M_, K_);
        opts.shape_b = TensorShape(batch_, 1, K_, N_);
        opts.shape_c = use_c_tensor_ ? TensorShape(batch_, 1, M_, N_) : TensorShape();
        opts.allow_fp16_computations = dt == DataType::eFp16;
        opts.type = GemmType::GemmType_AB;
        opts.use_dnnl_for_reference_calculations = true; // we expect perfect match!(set to false to compare with DML backend)
        opts.layout = DataLayout::eNCHW;
        opts.dt = dt;
        opts.alpha = alpha_;
        opts.beta = beta_;
        opts.b_managed = use_b_managed_;
        opts.c_managed = false;
        opts.b_transposed = false;
        opts.activation = act;
        return opts;
    }

private:
    bool use_c_tensor_ = false;
    bool use_b_managed_ = false;
    float alpha_ = 1.0f;
    float beta_ = 1.0f;

    std::uint32_t batch_ = 0;
    std::uint32_t M_ = 0;
    std::uint32_t K_ = 0;
    std::uint32_t N_ = 0;
};

TEST_P(DnnlPluginNext_GEMM_Activations, SmallSizes)
{
    set_batch(1);
    set_M(32);
    set_K(64);
    set_N(32);
    run();
}

TEST_P(DnnlPluginNext_GEMM_Activations, SmallSizesWithCtensor)
{
    set_batch(1);
    set_M(32);
    set_K(64);
    set_N(32);
    set_use_c_tensor();
    run();
}

TEST_P(DnnlPluginNext_GEMM_Activations, BigSizes)
{
    set_batch(1);
    set_M(256);
    set_K(512);
    set_N(256);
    set_use_c_tensor();
    set_alpha_value(1.5f);
    set_alpha_value(1.5f);
    run();
}


INSTANTIATE_TEST_SUITE_P(
    GeemmActivations, DnnlPluginNext_GEMM_Activations,
    testing::Combine(
        testing::Values(
            ActivationSettings{ ActivationType::eRelu },
            ActivationSettings{ ActivationType::eLeakyRelu, 0.1f },
            //ActivationSettings{ActivationType::eClip, 0.1f, 1.0f}, // crashing
            //ActivationSettings{ ActivationType::eGelu },
            ActivationSettings{ ActivationType::eSigmoid },
            ActivationSettings{ ActivationType::eLinear, 0.5f, 0.1f },
            ActivationSettings{ ActivationType::eTanh }
        ),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_GEMM_Activations::ParamType>& info) {
        return DnnlPluginNext_GEMM_Activations::params_to_str(info);
    });
