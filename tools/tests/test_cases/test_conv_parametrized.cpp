#include <tuple>

#include <gtest/gtest.h>

#include <conv.h>

#include "utils.h"
#include "test_conv_base.h"

class DnnlPluginNext_Convolution_Params : public ConvolutionBaseTestDispatcher, public testing::TestWithParam<std::tuple<
    std::int32_t, // batch
    std::int32_t, // IC
    std::int32_t, // OC
    std::int32_t, // HEIGHT
    std::int32_t, // WIDTH
    std::int32_t, // GROUPS
    bool, // transposed
    DataLayout,
    DataType
    >>
{
public:
    static std::string params_to_str(const testing::TestParamInfo<DnnlPluginNext_Convolution_Params::ParamType>& info)
    {
        const auto& params = info.param;

        const auto batch = std::get<TUPLE_ID_BATCH>(params);
        const auto ic = std::get<TUPLE_ID_IC>(params);
        const auto oc = std::get<TUPLE_ID_OC>(params);
        const auto h = std::get<TUPLE_ID_HEIGHT>(params);
        const auto w = std::get<TUPLE_ID_WIDTH>(params);
        const auto g = std::get<TUPLE_ID_GROUPS>(params);

        const auto is_transposed = std::get<TUPLE_ID_IS_TRANSPOSED>(params);

        const auto layout = std::get<TUPLE_ID_LAYOUT>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("batch_{}__ic_{}__oc_{}__h_{}__w_{}__w_{}__transposed_{}__layout_{}__datatype_{}",
            batch, ic, oc, h, w, g, is_transposed, data_layout_name(layout), get_data_type_str(dt));
        return fmt;
    }

protected:
    enum TupleID
    {
        TUPLE_ID_BATCH = 0,
        TUPLE_ID_IC = 1,
        TUPLE_ID_OC,
        TUPLE_ID_HEIGHT,
        TUPLE_ID_WIDTH,
        TUPLE_ID_GROUPS,
        TUPLE_ID_IS_TRANSPOSED,
        TUPLE_ID_LAYOUT,
        TUPLE_ID_DT,
    };


protected:
    DnnlPluginNext_Convolution_Params() {
        // You can do set-up work for each test here.
    }

    ~DnnlPluginNext_Convolution_Params() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    void set_kernel_size(std::uint32_t ks) { kernel_size_ = ks; }
    void set_kernel_stride(std::uint32_t ks) { kernel_stride_ = ks; }
    void set_input_padding(std::uint32_t p) { input_pad_ = p; }
    void set_dilation(std::uint32_t d) { dilation_ = d; }
    void set_no_bias() { no_bias_ = true; }
    void set_activation_setting(ActivationSettings act) { activation_ = std::move(act); };

protected:
    ConvolutionBaseDispatcher::create_params_t get_params() override
    {
        const auto params = GetParam();
        const auto batch = std::get<TUPLE_ID_BATCH>(params);
        const auto ic = std::get<TUPLE_ID_IC>(params);
        const auto oc = std::get<TUPLE_ID_OC>(params);
        const auto h = std::get<TUPLE_ID_HEIGHT>(params);
        const auto w = std::get<TUPLE_ID_WIDTH>(params);
        const auto g = std::get<TUPLE_ID_GROUPS>(params);

        const auto is_transposed = std::get<TUPLE_ID_IS_TRANSPOSED>(params);

        const auto layout = std::get<TUPLE_ID_LAYOUT>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        ConvolutionBaseDispatcher::create_params_t opts{};
        opts.algo_winograd = false; // we should use auto anyway
        opts.use_dnnl_for_reference_calculations = true;
        opts.dt = dt;
        opts.input_layout = layout;
        opts.output_layout = layout;
        opts.filter_layout = layout;
        opts.in_pad = input_pad_;
        opts.groups = g;
        opts.stride = TensorShape(1u, 1u, kernel_stride_, kernel_stride_);
        opts.dilation = TensorShape(0u, 0u, dilation_, dilation_);
        opts.managed_weights = true;
        opts.transposed = is_transposed;
        opts.input_shape = TensorShape(batch, is_transposed ? oc : ic, h, w);
        opts.filter_shape = TensorShape(oc, ic, kernel_size_, kernel_size_);
        opts.no_bias = no_bias_;
        opts.allow_fp16_computations = dt == DataType::eFp16;
        opts.activation = activation_;
        return opts;
    }

private:
    bool no_bias_ = false;
    std::uint32_t kernel_size_ = 1;
    std::uint32_t kernel_stride_ = 1;
    std::uint32_t input_pad_ = 0;
    std::uint32_t dilation_ = 0;
    ActivationSettings activation_ = {};
};

TEST_P(DnnlPluginNext_Convolution_Params, Kernel1x1)
{
    set_kernel_size(1);
    set_kernel_stride(1);
    set_input_padding(0);
    run();
}

TEST_P(DnnlPluginNext_Convolution_Params, Kernel3x3Stride2x2WithPadding)
{
    set_kernel_size(3);
    set_kernel_stride(2);
    set_input_padding(1);
    run();
}

TEST_P(DnnlPluginNext_Convolution_Params, Kernel3x3Stride2x2WithPaddingAndNoBias)
{
    set_kernel_size(3);
    set_kernel_stride(2);
    set_input_padding(1);
    set_no_bias();
    run();
}

TEST_P(DnnlPluginNext_Convolution_Params, Kernel3x3Stride2x2WithPaddingAndReLuActivation)
{
    set_kernel_size(3);
    set_kernel_stride(2);
    set_input_padding(1);
    set_activation_setting(ActivationSettings{ ActivationType::eRelu });
    run();
}

TEST_P(DnnlPluginNext_Convolution_Params, Kernel3x3Stride2x2WithPaddingAndReLuActivationAndDilation1)
{
    set_kernel_size(3);
    set_kernel_stride(2);
    set_input_padding(1);
    set_activation_setting(ActivationSettings{ ActivationType::eRelu });
    set_dilation(1u);
    run();
}

INSTANTIATE_TEST_SUITE_P(
    DimensionsPowerOfTwo, DnnlPluginNext_Convolution_Params,
    testing::Combine(
        testing::Values(1, 16),
        testing::Values(16),
        testing::Values(32),
        testing::Values(64), testing::Values(64),  // height and width
        testing::Values(1, 16),  // groups
        testing::Values(false, true), // is transposed
        testing::Values(DataLayout::eNCHW, DataLayout::eNHWC),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_Params::ParamType>& info) {
        return DnnlPluginNext_Convolution_Params::params_to_str(info);
    });

INSTANTIATE_TEST_SUITE_P(
    DimensionsNonPowerOfTwo, DnnlPluginNext_Convolution_Params,
    testing::Combine(
        testing::Values(3),
        testing::Values(17),
        testing::Values(79),
        testing::Values(55), testing::Values(55),  // height and width
        testing::Values(1), // groups
        testing::Values(false, true), // is transposed
        testing::Values(DataLayout::eNCHW, DataLayout::eNHWC),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_Params::ParamType>& info) {
        return DnnlPluginNext_Convolution_Params::params_to_str(info);
    });



class DnnlPluginNext_Convolution_ParamsUnpackedCases : public ConvolutionBaseTestDispatcher, public testing::TestWithParam<std::tuple<
    DataLayout,  // in
    DataLayout,  // out
    bool, // transposed
    DataType
    >>
{
public:
    static std::string params_to_str(const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info)
    {
        const auto& params = info.param;

        const auto in_layout = std::get<TUPLE_ID_IN_LAYOUT>(params);
        const auto out_layout = std::get<TUPLE_ID_OUT_LAYOUT>(params);
        const auto is_transposed = std::get<TUPLE_ID_IS_TRANSPOSED>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("in_layout_{}__out_layout_{}__datatype_{}__is_transposed_{}",
            data_layout_name(in_layout), data_layout_name(out_layout), get_data_type_str(dt), is_transposed);
        return fmt;
    }

protected:
    enum TupleID
    {
        TUPLE_ID_IN_LAYOUT,
        TUPLE_ID_OUT_LAYOUT,
        TUPLE_ID_IS_TRANSPOSED,
        TUPLE_ID_DT,
    };


protected:
    DnnlPluginNext_Convolution_ParamsUnpackedCases() {
        // You can do set-up work for each test here.
    }

    ~DnnlPluginNext_Convolution_ParamsUnpackedCases() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

protected:
    ConvolutionBaseDispatcher::create_params_t get_params() override
    {
        const auto params = GetParam();
        const auto in_layout = std::get<TUPLE_ID_IN_LAYOUT>(params);
        const auto out_layout = std::get<TUPLE_ID_OUT_LAYOUT>(params);
        const auto is_transposed = std::get<TUPLE_ID_IS_TRANSPOSED>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        ConvolutionBaseDispatcher::create_params_t opts{};
        opts.algo_winograd = false; // we should use auto anyway
        opts.use_dnnl_for_reference_calculations = true;
        opts.dt = dt;
        opts.input_layout = in_layout;
        opts.output_layout = out_layout;
        opts.filter_layout = DataLayout::eNCHW;
        opts.in_pad = 0;
        opts.stride = TensorShape(1, 1, 1, 1);
        opts.managed_weights = true;
        opts.filter_shape = TensorShape(16, 32, 1, 1);
        opts.transposed = is_transposed;
        opts.input_shape = TensorShape(1, is_transposed ? opts.filter_shape.n : opts.filter_shape.c, 64, 64);
        opts.no_bias = false;
        opts.allow_fp16_computations = dt == DataType::eFp16;
        return opts;
    }
};


TEST_P(DnnlPluginNext_Convolution_ParamsUnpackedCases, UnpackedTests)
{
    run();
}

INSTANTIATE_TEST_SUITE_P(
    ConvolutionUnpackedInputs, DnnlPluginNext_Convolution_ParamsUnpackedCases,
    testing::Combine(
        testing::Values(DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48),
        testing::Values(DataLayout::eNCHW, DataLayout::eNHWC),
        testing::Values(false, true),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info) {
        return DnnlPluginNext_Convolution_ParamsUnpackedCases::params_to_str(info);
    });

INSTANTIATE_TEST_SUITE_P(
    ConvolutionUnpackedOutputs, DnnlPluginNext_Convolution_ParamsUnpackedCases,
    testing::Combine(
        testing::Values(DataLayout::eNCHW, DataLayout::eNHWC),
        testing::Values(DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48),
        testing::Values(false, true),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info) {
        return DnnlPluginNext_Convolution_ParamsUnpackedCases::params_to_str(info);
    });

INSTANTIATE_TEST_SUITE_P(
    ConvolutionUnpackedIntputAndOutputs, DnnlPluginNext_Convolution_ParamsUnpackedCases,
    testing::Combine(
        testing::Values(DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48),
        testing::Values(DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48),
        testing::Values(false, true),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info) {
        return DnnlPluginNext_Convolution_ParamsUnpackedCases::params_to_str(info);
    });




class DnnlPluginNext_Convolution_Activations : public ConvolutionBaseTestDispatcher, public testing::TestWithParam<std::tuple<
    ActivationSettings,
    DataLayout,
    bool, //transposed
    DataType
    >>
{
public:
    static std::string params_to_str(const testing::TestParamInfo<DnnlPluginNext_Convolution_Activations::ParamType>& info)
    {
        const auto& params = info.param;

        const auto act = std::get<TUPLE_ID_ACTIVATION>(params);
        const auto layout = std::get<TUPLE_ID_LAYOUT>(params);
        const auto is_transposed = std::get<TUPLE_ID_IS_TRANSPOSED>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("activation_{}__layout_{}__datatype_{}__is_transposed_{}",
            get_activation_type_str(act.type), data_layout_name(layout), get_data_type_str(dt), is_transposed);
        return fmt;
    }

protected:
    enum TupleID
    {
        TUPLE_ID_ACTIVATION = 0,
        TUPLE_ID_LAYOUT,
        TUPLE_ID_IS_TRANSPOSED,
        TUPLE_ID_DT,
    };


protected:
    DnnlPluginNext_Convolution_Activations() 
    {
    }

    ~DnnlPluginNext_Convolution_Activations() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    // Sets up the test fixture.
    void SetUp() override
    {
        const auto params = GetParam();
        const auto activation = std::get<TUPLE_ID_ACTIVATION>(params);
        if (g_test_config.run_dml && activation.type == ActivationType::eGelu)
        {
            GTEST_SKIP() <<"DirectML does not fuse GELU activation to metacommand. Skipping test.";
        }
    }

    void set_input_tensor_shape(TensorShape shape) { input_shape_ = std::move(shape); }
    void set_filter_tensor_shape(TensorShape shape) { filter_shape_ = std::move(shape); }
    void set_kernel_stride(std::uint32_t ks) { kernel_stride_ = ks; }
    void set_input_padding(std::uint32_t p) { input_pad_ = p; }
    void set_no_bias() { no_bias_ = true; }
    void set_groups(std::uint32_t g) { groups_ = g; }

protected:
    ConvolutionBaseDispatcher::create_params_t get_params() override
    {
        const auto params = GetParam();
        const auto activation = std::get<TUPLE_ID_ACTIVATION>(params);
        const auto layout = std::get<TUPLE_ID_LAYOUT>(params);
        const auto is_transposed = std::get<TUPLE_ID_IS_TRANSPOSED>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        if (is_transposed)
        {
            input_shape_.c = filter_shape_.n;
        }

        ConvolutionBaseDispatcher::create_params_t opts{};
        opts.algo_winograd = false; // we should use auto anyway
        opts.use_dnnl_for_reference_calculations = true;
        opts.dt = dt;
        opts.groups = groups_;
        opts.input_layout = layout;
        opts.output_layout = layout;
        opts.filter_layout = layout;
        opts.in_pad = input_pad_;
        opts.transposed = is_transposed;
        opts.stride = TensorShape(1, 1, kernel_stride_, kernel_stride_);
        opts.managed_weights = true;
        opts.filter_shape = filter_shape_;
        opts.input_shape = input_shape_;
        opts.no_bias = no_bias_;
        opts.allow_fp16_computations = dt == DataType::eFp16;
        opts.activation = activation;
        return opts;
    }

private:
    bool no_bias_ = false;
    std::uint32_t kernel_stride_ = 1;
    std::uint32_t input_pad_ = 0;
    std::uint32_t groups_ = 1;
    TensorShape input_shape_ = {};
    TensorShape filter_shape_ = {};
};


TEST_P(DnnlPluginNext_Convolution_Activations, Kernel1x1)
{
    set_kernel_stride(1);
    set_input_padding(0);
    set_input_tensor_shape(TensorShape{1, 32, 64, 64});
    set_filter_tensor_shape(TensorShape{64, 32, 1, 1});
    run();
}

TEST_P(DnnlPluginNext_Convolution_Activations, Kernel3x3)
{
    set_kernel_stride(2);
    set_input_padding(1);
    set_input_tensor_shape(TensorShape{ 1, 32, 64, 64 });
    set_filter_tensor_shape(TensorShape{ 64, 32, 3, 3 });
    run();
}

TEST_P(DnnlPluginNext_Convolution_Activations, Kernel3x3AndGroups2)
{
    set_kernel_stride(2);
    set_input_padding(1);
    set_input_tensor_shape(TensorShape{ 1, 32, 64, 64 });
    set_filter_tensor_shape(TensorShape{ 32, 32, 3, 3 });
    set_groups(2);
    run();
}

TEST_P(DnnlPluginNext_Convolution_Activations, Kernel3x3Depthwise)
{
    set_kernel_stride(2);
    set_input_padding(1);
    set_input_tensor_shape(TensorShape{ 1, 32, 64, 64 });
    set_filter_tensor_shape(TensorShape{ 32, 32, 3, 3 });
    set_groups(32);
    run();
}

TEST_P(DnnlPluginNext_Convolution_Activations, Kernel1x1Depthwise)
{
    set_kernel_stride(1);
    set_input_tensor_shape(TensorShape{ 1, 32, 64, 64 });
    set_filter_tensor_shape(TensorShape{ 32, 32, 1, 1 });
    set_groups(32);
    run();
}

INSTANTIATE_TEST_SUITE_P(
    ConvolutionActivations, DnnlPluginNext_Convolution_Activations,
    testing::Combine(
        testing::Values(
            ActivationSettings{ ActivationType::eRelu },
            ActivationSettings{ ActivationType::eLeakyRelu, 0.1f },
            //ActivationSettings{ActivationType::eClip, 0.1f, 1.0f}, // crashing
            ActivationSettings{ ActivationType::eGelu },
            ActivationSettings{ ActivationType::eSigmoid},
            ActivationSettings{ ActivationType::eLinear, 0.5f, 0.1f},
            ActivationSettings{ ActivationType::eTanh }
        ),
        testing::Values(DataLayout::eNCHW, DataLayout::eNHWC),
        testing::Values(false, true), // is transposed
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_Activations::ParamType>& info) {
        return DnnlPluginNext_Convolution_Activations::params_to_str(info);
    });
