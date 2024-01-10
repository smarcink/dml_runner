#include <tuple>

#include <gtest/gtest.h>

#include <conv.h>

#include "utils.h"

class DnnlPluginNext_Convolution_Params : public NodeDispatcherBase, public testing::TestWithParam<std::tuple<
    std::int32_t, // batch
    std::int32_t, // IC
    std::int32_t, // OC
    std::int32_t, // HEIGHT
    std::int32_t, // WIDTH
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

        const auto layout = std::get<TUPLE_ID_LAYOUT>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("batch_{}__ic_{}__oc_{}__h_{}__w_{}__layout_{}__datatype_{}",
            batch, ic, oc, h, w, data_layout_name(layout), get_data_type_str(dt));
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
    std::unique_ptr<NodeDispatcher> create_dispatcher_impl() override
    {
        const auto params = GetParam();
        const auto batch = std::get<TUPLE_ID_BATCH>(params);
        const auto ic = std::get<TUPLE_ID_IC>(params);
        const auto oc = std::get<TUPLE_ID_OC>(params);
        const auto h = std::get<TUPLE_ID_HEIGHT>(params);
        const auto w = std::get<TUPLE_ID_WIDTH>(params);

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
        opts.stride = TensorShape(1u, 1u, kernel_stride_, kernel_stride_);
        opts.dilation = TensorShape(0u, 0u, dilation_, dilation_);
        opts.managaed_weights = true;
        opts.input_shape = TensorShape(batch, ic, h, w);
        opts.filter_shape = TensorShape(oc, ic, kernel_size_, kernel_size_);
        opts.no_bias = no_bias_;
        opts.allow_fp16_computations = dt == DataType::eFp16;
        opts.activation = activation_;
        auto node = std::make_unique<ConvolutionUmdD3d12Dispatcher>(std::move(opts),
            ConvolutionUmdD3d12Dispatcher::conv_umdd3d12_params_t{},
            g_dx12_engine.intel_extension_d3d12,
            g_dx12_engine.d3d12_device.Get(),
            g_dx12_engine.dml_device.Get(),
            g_dx12_engine.dml_command_recorder.Get(),
            g_dx12_engine.command_list.Get());
        return node;
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
        testing::Values(DataLayout::eNCHW, DataLayout::eNHWC),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_Params::ParamType>& info) {
        return DnnlPluginNext_Convolution_Params::params_to_str(info);
    });



class DnnlPluginNext_Convolution_ParamsUnpackedCases : public NodeDispatcherBase, public testing::TestWithParam<std::tuple<
    DataLayout,  // in
    DataLayout,  // out
    DataType
    >>
{
public:
    static std::string params_to_str(const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info)
    {
        const auto& params = info.param;

        const auto in_layout = std::get<TUPLE_ID_IN_LAYOUT>(params);
        const auto out_layout = std::get<TUPLE_ID_OUT_LAYOUT>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("in_layout_{}__out_layout_{}__datatype_{}",
            data_layout_name(in_layout), data_layout_name(out_layout), get_data_type_str(dt));
        return fmt;
    }

protected:
    enum TupleID
    {
        TUPLE_ID_IN_LAYOUT,
        TUPLE_ID_OUT_LAYOUT,
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
    std::unique_ptr<NodeDispatcher> create_dispatcher_impl() override
    {
        const auto params = GetParam();
        const auto in_layout = std::get<TUPLE_ID_IN_LAYOUT>(params);
        const auto out_layout = std::get<TUPLE_ID_OUT_LAYOUT>(params);
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
        opts.managaed_weights = true;
        opts.input_shape = TensorShape(1, 32, 64, 64);
        opts.filter_shape = TensorShape(16, 32, 1, 1);
        opts.no_bias = false;
        opts.allow_fp16_computations = dt == DataType::eFp16;
        auto node = std::make_unique<ConvolutionUmdD3d12Dispatcher>(std::move(opts),
            ConvolutionUmdD3d12Dispatcher::conv_umdd3d12_params_t{},
            g_dx12_engine.intel_extension_d3d12,
            g_dx12_engine.d3d12_device.Get(),
            g_dx12_engine.dml_device.Get(),
            g_dx12_engine.dml_command_recorder.Get(),
            g_dx12_engine.command_list.Get());
        return node;
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
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info) {
        return DnnlPluginNext_Convolution_ParamsUnpackedCases::params_to_str(info);
    });

INSTANTIATE_TEST_SUITE_P(
    ConvolutionUnpackedOutputs, DnnlPluginNext_Convolution_ParamsUnpackedCases,
    testing::Combine(
        testing::Values(DataLayout::eNCHW, DataLayout::eNHWC),
        testing::Values(DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info) {
        return DnnlPluginNext_Convolution_ParamsUnpackedCases::params_to_str(info);
    });

INSTANTIATE_TEST_SUITE_P(
    ConvolutionUnpackedIntputAndOutputs, DnnlPluginNext_Convolution_ParamsUnpackedCases,
    testing::Combine(
        testing::Values(DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48),
        testing::Values(DataLayout::eNCHW_AlignW320, DataLayout::eNHWC_AlignH48),
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_ParamsUnpackedCases::ParamType>& info) {
        return DnnlPluginNext_Convolution_ParamsUnpackedCases::params_to_str(info);
    });




class DnnlPluginNext_Convolution_Activations : public NodeDispatcherBase, public testing::TestWithParam<std::tuple<
    ActivationSettings,
    DataLayout,
    DataType
    >>
{
public:
    static std::string params_to_str(const testing::TestParamInfo<DnnlPluginNext_Convolution_Activations::ParamType>& info)
    {
        const auto& params = info.param;

        const auto act = std::get<TUPLE_ID_ACTIVATION>(params);
        const auto layout = std::get<TUPLE_ID_LAYOUT>(params);
        const auto dt = std::get<TUPLE_ID_DT>(params);

        const auto fmt = std::format("activation_{}__layout_{}__datatype_{}",
            get_activation_type_str(act.type), data_layout_name(layout), get_data_type_str(dt));
        return fmt;
    }

protected:
    enum TupleID
    {
        TUPLE_ID_ACTIVATION = 0,
        TUPLE_ID_LAYOUT,
        TUPLE_ID_DT,
    };


protected:
    DnnlPluginNext_Convolution_Activations() {
        // You can do set-up work for each test here.
    }

    ~DnnlPluginNext_Convolution_Activations() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    void set_input_tensor_shape(TensorShape shape) { input_shape_ = std::move(shape); }
    void set_filter_tensor_shape(TensorShape shape) { filter_shape_ = std::move(shape); }
    void set_kernel_stride(std::uint32_t ks) { kernel_stride_ = ks; }
    void set_input_padding(std::uint32_t p) { input_pad_ = p; }
    void set_no_bias() { no_bias_ = true; }

protected:
    std::unique_ptr<NodeDispatcher> create_dispatcher_impl() override
    {
        const auto params = GetParam();
        const auto activation = std::get<TUPLE_ID_ACTIVATION>(params);
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
        opts.stride = TensorShape(1, 1, kernel_stride_, kernel_stride_);
        opts.managaed_weights = true;
        opts.input_shape = input_shape_;
        opts.filter_shape = filter_shape_;
        opts.no_bias = no_bias_;
        opts.allow_fp16_computations = dt == DataType::eFp16;
        opts.activation = activation;
        auto node = std::make_unique<ConvolutionUmdD3d12Dispatcher>(std::move(opts),
            ConvolutionUmdD3d12Dispatcher::conv_umdd3d12_params_t{},
            g_dx12_engine.intel_extension_d3d12,
            g_dx12_engine.d3d12_device.Get(),
            g_dx12_engine.dml_device.Get(),
            g_dx12_engine.dml_command_recorder.Get(),
            g_dx12_engine.command_list.Get());
        return node;
    }

private:
    bool no_bias_ = false;
    std::uint32_t kernel_stride_ = 1;
    std::uint32_t input_pad_ = 0;
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
    set_kernel_stride(1);
    set_input_padding(1);
    set_input_tensor_shape(TensorShape{ 1, 32, 64, 64 });
    set_filter_tensor_shape(TensorShape{ 64, 32, 3, 3 });
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
        testing::Values(DataType::eFp32, DataType::eFp16)),
    [](const testing::TestParamInfo<DnnlPluginNext_Convolution_Activations::ParamType>& info) {
        return DnnlPluginNext_Convolution_Activations::params_to_str(info);
    });
