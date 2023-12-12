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
    void set_no_bias() { no_bias_ = true; }

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
        opts.dt = dt;
        opts.input_layout = layout;
        opts.output_layout = layout;
        opts.filter_layout = layout;
        opts.in_pad = input_pad_;
        opts.stride = TensorShape(1, 1, kernel_stride_, kernel_stride_);
        opts.managaed_weights = true;
        opts.input_shape = TensorShape(batch, ic, h, w);
        opts.filter_shape = TensorShape(oc, ic, kernel_size_, kernel_size_);
        opts.no_bias = no_bias_;
        auto node = std::make_unique<ConvolutionUmdD3d12Dispatcher>(std::move(opts),
            ConvolutionUmdD3d12Dispatcher::conv_umdd3d12_params_t{},
            g_dx12_engine.intel_extension_d3d12,
            g_dx12_engine.d3d12_device.Get(),
            g_dx12_engine.command_list.Get());
        return node;
    }

private:
    bool no_bias_ = false;
    std::uint32_t kernel_size_ = 1;
    std::uint32_t kernel_stride_ = 1;
    std::uint32_t input_pad_ = 0;
};

TEST_P(DnnlPluginNext_Convolution_Params, Kernel1x1)
{
    set_kernel_size(1);
    set_kernel_stride(1);
    set_input_padding(0);
    EXPECT_TRUE(run());
}

TEST_P(DnnlPluginNext_Convolution_Params, Kernel3x3Stride2x2WithPadding)
{
    set_kernel_size(3);
    set_kernel_stride(2);
    set_input_padding(1);
    EXPECT_TRUE(run());
}

TEST_P(DnnlPluginNext_Convolution_Params, Kernel3x3Stride2x2WithPaddingAndNoBias)
{
    set_kernel_size(3);
    set_kernel_stride(2);
    set_input_padding(1);
    set_no_bias();
    EXPECT_TRUE(run());
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