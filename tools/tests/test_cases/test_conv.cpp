#include <tuple>

#include <gtest/gtest.h>

#include <conv.h>

#include "utils.h"
#include "test_conv_base.h"



class DnnlPluginNext_Convolution : public ConvolutionBaseTestDispatcher, public testing::Test
{

protected:
    DnnlPluginNext_Convolution() {
        // You can do set-up work for each test here.
        params_.managed_weights = true;
        params_.algo_winograd = false; // we should use auto anyway
        params_.use_dnnl_for_reference_calculations = true;
    }

    ~DnnlPluginNext_Convolution() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

protected:
    ConvolutionBaseDispatcher::create_params_t get_params() override
    {
        return params_;
    }

protected:
    ConvolutionBaseDispatcher::create_params_t params_{};
};

TEST_F(DnnlPluginNext_Convolution, SD_dims_0)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNCHW;
    params_.output_layout = DataLayout::eNCHW;
    params_.filter_layout = DataLayout::eNCHW;
    params_.in_pad = 0;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(2, 320, 640, 640);
    params_.filter_shape = TensorShape(32, 320, 1, 1);
    run();
}

TEST_F(DnnlPluginNext_Convolution, SD_dims_1)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNCHW;
    params_.output_layout = DataLayout::eNCHW;
    params_.filter_layout = DataLayout::eNCHW;
    params_.in_pad = 0;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(2, 320, 640, 640);
    params_.filter_shape = TensorShape(32, 320, 1, 1);
    run();
}

TEST_F(DnnlPluginNext_Convolution, SD_dims_2)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNCHW;
    params_.output_layout = DataLayout::eNCHW;
    params_.filter_layout = DataLayout::eNCHW;
    params_.in_pad = 1;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(2, 1280, 16, 16);
    params_.filter_shape = TensorShape(1280, 1280, 3, 3);
    run();
}

TEST_F(DnnlPluginNext_Convolution, SD_dims_3)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNCHW;
    params_.output_layout = DataLayout::eNCHW;
    params_.filter_layout = DataLayout::eNCHW;
    params_.in_pad = 1;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(2, 320, 64, 64);
    params_.filter_shape = TensorShape(4, 320, 3, 3);
    run();
}

TEST_F(DnnlPluginNext_Convolution, denoirser_xxx_ts512_case_0)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNCHW;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 0;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(1, 128, 512, 512);
    params_.filter_shape = TensorShape(3, 128, 1, 1);
    run();
}

TEST_F(DnnlPluginNext_Convolution, denoirser_xxx_ts512_case_1)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 0;
    params_.stride = TensorShape(1, 1, 2, 2);
    params_.input_shape = TensorShape(1, 128, 512, 512);
    params_.filter_shape = TensorShape(192, 128, 2, 2);
    run();
}


TEST_F(DnnlPluginNext_Convolution, denoirser_xxx_ts704_case_0)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 1;
    params_.stride = TensorShape(1, 1, 3, 3);
    params_.input_shape = TensorShape(1, 128, 44, 44);
    params_.filter_shape = TensorShape(128, 128, 3, 3);
    run();
}

TEST_F(DnnlPluginNext_Convolution, denoirser_xxx_ts704_case_1)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 1;
    params_.stride = TensorShape(1, 1, 3, 3);
    params_.input_shape = TensorShape(1, 128, 44, 44);
    params_.filter_shape = TensorShape(128, 128, 3, 3);
    run();
}

TEST_F(DnnlPluginNext_Convolution, sr_xtrans_win_case_0)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 0;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(1, 320, 255, 255);
    params_.filter_shape = TensorShape(128, 320, 1, 1);
    params_.no_bias = true;
    run();
}

TEST_F(DnnlPluginNext_Convolution, lensblur_de_fp16_512_case_0)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 0;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(1, 64, 1, 6400);
    params_.filter_shape = TensorShape(768, 64, 1, 1);
    run();
}

TEST_F(DnnlPluginNext_Convolution, lensblur_de_fp16_512_case_1)
{
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 1;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(1, 256, 20, 20);
    params_.filter_shape = TensorShape(256, 256, 3, 3);
    run();
}

TEST_F(DnnlPluginNext_Convolution, lensblur_dr_v3_case_0)
{
    params_.dt = DataType::eFp32;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 3;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(1, 4, 1024, 1024);
    params_.filter_shape = TensorShape(64, 4, 7, 7);
    run();
}

TEST_F(DnnlPluginNext_Convolution, lensblur_dr_v3_case_1)
{
    params_.dt = DataType::eFp32;
    params_.input_layout = DataLayout::eNHWC;
    params_.output_layout = DataLayout::eNHWC;
    params_.filter_layout = DataLayout::eNHWC;
    params_.in_pad = 1;
    params_.stride = TensorShape(1, 1, 1, 1);
    params_.input_shape = TensorShape(1, 768, 128, 128);
    params_.filter_shape = TensorShape(96, 768, 1, 1);
    run();
}