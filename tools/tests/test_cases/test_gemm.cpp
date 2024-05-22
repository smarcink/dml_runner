#include <tuple>

#include <gtest/gtest.h>

#include <gemm.h>

#include "utils.h"
#include "config.h"
#include "test_gemm_base.h"

class DnnlPluginNext_GEMM : public GemmBaseTestDispatcher, public testing::Test
{

protected:
    DnnlPluginNext_GEMM() {
        // You can do set-up work for each test here.
        params_.type = GemmType::GemmType_AB;
        params_.layout = DataLayout::eNCHW;
        params_.alpha = 1.0f;
        params_.beta = 1.0f;
        params_.b_managed = false;
        params_.c_managed = false;
        params_.a_transposed = false;
        params_.b_transposed = false;
        params_.use_dnnl_for_reference_calculations = true;
    }

    ~DnnlPluginNext_GEMM() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }


protected:
    GemmBaseDispatcher::create_params_t get_params() override
    {
        return params_;
    }

protected:
    GemmBaseDispatcher::create_params_t params_{};
};

TEST_F(DnnlPluginNext_GEMM, Resnet50_Fp16)
{
    if (!g_test_config.run_dml)
    {
        GTEST_SKIP() << "Enabled only for DML path. ToDo: add support for non-DML path.";
    }
    params_.shape_a = TensorShape(1, 1, 1, 2048);
    params_.shape_b = TensorShape(1, 1, 2048, 1000);
    params_.shape_c = TensorShape(1, 1, 1, 1000);
    params_.b_transposed = true;
    params_.b_managed = true;
    params_.c_managed = true;
    params_.layout = DataLayout::eNCHW;
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    run();
}


TEST_F(DnnlPluginNext_GEMM, PackedTensor_fp16)
{
    params_.shape_a = TensorShape(1, 1, 32, 128);
    params_.shape_b = TensorShape(1, 1, 128, 64);
    params_.layout = DataLayout::eNCHW;
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    run();
}


TEST_F(DnnlPluginNext_GEMM, PackedTensor_fp32)
{
    params_.shape_a = TensorShape(1, 1, 32, 128);
    params_.shape_b = TensorShape(1, 1, 128, 64);
    params_.layout = DataLayout::eNCHW;
    run();
}

TEST_F(DnnlPluginNext_GEMM, UnpackedTensor_fp32)
{
    params_.shape_a = TensorShape(1, 1, 32, 128);
    params_.shape_b = TensorShape(1, 1, 128, 64);
    params_.layout = DataLayout::eNCHW_AlignW320;
    params_.dt = DataType::eFp32;
    run();
}


TEST_F(DnnlPluginNext_GEMM, SmallMandNBigK)
{
    params_.shape_a = TensorShape(1, 1, 20, 20000);
    params_.shape_b = TensorShape(1, 1, 20000, 20);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, SmallMandNBigKWithActivation)
{
    params_.shape_a = TensorShape(1, 1, 20, 20000);
    params_.shape_b = TensorShape(1, 1, 20000, 20);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.activation = ActivationSettings{ ActivationType::eRelu };
    run();
}

TEST_F(DnnlPluginNext_GEMM, SD_dims_0)
{
    params_.shape_a = TensorShape(2, 1, 4096, 320);
    params_.shape_b = TensorShape(2, 1, 320, 4096);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.b_managed = true;
    run();
}


TEST_F(DnnlPluginNext_GEMM, SD_dims_1)
{
    params_.shape_a = TensorShape(2, 1, 64, 1280);
    params_.shape_b = TensorShape(2, 1, 1280, 1280);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.b_managed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, SD_dims_2)
{
    params_.shape_a = TensorShape(2, 1, 1024, 640);
    params_.shape_b = TensorShape(2, 1, 640, 1920);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.b_managed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, SD_dims_2_with_alpha)
{
    params_.shape_a = TensorShape(2, 1, 1024, 640);
    params_.shape_b = TensorShape(2, 1, 640, 1920);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.alpha = 0.35f;
    params_.b_managed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, SD_dims_2_fp32)
{
    params_.shape_a = TensorShape(2, 1, 1024, 640);
    params_.shape_b = TensorShape(2, 1, 640, 1920);
    params_.dt = DataType::eFp32;
    params_.b_managed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, LLaMa_v2_case_0)
{
    params_.shape_a = TensorShape(1, 1, 1, 4096);
    params_.shape_b = TensorShape(1, 1, 4096, 4096);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, LLaMa_v2_case_1)
{
    params_.shape_a = TensorShape(1, 1, 1, 4096);
    params_.shape_b = TensorShape(1, 1, 4096, 11008);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, LLaMa_v2_case_2)
{
    params_.shape_a = TensorShape(1, 1, 1, 11008);
    params_.shape_b = TensorShape(1, 1, 11008, 4096);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposedCases_Normal_FP32)
{
    params_.shape_a = TensorShape(1, 1, 32, 128);
    params_.shape_b = TensorShape(1, 1, 128, 64);
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposedCases_A_Transpose_FP32)
{
    params_.shape_a = TensorShape(1, 1, 256, 512);
    params_.shape_b = TensorShape(1, 1, 512, 1024);
    params_.a_transposed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposedCases_B_Transpose_FP32)
{
    params_.shape_a = TensorShape(1, 1, 1024, 2048);
    params_.shape_b = TensorShape(1, 1, 2048, 4096);
    params_.b_transposed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposeCases_A_B_Transpose_FP32)
{
    params_.shape_a = TensorShape(1, 1, 8192, 4096);
    params_.shape_b = TensorShape(1, 1, 4096, 2048);
    params_.a_transposed = true;
    params_.b_transposed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposedCases_Normal_FP16)
{
    params_.shape_a = TensorShape(1, 1, 8192, 4096);
    params_.shape_b = TensorShape(1, 1, 4096, 2048);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposedCases_A_Transpose_FP16)
{
    params_.shape_a = TensorShape(1, 1, 512, 2048);
    params_.shape_b = TensorShape(1, 1, 2048, 128);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.a_transposed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposedCases_B_Transpose_FP16)
{
    params_.shape_a = TensorShape(1, 1, 128, 256);
    params_.shape_b = TensorShape(1, 1, 256, 512);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.b_transposed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, TransposeCases_A_B_Transpose_FP16)
{
    params_.shape_a = TensorShape(1, 1, 32, 128);
    params_.shape_b = TensorShape(1, 1, 128, 64);
    params_.dt = DataType::eFp16;
    params_.allow_fp16_computations = true;
    params_.a_transposed = true;
    params_.b_transposed = true;
    run();
}