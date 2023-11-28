#include <tuple>

#include <gtest/gtest.h>

#include <gemm.h>

#include "utils.h"



class DnnlPluginNext_GEMM : public NodeDispatcherBase, public testing::Test
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
        params_.b_transposed = false;
        params_.use_dnnl_for_reference_calculations = true;
    }

    ~DnnlPluginNext_GEMM() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }


protected:
    std::unique_ptr<NodeDispatcher> create_dispatcher_impl() override
    {
        params_.allow_fp16_computations = params_.dt == DataType::eFp16;
        auto node = std::make_unique<GemmUmdD3d12Dispatcher>(std::move(params_),
            g_dx12_engine.intel_extension_d3d12,
            g_dx12_engine.d3d12_device.Get(),
            g_dx12_engine.dml_device.Get(),
            g_dx12_engine.dml_command_recorder.Get(),
            g_dx12_engine.command_list.Get());
        return node;
    }

protected:
    GemmBaseDispatcher::create_params_t params_{};
};

TEST_F(DnnlPluginNext_GEMM, SD_dims_0)
{
    params_.shape_a = TensorShape(2, 1, 4096, 320);
    params_.shape_b = TensorShape(2, 1, 320, 4096);
    params_.dt = DataType::eFp16;
    params_.b_managed = true;
    run();
}


TEST_F(DnnlPluginNext_GEMM, SD_dims_1)
{
    params_.shape_a = TensorShape(2, 1, 64, 1280);
    params_.shape_b = TensorShape(2, 1, 1280, 1280);
    params_.dt = DataType::eFp16;
    params_.b_managed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, SD_dims_2)
{
    params_.shape_a = TensorShape(2, 1, 1024, 640);
    params_.shape_b = TensorShape(2, 1, 640, 1920);
    params_.dt = DataType::eFp16;
    params_.b_managed = true;
    run();
}

TEST_F(DnnlPluginNext_GEMM, SD_dims_2_with_alpha)
{
    params_.shape_a = TensorShape(2, 1, 1024, 640);
    params_.shape_b = TensorShape(2, 1, 640, 1920);
    params_.dt = DataType::eFp16;
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

TEST_F(DnnlPluginNext_GEMM, SmallMandNBigK)
{
    params_.shape_a = TensorShape(1, 1, 20, 20000);
    params_.shape_b = TensorShape(1, 1, 20000, 20);
    params_.dt = DataType::eFp16;
    run();
}
