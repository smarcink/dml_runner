
#include <gtest/gtest.h>

struct TestShapes
{
    std::size_t B;
    std::size_t M;
    std::size_t K;
    std::size_t N;
};


// The fixture for testing class Foo.
class DnnlPluginNext_GEMM : public testing::TestWithParam<int>
{
protected:
    DnnlPluginNext_GEMM() {
        // You can do set-up work for each test here.
    }

    ~DnnlPluginNext_GEMM() override {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    bool run()
    {
        return true;
    }

private:

};


TEST_P(DnnlPluginNext_GEMM, TestParametrized)
{
    EXPECT_TRUE(run());
}

//INSTANTIATE_TEST_SUITE_P(TestsFP32, DnnlPluginNext_GEMM, testing::Range(0, 10),
//    testing::PrintToStringParamName());