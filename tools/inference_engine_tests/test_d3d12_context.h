#pragma once
#include <inference_engine.h>

#include <utility>

namespace test_ctx
{
    class TestGpuContext
    {
    public:
        TestGpuContext();

        TestGpuContext(const TestGpuContext&& rhs) = delete;
        TestGpuContext& operator=(const TestGpuContext&& rhs) = delete;
        TestGpuContext(TestGpuContext&& rhs) noexcept
        {
            std::swap(ctx_, rhs.ctx_);
            std::swap(device_, rhs.device_);
        }
        TestGpuContext& operator=(TestGpuContext&& rhs) noexcept
        {
            if (this != &rhs)
            {
                std::swap(this->ctx_, rhs.ctx_);
                std::swap(this->device_, rhs.device_);
            }
            return *this;
        }

        ~TestGpuContext();

        inference_engine_context_handle_t get()
        {
            return ctx_;
        }    

    private:
        inference_engine_context_handle_t ctx_ = nullptr;
        inference_engine_device_t device_ = nullptr;
    };
}
