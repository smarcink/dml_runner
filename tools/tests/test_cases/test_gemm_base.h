#pragma once

#include <gemm.h>
#include "utils.h"
#include "config.h"

class GemmBaseTestDispatcher : public NodeDispatcherBase
{
public:
protected:
    std::unique_ptr<NodeDispatcher> create_dispatcher_impl() override
    {
        std::unique_ptr<NodeDispatcher> node = nullptr;
        if (g_test_config.run_dml)
        {
            node = std::make_unique<GemmDmlDispatcher>(std::move(get_params()),
                true,
                g_dx12_engine.d3d12_device.Get(),
                g_dx12_engine.dml_device.Get(),
                g_dx12_engine.dml_command_recorder.Get(),
                g_dx12_engine.command_list.Get());
        }
        else
        {
            node = std::make_unique<GemmUmdD3d12Dispatcher>(std::move(get_params()),
                GemmUmdD3d12Dispatcher::gemm_umdd3d12_params_t{},
                g_dx12_engine.intel_extension_d3d12,
                g_dx12_engine.d3d12_device.Get(),
                g_dx12_engine.dml_device.Get(),
                g_dx12_engine.dml_command_recorder.Get(),
                g_dx12_engine.command_list.Get());
        }
        return node;
    }

    virtual GemmBaseDispatcher::create_params_t get_params() = 0;
};