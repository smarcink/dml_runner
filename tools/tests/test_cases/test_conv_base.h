#pragma once

#include <conv.h>
#include "utils.h"
#include "config.h"

class ConvolutionBaseTestDispatcher : public NodeDispatcherBase
{
public:
protected:
    std::unique_ptr<NodeDispatcher> create_dispatcher_impl() override
    {
        std::unique_ptr<NodeDispatcher> node = nullptr;
        if (g_test_config.run_dml)
        {
            node = std::make_unique<ConvolutionDirectMLDispatcher>(get_params(),
                false,
                g_dx12_engine.d3d12_device.Get(),
                g_dx12_engine.dml_device.Get(),
                g_dx12_engine.dml_command_recorder.Get(),
                g_dx12_engine.command_list.Get());
        }
        else
        {
            node = std::make_unique<ConvolutionUmdD3d12Dispatcher>(get_params(),
                ConvolutionUmdD3d12Dispatcher::conv_umdd3d12_params_t{},
                g_dx12_engine.intel_extension_d3d12,
                g_dx12_engine.d3d12_device.Get(),
                g_dx12_engine.dml_device.Get(),
                g_dx12_engine.dml_command_recorder.Get(),
                g_dx12_engine.command_list.Get());
        }

        return node;
    }

    virtual ConvolutionBaseDispatcher::create_params_t get_params() = 0;
};