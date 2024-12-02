#include "inference_engine_model.h"
#include "impl/gpu_context.h"
#include "impl/model.h"

#include <iostream>
#include <cassert>

INFERENCE_ENGINE_API inference_engine_model_descriptor_t inferenceEngineCreateModelDescriptor()
{
    auto md = new inference_engine::ModelDescriptor();
    return reinterpret_cast<inference_engine_model_descriptor_t>(md);
}

INFERENCE_ENGINE_API void inferenceEngineDestroyModelDescriptor(inference_engine_model_descriptor_t md)
{
    std::cout << "Destroyed Model Descriptor" << std::endl;
    auto typed_md = reinterpret_cast<inference_engine::ModelDescriptor*>(md);
    delete typed_md;
}

INFERENCE_ENGINE_API void inferenceEngineDestroyModel(inference_engine_model_t model)
{
    std::cout << "Destroyed Model " << std::endl;
    auto typed_md = reinterpret_cast<inference_engine::ExecutableModel*>(model);
    delete typed_md;
}

INFERENCE_ENGINE_API inference_engine_model_t inferenceEngineCompileModelDescriptor(inference_engine_context_handle_t context, inference_engine_stream_t stream, inference_engine_model_descriptor_t model_desc, inference_engine_tensor_mapping_t* input_mapping_list, size_t input_mapping_size)
{
    std::cout << "inferenceEngineCompileModelDescriptor" << std::endl;
    try
    {
        auto ctx = reinterpret_cast<inference_engine::GpuContext*>(context);
        auto md = reinterpret_cast<inference_engine::ModelDescriptor*>(model_desc);
        auto str = reinterpret_cast<inference_engine::GpuStream*>(stream);

        if (!input_mapping_list || input_mapping_size == 0)
        {
            std::cout << "Wrong param input_mapping_list is nullptr or input_mapping_size is 0 " << std::endl;
            inference_engine::set_last_error(INFERENCE_ENGINE_RESULT_INVALID_ARGUMENT);
            return nullptr;
        }
        std::vector<inference_engine::TensorMapping> im{};
        for (auto i = 0; i < input_mapping_size; i++)
        {
            im.push_back({ input_mapping_list[i].id, inference_engine::Tensor(input_mapping_list[i].tensor) });
        }
        auto* exec_model = new inference_engine::ExecutableModel(md->compile(*ctx, *str, im));
        return reinterpret_cast<inference_engine_model_t>(exec_model);
    }
    catch (const inference_engine::inference_engine_exception& ex) 
    {
        inference_engine::set_last_error(ex.val_);
    }
    catch (...)
    {
        inference_engine::set_last_error(INFERENCE_ENGINE_RESULT_OTHER);
    }
    return nullptr;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineModelSetResource(inference_engine_model_t model, inference_engine_node_id_t id, inference_engine_resource_t resource)
{
    std::cout << "inferenceEngineModelSetResource" << std::endl;
    try
    {
        auto md = reinterpret_cast<inference_engine::ExecutableModel*>(model);
        md->set_resource(id, std::make_unique<inference_engine::GpuResource>(resource));
    }
    catch (...)
    {
        return INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN;
    }
    return INFERENCE_ENGINE_RESULT_SUCCESS;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineModelGetOutputs(inference_engine_model_t model, inference_engine_tensor_mapping_t* list, size_t* size)
{
    std::cout << "inferenceEngineModelGetOutputs" << std::endl;
    try
    {
        auto md = reinterpret_cast<inference_engine::ExecutableModel*>(model);
        const auto outputs = md->get_outputs();
        if (!list && size)
        {
            *size = outputs.size();
        }
        else if (list)
        {
            for (auto i = 0; i < outputs.size(); i++)
            {
                list[i] = inference_engine_tensor_mapping_t{ outputs[i].id, outputs[i].tensor };
            }
        }
        else
        {
            return INFERENCE_ENGINE_RESULT_INVALID_ARGUMENT;
        }
    }
    catch (...)
    {
        return INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN;
    }
    return INFERENCE_ENGINE_RESULT_SUCCESS;
}

INFERENCE_ENGINE_API inference_engine_result_t inferenceEngineExecuteModel(inference_engine_model_t model, inference_engine_stream_t stream)
{
    std::cout << "inferenceEngineExecuteModel" << std::endl;
    try
    {
        auto md = reinterpret_cast<inference_engine::ExecutableModel*>(model);
        auto str = inference_engine::GpuStream(stream);  // temporary object, handle passed by user, library do not own this.
        md->execute(str);
    }
    catch (...)
    {
        return INFERENCE_ENGINE_RESULT_ERROR_UNKNOWN;
    }
    return INFERENCE_ENGINE_RESULT_SUCCESS;
}

