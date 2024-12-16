#include "ai_driver_model.h"
#include "impl/gpu_context.h"
#include "impl/model.h"

#include <iostream>
#include <cassert>

AI_DRIVER_API ai_driver_model_descriptor_t aiDriverCreateModelDescriptor()
{
    auto md = new ai_driver::ModelDescriptorDAG();
    return reinterpret_cast<ai_driver_model_descriptor_t>(md);
}

AI_DRIVER_API void aiDriverDestroyModelDescriptor(ai_driver_model_descriptor_t md)
{
    std::cout << "Destroyed Model Descriptor" << std::endl;
    auto typed_md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(md);
    delete typed_md;
}

AI_DRIVER_API void aiDriverDestroyModel(ai_driver_model_t model)
{
    std::cout << "Destroyed Model " << std::endl;
    auto typed_md = reinterpret_cast<ai_driver::ExecutableModel*>(model);
    delete typed_md;
}

AI_DRIVER_API ai_driver_model_t aiDriverCompileModelDescriptor(ai_driver_context_handle_t context, ai_driver_stream_t stream, ai_driver_model_descriptor_t model_desc, ai_driver_tensor_mapping_t* input_mapping_list, size_t input_mapping_size)
{
    std::cout << "aiDriverCompileModelDescriptor" << std::endl;
    try
    {
        auto ctx = reinterpret_cast<ai_driver::GpuContext*>(context);
        auto md = reinterpret_cast<ai_driver::ModelDescriptorDAG*>(model_desc);
        auto str = reinterpret_cast<ai_driver::GpuStream*>(stream);

        if (!input_mapping_list || input_mapping_size == 0)
        {
            std::cout << "Wrong param input_mapping_list is nullptr or input_mapping_size is 0 " << std::endl;
            return nullptr;
        }
        ai_driver::TensorMapping im{};
        for (auto i = 0; i < input_mapping_size; i++)
        {
            im[input_mapping_list[i].id] = ai_driver::Tensor(input_mapping_list[i].tensor);
        }
        auto* exec_model = new ai_driver::ExecutableModel(md->compile(*ctx, *str, im));
        return reinterpret_cast<ai_driver_model_t>(exec_model);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "exception: " << ex.what() << '\n';
    }
    catch (...)
    {
        std::cerr << "unknown exception!\n";
    }
    return nullptr;
}

AI_DRIVER_API bool aiDriverModelSetResource(ai_driver_model_t model, ai_driver_node_id_t id, ai_driver_resource_t resource)
{
    std::cout << "aiDriverModelSetResource" << std::endl;
    try
    {
        auto md = reinterpret_cast<ai_driver::ExecutableModel*>(model);
        md->set_resource(id, std::make_unique<ai_driver::GpuResource>(resource));
    }
    catch (...)
    {
        return false;
    }
    return true;
}

AI_DRIVER_API bool aiDriverModelGetOutputs(ai_driver_model_t model, ai_driver_tensor_mapping_t* list, size_t* size)
{
    std::cout << "aiDriverModelGetOutputs" << std::endl;
    try
    {
        auto md = reinterpret_cast<ai_driver::ExecutableModel*>(model);
        const auto outputs = md->get_outputs();
        if (!list && size)
        {
            *size = outputs.size();
        }
        else if (list)
        {
            std::size_t idx = 0;
            for (const auto& [id, tensor] : outputs)
            {
                list[idx++] = ai_driver_tensor_mapping_t{ id, tensor };
            }
        }
        else
        {
            std::cerr << "invalid argument\n";
            return false;
        }
    }
    catch (...)
    {
        std::cerr << "unknown exception!\n";
        return false;
    }
    return true;
}

AI_DRIVER_API bool aiDriverExecuteModel(ai_driver_model_t model, ai_driver_stream_t stream)
{
    std::cout << "aiDriverExecuteModel" << std::endl;
    try
    {
        auto md = reinterpret_cast<ai_driver::ExecutableModel*>(model);
        auto str = ai_driver::GpuStream(stream);  // temporary object, handle passed by user, library do not own this.
        md->execute(str);
    }
    catch (...)
    {
        std::cerr << "unknown exception!\n";
        return false;
    }
    return true;
}

