#include "iumd_d3d12_impl.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////
// Custom Metacommand
// {9C365CB6-AF13-49B6-BA9C-4B74E10FCDE1}
static constexpr GUID GUID_CUSTOM =
{ 0x9c365cb6, 0xaf13, 0x49b6,{ 0xba, 0x9c, 0x4b, 0x74, 0xe1, 0xf, 0xcd, 0xe1 } };

//////////////////////////////////////////////////////////////////////////
enum META_COMMAND_CUSTOM_SHADER_LANGUAGE : UINT64
{
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE = 0,
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL,
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL_STATELESS,
    META_COMMAND_CUSTOM_SHADER_LANGUAGE_ZEBIN_ELF,
};

//////////////////////////////////////////////////////////////////////////
enum META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE : UINT64
{
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_NONE = 0,
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_HANDLE,
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_ADDRESS
};

//////////////////////////////////////////////////////////////////////////
struct META_COMMAND_CREATE_CUSTOM_DESC
{
    UINT64 ShaderSourceCode;
    UINT64 ShaderSourceCodeSize;
    UINT64 BuildOptionString;
    UINT64 BuildOptionStringSize;
    META_COMMAND_CUSTOM_SHADER_LANGUAGE ShaderLanguage;
};

//////////////////////////////////////////////////////////////////////////
struct META_COMMAND_INITIALIZE_CUSTOM_DESC
{
    D3D12_GPU_DESCRIPTOR_HANDLE Resources[10];
};

//////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
struct META_COMMAND_EXECUTE_CUSTOM_DESC
{
    UINT64                      ResourceCount;
    META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE ResourceBindType[50];
    UINT64                      ResourceBindOffset[50];
    D3D12_GPU_DESCRIPTOR_HANDLE Resources[50];            // use address or handles
    D3D12_GPU_VIRTUAL_ADDRESS   ResourcesAddress[50];     // use address or handles
    UINT64                      ResourcesByteOffset[50];  // works only in stateless mode

    UINT64 RuntimeConstants;      // buffer with constants
    UINT64 RuntimeConstantsCount; // how many runtime constants in total
    UINT64 RuntimeConstantsBindOffsets[40];  // offsets in bindings
    UINT64 RuntimeConstantsMemorySizes[40];   // how much bytes to copy
    UINT64 RuntimeConstantsMemoryOffsets[40]; // bytes offset into "RuntimeConstants" buffer

    UINT64 DispatchGlobalWorkSize[3];
    UINT64 DispatchLocalWorkSize[3];
    UINT64 SharedLocalMemorySize;
};


iumd::custom_metacommand::UmdD3d12Device::UmdD3d12Device(ID3D12Device* device, INTCExtensionInfo extension_info)
    : impl_(device)
{
    assert(device);
    ComPtr<IDXGIFactory4> dxgi_factory;
    throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(dxgi_factory.ReleaseAndGetAddressOf())), "dxgi factory");
    dxgi_factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter_));
    const auto adapter_desc = get_adapter_desc();

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    if (SUCCEEDED(device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1)))) {
        sku_.max_simd_size = options1.WaveLaneCountMax;
    }
    const auto wstr_name = std::wstring(adapter_desc.Description);
    //sku_.name = std::string(wstr_name.begin(), wstr_name.end());  // not ideal, enough for now
    sku_.name = std::string(wstr_name.length(), 0);  // not ideal, enough for now
    std::transform(wstr_name.begin(), wstr_name.end(), sku_.name.begin(), [](wchar_t ch) {return (char)ch; }); // good enough for latin characters...
    sku_.eu_count = extension_info.IntelDeviceInfo.EUCount;
    sku_.igfx = [](const auto name, int eu_count)
    {
        if (std::wcscmp(name, L"Tigerlake") == 0)
        {
            return UMD_IGFX::eTIGERLAKE_LP;
        }
        else if (std::wcscmp(name, L"Meteorlake") == 0)
        {
            return UMD_IGFX::eMETEORLAKE;
        }
        else if (std::wcscmp(name, L"DG2") == 0)
        {
            return UMD_IGFX::eDG2;
        }
        else if (std::wcscmp(name, L"Battlemage") == 0)
        {
            return UMD_IGFX::eBMG;
        }
        else if (std::wcscmp(name, L"Lunarlake") == 0)
        {
            return UMD_IGFX::eLNL;
        }
        else if (std::wcscmp(name, L"Arrowlake") == 0)
        {
            //reference: https://gfxspecs.intel.com/Predator/Home/Index/55414
            if (eu_count == 128)
            {
				return UMD_IGFX::eARLH;
			}
            if (eu_count == 64)
			{
				return UMD_IGFX::eARLS;
			}
        }
        return UMD_IGFX::eUNKNOWN;
    }(extension_info.IntelDeviceInfo.GTGenerationName, sku_.eu_count);

    if (sku_.igfx == UMD_IGFX::eDG2 || sku_.igfx == UMD_IGFX::eMETEORLAKE || sku_.igfx == UMD_IGFX::eARLH || sku_.igfx == UMD_IGFX::eARLS)
    {
        sku_.eu_per_dss = 16;
        sku_.threads_per_eu = 8;
        sku_.hw_simd_size = 8;

    }
    else if (sku_.igfx == UMD_IGFX::eTIGERLAKE_LP)
    {
        sku_.eu_per_dss = 16;
        sku_.threads_per_eu = 7;
        sku_.hw_simd_size = 8;
    }
    else if (sku_.igfx == UMD_IGFX::eLNL || sku_.igfx == UMD_IGFX::eBMG)
    {
        sku_.eu_per_dss = 8;
        sku_.threads_per_eu = 8;
        sku_.hw_simd_size = 16;
    }

    switch(sku_.igfx)
    {
    case UMD_IGFX::eARLH:
        sku_.l3_cache_size = 8192 * 1024;
        break;
    default:
        break;  // to-do - add more cases
    }

}


bool iumd::custom_metacommand::UmdD3d12Device::fill_memory(IUMDCommandList* cmd_list, const IUMDMemory* dst_mem, std::size_t size, const void* pattern, std::size_t pattern_size,
    const std::vector<IUMDEvent*>& deps, std::shared_ptr<IUMDEvent>* out)
{
    if (!cmd_list || !dst_mem || !pattern || (pattern_size == 0))
    {
        return false;
    }
    if (size == 0)
    {
        // no work to do, but it's not cosidered error
        return true;
    }

    // compile copy kernel, if it's first time hitting this function
    if (!buffer_filler_) {
        const char* code_string
            = "__attribute__((reqd_work_group_size(1,1, 1))) "
            "__kernel void buffer_pattern_filler(__global unsigned char* output, unsigned char pattern)"
            "{"
            "const uint id = get_global_id(0);"
            "output[id] = pattern;"
            "}";

        buffer_filler_ = create_pipeline_state_object("buffer_pattern_filler", code_string, std::strlen(code_string), "", UMD_SHADER_LANGUAGE::eOCL_STATELESS);
    }
    assert(buffer_filler_);

    auto typed_cmd_list = dynamic_cast<iumd::custom_metacommand::UmdD3d12CommandList*>(cmd_list);
    
    buffer_filler_->set_kernel_arg(0, dst_mem);
    buffer_filler_->set_kernel_arg(1, IUMDPipelineStateObject::ScalarArgType{pattern_size, pattern});
    const auto lws = std::array<std::size_t, 3>{1, 1, 1};
    const auto gws = std::array<std::size_t, 3>{size, 1, 1};

    return typed_cmd_list->dispatch(buffer_filler_.get(), gws, lws, deps, out);
}

bool iumd::custom_metacommand::UmdD3d12Device::do_support_extension(UMD_EXTENSIONS ext) const
{
    switch (ext)
    {
    case UMD_EXTENSIONS::eSUBGROUP: return true;
    case UMD_EXTENSIONS::eFP16: return true;
    case UMD_EXTENSIONS::eFP64: return false;  //ToDo: check this
    case UMD_EXTENSIONS::eDP4A: return sku_.igfx >= UMD_IGFX::eTIGERLAKE_LP;
    case UMD_EXTENSIONS::eDPAS:  return can_use_systolic();
    case UMD_EXTENSIONS::eSDPAS:  return can_use_systolic();
    case UMD_EXTENSIONS::eVARIABLE_THREAD_PER_EU: return false;
    case UMD_EXTENSIONS::eGLOBAL_FLOAT_ATOMICS: return true;
    case UMD_EXTENSIONS::eGLOBAL_INT32_ATOMICS: return true;
    case UMD_EXTENSIONS::eLOCAL_INT32_ATOMICS: return true;
    case UMD_EXTENSIONS::eINT64_ATOMICS: return true;
    default:
        assert(false);
    }
    return false;
}

bool iumd::custom_metacommand::UmdD3d12Device::do_support_ngen_kernels() const
{
    // look at OneDNN for OpenCL runtime - they compile dummy kernel to get information if NGEN is supported - this is good and roboust solution.
    // For arrowlake: only arl-h which has DPAS.
    printf("Add proper NGEN support detection mechanism!\n"); 
    if (sku_.igfx == UMD_IGFX::eDG2 || sku_.igfx == UMD_IGFX::eMETEORLAKE || 
        sku_.igfx == UMD_IGFX::eLNL || sku_.igfx == UMD_IGFX::eBMG || sku_.igfx == UMD_IGFX::eARLH)
    {
        return true;
    }
    return false;
}

iumd::custom_metacommand::UmdD3d12PipelineStateObject::UmdD3d12PipelineStateObject(iumd::custom_metacommand::UmdD3d12Device* device, const char* kernel_name, const void* kernel_code, std::size_t kernel_code_size, const char* build_options, UMD_SHADER_LANGUAGE language)
    : name_(kernel_name)
    , device_(device)
{
    auto d3d12_dev = device->get_d3d12_device();
    ID3D12Device5* dev5 = nullptr;
    throw_if_failed(d3d12_dev->QueryInterface(&dev5), "cant cast d3d12 device to ID3D12Device5");

    if (kernel_code == nullptr || kernel_code_size == 0)
    {
        throw std::runtime_error("Code string is empty. Please provide valid kernel/binary data.\n");
    }

    META_COMMAND_CREATE_CUSTOM_DESC create_desc{};
    create_desc.ShaderSourceCode = reinterpret_cast<UINT64>(kernel_code);
    create_desc.ShaderSourceCodeSize = kernel_code_size;
    create_desc.BuildOptionString = reinterpret_cast<UINT64>(build_options);
    create_desc.BuildOptionStringSize = build_options ? std::strlen(build_options) : 0ull;
    switch (language)
    {
    case UMD_SHADER_LANGUAGE::eOCL:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL;
        break;
    case UMD_SHADER_LANGUAGE::eOCL_STATELESS:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_OCL_STATELESS;
        break;
    case UMD_SHADER_LANGUAGE::eZEBIN_ELF:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_ZEBIN_ELF;
        break;
    default:
        create_desc.ShaderLanguage = META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE;
        
    }
    assert(create_desc.ShaderLanguage != META_COMMAND_CUSTOM_SHADER_LANGUAGE_NONE);
    
    throw_if_failed(dev5->CreateMetaCommand(GUID_CUSTOM, 0, &create_desc, sizeof(create_desc), IID_PPV_ARGS(mc_.ReleaseAndGetAddressOf())), "cant create custom metacommand");

    // Cache kernel. It's not ideal, but should be enough to represent needs of real mechanism. In driver this would store/read to disk.
    cached_kernel_.resize(kernel_code_size);
    std::memcpy(cached_kernel_.data(), kernel_code, kernel_code_size);
}

const char* iumd::custom_metacommand::UmdD3d12PipelineStateObject::get_name() const
{
    return name_.c_str();
}

iumd::IUMDDevice* iumd::custom_metacommand::UmdD3d12PipelineStateObject::get_parent_device()
{
    return device_;
}

bool iumd::custom_metacommand::UmdD3d12PipelineStateObject::set_kernel_arg(std::size_t index, const IUMDMemory* memory, std::size_t offset)
{
    auto typed_mem = memory ? dynamic_cast<const iumd::custom_metacommand::UmdD3d12Memory*>(memory) : nullptr;
    resources_[index] = { typed_mem, offset };
    return true;
}

const void* iumd::custom_metacommand::UmdD3d12PipelineStateObject::get_binary() const
{
    assert(!cached_kernel_.empty());
    return cached_kernel_.data();
}

const std::size_t iumd::custom_metacommand::UmdD3d12PipelineStateObject::get_binary_size() const
{
    assert(!cached_kernel_.empty());
    return cached_kernel_.size();
}

bool iumd::custom_metacommand::UmdD3d12PipelineStateObject::set_kernel_arg(std::size_t index, IUMDPipelineStateObject::ScalarArgType scalar)
{
    scalars_[index] = scalar;
    return true;
}

bool iumd::custom_metacommand::UmdD3d12PipelineStateObject::set_kernel_arg(std::size_t index, std::size_t slm_size)
{
    locals_[index] = slm_size;
    return true;
}

bool iumd::custom_metacommand::UmdD3d12PipelineStateObject::execute(ID3D12GraphicsCommandList4* cmd_list, const std::array<std::size_t, 3>& gws, const std::array<std::size_t, 3>& lws)
{
    assert(gws.size() == lws.size());

    // [0] Calculate dispatch thread size
    META_COMMAND_EXECUTE_CUSTOM_DESC exec_desc{};
    for (std::size_t i = 0; i < gws.size(); i++)
    {
        if (gws[i] == 0 || lws[i] == 0)
        {
            return false;
        }
        exec_desc.DispatchGlobalWorkSize[i] = gws[i];
        exec_desc.DispatchLocalWorkSize[i] = lws[i];
    }
    exec_desc.ResourceCount = resources_.size();
    if (exec_desc.ResourceCount >= std::size(exec_desc.Resources))
    {
        assert(!"Please extend number of supported resources for custom metacommand!");
        return false;
    }

    // [1] Prepare resoruces pointer handles 
    for (std::size_t idx = 0; const auto& [bind_indx, memory] : resources_)
    {
        exec_desc.ResourceBindOffset[idx] = bind_indx;

        const auto& [mem_ptr, base_offset] = memory;
        if (mem_ptr)
        {
            // set offset no matter what type of resource
            exec_desc.ResourcesByteOffset[idx] = base_offset;

            // use type of binding based on memory type
            const auto type = mem_ptr->get_type();
            if (UmdD3d12Memory::Type::eHandle == type)
            {
                exec_desc.ResourceBindType[idx] = META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_HANDLE;
                exec_desc.Resources[idx] = mem_ptr->get_gpu_descriptor_handle();
            }
            else if (UmdD3d12Memory::Type::eResource == type)
            {
                exec_desc.ResourceBindType[idx] = META_COMMAND_CUSTOM_RESOURCE_BIND_TYPE_ADDRESS;
                exec_desc.ResourcesAddress[idx] = mem_ptr->get_resource()->GetGPUVirtualAddress();
            }
            else
            {
                assert(!"Unsupported memory type!");
                return false;
            }
        }
        idx++;
    }


    // [2] Build execution time constants 
    std::vector<std::byte> execution_time_constants;
    for (std::size_t i = 0; const auto& [idx, scalar] : scalars_)
    {
        exec_desc.RuntimeConstantsBindOffsets[i] = idx;
        exec_desc.RuntimeConstantsMemorySizes[i] = scalar.size;
        exec_desc.RuntimeConstantsMemoryOffsets[i] = execution_time_constants.size();;
        execution_time_constants.resize(execution_time_constants.size() + scalar.size);
        i++;
    }

    auto* ptr_to_copy_data = execution_time_constants.data();
    for (const auto& [idx, scalar] : scalars_)
    {
        std::memcpy(ptr_to_copy_data, scalar.data, scalar.size);
        ptr_to_copy_data += scalar.size;
    }
    exec_desc.RuntimeConstantsCount = scalars_.size();
    exec_desc.RuntimeConstants = reinterpret_cast<UINT64>(execution_time_constants.data());

    // [3] Build slm
    if (locals_.size() > 1)
    {
        assert("!Unsupported case. Please remove this check and test - if it fails most probably driver need changes!");
        return false;
    }
    for (const auto& [idx, slm_size] : locals_)
    {
        exec_desc.SharedLocalMemorySize += slm_size;
    }

    cmd_list->ExecuteMetaCommand(mc_.Get(), &exec_desc, sizeof(exec_desc));
    return true;
}

bool iumd::custom_metacommand::UmdD3d12CommandList::dispatch(iumd::IUMDPipelineStateObject* pso, const std::array<std::size_t, 3>& gws, const std::array<std::size_t, 3>& lws, const std::vector<iumd::IUMDEvent*>& deps, std::shared_ptr<iumd::IUMDEvent>* out)
{
    wait_for_deps(deps);

    auto typed_pso = dynamic_cast<iumd::custom_metacommand::UmdD3d12PipelineStateObject*>(pso);
    const auto result = typed_pso->execute(impl_, gws, lws);
 
    put_barrier(out);
    return result;
}


void iumd::custom_metacommand::UmdD3d12CommandList::wait_for_deps(const std::vector<IUMDEvent*>& deps)
{
    if (!deps.empty())
    {
        // Single global barrier is enough for now, because we use b.UAV.pResource = nullptr;
        // If we would support concrete resources barriers, than we need a way to specify resource pointer to b.UAV.pResource
        constexpr const bool single_barrier_is_enough = true;
        std::vector<D3D12_RESOURCE_BARRIER> barriers(single_barrier_is_enough ? 1u : deps.size());
        for (auto& b : barriers)
        {
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            b.UAV.pResource = nullptr;
        }
        impl_->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }
}

void iumd::custom_metacommand::UmdD3d12CommandList::put_barrier(std::shared_ptr<IUMDEvent>* out)
{
    if (out)
    {
        *out = std::make_shared<iumd::custom_metacommand::UmdD3d12Event>();
    }
}