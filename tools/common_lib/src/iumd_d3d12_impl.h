#pragma once

#include "dx12_utils.h"

#include <oneapi/dnnl/iumd/iumd.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#define INTC_IGDEXT_D3D12
#include <igdext.h>

#include <cassert>
#include <string>
#include <variant>

namespace iumd
{
namespace custom_metacommand
{
    class UmdD3d12Memory : public IUMDMemory
    {
    public:
        enum class Type
        {
            eHandle,
            eResource,
            eUnknown,
        };
    public:
        UmdD3d12Memory() = default;
        UmdD3d12Memory(D3D12_GPU_DESCRIPTOR_HANDLE handle)
            : v_(handle)
            , type_(Type::eHandle)
        {
        }

        UmdD3d12Memory(ComPtr<ID3D12Resource> rsc)
            : v_(rsc)
            , type_(Type::eResource)
        {
        }

        D3D12_GPU_DESCRIPTOR_HANDLE get_gpu_descriptor_handle() const { return std::get<static_cast<std::size_t>(Type::eHandle)>(v_); }
        ID3D12Resource* get_resource() const { return std::get<static_cast<std::size_t>(Type::eResource)>(v_).Get(); }

        Type get_type() const
        {
            return type_;
        }
    private:
        Type type_ = Type::eUnknown;
        std::variant<D3D12_GPU_DESCRIPTOR_HANDLE, ComPtr<ID3D12Resource>> v_;
    };

    class UmdD3d12PipelineStateObject : public IUMDPipelineStateObject
    {
    public:
        friend class UmdD3d12Device;

        const char* get_name() const override;
        IUMDDevice* get_parent_device() override;

        bool set_kernel_arg(std::size_t index, const IUMDMemory* memory, std::size_t offset) override;
        bool set_kernel_arg(std::size_t index, IUMDPipelineStateObject::ScalarArgType scalar) override;

        bool execute(ID3D12GraphicsCommandList4* cmd_list, const std::array<std::size_t, 3>& gws, const std::array<std::size_t, 3>& lws);

    private:
        UmdD3d12PipelineStateObject(class UmdD3d12Device* device, const char* kernel_name,
            const char* code_string, const char* build_options,
            UMD_SHADER_LANGUAGE language);

    private:
        UmdD3d12Device* device_ = nullptr;
        ComPtr<ID3D12MetaCommand> mc_ = nullptr;
        std::unordered_map<std::size_t, std::pair<const UmdD3d12Memory*, std::size_t>> resources_;
        std::unordered_map<std::size_t, ScalarArgType> scalars_;
        std::string name_;
    };

    class UmdD3d12Event : public IUMDEvent
    {
    public:

    };

    class UmdD3d12Device : public IUMDDevice
    {
    public:
        UmdD3d12Device(ID3D12Device* device, INTCExtensionInfo extension_info);

        IUMDPipelineStateObject::Ptr
            create_pipeline_state_object(const char* kernel_name,
                const char* code_string, const char* build_options,
                UMD_SHADER_LANGUAGE language) override
        {
            return std::unique_ptr<UmdD3d12PipelineStateObject>(new UmdD3d12PipelineStateObject(this, kernel_name, code_string, build_options, language));
        }

        IUMDMemory::Ptr allocate_memory(std::size_t size) override
        {
            ComPtr<ID3D12Resource> ret;
            const auto heap_props = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            const auto buffer_desc = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            throw_if_failed(impl_->CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(ret.ReleaseAndGetAddressOf())), "create commited resource");
            return std::make_unique<UmdD3d12Memory>(ret);
        }


        bool fill_memory(IUMDCommandList* cmd_list, const IUMDMemory* dst_mem, std::size_t size,
            const void* pattern, std::size_t pattern_size,
            const std::vector<IUMDEvent*>& deps = {},
            std::shared_ptr<IUMDEvent>* out = nullptr) override;

        std::uint32_t get_eu_count() const override
        {
            return sku_.eu_count;
        };

        std::uint32_t get_max_wg_size() const override
        {
            return sku_.threads_per_eu * sku_.eu_per_dss * sku_.max_simd_size;
        };

        UMD_IGFX get_umd_igfx() const override
        {
            return sku_.igfx;
        }

        const char* get_name() const override
        {
            return sku_.name.c_str();
        }

        bool can_use_systolic() const override
        {
            if (sku_.igfx == UMD_IGFX_DG2)
            {
                return true;
            }
            return false;
        }

        virtual std::uint32_t get_vendor_id() const override
        {
            const auto desc = get_adapter_desc();
            return desc.VendorId;
        }

        bool do_support_extension(UMD_EXTENSIONS ext) const
        {
            if (ext & UMD_EXTENSIONS_SUBGROUP)
            {
                return true;
            }
            else if (ext & UMD_EXTENSIONS_FP16)
            {
                return true;
            }
            else if (ext & UMD_EXTENSIONS_FP64)
            {
                return false;  //ToDo: check this
            }
            else if (ext & UMD_EXTENSIONS_DP4A)
            {
                return sku_.igfx >= UMD_IGFX_TIGERLAKE_LP;
            }
            else if (ext & UMD_EXTENSIONS_DPAS)
            {
                return sku_.igfx >= UMD_IGFX_DG2;
            }
            else if (ext & UMD_EXTENSIONS_SDPAS)
            {
                return sku_.igfx == UMD_IGFX_DG2;
            }
            else if (ext & UMD_EXTENSIONS_VARIABLE_THREAD_PER_EU)
            {
                return false;
            }
            else if (ext & UMD_EXTENSIONS_GLOBAL_FLOAT_ATOMICS)
            {
                return false;
            }
            assert(false);
            return false;
        }

        const ID3D12Device* get_d3d12_device() const { return impl_; }
        ID3D12Device* get_d3d12_device() { return impl_; }

    private:
        DXGI_ADAPTER_DESC get_adapter_desc() const
        {
            DXGI_ADAPTER_DESC desc{};
            throw_if_failed(adapter_->GetDesc(&desc), "get adapter desc");
            return desc;
        }

    private:
        struct SKU
        {
            std::string name = "";
            std::uint32_t eu_count = 0;
            std::uint32_t threads_per_eu = 0;
            std::uint32_t eu_per_dss = 0;
            std::uint32_t max_simd_size = 0;
            UMD_IGFX igfx = UMD_IGFX_UNKNOWN;
        };

    private:
        ID3D12Device* impl_;
        IDXGIAdapter* adapter_;
        SKU sku_{};
        iumd::IUMDPipelineStateObject::Ptr buffer_filler_ = nullptr;
    };

    class UmdD3d12CommandList : public IUMDCommandList
    {
    public:
        UmdD3d12CommandList(ID3D12GraphicsCommandList4* cmd_list)
            : impl_(cmd_list)
        {}

        UmdD3d12CommandList(ID3D12GraphicsCommandList* cmd_list)
        {
            throw_if_failed(cmd_list->QueryInterface(&impl_), "cant cast d3d12 device to ID3D12Device5");
        }

        bool dispatch(IUMDPipelineStateObject* pso, const std::array<std::size_t, 3>& gws, const std::array<std::size_t, 3>& lws, const std::vector<IUMDEvent*>& deps = {}, std::shared_ptr<IUMDEvent>* out = nullptr) override;

        bool supports_out_of_order() const { return false; }
        bool supports_profiling() const { return false; }

    private:
        void wait_for_deps(const std::vector<IUMDEvent*>& deps);
        void put_barrier(std::shared_ptr<IUMDEvent>* out = nullptr);

    private:
        ID3D12GraphicsCommandList4* impl_;
    };
} // namespace custom_metacommand
} // namespace umd