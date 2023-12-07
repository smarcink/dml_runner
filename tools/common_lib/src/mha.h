#pragma once
#include <vector>
#include <random>

#include "dml_base_node.h"
#include "softmax.h"

enum class MhaType
{  
    // qkv
    MhaType_QKV = 1,
};
namespace gpu_op
{
    class Mha : public DirectMlBaseNode 
    {
    public:
        Mha(MhaType mha_type, const DML_TENSOR_DATA_TYPE data_type, const dml::TensorPolicy& tensor_policy,
            const TensorShape& shape_input,  const TensorShape& shape_out,
            IDMLDevice* dml_device, ID3D12Device* d3d12_device, bool disable_mc = false)
            :DirectMlBaseNode(dml_device, d3d12_device)
        {
            if (mha_type == MhaType::MhaType_QKV)  // todo: move some codes out as general code for other mha types
            {

                const dml::TensorDimensions input_dims{ shape_input.n, shape_input.c, shape_input.d, shape_input.h, shape_input.w};
                const dml::TensorDimensions output_dims{ shape_out.n, shape_out.h, shape_out.w };

                dml::TensorProperties stacked_qkv_tensor_properites{};
                {
                    tensor_stacked_qkv_desc_.DataType = data_type;
                    tensor_stacked_qkv_desc_.Flags = DML_TENSOR_FLAG_NONE;
                    tensor_stacked_qkv_desc_.DimensionCount = static_cast<std::uint32_t>(input_dims.size());
                    tensor_stacked_qkv_desc_.Sizes = input_dims.data();

                    //stacked_qkv_tensor_properites = input_tensor_policy.Get(tensor_stacked_qkv_desc_.DataType, tensor_stacked_qkv_desc_.Flags, input_dims);
                    tensor_stacked_qkv_desc_.Strides = nullptr; // stacked_qkv_tensor_properites.strides.has_value() ? stacked_qkv_tensor_properites.strides->data() : nullptr;
                    tensor_stacked_qkv_desc_.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
                        tensor_stacked_qkv_desc_.DataType,
                        tensor_stacked_qkv_desc_.DimensionCount,
                        tensor_stacked_qkv_desc_.Sizes,
                        tensor_stacked_qkv_desc_.Strides);                      //stacked_qkv_tensor_properites.totalTensorSizeInBytes;
                    tensor_stacked_qkv_desc_.GuaranteedBaseOffsetAlignment = 0; // stacked_qkv_tensor_properites.guaranteedBaseOffsetAlignment;

                }

                dml::TensorProperties output_tensor_properites;
                {
                    tensor_out_desc_.DataType = data_type;
                    tensor_out_desc_.Flags = DML_TENSOR_FLAG_NONE;
                    tensor_out_desc_.DimensionCount = static_cast<std::uint32_t>(output_dims.size());
                    tensor_out_desc_.Sizes = output_dims.data();

                    tensor_out_desc_.Strides = nullptr; 
                    tensor_out_desc_.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
                        tensor_out_desc_.DataType,
                        tensor_out_desc_.DimensionCount,
                        tensor_out_desc_.Sizes,
                        tensor_out_desc_.Strides);
                    tensor_out_desc_.GuaranteedBaseOffsetAlignment = 0; 
                }

                DML_TENSOR_DESC output_desc{};
                output_desc.Desc = &tensor_out_desc_;
                output_desc.Type = DML_TENSOR_TYPE_BUFFER;

                DML_TENSOR_DESC stacked_qkv_tensor_desc = {};
                stacked_qkv_tensor_desc.Desc = &tensor_stacked_qkv_desc_;
                stacked_qkv_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
                
                DML_MULTIHEAD_ATTENTION_OPERATOR_DESC desc = {};
                desc.StackedQueryKeyValueTensor = &stacked_qkv_tensor_desc;
                desc.OutputTensor = &output_desc;
                desc.Scale = 0.001f;
                desc.MaskFilterValue = -10000000;
                desc.HeadCount = 8;
                desc.MaskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;

                DML_OPERATOR_DESC dml_operator_desc{};
                dml_operator_desc.Type = DML_OPERATOR_MULTIHEAD_ATTENTION;
                dml_operator_desc.Desc = &desc;

                throw_if_failed(dml_device->CreateOperator(
                    &dml_operator_desc, IID_PPV_ARGS(dml_operator_.ReleaseAndGetAddressOf())), "create Multihead Attention operator");

                DML_EXECUTION_FLAGS exec_flags = DML_EXECUTION_FLAG_NONE; //| DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;
                if (data_type == DML_TENSOR_DATA_TYPE_FLOAT16)
                {
                    exec_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
                }

                if (disable_mc)
                {
                    exec_flags |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
                }

                throw_if_failed(dml_device->CompileOperator(
                    dml_operator_.Get(),
                    exec_flags,
                    IID_PPV_ARGS(dml_op_executor_.ReleaseAndGetAddressOf())), "create Multihead Attention compiled operator");

                create_operator_impl();
            }else{
                assert(false && "Unsupported MHA type!");
            }
        }

        dml::TensorDesc get_tensor_input_desc() const
        {
            return tensor_stacked_qkv_desc_;
        }

        dml::TensorDesc get_tensor_out_desc() const
        {
            return tensor_out_desc_;
        }

        void record_execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list,
            ID3D12Resource* resource_out, ID3D12Resource* resource_input)
        {

            assert(resource_input);
            assert(resource_out);
            DML_BUFFER_BINDING input_buffer_binding{ resource_input, 0, resource_input->GetDesc().Width };
            std::vector<DML_BINDING_DESC> input_bindings;
            input_bindings.reserve(11);
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &input_buffer_binding });  
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });
            input_bindings.push_back({ DML_BINDING_TYPE_NONE , nullptr });

            std::vector<DML_BINDING_DESC> output_bindings;
            output_bindings.reserve(3);
            DML_BUFFER_BINDING output_buffer_binding{ resource_out, 0, resource_out->GetDesc().Width };
            DML_BINDING_DESC output_binding_desc{ DML_BINDING_TYPE_BUFFER, &output_buffer_binding };
            output_bindings.push_back({ DML_BINDING_TYPE_BUFFER, &output_buffer_binding });
            output_bindings.push_back({ DML_BINDING_TYPE_NONE, nullptr });
            output_bindings.push_back({ DML_BINDING_TYPE_NONE, nullptr });
           
            record_execute_impl(dml_cmd_recorder, cmd_list, input_bindings, output_bindings);
        }

        virtual void record_initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        {
            const auto initialize_binding_properties = dml_op_initializer_->GetBindingProperties();
            if (initialize_binding_properties.TemporaryResourceSize > 0 && temporary_buffer_)
            {
                DML_BUFFER_BINDING buffer_binding{ temporary_buffer_.Get(), 0, temporary_buffer_->GetDesc().Width };
                DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
                dml_init_binding_table->BindTemporaryResource(&binding_desc);
            }

            if (persistent_buffer_)
            {
                // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
                DML_BUFFER_BINDING buffer_binding{ persistent_buffer_.Get(), 0, persistent_buffer_->GetDesc().Width };
                DML_BINDING_DESC binding_desc{ DML_BINDING_TYPE_BUFFER, &buffer_binding };
                dml_init_binding_table->BindOutputs(1, &binding_desc);
            }

            dml_cmd_recorder->RecordDispatch(
                cmd_list,
                dml_op_initializer_.Get(),
                dml_init_binding_table.Get());

        }
    private:
        ComPtr<IDMLOperator> dml_operator_;
        DML_BUFFER_TENSOR_DESC tensor_stacked_qkv_desc_;
        DML_BUFFER_TENSOR_DESC tensor_out_desc_;
        MhaType type_{};

    };
}

class MhaBaseDispatcher: public NodeDispatcher
{
    public:
        struct create_params_t
        {
            MhaType type;
            DataType dt;
            DataLayout layout;

            TensorShape shape_input;

            inline static void add_cli_options(CLI::App* opts, create_params_t& params)
            {
                add_data_type_cli_option(opts, "--data_type", params.dt)->required();
                add_data_layout_cli_option(opts, "--layout", params.layout)->required();
                opts->add_option("--shape_input", params.shape_input)->required();

                opts->add_option("--mha_type", params.type, "Name of the type of MHA to run.")
                    ->check(CLI::IsMember({ MhaType::MhaType_QKV }))->
                    transform(CLI::Transformer(std::map<std::string, MhaType>{
                        { "qkv", MhaType::MhaType_QKV },

                }, CLI::ignore_case))->required();
            }
        };

    public:
    MhaBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , dml_cmd_recorder_(dml_cmd_recorder)
        , dml_device_(dml_device)
        , d3d12_device_(d3d12_device)

    {
        input_data_.resize(params_.shape_input.get_elements_count() * get_data_type_bytes_width(params_.dt));
        if (params_.type == MhaType::MhaType_QKV)
        {
            assert(params_.shape_input.get_dims_count() == 5);
            assert(!input_data_.empty());
        } else
        {
            assert(false && "Not supported MHA type!");
        }

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-1.0f, 1.0f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_);
          
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_);
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        const auto tensor_input_bytes_width = input_data_.size();
      
        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = out_shape.get_elements_count() * get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_input_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
      
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
      
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
      
        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
       
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    
    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue, ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list, bool print_mismatches)
    {
        const auto out_shape = get_shape_output();
        const auto tensor_out_bytes_width = out_shape.get_elements_count() * get_data_type_bytes_width(params_.dt);

        // readback data and validate
        auto readback_buffer = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> data_out(tensor_out_bytes_width);
        std::byte* readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
        readback_buffer->Unmap(0, nullptr);

        //
        //  calc reference with dml non-mc 
        //
        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        command_list->ResourceBarrier(1, &readback_output_barrirer);

        gpu_op::Mha mha_ref(params_.type, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout),
            params_.shape_input, get_shape_output(),
            dml_device_, d3d12_device_, true /*disable metacommand*/);

        // bind descriptor heap
        auto descriptor_heap = create_descriptor_heap(d3d12_device_, mha_ref.get_total_descriptor_count());
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        mha_ref.create_binding_tables(descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        mha_ref.record_initialize(dml_cmd_recorder_, command_list);
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);
        mha_ref.record_execute(dml_cmd_recorder_, command_list, output_buffer_.Get(), input_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> ref_untyped_result(tensor_out_bytes_width);
        readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(ref_untyped_result.data(), readback_mapped_ptr, ref_untyped_result.size());
        readback_buffer->Unmap(0, nullptr);



        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, ref_untyped_result, 0.05f, print_mismatches);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, ref_untyped_result, 0.05f, print_mismatches);
        }
        assert(false && "Unsupported output data type!");
        ConformanceResult ret{};
        return ret;

    }
    protected:
    TensorShape get_shape_output() const
    {
        TensorShape ret{};
        ret.n = get_batch();
        ret.h = get_height();
        ret.w = get_width();
        return ret;
    }

    std::uint32_t get_batch() const
    {
        return params_.shape_input.n;
    }


    std::uint32_t get_height() const
    {
        if (params_.type == MhaType::MhaType_QKV)
        {
            return params_.shape_input.c;
        }
        assert(false && "Not supported");
        return 0;
    }

    std::uint32_t get_width() const
    {
 
         if (params_.type == MhaType::MhaType_QKV)
        {
            return params_.shape_input.d * params_.shape_input.w;
        }
       
        assert(false && "Not supported");
        return 0;
    }

    protected:
    create_params_t params_;
    ID3D12Device* d3d12_device_;
    IDMLDevice* dml_device_;
    IDMLCommandRecorder* dml_cmd_recorder_;

    std::vector<std::byte> input_data_;

    ComPtr<ID3D12Resource> input_buffer_;

    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;

};

class MhaDmlDispatcher : public MhaBaseDispatcher
{
public:
    MhaDmlDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
        : MhaBaseDispatcher(std::move(params), d3d12_device, dml_device, dml_cmd_recorder, cmd_list)
        , mha_(params_.type, to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout),  params_.shape_input,  get_shape_output(),
            dml_device, d3d12_device, false)
    {

    }
    
    std::uint32_t get_total_descriptor_count() override
    {
        return mha_.get_total_descriptor_count();
    }

  
    void initialize(ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        mha_.create_binding_tables(cpu_handle, gpu_handle);
        mha_.record_initialize(dml_cmd_recorder_, cmd_list);
    }

    void execute(ID3D12GraphicsCommandList* cmd_list)
    {
        mha_.record_execute(dml_cmd_recorder_, cmd_list,
            output_buffer_.Get(), input_buffer_.Get());
    }
private:
    gpu_op::Mha mha_;
};

