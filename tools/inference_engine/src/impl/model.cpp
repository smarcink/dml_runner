#include "model.h"
#include "inference_engine_model.h"
#include "inference_engine_tensor.h"
#include "nodes/port.h"
#include <exception>
#include <algorithm>
#include "nodes/matmul.h"
#include "nodes/activation.h"
#include "nodes/convolution.h"
#include "nodes/elementwise_add.h"
#include "gpu_visitor.h"

namespace inference_engine
{
    namespace
    {
        std::vector<TensorMapping> build_output_mapping(const std::vector<std::unique_ptr<GpuNode>>& nodes)
        {
            std::vector<TensorMapping> ret{};
            for (const auto& n : nodes)
            {
                if (n->get_outputs().empty())
                {
                    ret.push_back({ n->get_id(), n->get_output_tensor() });
                }
            }
            return ret;
        }
    }

    DAG::DAG()
    {
        //preallocate some memory
        nodes_.reserve(1024);
    }

    std::vector<std::unique_ptr<inference_engine::GpuNode>> DAG::compile(std::span<TensorMapping> input_mappings)
    {
        create_adjacency_list();

        auto topological_sorted = topological_sort();
        
        // construct return list of GpuNodes
        std::vector<std::unique_ptr<GpuNode>> ret(topological_sorted.size());
        std::unordered_map<size_t, inference_engine::GpuNode*> id_to_node_map;
        for (auto i = 0; i < topological_sorted.size(); i++)
        {
            auto& ts = topological_sorted[i];
            std::vector<GpuNode*> inputs;

            // gpu version of inputs should be already created, so match them based on INode id...
            for (auto& input_id : ts->inputs()) {                
                if (auto iter = id_to_node_map.find(input_id); iter != id_to_node_map.end())
                    inputs.push_back(iter->second);
            }

            ret[i] = ts->create_gpu_node(inputs);
            id_to_node_map[ret[i]->get_id()] = ret[i].get();
            
            for (auto& in : inputs)
                in->add_output(ret[i].get());

            // maybe we can move it to a separate function?
            // set input tensor if this is port (important: we have topological sorted, so we assume here that all inputs are traversed first)!
            auto it = std::find_if(std::begin(input_mappings), std::end(input_mappings), [&](const TensorMapping& im)
                {
                    return im.id == ret[i]->get_id();
                });
            if (it != std::end(input_mappings))
            {
                auto port = dynamic_cast<GpuPort*>(ret[i].get());
                if (port)
                {
                    port->set_tensor(it->tensor);
                }
                else
                {
                    std::cout << "Cant set input for id: " << it->id << ", because it is not Port. It is: " << ret[i]->to_str() << std::endl;
                }
            }
        }
        return ret;
    }

    void DAG::create_adjacency_list()
    {
        for (auto& n : nodes_)
        {
            for (auto& in : n->inputs())
            {
                auto input = nodes_.at(in).get();
                adjacency_list_[input].push_back(n.get());
            }
        }
    }

    std::vector<INode*> DAG::topological_sort()
    {
        std::unordered_set<INode*> visited;
        std::stack<INode*> stack;
        for (const auto& node : nodes_) {
            if (!visited.contains(node.get())) {
                topological_sort_util(node.get(), visited, stack);
            }
        }

        std::vector<INode*> sorted;
        while (!stack.empty()) {
            sorted.push_back(stack.top());
            stack.pop();
        }
        return sorted;
    }

    void DAG::topological_sort_util(INode* node, std::unordered_set<INode*>& visited, std::stack<INode*>& stack)
    {
        visited.insert(node);
        for (INode* adjacent : adjacency_list_[node]) {
            if (!visited.contains(adjacent)) {
                topological_sort_util(adjacent, visited, stack);
            }
        }
        stack.push(node);
    }

    ExecutableModel::ExecutableModel(std::vector<std::unique_ptr<GpuNode>>&& nodes)
        : nodes_(std::move(nodes))
        , output_mappings_(build_output_mapping(nodes_))
    {
        std::cout << "ExecutableModel:" << std::endl;
    }

ExecutableModel::~ExecutableModel()
{
    std::cout << "~ExecutableModel:" << std::endl;
}

void ExecutableModel::execute(GpuStream& stream)
{
    std::cout << "ExecutableModel execute()" << std::endl;

    // We should have topological order of nodes here, we know that model desc just reversed list. For now it should work.
    for (auto& n : nodes_)
    {
        std::cout << "\t[Executing] " << n->to_str() << std::endl;
        auto out_resource = n->execute(stream);

        // aggregate resources and dispatch barrier (sync point) - this is naive, as it will add sync point after each node
        if (out_resource)
        {
            // we should know dependency graph and when to put resource barriers, but for now always put barrier, after every primitive
            stream.dispatch_resource_barrier(*out_resource);
        }

    }
}

void ExecutableModel::set_resource(inference_engine_node_id_t id, GpuResource::Ptr rsc)
{
    auto it = std::find_if(std::begin(nodes_), std::end(nodes_), [&id](const auto& node)
        {
            return id == node->get_id();
        });
    if (it != std::end(nodes_))
    {
        it->get()->set_resource(std::move(rsc));
    }
    else
    {
        std::cout << "Trying to set resource for node: " << id << " but it does not exist in executable model" << std::endl;
        throw std::invalid_argument("");
    }
    }

const std::vector<inference_engine::TensorMapping>& ExecutableModel::get_outputs() const
{
    return output_mappings_;
}

ModelDescriptor::ModelDescriptor()
{
    std::cout << "C-TOR ModelDescriptor()" << std::endl;
}

ModelDescriptor::~ModelDescriptor()
{
    std::cout << "D-TOR ~ModelDescriptor()" << std::endl;
}

class FusionVisitor : public GpuVisitor 
{
    std::unordered_set<GpuNode*> to_delete_;
    std::unordered_map<GpuNode*, size_t> node_to_index_;

    void mark_node_for_deletion(GpuNode* node) 
    {
        to_delete_.insert(node);
    }
    void update_indices(std::vector<std::unique_ptr<GpuNode>>& sorted_nodes) 
    {
        for (size_t i = 0; i < sorted_nodes.size(); ++i)
            node_to_index_[sorted_nodes[i].get()] = i;
    }
public:
    void process_sorted_nodes(std::vector<std::unique_ptr<GpuNode>>& sorted_nodes) override 
    {
        update_indices(sorted_nodes);
        for (auto it = std::begin(sorted_nodes); it != std::end(sorted_nodes);) 
        {
            to_delete_.clear();
            (*it)->accept(this);
            if (!to_delete_.empty()) 
            {
                if (to_delete_.size() == 1) 
                {
                    auto index = node_to_index_[*to_delete_.begin()];
                    it = sorted_nodes.erase(std::begin(sorted_nodes) + index); // continue the iteration from node after deleted one
                }
                else 
                {
                    std::vector<size_t> ids_to_delete;
                    ids_to_delete.reserve(to_delete_.size());
                    for (auto& elem : to_delete_)
                        ids_to_delete.push_back(node_to_index_[elem]);
                    std::sort(std::begin(ids_to_delete), std::end(ids_to_delete));
                    // remove nodes from sorted list
                    for (auto& index : ids_to_delete)
                        it = sorted_nodes.erase(std::begin(sorted_nodes) + index);  // continue the iteration from node after the last deleted node
                }
                update_indices(sorted_nodes);
            }
            else
                ++it;
        }
    }

    void visit(GpuPort* pn) override 
    {
        std::cout << "visiting port...\n";
    }
    void visit(GpuActivation* pn) override 
    {
        std::cout << "visiting activation...\n";

        // check matmul/conv + activation and fuse?
        // if we have multiple activations in a row, we do it one by one

        auto& inputs = pn->get_inputs();
        if (inputs.size() == 1) {
            if (auto matmul = dynamic_cast<GpuMatMul*>(inputs[0])) {
                mark_node_for_deletion(pn);
                matmul->fuse_with(pn);
            }
            else if (auto conv = dynamic_cast<GpuConvolution*>(inputs[0])) {
                mark_node_for_deletion(pn);
                conv->fuse_with(pn);
            }
        }
    }
    void visit(GpuMatMul* pn) override 
    {
        std::cout << "visiting matmul...\n";        
    }
    void visit(class GpuElementwiseAdd*) override
    {
        std::cout << "visiting elementwise add...\n";
    }
    void visit(class GpuConvolution*) override
    {
        std::cout << "visiting convolution...\n";
    }
};

inference_engine::ExecutableModel ModelDescriptor::compile(GpuContext& ctx, GpuStream& stream, std::span<TensorMapping> input_mappings)
{
    //ToDo: we need some data structure to represent graph (random order of example features below)
    // 1) Sorting graph
    // 2) Traversing graph (i.e. layout prorogation)
    // 2b) Memory allocations and ping-pong (memory reuse)
    // 3) Graph optimization passes (layers fusions) - we should be able to register and unregister passes etc.
    // 4) Uploading constant data to gpu
    // 5) compiling shaders (picking optimized implementations)

    std::cout << "[Compile][Pass-X] -- Topological sort\n";
    auto sorted_nodes = dag_.compile(input_mappings);

    FusionVisitor v;
    v.process_sorted_nodes(sorted_nodes);

    std::cout << "[Compile][Pass-Q] -- Memory allocations" << std::endl;
    for (auto& n : sorted_nodes)
    {
        const auto has_resource = n->get_resource();
        const auto is_intermidate_node = !n->get_inputs().empty() && !n->get_outputs().empty();
        if (!has_resource && is_intermidate_node)
        {
            n->set_resource(std::make_shared<GpuResource>(ctx.allocate_resource(n->get_output_tensor().bytes_width())));
        }
    }
    std::cout << "[Compile][Pass-Z] -- Compile" << std::endl;
    for (auto& n : sorted_nodes)
    {
        n->compile(ctx);
    }
    std::cout << "[Compile][Pass-W] -- Initialize" << std::endl;
    for (auto& n : sorted_nodes)
    {
        n->initalize(stream);
    }
    return ExecutableModel(std::move(sorted_nodes));
}


} // namespace inference_engine