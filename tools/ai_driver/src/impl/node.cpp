#include "node.h"
#include <format>

std::string ai_driver::node_utils::create_name(std::string_view node_type_name, const std::string& name)
{
    if (name.empty())
        return std::string(node_type_name);

    return std::format("{} \"{}\"", node_type_name, name);
}
