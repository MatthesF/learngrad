#include "value.h"
#include <set>

Value operator+(const Value& lhs, const Value& rhs)
{
    Value new_node{Value(lhs.ptr->value + rhs.ptr->value)};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "+";
    new_node.ptr->backward = [=]() {
        lhs.ptr->grad += new_node.ptr->grad;
        rhs.ptr->grad += new_node.ptr->grad;
    };

    return new_node;
}

Value operator*(const Value& lhs, const Value& rhs)
{
    Value new_node{Value(lhs.ptr->value * rhs.ptr->value)};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "*";
    new_node.ptr->backward = [=]() {
        lhs.ptr->grad += rhs.ptr->value * new_node.ptr->grad;
        rhs.ptr->grad += lhs.ptr->value * new_node.ptr->grad;
    };

    return new_node;
}

std::vector<std::shared_ptr<ValueImpl>> build_topo(const Value& node){

    std::vector<std::shared_ptr<ValueImpl>> topo;
    std::set<std::shared_ptr<ValueImpl>> visited;

    std::function<void(const std::shared_ptr<ValueImpl>&)> build = [&](const std::shared_ptr<ValueImpl>& node){
        if (visited.contains(node)) return;
    
        visited.insert(node);
        for (const auto& prev_node : node->prev) {
            build(prev_node);
            }
        topo.push_back(node);

    };

    build(node.ptr);

    return topo;
}

void backprop(const Value& root){
    root.ptr->grad = 1;
    auto nodes = build_topo(root);
    for (auto it{nodes.rbegin()}; it != nodes.rend(); ++it){
        (*it)->backward();
    }
}
