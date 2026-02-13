#include "value.h"
#include <unordered_set>
#include <cmath>

Value operator+(const Value& lhs, const Value& rhs)
{
    Value new_node{lhs.ptr->value + rhs.ptr->value};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "+";
    new_node.ptr->vjp_dependencies.reserve(2);
    new_node.ptr->vjp_dependencies.emplace_back(lhs.ptr, 1.0);
    new_node.ptr->vjp_dependencies.emplace_back(rhs.ptr, 1.0);

    return new_node;
}

Value operator*(const Value& lhs, const Value& rhs)
{
    Value new_node{lhs.ptr->value * rhs.ptr->value};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "*";
    new_node.ptr->vjp_dependencies.reserve(2);
    new_node.ptr->vjp_dependencies.emplace_back(lhs.ptr, rhs.ptr->value);
    new_node.ptr->vjp_dependencies.emplace_back(rhs.ptr, lhs.ptr->value);

    return new_node;
}

Value operator-(const Value& node)
{
    Value new_node{node.ptr->value * -1};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {node.ptr});
    new_node.ptr->op = "-";
    new_node.ptr->vjp_dependencies.reserve(1);
    new_node.ptr->vjp_dependencies.emplace_back(node.ptr, -1.0);
    return new_node;
}

Value operator-(const Value& lhs, const Value& rhs)
{
    Value new_node{lhs.ptr->value - rhs.ptr->value};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "-";
    new_node.ptr->vjp_dependencies.reserve(2);
    new_node.ptr->vjp_dependencies.emplace_back(lhs.ptr, 1.0);
    new_node.ptr->vjp_dependencies.emplace_back(rhs.ptr, -1.0);
    return new_node;
}

Value operator/(const Value& lhs, const Value& rhs)
{
    Value new_node{lhs.ptr->value / rhs.ptr->value};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "/";
    new_node.ptr->vjp_dependencies.reserve(2);
    new_node.ptr->vjp_dependencies.emplace_back(lhs.ptr, 1/rhs.ptr->value);
    new_node.ptr->vjp_dependencies.emplace_back(rhs.ptr, - lhs.ptr->value / (rhs.ptr->value*rhs.ptr->value));
    return new_node;
}

std::vector<std::shared_ptr<ValueImpl>> build_topo(const Value& node){

    std::vector<std::shared_ptr<ValueImpl>> topo;
    std::unordered_set<std::shared_ptr<ValueImpl>> visited;

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
        for (const auto& vjp_dependency : (*it)->vjp_dependencies){
            vjp_dependency.node->grad += vjp_dependency.local_grad * (*it)->grad;
        }
    }
}

Value Value::tanh(){
    Value new_node{std::tanh(ptr->value)};
    new_node.ptr->prev = { ptr };
    new_node.ptr->op = "tanh";
    new_node.ptr->vjp_dependencies.reserve(1);
    new_node.ptr->vjp_dependencies.emplace_back(ptr, 1.0 - (new_node.ptr->value * new_node.ptr->value));
    return new_node;
}

Value Value::pow(double exponent){
    Value new_node{std::pow(ptr->value,exponent)};
    new_node.ptr->prev = { ptr };
    new_node.ptr->op = "pow";
    new_node.ptr->vjp_dependencies.reserve(1);
    new_node.ptr->vjp_dependencies.emplace_back(ptr, exponent * std::pow(ptr->value,exponent-1));
    return new_node;
}

void Value::zero_grad() { ptr->grad = 0.0; }

void Value::update(double lr) { ptr->value -= lr * ptr->grad; }