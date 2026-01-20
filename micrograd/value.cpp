#include "value.h"
#include <set>
#include <cmath>

Value operator+(const Value& lhs, const Value& rhs)
{
    Value new_node{lhs.ptr->value + rhs.ptr->value};
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
    Value new_node{lhs.ptr->value * rhs.ptr->value};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "*";
    new_node.ptr->backward = [=]() {
        lhs.ptr->grad += rhs.ptr->value * new_node.ptr->grad;
        rhs.ptr->grad += lhs.ptr->value * new_node.ptr->grad;
    };

    return new_node;
}

Value operator-(const Value& node)
{
    return node*-1;
}

Value operator-(const Value& lhs, const Value& rhs)
{
    return lhs+(rhs*-1);
}

Value operator/(const Value& lhs, const Value& rhs)
{
    Value new_node{lhs.ptr->value / rhs.ptr->value};
    new_node.ptr->prev.insert(new_node.ptr->prev.end(), {lhs.ptr, rhs.ptr});
    new_node.ptr->op = "/";
    new_node.ptr->backward = [=]() {
        lhs.ptr->grad += 1/rhs.ptr->value * new_node.ptr->grad;
        rhs.ptr->grad += - lhs.ptr->value / (rhs.ptr->value*rhs.ptr->value) * new_node.ptr->grad;
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

Value Value::tanh(){
    Value new_node{std::tanh(ptr->value)};
    new_node.ptr->prev = { ptr };
    new_node.ptr->op = "tanh";
    new_node.ptr->backward = [self = ptr, out = new_node.ptr]() {
        self->grad += out->grad * (1.0 - (out->value * out->value));
    };
    return new_node;
}

Value Value::pow(double exponent){
    Value new_node{std::pow(ptr->value,exponent)};
    new_node.ptr->prev = { ptr };
    new_node.ptr->op = "pow";
    new_node.ptr->backward = [self = ptr, exp = exponent, out = new_node.ptr]() {
        self->grad += out->grad * exp * std::pow(self->value,exp-1);
    };
    return new_node;
}

