#include "value.h"
#include <set>
#include <iostream>

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs)
{
    std::shared_ptr<Value> new_node{std::make_shared<Value>(lhs->value + rhs->value)};
    new_node->prev.insert(new_node->prev.end(), {lhs, rhs});
    new_node->op = "+";
    new_node->backward = [=]() {
        lhs->grad += new_node->grad;
        rhs->grad += new_node->grad;
    };

    return new_node;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs)
{
    std::shared_ptr<Value> new_node{std::make_shared<Value>(lhs->value * rhs->value)};
    new_node->prev.insert(new_node->prev.end(), {lhs, rhs});
    new_node->op = "*";
    new_node->backward = [=]() {
        lhs->grad += rhs->value * new_node->grad;
        rhs->grad += lhs->value * new_node->grad;
    };

    return new_node;
}

std::vector<std::shared_ptr<Value>> build_topo(const std::shared_ptr<Value>& node){

    std::vector<std::shared_ptr<Value>> topo;
    std::set<std::shared_ptr<Value>> visited;

    std::function<void(const std::shared_ptr<Value>&)> build = [&](const std::shared_ptr<Value>& node){
        if (visited.contains(node)) return;
    
        visited.insert(node);
        for (const auto& prev_node : node->prev) {
            build(prev_node);
            }
        
        topo.push_back(node);

    };

    build(node);

    return topo;


}

void backprop(const std::shared_ptr<Value>& root){
    root->grad = 1;
    auto nodes = build_topo(root);
    for (auto it{nodes.rbegin()}; it != nodes.rend(); ++it){
        (*it)->backward();
    }
}

int main() {
    // 1. Inputs
    auto a = std::make_shared<Value>(-4.0);
    auto b = std::make_shared<Value>(2.0);

    // 2. Math: d = a * (a + b) + b
    // Intermediate: c = a + b
    auto c = a + b;
    // Result: d = a * c + b
    auto d = a * c + b;

    // 3. Backprop
    backprop(d);

    // 4. Print Results
    // Expected: d = -4 * (-2) + 2 = 10
    std::cout << "d value: " << d->value << " (Expected: 10)" << std::endl;
    
    // Expected: da = c + a*1 = -2 + (-4) = -6
    std::cout << "a grad: "  << a->grad  << " (Expected: -6)" << std::endl;
    
    // Expected: db = a*1 + 1 = -4 + 1 = -3
    std::cout << "b grad: "  << b->grad  << " (Expected: -3)" << std::endl;

    return 0;
}