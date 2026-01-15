#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct Value {
    double value;
    double grad{0};

    std::string op;
    std::vector<std::shared_ptr<Value>> prev;
    std::function<void()> backward;

    explicit Value(double val) : value{val}, backward{[](){}} {}
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs,
                                 const std::shared_ptr<Value>& rhs);

std::vector<std::shared_ptr<Value>> build_topo(const std::shared_ptr<Value>& node);

void backprop(const std::shared_ptr<Value>& root);
