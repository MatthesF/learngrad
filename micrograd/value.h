#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ValueImpl {
    double value;
    double grad{0};

    std::string op;
    std::vector<std::shared_ptr<ValueImpl>> prev;
    std::function<void()> backward;

    ValueImpl(double val) : value{val}, backward{[](){}} {}
};


class Value {
    std::shared_ptr<ValueImpl> ptr;

    public:
        Value(double data): ptr{std::make_shared<ValueImpl>(data)}{};

        friend Value operator+(const Value& lhs, const Value& rhs);
        friend Value operator*(const Value& lhs, const Value& rhs);
        friend std::vector<std::shared_ptr<ValueImpl>> build_topo(const Value& node);
        friend void backprop(const Value& root);

        double val() const { return ptr->value; }
        double grad() const { return ptr->grad; }

        Value tanh();
};

Value operator+(const Value& lhs, const Value& rhs);

Value operator*(const Value& lhs, const Value& rhs);


std::vector<std::shared_ptr<ValueImpl>> build_topo(const Value& node);

void backprop(const Value& root);
