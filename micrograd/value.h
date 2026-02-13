#pragma once

#include <memory>
#include <vector>

struct ValueImpl;

struct VJP_dependency {
    std::shared_ptr<ValueImpl> node;
    double local_grad;
};

struct ValueImpl {
    double value;
    double grad{0};

    std::vector<VJP_dependency> vjp_dependencies;

    ValueImpl(double val) : value{val} {}
};

class Value {
    std::shared_ptr<ValueImpl> ptr;

    public:
        Value(double data): ptr{std::make_shared<ValueImpl>(data)}{};

        friend Value operator+(const Value& lhs, const Value& rhs);
        friend Value operator*(const Value& lhs, const Value& rhs);
        friend Value operator-(const Value& node);
        friend Value operator-(const Value& lhs, const Value& rhs);
        friend Value operator/(const Value& lhs, const Value& rhs);

        friend std::vector<std::shared_ptr<ValueImpl>> build_topo(const Value& node);
        friend void backprop(const Value& root);

        double val() const { return ptr->value; }
        double grad() const { return ptr->grad; }

        Value tanh();
        Value pow(double exponent);
        
        void zero_grad();
        void update(double lr);
};

Value operator+(const Value& lhs, const Value& rhs);

Value operator*(const Value& lhs, const Value& rhs);

Value operator-(const Value& lhs, const Value& rhs);

Value operator-(const Value& node);

Value operator/(const Value& lhs, const Value& rhs);

std::vector<std::shared_ptr<ValueImpl>> build_topo(const Value& node);

void backprop(const Value& root);
