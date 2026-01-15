#include <string>
#include <set>
#include <memory>
#include <functional>
#include <vector>

struct Value{

    double value;
    double grad{0};

    std::string op;
    std::set<std::shared_ptr<Value>> prev;
    std::function<void()> backward;

    Value(double val)
        : value{val}, backward{[](){}}{}
    };

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs)
    {
         std::shared_ptr<Value> new_node{std::make_shared<Value>(lhs->value + rhs->value)};
         new_node->prev.insert({lhs,rhs});
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
         new_node->prev.insert({lhs,rhs});
         new_node->op = "*";
         new_node->backward = [=]() {
            lhs->grad += rhs->value*new_node->grad;
            rhs->grad += lhs->value*new_node->grad;
            };

        return new_node;
    }


    std::vector<std::shared_ptr<Value>> topo(const std::shared_ptr<Value>& node){


    }
