#include "value.h"

#include <iostream>
#include <memory>

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
