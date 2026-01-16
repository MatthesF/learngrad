#include "value.h"

#include <iostream>
#include <memory>

int main() {
    // 1. Inputs
    Value a = -4.0;
    Value b = 2.0;

    // 2. Math: d = a * (a + b) + b
    // Intermediate: c = a + b
    Value c = a + b;
    // Result: d = a * c + b
    Value d = a * c + b;

    // 3. Backprop
    backprop(d);

    // 4. Print Results
    // Expected: d = -4 * (-2) + 2 = 10
    std::cout << "d value: " << d.val() << " (Expected: 10)" << std::endl;
    
    // Expected: da = c + a*1 = -2 + (-4) = -6
    std::cout << "a grad: "  << a.grad()  << " (Expected: -6)" << std::endl;
    
    // Expected: db = a*1 + 1 = -4 + 1 = -3
    std::cout << "b grad: "  << b.grad()  << " (Expected: -3)" << std::endl;

    return 0;
}
