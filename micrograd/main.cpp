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

    Value f = d.tanh();

    Value n = -f;

    Value m = n-a;

    // 3. Backprop
    backprop(m);

    // 4. Print Results
    // Expected: d = -4 * (-2) + 2 = 10
    std::cout << "d value: " << d.val() << " (Expected: 10)" << std::endl;

    std::cout << "f value: " << f.val() << " (Expected: ca. 1)" << std::endl;

    std::cout << "n value: " << n.val() << " (Expected: ca. -1)" << std::endl;

    std::cout << "m value: " << m.val() << " (Expected: ca. 3)" << std::endl;

    
    std::cout << "a grad: "  << a.grad()  << std::endl;
    
    std::cout << "b grad: "  << b.grad() << std::endl;

    std::cout << "f grad: "  << f.grad() << std::endl;

    return 0;
}
