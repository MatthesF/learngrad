#include <iostream>
#include <vector>
#include "value.h"
#include "nn.h"

int main() {
    // 1. DATA (XOR: 0,0->-1 | 0,1->1 | 1,0->1 | 1,1->-1)
    std::vector<std::vector<Value>> inputs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    std::vector<Value> targets = {-1.0, 1.0, 1.0, -1.0};

    // 2. MODEL (2 inputs -> 4 neurons in hidden -> 1 output)
    MLP model(2, {4, 1});

    // 3. TRAINING
    std::cout << "Starting training..." << std::endl;
    
    for (int k = 0; k < 500; ++k) {
        
        // A. Forward pass and calculate error (Loss)
        Value loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Value prediction = model(inputs[i])[0];
            Value diff = prediction - targets[i];
            loss = loss + diff.pow(2);
        }

        // B. Update network
        model.zero_grad();          // Reset old gradients
        backprop(loss);             // Calculate new gradients
        model.update(0.1);          // Adjust weights (Learning rate = 0.1)

        // Print status
        if (k % 50 == 0) std::cout << "Step " << k << " | Loss: " << loss.val() << std::endl;
    }

    // 4. TEST
    std::cout << "\n--- Results ---" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        double pred = model(inputs[i])[0].val();
        double target = targets[i].val();
        
        std::cout << "Input: " << inputs[i][0].val() << "," << inputs[i][1].val() 
                  << " -> Target: " << target 
                  << " -> Pred: " << pred << std::endl;
    }

    return 0;
}