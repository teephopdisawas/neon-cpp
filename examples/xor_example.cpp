// XOR Neural Network Example
// Trains a 2-layer MLP to solve XOR — the "hello world" of neural nets

#include "neon/neon.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace neon;

int main() {
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  Neon AI Framework — XOR Example\n";
    std::cout << "═══════════════════════════════════════\n\n";

    // XOR dataset
    Tensor X(Shape({4, 2}), {0, 0,
                               0, 1,
                               1, 0,
                               1, 1});
    Tensor Y(Shape({4, 1}), {0, 1, 1, 0});

    // Build model: 2 -> 8 -> 1
    Sequential model;
    model.add(std::make_shared<Linear>(2, 8));
    model.add(std::make_shared<Tanh>());
    model.add(std::make_shared<Linear>(8, 1));
    model.add(std::make_shared<Sigmoid>());

    std::cout << "Model: " << model.name() << "\n\n";

    // Optimizer
    auto optimizer = make_sgd(model, 0.5f, 0.9f);

    // Training loop
    const int epochs = 5000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        Tensor pred = model.forward(X);
        float32 loss_val = loss::binary_cross_entropy(pred, Y);

        // Backward pass
        Tensor grad = loss::binary_cross_entropy_grad(pred, Y);
        model.backward(grad);

        // Update weights
        optimizer->step();
        optimizer->zero_grad();

        // Print progress
        if (epoch % 500 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss_val
                      << "\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Final predictions
    std::cout << "\n─── Final Predictions ───\n";
    model.eval();
    Tensor pred = model.forward(X);
    for (int i = 0; i < 4; ++i) {
        std::cout << "  " << (int)X(i, 0) << " XOR " << (int)X(i, 1)
                  << " = " << std::fixed << std::setprecision(4) << pred(i, 0)
                  << " (expected: " << (int)Y(i, 0) << ")\n";
    }

    std::cout << "\nTraining time: " << ms << "ms\n";
    std::cout << "═══════════════════════════════════════\n";

    return 0;
}
