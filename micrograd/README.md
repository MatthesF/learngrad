# Micrograd (C++)

This directory contains the first stage of LearnGrad: a scalar-value autograd engine implemented from scratch in C++.

It is a port/adaptation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), designed to teach the fundamental concepts of backpropagation.

## Features

-   **Scalar Autograd**: Tracks operations on single floating-point values.
-   **Dynamic Computational Graph**: Builds the graph at runtime.
-   **Backpropagation**: Implements topological sort and reverse-mode automatic differentiation to calculate gradients.
