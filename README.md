# LearnGrad

A deep dive into building deep learning systems from scratch in C++. This project is inspired by Andrej Karpathy's educational materials but implemented in C++ to understand the low-level details of autograd engines and optimization.

## Current Status

**Working on:** `micrograd` (Scalar-level autograd)

## Roadmap

1.  **Micrograd**:
    -   Implement a scalar-value autograd engine.
    -   Understand the basics of backpropagation and computational graphs.

2.  **Tensor Autograd**:
    -   Extend autograd to support N-dimensional tensors.
    -   Implement matrix multiplication and broadcasting.

3.  **CUDA Optimization**:
    -   Rewrite the engine to leverage GPU acceleration using CUDA.
    -   Focus on performance and parallel computation.

4.  **Applications (Transformers)**:
    -   Use the custom autograd engines built above to implement a Transformer from scratch.
    -   Build and train neural networks without relying on external deep learning frameworks (like PyTorch or TensorFlow).
