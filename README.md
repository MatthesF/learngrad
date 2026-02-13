# LearnGrad

LearnGrad is a deep dive into building deep learning systems from scratch in C++.
It is inspired by Andrej Karpathy's educational approach, but all core autograd
engines in this repository are implemented directly in C++.

## Current Status

**Completed:** `micrograd` (scalar autograd)

**Working on:** `tensorgrad` (tensor autograd)

## Roadmap

1. **Micrograd**
   - Implement a scalar-value autograd engine.
   - Understand backpropagation and computational graphs.

2. **Tensor Autograd**
   - Extend autograd to support N-dimensional tensors.
   - Implement matrix multiplication and broadcasting.

3. **CUDA Optimization**
   - Rewrite the engine for GPU acceleration with CUDA.
   - Focus on performance and parallel computation.

4. **Applications (Transformers)**
   - Use the custom autograd engines above to implement Transformers from scratch.
   - Build and train models without relying on deep learning frameworks such as PyTorch or TensorFlow.
