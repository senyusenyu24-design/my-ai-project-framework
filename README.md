# mininn

A minimal neural network framework writen from scratch in C++:
custom tensors, computation graph, automatic differentiation
(`backward()`), basic layers, loss functions, and one optimizer.

This is not a PyTorch replacement and it wasn't created to be one. The goal is to show how autograd and nerual network training work under the hood. However, it is still capable of training small networks on real data. 

## Capabilies

- **Tensor** — 2D-matrix (rows × cols) with automatic graph of calculations; 
    each operation creates a graph node with references to its parents and closure for backward pass.
- **Autograd** — `loss.backward()` shows the graph in reverse topological order + accumulates gradients for every tensor using `requires_grad=true`
- **Operations** - +, -, elementwise multiply (mul), matmul, transpose, relu, sigmoid, sum, mean, scalar multiply; + supports broadcasting a bias vector. 
- **Layers**: `Linear`, `ReLU`, `Sigmoid`, `Sequential`
- **Loss**: `mse_loss` (which is regression), `cross_entropy_loss`
  (softmax + NLL, с log-sum-exp for log-sum-exp trick for numerical stability)
- **Optimizer**: `SGD` with optional momentum

Correctness of `backward()` has been verified against numerical gradients; the discrepancy was about 1e-9.

## Project Structure

```
mininn/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── tensor.hpp   — Tensor, computation graph, autograd
│   ├── nn.hpp        — Module, Linear, ReLU, Sigmoid, Sequential
│   ├── loss.hpp      — mse_loss, cross_entropy_loss
│   └── optim.hpp      — SGD
└── src/
    └── main.cpp      — demo: training an MLP on a synthetic dataset
```

Framework header-only — to use it, just include the headers `.hpp` from `include/`, nothing needs to be additionally compiled. 

## Build and run

Directly with g++:

```bash
g++ -std=c++17 -O2 -Iinclude src/main.cpp -o mininn_demo
./mininn_demo
```

Also can do it through CMake:

```bash
mkdir build && cd build
cmake ..
cmake --build .
./mininn_demo
```

The demo trains an MLP(2->16->16->2) on a synthetic nonlinear dataset (two noisy concentretic circles - a linearly non-seperable problem). It prints loss and accuracy for each epoch. The model reaches 100% test accuracy within ~ 20 epochs. 

## Quick usage example

```cpp
#include "tensor.hpp"
#include "nn.hpp"
#include "loss.hpp"
#include "optim.hpp"
using namespace mininn;

// Model
Sequential model({
    std::make_shared<Linear>(2, 16),
    std::make_shared<ReLU>(),
    std::make_shared<Linear>(16, 2)
});

// Optimizer
SGD optimizer(model.parameters(), /*lr=*/0.1, /*momentum=*/0.9);

// One training step
Tensor logits = model.forward(x_batch);            // forward, makes the graph
Tensor loss = cross_entropy_loss(logits, labels);   // or mse_loss for regression

optimizer.zero_grad();
loss.backward();                                     // backward, autograd
optimizer.step();                                     // update weights
```

To use your own data, fill a `Tensor` your own numbers:

```cpp
Tensor X = Tensor::create(num_samples, num_features);
X.at(i, j) = value; // manually or CSV/file. //////
```

## Limitations

This is a research tool, not a production framework:

- Only 2D-tensors (matrices) — works for fully-connected networks (MLPs),
  not for CNN/RNN/Transformer
- CPU only, no vectorization (SIMD) for multithreading - on larger datasets (tens of thousands of samples and up); it is obviously slower than any PyTorch or TensorFlow. 
- Only SGD is implemented as an optimizer (Adam and others are straightforward to add following the pattern n optim.hpp)
- Every `forward()` call rebuilds the graph from scratch. It is simple and easy to follow for learning purposes, but less efficient than a reusable graph.  

Good for: small tabular datasets (hundreds to thousands of samples, a handful to a few dozen features); classification and regression; understanding how backdrop works, experimenting with MLP architectures. 

Not good for: images, text/NLP, large datasets, anything requiring GPU or production-level speed. 

## License

MIT license. 

