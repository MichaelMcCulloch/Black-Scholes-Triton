# bscholes

Black-Scholes option pricing model with multiple high-performance backend implementations.

## Overview

This package provides efficient implementations of the Black-Scholes model for European options pricing and Greeks calculations. It supports multiple computational backends optimized for different use cases, from CPU-based NumPy operations to GPU-accelerated PyTorch and Triton kernels.

## Features

- **Multiple Backends**: Choose from NumPy, PyTorch (stateless/stateful), TorchScript JIT, or Triton implementations
- **GPU Acceleration**: Leverage CUDA for massive parallel computation
- **Greeks Calculation**: Compute all option Greeks (delta, gamma, vega, theta, rho)
- **Unified API**: Consistent interface across all backends
- **Performance Optimized**: Fused kernels minimize memory transfers and maximize throughput

## Installation

```bash
uv pip install -e .
```

For development:
```bash
uv pip install -e ".[dev]"
```

## Backend Implementations

The package provides five different backend implementations, each optimized for different scenarios:

| Backend | File | Description |
|---------|------|-------------|
| **NumPy** | [black_scholes.py](src/bscholes/black_scholes.py) | Pure NumPy + SciPy implementation. CPU-based, no external dependencies. Good baseline performance. |
| **Torch Stateless** | [black_scholes_torch.py](src/bscholes/black_scholes_torch.py) | Functional PyTorch implementation with NumPy API wrapper. GPU-accelerated, stateless functions. |
| **Torch Stateful** | [torch_model.py](src/bscholes/torch_model.py) | Stateful PyTorch class that holds data on GPU. Minimizes host-device transfers. Optimized for batch processing. |
| **Torch JIT** | [torch_jit_kernels.py](src/bscholes/torch_jit_kernels.py) | TorchScript JIT-compiled fused kernel. Reduced kernel launch overhead. Optimized for Greeks calculation. |
| **Triton** | [triton_kernels.py](src/bscholes/triton_kernels.py) | Custom Triton kernel with manual memory hierarchy control. Maximum performance for large batches. Fused Greeks computation. |

Additional files:
- [api.py](src/bscholes/api.py) - Unified API and backend factory
- [__init__.py](src/bscholes/__init__.py) - Package exports

## Usage

### Basic Example

```python
import numpy as np
from bscholes import get_backend

# Select a backend
backend = get_backend("numpy")  # or "torch_functional", "torch_stateful", "torch_jit", "torch_triton"

# Define option parameters
S = np.array([100.0, 105.0, 110.0])  # Spot prices
K = np.array([100.0, 100.0, 100.0])  # Strike prices
T = np.array([1.0, 1.0, 1.0])        # Time to maturity (years)
r = np.array([0.05, 0.05, 0.05])     # Risk-free rate
sigma = np.array([0.2, 0.2, 0.2])    # Volatility

# Calculate option prices
call_prices = backend.black_scholes_call(S, K, T, r, sigma)
put_prices = backend.black_scholes_put(S, K, T, r, sigma)

# Calculate Greeks
delta_c = backend.delta_call(S, K, T, r, sigma)
gamma = backend.gamma(S, K, T, r, sigma)
vega = backend.vega(S, K, T, r, sigma)
```

### Stateful PyTorch Backend (Efficient for Batch Operations)

```python
from bscholes.torch_model import BlackScholesPyTorch

# Create model instance (data transferred to GPU once)
model = BlackScholesPyTorch(S, K, T, r, sigma)

# Calculate prices
call_prices = model.price_call()
put_prices = model.price_put()

# Calculate all Greeks in a single optimized pass
greeks = model.calculate_greeks()
print(greeks['delta_call'])
print(greeks['gamma'])
print(greeks['vega'])
```

### Triton Fused Kernel (Maximum Performance)

```python
from bscholes.triton_kernels import calculate_greeks_triton

# Calculate all Greeks in a single fused kernel
greeks = calculate_greeks_triton(S, K, T, r, sigma)
```

## Testing

Run the test suite using pytest:

```bash
uv run pytest -s
```

This will run all unit tests across all backends, validating correctness and numerical accuracy.

## Performance

Benchmark: 1,000,000 options, all Greeks (RTX 4090)

| Backend | Time (ms) | Throughput (ops/sec) | Speedup |
|---------|-----------|----------------------|---------|
| NumPy (CPU) | 286.52 | 3.49M | 1.0x |
| Torch Stateless | 32.28 | 30.98M | 8.9x |
| Torch Stateful | 11.67 | 85.71M | 24.5x |
| Torch JIT | 103.99 | 9.62M | 2.8x |
| Triton | 11.80 | 84.75M | 24.3x |

## Requirements

- Python >= 3.12
- NumPy >= 2.3.4
- SciPy >= 1.16.3
- PyTorch (for torch backends)
- Triton (for triton backend)

## License

See LICENSE file for details.
