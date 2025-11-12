# ./tests/performance/test_profiler.py
"""
Performance tests for the bscholes package, including:
1. A unified throughput benchmark for calculating all Greeks across all backends.
2. Detailed hotspot profiling for the stateful PyTorch and Triton models.
"""

import numpy as np
import pytest
import time
from types import SimpleNamespace

# --- Backend Imports ---
import bscholes.black_scholes as np_backend
import bscholes.black_scholes_torch as torch_functional_backend
from bscholes.torch_jit_kernels import calculate_greeks_jit
from bscholes.torch_model import BlackScholesPyTorch
from bscholes.triton_kernels import calculate_greeks_triton

import torch
from torch.profiler import profile, record_function, ProfilerActivity


# ==============================================================================
# == Unified Throughput Benchmark (All Greeks)
# ==============================================================================


# --- Runner function definitions for a unified benchmark ---
def numpy_greeks_runner(S, K, T, r, sigma):
    """Runner for the NumPy backend, calling each Greek function individually."""
    return {
        "delta_call": np_backend.delta_call(S, K, T, r, sigma),
        "delta_put": np_backend.delta_put(S, K, T, r, sigma),
        "gamma": np_backend.gamma(S, K, T, r, sigma),
        "vega": np_backend.vega(S, K, T, r, sigma),
        "theta_call": np_backend.theta_call(S, K, T, r, sigma),
        "theta_put": np_backend.theta_put(S, K, T, r, sigma),
        "rho_call": np_backend.rho_call(S, K, T, r, sigma),
        "rho_put": np_backend.rho_put(S, K, T, r, sigma),
    }


def torch_functional_greeks_runner(S, K, T, r, sigma):
    """Runner for the stateless PyTorch backend, calling each function."""
    return {
        "delta_call": torch_functional_backend.delta_call(S, K, T, r, sigma),
        "delta_put": torch_functional_backend.delta_put(S, K, T, r, sigma),
        "gamma": torch_functional_backend.gamma(S, K, T, r, sigma),
        "vega": torch_functional_backend.vega(S, K, T, r, sigma),
        "theta_call": torch_functional_backend.theta_call(S, K, T, r, sigma),
        "theta_put": torch_functional_backend.theta_put(S, K, T, r, sigma),
        "rho_call": torch_functional_backend.rho_call(S, K, T, r, sigma),
        "rho_put": torch_functional_backend.rho_put(S, K, T, r, sigma),
    }


def stateful_torch_greeks_runner(S, K, T, r, sigma):
    """Runner for the stateful model's optimized `calculate_greeks` method."""
    model = BlackScholesPyTorch(S, K, T, r, sigma)
    return model.calculate_greeks()


# --- Parametrized test setup ---
greeks_throughput_backends = [
    pytest.param(
        SimpleNamespace(name="NumPy (CPU)", runner=numpy_greeks_runner),
        id="numpy_cpu",
    ),
    pytest.param(
        SimpleNamespace(
            name="PyTorch Functional (Stateless)",
            runner=torch_functional_greeks_runner,
        ),
        id="torch_stateless",
    ),
    pytest.param(
        SimpleNamespace(
            name="PyTorch Class-Based (Stateful)",
            runner=stateful_torch_greeks_runner,
        ),
        id="torch_stateful",
    ),
    pytest.param(
        SimpleNamespace(name="PyTorch JIT Fused", runner=calculate_greeks_jit),
        id="torch_jit",
    ),
    pytest.param(
        SimpleNamespace(name="Triton Fused Kernel", runner=calculate_greeks_triton),
        id="triton_fused",
    ),
]


@pytest.mark.parametrize("backend", greeks_throughput_backends)
def test_greeks_throughput_comparison(backend):
    """
    Measures and compares the throughput of all backends for calculating the
    full set of Black-Scholes Greeks.
    """
    n = 1_000_000
    S = np.random.uniform(80, 120, n)
    K = np.full(n, 100.0)
    T = np.random.uniform(0.1, 2.0, n)
    r = np.full(n, 0.05)
    sigma = np.random.uniform(0.1, 0.4, n)

    # Warmup
    backend.runner(S, K, T, r, sigma)
    if "cpu" not in backend.name.lower():
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    greeks = backend.runner(S, K, T, r, sigma)
    if "cpu" not in backend.name.lower():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = n / elapsed
    print(
        f"\n[{backend.name} - All Greeks] Processed {n:,} options in {elapsed*1000:.2f}ms | Throughput: {throughput:,.0f} ops/sec"
    )

    # Validation
    assert "delta_call" in greeks
    assert greeks["delta_call"].shape == (n,)
    assert np.all(np.isfinite(greeks["gamma"]))


# ==============================================================================
# == Profiler for Hotspot Analysis (Stateful PyTorch vs. Triton)
# ==============================================================================


def test_torch_model_profiler_hotspots():
    """Profiles the stateful BlackScholesPyTorch model to identify hotspots."""
    print("\n--- Running PyTorch Profiler Test for Stateful Model ---")
    n = 250_000
    S = np.random.uniform(80, 120, n)
    K = np.full(n, 100.0)
    T = np.random.uniform(0.1, 2.0, n)
    r = np.full(n, 0.05)
    sigma = np.random.uniform(0.1, 0.4, n)

    # Warmup and one-time data transfer
    model = BlackScholesPyTorch(S, K, T, r, sigma)
    _ = model.calculate_greeks()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("greeks_computation_pytorch"):
            _ = model.calculate_greeks()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def test_triton_model_profiler_hotspots():
    """Profiles the fused Triton kernel to identify hotspots."""
    print("\n--- Running PyTorch Profiler Test for Triton Kernel ---")
    n = 250_000
    S = np.random.uniform(80, 120, n)
    K = np.full(n, 100.0)
    T = np.random.uniform(0.1, 2.0, n)
    r = np.full(n, 0.05)
    sigma = np.random.uniform(0.1, 0.4, n)

    # Warmup
    _ = calculate_greeks_triton(S, K, T, r, sigma)
    torch.cuda.synchronize()

    # Data transfer happens inside the wrapper, so we profile the whole call
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function("greeks_computation_triton"):
            _ = calculate_greeks_triton(S, K, T, r, sigma)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
