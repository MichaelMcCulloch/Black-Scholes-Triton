# ./src/bscholes/api.py
"""
Defines a unified API for Black-Scholes calculations and provides a factory
to access different backend implementations (NumPy, PyTorch).
"""

import numpy as np
from typing import Protocol, Literal

from bscholes import torch_jit_kernels

# --- Backend Implementations ---
from . import black_scholes as np_backend
from . import black_scholes_torch as torch_functional_backend
from .torch_model import BlackScholesPyTorch
from . import triton_kernels

BackendName = Literal[
    "numpy", "torch_functional", "torch_stateful", "torch_triton", "torch_jit"
]


class PyTorchJitBackend:
    """
    Adapter for the fused torch.jit.script function to conform to the API.
    """

    def black_scholes_call(self, S, K, T, r, sigma):
        # JIT kernel doesn't calculate price, fall back to functional torch
        return torch_functional_backend.black_scholes_call(S, K, T, r, sigma)

    def black_scholes_put(self, S, K, T, r, sigma):
        return torch_functional_backend.black_scholes_put(S, K, T, r, sigma)

    def _calculate_all_greeks(self, S, K, T, r, sigma):
        return torch_jit_kernels.calculate_greeks_jit(S, K, T, r, sigma)

    def delta_call(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["delta_call"]

    def delta_put(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["delta_put"]

    def gamma(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["gamma"]

    def vega(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["vega"]

    def theta_call(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["theta_call"]

    def theta_put(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["theta_put"]

    def rho_call(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["rho_call"]

    def rho_put(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["rho_put"]


class BlackScholesAPI(Protocol):
    """
    A unified, functional interface for any Black-Scholes backend.

    All methods must accept NumPy arrays or scalars and return NumPy arrays.
    """

    def black_scholes_call(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def black_scholes_put(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def delta_call(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def delta_put(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def gamma(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def vega(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def theta_call(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def theta_put(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def rho_call(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...

    def rho_put(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray: ...


class NumPyBackend:
    """NumPy implementation of the BlackScholesAPI."""

    def black_scholes_call(self, S, K, T, r, sigma):
        return np_backend.black_scholes_call(S, K, T, r, sigma)

    def black_scholes_put(self, S, K, T, r, sigma):
        return np_backend.black_scholes_put(S, K, T, r, sigma)

    def delta_call(self, S, K, T, r, sigma):
        return np_backend.delta_call(S, K, T, r, sigma)

    def delta_put(self, S, K, T, r, sigma):
        return np_backend.delta_put(S, K, T, r, sigma)

    def gamma(self, S, K, T, r, sigma):
        return np_backend.gamma(S, K, T, r, sigma)

    def vega(self, S, K, T, r, sigma):
        return np_backend.vega(S, K, T, r, sigma)

    def theta_call(self, S, K, T, r, sigma):
        return np_backend.theta_call(S, K, T, r, sigma)

    def theta_put(self, S, K, T, r, sigma):
        return np_backend.theta_put(S, K, T, r, sigma)

    def rho_call(self, S, K, T, r, sigma):
        return np_backend.rho_call(S, K, T, r, sigma)

    def rho_put(self, S, K, T, r, sigma):
        return np_backend.rho_put(S, K, T, r, sigma)


class PyTorchFunctionalBackend:
    """Stateless PyTorch implementation of the BlackScholesAPI."""

    def black_scholes_call(self, S, K, T, r, sigma):
        return torch_functional_backend.black_scholes_call(S, K, T, r, sigma)

    def black_scholes_put(self, S, K, T, r, sigma):
        return torch_functional_backend.black_scholes_put(S, K, T, r, sigma)

    def delta_call(self, S, K, T, r, sigma):
        return torch_functional_backend.delta_call(S, K, T, r, sigma)

    def delta_put(self, S, K, T, r, sigma):
        return torch_functional_backend.delta_put(S, K, T, r, sigma)

    def gamma(self, S, K, T, r, sigma):
        return torch_functional_backend.gamma(S, K, T, r, sigma)

    def vega(self, S, K, T, r, sigma):
        return torch_functional_backend.vega(S, K, T, r, sigma)

    def theta_call(self, S, K, T, r, sigma):
        return torch_functional_backend.theta_call(S, K, T, r, sigma)

    def theta_put(self, S, K, T, r, sigma):
        return torch_functional_backend.theta_put(S, K, T, r, sigma)

    def rho_call(self, S, K, T, r, sigma):
        return torch_functional_backend.rho_call(S, K, T, r, sigma)

    def rho_put(self, S, K, T, r, sigma):
        return torch_functional_backend.rho_put(S, K, T, r, sigma)


class PyTorchStatefulBackend:
    """
    Adapter for the stateful PyTorch class to conform to the functional BlackScholesAPI.

    NOTE: This adapter is inefficient for unit testing as it reinstantiates the
    model on every call. It serves to validate correctness against a unified API.
    For performance, use the BlackScholesPyTorch class directly.
    """

    def black_scholes_call(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).price_call()

    def black_scholes_put(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).price_put()

    def delta_call(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["delta_call"]

    def delta_put(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["delta_put"]

    def gamma(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["gamma"]

    def vega(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["vega"]

    def theta_call(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["theta_call"]

    def theta_put(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["theta_put"]

    def rho_call(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["rho_call"]

    def rho_put(self, S, K, T, r, sigma):
        return BlackScholesPyTorch(S, K, T, r, sigma).calculate_greeks()["rho_put"]


class PyTorchTritonBackend:
    """
    Adapter for the fused Triton kernel to conform to the functional API.

    NOTE: Calling individual Greek functions is highly inefficient as it runs
    the fused kernel for all Greeks and discards the rest. This adapter is
    primarily for correctness testing. For performance, a direct call to a
    dedicated 'calculate_greeks' method should be used.
    """

    def black_scholes_call(self, S, K, T, r, sigma):
        # Triton kernel doesn't calculate price, fall back to functional torch
        return torch_functional_backend.black_scholes_call(S, K, T, r, sigma)

    def black_scholes_put(self, S, K, T, r, sigma):
        # Triton kernel doesn't calculate price, fall back to functional torch
        return torch_functional_backend.black_scholes_put(S, K, T, r, sigma)

    def _calculate_all_greeks(self, S, K, T, r, sigma):
        return triton_kernels.calculate_greeks_triton(S, K, T, r, sigma)

    def delta_call(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["delta_call"]

    def delta_put(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["delta_put"]

    def gamma(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["gamma"]

    def vega(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["vega"]

    def theta_call(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["theta_call"]

    def theta_put(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["theta_put"]

    def rho_call(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["rho_call"]

    def rho_put(self, S, K, T, r, sigma):
        return self._calculate_all_greeks(S, K, T, r, sigma)["rho_put"]


# --- Factory ---

_backends = {
    "numpy": NumPyBackend(),
    "torch_functional": PyTorchFunctionalBackend(),
    "torch_stateful": PyTorchStatefulBackend(),
    "torch_triton": PyTorchTritonBackend(),
    "torch_jit": PyTorchJitBackend(),
}


def get_backend(name: BackendName) -> BlackScholesAPI:
    """
    Factory function to retrieve a specific Black-Scholes backend.

    Args:
        name: The name of the backend. One of 'numpy', 'torch_functional',
              'torch_stateful', or 'torch_triton'.

    Returns:
        An object that conforms to the BlackScholesAPI protocol.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    backend = _backends.get(name)
    if backend is None:
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {list(_backends.keys())}"
        )
    return backend
