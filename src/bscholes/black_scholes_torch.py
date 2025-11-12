"""Black-Scholes option pricing model using PyTorch.

This module implements the Black-Scholes model for European options pricing
and the Greeks (sensitivities). It uses a PyTorch backend for potential GPU
acceleration but exposes a NumPy-based API. All public functions accept
NumPy arrays or scalars and return NumPy arrays.
"""

import torch
import numpy as np
from functools import wraps

# Set device globally for the module
device = torch.device("cuda")
__all__ = [
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes",
    "delta_call",
    "delta_put",
    "gamma",
    "vega",
    "theta_call",
    "theta_put",
    "rho_call",
    "rho_put",
]


def numpy_api(func):
    """
    Decorator to wrap a PyTorch function with a NumPy API.
    - Converts NumPy/scalar inputs to PyTorch tensors on the target device.
    - Executes the core PyTorch function.
    - Converts the resulting PyTorch tensor back to a NumPy array.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert positional arguments
        torch_args = [
            (
                torch.as_tensor(arg, dtype=torch.float64, device=device)
                if isinstance(arg, (np.ndarray, int, float, list))
                else arg
            )
            for arg in args
        ]
        # Convert keyword arguments
        torch_kwargs = {
            k: (
                torch.as_tensor(v, dtype=torch.float64, device=device)
                if isinstance(v, (np.ndarray, int, float, list))
                else v
            )
            for k, v in kwargs.items()
        }

        # Execute the core function with PyTorch tensors
        result_tensor = func(*torch_args, **torch_kwargs)

        # Convert result back to NumPy array
        if isinstance(result_tensor, torch.Tensor):
            return result_tensor.cpu().numpy()
        return result_tensor

    return wrapper


# Internal functions operate on torch.Tensor
def _d1(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))


def _d2(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    return d1_val - sigma * torch.sqrt(T)


# Public functions are decorated to handle NumPy I/O
@numpy_api
def black_scholes_call(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    call_price = S * normal.cdf(d1_val) - K * torch.exp(-r * T) * normal.cdf(d2_val)
    return call_price


@numpy_api
def black_scholes_put(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    put_price = K * torch.exp(-r * T) * normal.cdf(-d2_val) - S * normal.cdf(-d1_val)
    return put_price


@numpy_api
def black_scholes(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
    option_type: str = "call",
) -> torch.Tensor:
    if option_type.lower() == "call":
        # Note: We call the internal tensor-based function, not the decorated one, to avoid double conversion
        return black_scholes_call.__wrapped__(S, K, T, r, sigma)
    elif option_type.lower() == "put":
        return black_scholes_put.__wrapped__(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# Greeks
@numpy_api
def delta_call(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    return normal.cdf(d1_val)


@numpy_api
def delta_put(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    return normal.cdf(d1_val) - 1


@numpy_api
def gamma(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    return torch.exp(normal.log_prob(d1_val)) / (S * sigma * torch.sqrt(T))


@numpy_api
def vega(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    return S * torch.exp(normal.log_prob(d1_val)) * torch.sqrt(T) / 100


@numpy_api
def theta_call(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    term1 = -(S * torch.exp(normal.log_prob(d1_val)) * sigma) / (2 * torch.sqrt(T))
    term2 = -r * K * torch.exp(-r * T) * normal.cdf(d2_val)
    return (term1 + term2) / 365


@numpy_api
def theta_put(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    term1 = -(S * torch.exp(normal.log_prob(d1_val)) * sigma) / (2 * torch.sqrt(T))
    term2 = r * K * torch.exp(-r * T) * normal.cdf(-d2_val)
    return (term1 + term2) / 365


@numpy_api
def rho_call(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d2_val = _d2(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    return K * T * torch.exp(-r * T) * normal.cdf(d2_val) / 100


@numpy_api
def rho_put(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    d2_val = _d2(S, K, T, r, sigma)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.float64),
        torch.tensor(1.0, device=device, dtype=torch.float64),
    )
    return -K * T * torch.exp(-r * T) * normal.cdf(-d2_val) / 100
