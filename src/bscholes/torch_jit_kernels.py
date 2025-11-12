# ./src/bscholes/torch_jit_kernels.py
"""
Fused Black-Scholes Greeks calculation using a torch.jit.script function.
"""

import torch
import numpy as np
from typing import Dict, Tuple


@torch.jit.script
def _calculate_greeks_jit(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    JIT-compiled function for fused calculation of all Black-Scholes Greeks.
    Operates purely on PyTorch Tensors.
    """
    # --- Constants must be defined inside the JIT function scope ---
    M_SQRT1_2 = 0.7071067811865476  # 1/sqrt(2)
    M_SQRT_2_PI_INV = 0.3989422804014327  # 1/sqrt(2*pi)
    INV_365 = 1.0 / 365.0

    # --- 1. Pre-compute common terms ---
    sqrt_T = torch.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    d1 = (torch.log(S / K) + (r + 0.5 * sigma.pow(2)) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    # --- 2. Pre-compute PDF and CDF values using JIT-compatible math ---
    # norm_cdf(x) = 0.5 * (1.0 + torch.erf(x * M_SQRT1_2))
    cdf_d1 = 0.5 * (1.0 + torch.erf(d1 * M_SQRT1_2))
    cdf_d2 = 0.5 * (1.0 + torch.erf(d2 * M_SQRT1_2))
    cdf_neg_d2 = 0.5 * (1.0 + torch.erf(-d2 * M_SQRT1_2))

    # norm_pdf(x) = M_SQRT_2_PI_INV * torch.exp(-0.5 * x * x)
    pdf_d1 = M_SQRT_2_PI_INV * torch.exp(-0.5 * d1 * d1)

    # Pre-compute other common terms
    K_exp_rt = K * torch.exp(-r * T)

    # --- 3. Calculate all Greeks ---
    delta_call = cdf_d1
    delta_put = cdf_d1 - 1.0
    gamma = pdf_d1 / (S * sigma_sqrt_T)
    vega = S * pdf_d1 * sqrt_T * 0.01

    theta_term1 = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
    theta_call = (theta_term1 - r * K_exp_rt * cdf_d2) * INV_365
    theta_put = (theta_term1 + r * K_exp_rt * cdf_neg_d2) * INV_365

    rho_call = K_exp_rt * T * cdf_d2 * 0.01
    rho_put = -K_exp_rt * T * cdf_neg_d2 * 0.01

    return (
        delta_call,
        delta_put,
        gamma,
        vega,
        theta_call,
        theta_put,
        rho_call,
        rho_put,
    )


def calculate_greeks_jit(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Python wrapper for the fused JIT kernel. Handles NumPy/Tensor conversion.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Move data to device ---
    S_t = torch.as_tensor(S, device=device, dtype=torch.float64)
    K_t = torch.as_tensor(K, device=device, dtype=torch.float64)
    T_t = torch.as_tensor(T, device=device, dtype=torch.float64)
    r_t = torch.as_tensor(r, device=device, dtype=torch.float64)
    sigma_t = torch.as_tensor(sigma, device=device, dtype=torch.float64)

    # --- 2. Call the JIT-compiled function ---
    greeks_tuple = _calculate_greeks_jit(S_t, K_t, T_t, r_t, sigma_t)

    # --- 3. OPTIMIZED: Stack on GPU, then single DtoH transfer ---
    stacked_greeks = torch.stack(greeks_tuple)
    greeks_np = stacked_greeks.cpu().numpy()

    # --- 4. Unpack NumPy array on CPU ---
    return {
        "delta_call": greeks_np[0],
        "delta_put": greeks_np[1],
        "gamma": greeks_np[2],
        "vega": greeks_np[3],
        "theta_call": greeks_np[4],
        "theta_put": greeks_np[5],
        "rho_call": greeks_np[6],
        "rho_put": greeks_np[7],
    }
