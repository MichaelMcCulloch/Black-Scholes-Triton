# ./src/bscholes/triton_kernels.py
"""
Fused Black-Scholes Greeks calculation using a custom Triton kernel.
"""

import torch
import triton
import triton.language as tl
from typing import Dict
import numpy as np


@triton.jit
def norm_cdf(x):
    """Triton JIT implementation of the standard normal CDF."""
    # Constant must be defined inside the JIT function.
    M_SQRT1_2 = 0.7071067811865476  # 1/sqrt(2)
    return 0.5 * (1.0 + tl.erf(x * M_SQRT1_2))


@triton.jit
def norm_pdf(x):
    """Triton JIT implementation of the standard normal PDF."""
    # Constant must be defined inside the JIT function.
    M_SQRT_2_PI_INV = 0.3989422804014327  # 1/sqrt(2*pi)
    return M_SQRT_2_PI_INV * tl.exp(-0.5 * x * x)


@triton.jit
def greeks_kernel(
    S_ptr,
    K_ptr,
    T_ptr,
    r_ptr,
    sigma_ptr,
    delta_call_ptr,
    delta_put_ptr,
    gamma_ptr,
    vega_ptr,
    theta_call_ptr,
    theta_put_ptr,
    rho_call_ptr,
    rho_put_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused calculation of all Black-Scholes Greeks.
    """
    # --- 1. Calculate offsets and apply mask for parallel processing ---
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # --- 2. Load a block of input data from HBM into SRAM ---
    S = tl.load(S_ptr + offsets, mask=mask)
    K = tl.load(K_ptr + offsets, mask=mask)
    T = tl.load(T_ptr + offsets, mask=mask)
    r = tl.load(r_ptr + offsets, mask=mask)
    sigma = tl.load(sigma_ptr + offsets, mask=mask)

    # --- 3. Perform all calculations in SRAM/registers ---
    # Pre-compute common terms
    sqrt_T = tl.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T
    d1 = (tl.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    # Pre-compute PDF and CDF values
    pdf_d1 = norm_pdf(d1)
    cdf_d1 = norm_cdf(d1)
    cdf_d2 = norm_cdf(d2)
    cdf_neg_d2 = norm_cdf(-d2)

    # Pre-compute other common terms
    K_exp_rt = K * tl.exp(-r * T)

    # Calculate all Greeks
    delta_call = cdf_d1
    delta_put = cdf_d1 - 1.0
    gamma = pdf_d1 / (S * sigma_sqrt_T)
    vega = S * pdf_d1 * sqrt_T * 0.01

    theta_term1 = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
    INV_365 = 1.0 / 365.0
    theta_call = (theta_term1 - r * K_exp_rt * cdf_d2) * INV_365
    theta_put = (theta_term1 + r * K_exp_rt * cdf_neg_d2) * INV_365

    rho_call = K_exp_rt * T * cdf_d2 * 0.01
    rho_put = -K_exp_rt * T * cdf_neg_d2 * 0.01

    # --- 4. Write final results from SRAM back to HBM ---
    tl.store(delta_call_ptr + offsets, delta_call, mask=mask)
    tl.store(delta_put_ptr + offsets, delta_put, mask=mask)
    tl.store(gamma_ptr + offsets, gamma, mask=mask)
    tl.store(vega_ptr + offsets, vega, mask=mask)
    tl.store(theta_call_ptr + offsets, theta_call, mask=mask)
    tl.store(theta_put_ptr + offsets, theta_put, mask=mask)
    tl.store(rho_call_ptr + offsets, rho_call, mask=mask)
    tl.store(rho_put_ptr + offsets, rho_put, mask=mask)


def calculate_greeks_triton(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Python wrapper for the fused Triton kernel.
    """
    # ... (data transfer and kernel launch are unchanged) ...
    # --- 1. Move data to GPU ---
    device = torch.device("cuda")
    S_t = torch.as_tensor(S, device=device, dtype=torch.float64)
    K_t = torch.as_tensor(K, device=device, dtype=torch.float64)
    T_t = torch.as_tensor(T, device=device, dtype=torch.float64)
    r_t = torch.as_tensor(r, device=device, dtype=torch.float64)
    sigma_t = torch.as_tensor(sigma, device=device, dtype=torch.float64)
    n_elements = S_t.numel()

    # --- 2. Allocate output tensors on GPU ---
    delta_call_t = torch.empty_like(S_t)
    delta_put_t = torch.empty_like(S_t)
    gamma_t = torch.empty_like(S_t)
    vega_t = torch.empty_like(S_t)
    theta_call_t = torch.empty_like(S_t)
    theta_put_t = torch.empty_like(S_t)
    rho_call_t = torch.empty_like(S_t)
    rho_put_t = torch.empty_like(S_t)

    # --- 3. Launch Triton kernel ---
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    greeks_kernel[grid](
        S_t,
        K_t,
        T_t,
        r_t,
        sigma_t,
        delta_call_t,
        delta_put_t,
        gamma_t,
        vega_t,
        theta_call_t,
        theta_put_t,
        rho_call_t,
        rho_put_t,
        n_elements,
        BLOCK_SIZE=1024,
    )

    # --- 4. OPTIMIZED: Stack on GPU, then single DtoH transfer ---
    greeks_tensors = [
        delta_call_t,
        delta_put_t,
        gamma_t,
        vega_t,
        theta_call_t,
        theta_put_t,
        rho_call_t,
        rho_put_t,
    ]
    stacked_greeks = torch.stack(greeks_tensors)
    greeks_np = stacked_greeks.cpu().numpy()

    # --- 5. Unpack NumPy array on CPU ---
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
