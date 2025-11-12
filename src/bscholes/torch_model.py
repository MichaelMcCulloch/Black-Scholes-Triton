import torch
import numpy as np
from typing import Dict

# Internal functions remain the same but operate on torch.Tensor
from .black_scholes_torch import _d1, _d2


class BlackScholesPyTorch:
    """
    A stateful Black-Scholes calculator that holds data on a target device.
    """

    def __init__(
        self,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ):
        """
        Initializes the model and transfers all input data to the target device.

        Args:
            S, K, T, r, sigma: NumPy arrays or scalars for the option parameters.
        """
        self.device = torch.device("cuda")

        # Transfer data to the device ONCE.
        self.S = torch.as_tensor(S, dtype=torch.float64, device=self.device)
        self.K = torch.as_tensor(K, dtype=torch.float64, device=self.device)
        self.T = torch.as_tensor(T, dtype=torch.float64, device=self.device)
        self.r = torch.as_tensor(r, dtype=torch.float64, device=self.device)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float64, device=self.device)

        self.normal = torch.distributions.Normal(
            torch.tensor(0.0, device=self.device, dtype=torch.float64),
            torch.tensor(1.0, device=self.device, dtype=torch.float64),
        )

    def _price(self, option_type: str = "call") -> torch.Tensor:
        """Internal calculation, returns a torch.Tensor on the device."""
        d1_val = _d1(self.S, self.K, self.T, self.r, self.sigma)
        d2_val = _d2(self.S, self.K, self.T, self.r, self.sigma)

        if option_type == "call":
            return self.S * self.normal.cdf(d1_val) - self.K * torch.exp(
                -self.r * self.T
            ) * self.normal.cdf(d2_val)
        elif option_type == "put":
            return self.K * torch.exp(-self.r * self.T) * self.normal.cdf(
                -d2_val
            ) - self.S * self.normal.cdf(-d1_val)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def price_call(self) -> np.ndarray:
        """Calculates call option price and returns it as a NumPy array."""
        return self._price("call").cpu().numpy()

    def price_put(self) -> np.ndarray:
        """Calculates put option price and returns it as a NumPy array."""
        return self._price("put").cpu().numpy()

    def calculate_greeks(self) -> Dict[str, np.ndarray]:
        """
        Calculates all primary Greeks in a single pass by computing common
        subexpressions once, minimizing redundant computation and kernel launches.

        Returns:
            A dictionary of NumPy arrays for all primary Greeks.
        """
        # --- 1. Pre-compute common terms ONCE on the GPU ---
        sqrt_T = torch.sqrt(self.T)
        sigma_sqrt_T = self.sigma * sqrt_T
        d1 = (
            torch.log(self.S / self.K) + (self.r + 0.5 * self.sigma.pow(2)) * self.T
        ) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T

        # Pre-compute PDF and CDF values
        pdf_d1 = torch.exp(self.normal.log_prob(d1))
        cdf_d1 = self.normal.cdf(d1)
        cdf_d2 = self.normal.cdf(d2)
        cdf_neg_d2 = self.normal.cdf(-d2)

        # Pre-compute other common terms
        K_exp_rt = self.K * torch.exp(-self.r * self.T)

        # --- 2. Calculate all Greeks using pre-computed terms ---
        delta_call = cdf_d1
        delta_put = cdf_d1 - 1.0
        gamma = pdf_d1 / (self.S * sigma_sqrt_T)
        vega = self.S * pdf_d1 * sqrt_T * 0.01  # mult by 0.01 is faster than div

        theta_term1 = -(self.S * pdf_d1 * self.sigma) / (2.0 * sqrt_T)
        INV_365 = 1.0 / 365.0
        theta_call = (theta_term1 - self.r * K_exp_rt * cdf_d2) * INV_365
        theta_put = (theta_term1 + self.r * K_exp_rt * cdf_neg_d2) * INV_365

        rho_call = K_exp_rt * self.T * cdf_d2 * 0.01
        rho_put = -K_exp_rt * self.T * cdf_neg_d2 * 0.01

        # --- 3. Stack results on GPU and perform one DtoH transfer ---
        greeks_tensors = [
            delta_call,
            delta_put,
            gamma,
            vega,
            theta_call,
            theta_put,
            rho_call,
            rho_put,
        ]
        stacked_greeks = torch.stack(greeks_tensors)
        greeks_np = stacked_greeks.cpu().numpy()

        # --- 4. Unpack NumPy array on CPU and return ---
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
