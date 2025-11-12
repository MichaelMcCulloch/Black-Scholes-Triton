"""Black-Scholes option pricing model.

This module implements the Black-Scholes model for European options pricing
and the Greeks (sensitivities).
"""

import numpy as np
from scipy import stats


def _d1(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate d1 parameter for Black-Scholes formula.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        d1 parameter as numpy array
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate d2 parameter for Black-Scholes formula.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        d2 parameter as numpy array
    """
    d1_val = _d1(S, K, T, r, sigma)
    return d1_val - sigma * np.sqrt(T)


def black_scholes_call(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate European call option price using Black-Scholes formula.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Call option price as numpy array
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)

    call_price = S * stats.norm.cdf(d1_val) - K * np.exp(-r * T) * stats.norm.cdf(
        d2_val
    )
    return call_price


def black_scholes_put(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate European put option price using Black-Scholes formula.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Put option price as numpy array
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)

    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2_val) - S * stats.norm.cdf(
        -d1_val
    )
    return put_price


def black_scholes(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    option_type: str = "call",
) -> np.ndarray:
    """Calculate European option price using Black-Scholes formula.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)
        option_type: "call" or "put"

    Returns:
        Option price as numpy array

    Raises:
        ValueError: If option_type is not "call" or "put"
    """
    if option_type.lower() == "call":
        return black_scholes_call(S, K, T, r, sigma)
    elif option_type.lower() == "put":
        return black_scholes_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# Greeks
def delta_call(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate delta (∂V/∂S) for a call option.

    Delta measures the rate of change of option price with respect to the underlying asset price.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Delta as numpy array
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    return stats.norm.cdf(d1_val)


def delta_put(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate delta (∂V/∂S) for a put option.

    Delta measures the rate of change of option price with respect to the underlying asset price.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Delta as numpy array
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    return stats.norm.cdf(d1_val) - 1


def gamma(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate gamma (∂²V/∂S²) for an option.

    Gamma measures the rate of change of delta with respect to the underlying asset price.
    Gamma is the same for both call and put options.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Gamma as numpy array
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    return stats.norm.pdf(d1_val) / (S * sigma * np.sqrt(T))


def vega(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate vega (∂V/∂σ) for an option.

    Vega measures the sensitivity of option price to volatility.
    Vega is the same for both call and put options.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Vega as numpy array (per 1% change in volatility)
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    return (
        S * stats.norm.pdf(d1_val) * np.sqrt(T) / 100
    )  # Divided by 100 for per 1% change


def theta_call(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate theta (∂V/∂T) for a call option.

    Theta measures the sensitivity of option price to the passage of time (time decay).

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Theta as numpy array (per day)
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)

    term1 = -(S * stats.norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2_val)

    return (term1 + term2) / 365  # Divided by 365 for per-day theta


def theta_put(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate theta (∂V/∂T) for a put option.

    Theta measures the sensitivity of option price to the passage of time (time decay).

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Theta as numpy array (per day)
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, r, sigma)

    term1 = -(S * stats.norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2_val)

    return (term1 + term2) / 365  # Divided by 365 for per-day theta


def rho_call(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate rho (∂V/∂r) for a call option.

    Rho measures the sensitivity of option price to interest rate changes.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Rho as numpy array (per 1% change in interest rate)
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d2_val = _d2(S, K, T, r, sigma)
    return (
        K * T * np.exp(-r * T) * stats.norm.cdf(d2_val) / 100
    )  # Divided by 100 for per 1% change


def rho_put(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Calculate rho (∂V/∂r) for a put option.

    Rho measures the sensitivity of option price to interest rate changes.

    Args:
        S: Spot price (current asset price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)

    Returns:
        Rho as numpy array (per 1% change in interest rate)
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    d2_val = _d2(S, K, T, r, sigma)
    return (
        -K * T * np.exp(-r * T) * stats.norm.cdf(-d2_val) / 100
    )  # Divided by 100 for per 1% change
