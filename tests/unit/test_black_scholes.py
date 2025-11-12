# ./tests/unit/test_black_scholes.py
"""
Unit tests for all Black-Scholes backends, validated against a unified API.
"""

import numpy as np
import pytest

from bscholes import get_backend

# --- Backend Fixture Setup ---

ALL_BACKEND_IDS = [
    "numpy",
    "torch_functional",
    "torch_stateful",
    "torch_triton",
    "torch_jit",
]


@pytest.fixture(params=ALL_BACKEND_IDS)
def backend(request):
    """Provides a parameterized backend instance via the factory."""
    return get_backend(request.param)


# --- Shared Test Data and Expected Values Fixtures ---
@pytest.fixture(scope="module")
def scalar_params():
    """Provides standard scalar parameters for an at-the-money option."""
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}


@pytest.fixture(scope="module")
def scalar_expected():
    """Provides expected results for the scalar parameters."""
    return {
        "call": 10.4506,
        "put": 5.5735,
        "delta_call": 0.6368,
        "delta_put": -0.3632,
        "gamma": 0.0188,
        "vega": 0.3753,
        "theta_call": -0.0192,
        "theta_put": -0.0045,
        "rho_call": 0.5319,
        "rho_put": -0.4193,
    }


@pytest.fixture(scope="module")
def vector_params():
    """Provides vectorized parameters for multiple options."""
    return {
        "S": np.array([90.0, 100.0, 110.0]),
        "K": np.array([100.0, 100.0, 100.0]),
        "T": np.array([1.0, 1.0, 1.0]),
        "r": np.array([0.05, 0.05, 0.05]),
        "sigma": np.array([0.2, 0.2, 0.2]),
    }


@pytest.fixture(scope="module")
def vector_expected():
    """Provides expected results for the vector parameters."""
    return {
        "call": np.array([5.0912, 10.4506, 17.6630]),
        "put": np.array([10.2141, 5.5735, 2.7859]),
        "delta_call": np.array([0.4298, 0.6368, 0.7958]),
        "gamma": np.array([0.0218, 0.0188, 0.0129]),
    }


# ==============================================================================
# == Unified Tests for All Backends
# ==============================================================================
class TestAllBackends:
    """Test suite for all backends using the unified functional API."""

    def test_call_price(self, backend, scalar_params, scalar_expected):
        result = backend.black_scholes_call(**scalar_params)
        assert np.isclose(result, scalar_expected["call"], rtol=1e-4)

    def test_put_price(self, backend, scalar_params, scalar_expected):
        result = backend.black_scholes_put(**scalar_params)
        assert np.isclose(result, scalar_expected["put"], rtol=1e-4)

    def test_delta_call(self, backend, scalar_params, scalar_expected):
        result = backend.delta_call(**scalar_params)
        assert np.isclose(result, scalar_expected["delta_call"], rtol=1e-3)

    def test_delta_put(self, backend, scalar_params, scalar_expected):
        result = backend.delta_put(**scalar_params)
        assert np.isclose(result, scalar_expected["delta_put"], rtol=1e-3)

    def test_gamma(self, backend, scalar_params, scalar_expected):
        result = backend.gamma(**scalar_params)
        assert np.isclose(result, scalar_expected["gamma"], rtol=1e-2)

    def test_vega(self, backend, scalar_params, scalar_expected):
        result = backend.vega(**scalar_params)
        assert np.isclose(result, scalar_expected["vega"], rtol=1e-3)

    def test_theta_call(self, backend, scalar_params, scalar_expected):
        result = backend.theta_call(**scalar_params)
        assert np.isclose(result, scalar_expected["theta_call"], rtol=0.1)

    def test_theta_put(self, backend, scalar_params, scalar_expected):
        result = backend.theta_put(**scalar_params)
        assert np.isclose(result, scalar_expected["theta_put"], rtol=0.1)

    def test_rho_call(self, backend, scalar_params, scalar_expected):
        result = backend.rho_call(**scalar_params)
        assert np.isclose(result, scalar_expected["rho_call"], rtol=1e-2)

    def test_rho_put(self, backend, scalar_params, scalar_expected):
        result = backend.rho_put(**scalar_params)
        assert np.isclose(result, scalar_expected["rho_put"], rtol=1e-2)

    def test_vectorization(self, backend, vector_params, vector_expected):
        result = backend.black_scholes_call(**vector_params)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.allclose(result, vector_expected["call"], rtol=1e-3)

    def test_put_call_parity(self, backend, scalar_params):
        """Test put-call parity relationship."""
        call = backend.black_scholes_call(**scalar_params)
        put = backend.black_scholes_put(**scalar_params)
        S, K, T, r = (
            scalar_params["S"],
            scalar_params["K"],
            scalar_params["T"],
            scalar_params["r"],
        )
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        assert np.isclose(lhs, rhs)
