"""
Test suite for pricing models
"""

import pytest
import jax.numpy as jnp
from gaussian_quadrature.models.bachelier import (
    bachelier_caplet_price_taylor,
    bachelier_caplet_price_laguerre
)
from gaussian_quadrature.pricing.pricer import compare_pricing_models


@pytest.mark.parametrize("params", [
    {
        "RtT": 0.05,    # 5% forward rate
        "K": 0.04,      # 4% strike
        "sigma_gtT": 0.01, # 1% volatility
        "tau": 0.5,     # 6 months
        "PtT": 0.98,    # Discount factor
    },
    {
        "RtT": 0.03,    # 3% forward rate
        "K": 0.035,     # 3.5% strike (out of money)
        "sigma_gtT": 0.015, # 1.5% volatility
        "tau": 0.25,    # 3 months
        "PtT": 0.99,    # Discount factor
    }
])
def test_pricing_models_consistency(params):
    """Test consistency between Taylor and Laguerre methods"""
    results = compare_pricing_models(
        params["RtT"],
        params["K"],
        params["sigma_gtT"],
        params["tau"],
        params["PtT"],
        n_points=30
    )
    
    taylor_price = results["Taylor Series"]["Price"]
    laguerre_price = results["Laguerre Quadrature"]["Price"]
    
    # Check relative difference is within tolerance
    rel_diff = jnp.abs(taylor_price - laguerre_price) / taylor_price
    assert rel_diff < 0.01, f"Large relative difference: {rel_diff:.4f}"


def test_price_bounds():
    """Test that prices respect theoretical bounds"""
    RtT = 0.05      # 5% forward rate
    K = 0.04        # 4% strike
    sigma_gtT = 0.01 # 1% volatility
    tau = 0.5       # 6 months
    PtT = 0.98      # Discount factor
    
    # Test Taylor price
    taylor_price = bachelier_caplet_price_taylor(RtT, K, sigma_gtT, tau, PtT)
    assert taylor_price >= 0, "Taylor price negative"
    assert taylor_price <= tau * PtT * RtT, "Taylor price exceeds maximum bound"
    
    # Test Laguerre price
    laguerre_price = bachelier_caplet_price_laguerre(tau, K, RtT, sigma_gtT, PtT)
    assert laguerre_price >= 0, "Laguerre price negative"
    assert laguerre_price <= tau * PtT * RtT, "Laguerre price exceeds maximum bound"


def test_volatility_sensitivity():
    """Test price sensitivity to volatility changes"""
    RtT = 0.05
    K = 0.04
    tau = 0.5
    PtT = 0.98
    
    # Test increasing volatility leads to higher prices
    sigma_low = 0.01
    sigma_high = 0.02
    
    price_low = bachelier_caplet_price_taylor(RtT, K, sigma_low, tau, PtT)
    price_high = bachelier_caplet_price_taylor(RtT, K, sigma_high, tau, PtT)
    
    assert price_high > price_low, "Price not increasing with volatility"


if __name__ == "__main__":
    pytest.main([__file__])