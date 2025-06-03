"""
Bachelier model implementation for caplet pricing
"""

import jax
import jax.numpy as jnp
from ..utils.transforms import compute_transformed_strike
from ..utils.stats import standard_normal_cdf, standard_normal_pdf
# Remove direct import to break circular dependency
# We'll use lazy import inside functions that need it


def dF(x, tau, K, Rt, v, PtT):
    """
    Compute the differential dF(x) for the caplet value integral.
    """
    K_hat = compute_transformed_strike(K, tau)
    exponent_term = jnp.exp(x) * jnp.exp(tau * K)
    z = (exponent_term - 1 - tau * Rt) / (tau * v)

    phi_z = standard_normal_pdf(z)
    dF_val = phi_z * exponent_term / (tau * v)
    return dF_val


def f(x, tau, K, Rt, v, PtT):
    K_hat = compute_transformed_strike(K, tau)
    z = (jnp.exp(x) * jnp.exp(tau * K) - 1 - tau * Rt) / (tau * v)

    phi_z = standard_normal_pdf(z)
    Phi_K_hat = standard_normal_cdf((K_hat - Rt) / v)

    term1 = PtT * (jnp.exp(2 * x + tau) / (tau * v)) * jnp.exp(-2 * x)
    term2 = -tau * v * Phi_K_hat * phi_z

    return jnp.where(z < 100, term1 + term2, 0.0)  # Cap large z


@jax.jit
def bachelier_caplet_price_laguerre(tau, K, Rt, v, PtT, n_points=20):
    """
    Compute the caplet price using improved Laguerre quadrature.
    
    Parameters:
    tau : float
        Year fraction or accrual period
    K : float
        Strike rate
    Rt : float
        Forward rate
    v : float
        Volatility term
    PtT : float
        Discount factor
    n_points : int
        Number of quadrature points (default: 20)
    """
    # Import inside the function to avoid circular dependency
    from ..utils.quadrature import generalized_laguerre_quadrature
    
    # Use alpha=1 for better convergence with caplet payoff structure
    caplet_value = generalized_laguerre_quadrature(
        lambda x: f(x, tau, K, Rt, v, PtT), 
        n_points=n_points,
        alpha=1.0
    )
    
    return jnp.maximum(caplet_value, 0.0)  # Ensure non-negative price


@jax.jit
def bachelier_caplet_price_taylor(RtT, K, sigma_gtT, tau, PtT):
    """
    Compute caplet price using Taylor series approximation.
    """
    K_hat = compute_transformed_strike(K, tau)
    d = (RtT - K_hat) / sigma_gtT
    Phi_d = standard_normal_cdf(d)
    phi_d = standard_normal_pdf(d)

    a, b = RtT, sigma_gtT
    integral_z2 = a**2 * (1 - Phi_d) - 2 * a * b * phi_d + b**2 * (d * phi_d + (1 - Phi_d))
    integral_z3 = (
        a**3 * (1 - Phi_d)
        - 3 * a**2 * b * phi_d
        + 3 * a * b**2 * (d * phi_d + (1 - Phi_d))
        - b**3 * ((d**2 + 1) * phi_d + d * (1 - Phi_d))
    )
    I = (-tau / 2) * integral_z2 + (tau**2 / 3) * integral_z3

    V_caplet = tau * PtT * ((RtT - K) * Phi_d - sigma_gtT * phi_d + I)
    return V_caplet