import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from functools import partial


def gaussian_legendre_quadrature(f, a, b, n_points=50):
    """
    Performs Gaussian-Legendre quadrature to approximate the integral of f from a to b.

    Parameters:
    f : callable
        The integrand function.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n_points : int
        Number of quadrature points.

    Returns:
    float
        Approximated integral of f from a to b.
    """
    # Get the roots (nodes) and weights for the standard interval [-1, 1]
    nodes, weights = jnp.polynomial.legendre.leggauss(n_points)

    # Change of variables from [a, b] to [-1, 1]
    transformed_nodes = 0.5 * (nodes + 1) * (b - a) + a
    transformed_weights = 0.5 * (b - a) * weights

    # Evaluate the function at the transformed nodes
    fx = f(transformed_nodes)

    # Compute the weighted sum
    integral = jnp.sum(transformed_weights * fx)

    return integral


def compute_transformed_strike(K, tau):
    """
    Computes the transformed strike \hat{K}.

    Parameters:
    K : float
        Strike rate of the caplet.
    tau : float
        Year fraction or accrual period.

    Returns:
    float
        Transformed strike \hat{K}.
    """
    return (jnp.exp(tau * K) - 1) / tau

def standard_normal_cdf(x):
    return norm.cdf(x)

def standard_normal_pdf(x):
    return norm.pdf(x)

def bachelier_caplet_price(RtT, K, sigma_gtT, tau, PtT):
    """
    Compute the caplet price using the Bachelier (normal) model with arithmetic average rates.

    Parameters:
    RtT : float or array
        Forward rate at time t for maturity T.
    K : float
        Strike rate of the caplet.
    sigma_gtT : float or array
        Volatility term (σ * g_t,T).
    tau : float
        Year fraction or accrual period.
    PtT : float or array
        Discount factor from time t to T.

    Returns:
    V_caplet : float or array
        The price of the caplet.
    """
    # Transformed strike
    K_hat = (jnp.exp(tau * K) - 1) / tau

    # d parameter
    d = (RtT - K_hat) / sigma_gtT

    # Standard normal CDF and PDF
    Phi_d = norm.cdf(d)
    phi_d = norm.pdf(d)

    # Integral Term Components
    # Compute moments required for the integral
    # I = ∫_{d}^{∞} [ -tau*(RtT - sigma_gtT*z)^2 / 2 + tau^2*(RtT - sigma_gtT*z)^3 / 3 ] * phi(z) dz

    # Let a = RtT, b = sigma_gtT
    a = RtT
    b = sigma_gtT

    # Compute integrals of (a - b z)^2 * phi(z) dz and (a - b z)^3 * phi(z) dz from d to infinity
    # Expand the terms:
    # (a - b z)^2 = a^2 - 2 a b z + b^2 z^2
    # (a - b z)^3 = a^3 - 3 a^2 b z + 3 a b^2 z^2 - b^3 z^3

    # Compute ∫ (a - b z)^2 phi(z) dz from d to ∞
    integral_z2 = a ** 2 * (1 - Phi_d) - 2 * a * b * phi_d + b ** 2 * (d * phi_d + (1 - Phi_d))

    # Compute ∫ (a - b z)^3 phi(z) dz from d to ∞
    integral_z3 = (
            a ** 3 * (1 - Phi_d)
            - 3 * a ** 2 * b * phi_d
            + 3 * a * b ** 2 * (d * phi_d + (1 - Phi_d))
            - b ** 3 * ((d ** 2 + 1) * phi_d + d * (1 - Phi_d))
    )

    # Now compute the integral term I
    I = (-tau / 2) * integral_z2 + (tau ** 2 / 3) * integral_z3

    # Combine all terms to get the caplet price
    V_caplet = tau * PtT * (
            (RtT - K) * Phi_d
            - sigma_gtT * phi_d
            + I
    )

    return V_caplet


# Example usage with JIT compilation for performance
bachelier_caplet_price_jit = jax.jit(bachelier_caplet_price)

# Test case
if __name__ == "__main__":
    # Sample input values
    RtT = 0.05  # 5% forward rate
    K = 0.04  # 4% strike
    sigma_gtT = 0.01  # 1% volatility
    tau = 0.5  # 6 months accrual
    PtT = 0.98  # Discount factor

    # Compute caplet price
    caplet_price = bachelier_caplet_price_jit(RtT, K, sigma_gtT, tau, PtT)

    print(f"Caplet Price: {caplet_price:.6f}")