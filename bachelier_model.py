import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from integrals import laguerre_quadrature
from utils import compute_transformed_strike, standard_normal_cdf, standard_normal_pdf


def dF(x, tau, K, Rt, v, PtT):
    """
    Compute the differential dF(x) for the caplet value integral.
    """
    K_hat = compute_transformed_strike(K, tau)
    exponent_term = jnp.exp(x) * jnp.exp(tau * K)
    z = (exponent_term - 1 - tau * Rt) / (tau * v)

    phi_z = norm.pdf(z)
    dF_val = phi_z * exponent_term / (tau * v)

    # Debugging logs
    print(f"dF Debug: K_hat={K_hat}, z={z}, phi_z={phi_z}, dF_val={dF_val}")

    return dF_val


def f(x, tau, K, Rt, v, PtT):
    K_hat = compute_transformed_strike(K, tau)
    z = (jnp.exp(x) * jnp.exp(tau * K) - 1 - tau * Rt) / (tau * v)

    phi_z = standard_normal_pdf(z)
    Phi_K_hat = standard_normal_cdf((K_hat - Rt) / v)

    # Normalized term1
    term1 = PtT * (jnp.exp(2 * x + tau) / (tau * v)) * jnp.exp(-2 * x)
    term2 = -tau * v * Phi_K_hat * phi_z

    # Debugging
    print(f"Adjusted f Debug: x={x}, K_hat={K_hat}, z={z}, phi_z={phi_z}, Phi_K_hat={Phi_K_hat}, term1={term1}, term2={term2}")

    return jnp.where(z < 100, term1 + term2, 0.0)  # Cap large z


def bachelier_caplet_price_laguerre(tau, K, Rt, v, PtT, n_points=10):
    """
    Compute the caplet price using Laguerre quadrature with normalization.
    """
    K_hat = compute_transformed_strike(K, tau)
    D = 1 - standard_normal_cdf((K_hat - Rt) / v)

    if D == 0:
        raise ValueError("D is zero, which indicates no positive payoff.")

    # Function for the integrand
    integrand = lambda x: f(x, tau, K, Rt, v, PtT)

    # Debugging integrand at sample points
    nodes = jnp.linspace(0, 10, 5)  # Test nodes for debugging
    for x in nodes:
        print(f"Integrand Debug: x={x}, value={integrand(x)}")

    # Perform Laguerre quadrature
    caplet_value = laguerre_quadrature(integrand, n_points)
    print(f"Laguerre Result: caplet_value={caplet_value}")
    return caplet_value



@jax.jit
def bachelier_caplet_price_taylor(RtT, K, sigma_gtT, tau, PtT):
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

    # Debugging logs
    print(f"Taylor Debug: K_hat={K_hat}, d={d}, Phi_d={Phi_d}, phi_d={phi_d}")
    print(f"Taylor Debug: integral_z2={integral_z2}, integral_z3={integral_z3}, I={I}")

    V_caplet = tau * PtT * ((RtT - K) * Phi_d - sigma_gtT * phi_d + I)
    print(f"Taylor Result: V_caplet={V_caplet}")
    return V_caplet