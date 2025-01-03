import jax.numpy as jnp
from jax.scipy.special import erf


# ----------------------------
# Helper Functions
# ----------------------------

def compute_transformed_strike(K, tau):
    """
    Computes the transformed strike price.

    Parameters:
    K (float or jnp.ndarray): Original strike price.
    tau (float): Time parameter.

    Returns:
    jnp.ndarray: Transformed strike price.
    """
    # Example implementation: Exponentially decayed strike
    K_hat = K * jnp.exp(-tau)
    return K_hat


def standard_normal_pdf(z):
    """
    Computes the standard normal probability density function (PDF).

    Parameters:
    z (float or jnp.ndarray): Input value(s).

    Returns:
    jnp.ndarray: PDF evaluated at z.
    """
    return (1.0 / jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * z ** 2)


def standard_normal_cdf(z):
    """
    Computes the standard normal cumulative distribution function (CDF).

    Parameters:
    z (float or jnp.ndarray): Input value(s).

    Returns:
    jnp.ndarray: CDF evaluated at z.
    """
    return 0.5 * (1 + erf(z / jnp.sqrt(2)))


# ----------------------------
# Integrand Function
# ----------------------------

def f(x, tau, K, Rt, v, PtT):
    """
    Computes the integrand function for Gaussian-Laguerre quadrature.

    Parameters:
    x (float or jnp.ndarray): Integration variable.
    tau (float): Time parameter.
    K (float): Strike price.
    Rt (float): Risk-free rate.
    v (float): Volatility.
    PtT (float): Present value or other scaling factor.

    Returns:
    jnp.ndarray: Evaluated integrand.
    """
    # Compute transformed strike
    K_hat = compute_transformed_strike(K, tau)

    # Compute z
    z = (jnp.exp(x) * jnp.exp(tau * K) - 1 - tau * Rt) / (tau * v)

    # Compute standard normal PDF and CDF
    phi_z = standard_normal_pdf(z)
    Phi_K_hat = standard_normal_cdf((K_hat - Rt) / v)

    # Compute terms
    term1 = PtT * (jnp.exp(2 * x + tau) / (tau * v)) * jnp.exp(-2 * x)
    term2 = -tau * v * Phi_K_hat * phi_z

    # Combine terms
    integrand_value = term1 + term2

    # Apply condition: if z < 100, keep the value; else, set to 0
    return jnp.where(z < 100, integrand_value, 0.0)