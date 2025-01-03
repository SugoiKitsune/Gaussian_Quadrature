import jax.numpy as jnp
from jax.scipy.stats import norm

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
    """
    Standard normal cumulative distribution function (CDF).
    """
    return norm.cdf(x)

def standard_normal_pdf(x):
    """
    Standard normal probability density function (PDF).
    """
    return norm.pdf(x)