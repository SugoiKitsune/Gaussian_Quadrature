"""
Statistical utility functions
"""

import jax.numpy as jnp
from jax.scipy.stats import norm


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