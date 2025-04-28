"""
Transform functions for financial calculations
"""

import jax.numpy as jnp


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