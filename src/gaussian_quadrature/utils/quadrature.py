"""
Core quadrature implementations with numerical stability improvements
"""

import jax.numpy as jnp
from jax.scipy.special import gamma


def laguerre_function_final(n, x, alpha=0.0):
    """
    Final version of scaled Laguerre functions with additional error control.
    """
    x = jnp.asarray(x)
    x_scaled = jnp.where(x < 1.0, x, x * (1.0 + jnp.log(x))/x)
    
    L0 = jnp.exp(-x_scaled/2)
    if n == 0:
        return L0, -0.5 * L0
    
    L1 = (alpha + 1 - x_scaled) * jnp.exp(-x_scaled/2)
    if n == 1:
        return L1, -(alpha + 3 - x_scaled)/2 * jnp.exp(-x_scaled/2)
    
    L_prev = L0
    L_curr = L1
    
    for k in range(1, n):
        scale = 1.0 / jnp.maximum(1.0, jnp.sqrt(k))
        term1 = (2*k + alpha + 1 - x_scaled) * L_curr * scale
        term2 = (k + alpha) * L_prev * scale
        L_next = (term1 - term2) / (k + 1)
        L_prev = L_curr
        L_curr = L_next
    
    deriv = -0.5 * L_curr - 0.5 * L_prev
    return L_curr, deriv


def compute_gauss_laguerre_points_final(n, alpha=0.0, max_iter=100, tol=1e-12):
    """
    Final version of Gauss-Laguerre quadrature points computation with stability improvements.
    """
    j = jnp.arange(1, n + 1)
    theta = jnp.pi * (j - 0.25) / (n + 0.5)
    nodes = (n + alpha/2 + 0.75) * jnp.cos(theta)**2
    
    for iter in range(max_iter):
        L_n, dL_n = laguerre_function_final(n, nodes, alpha)
        delta = L_n / dL_n
        damping = 1.0 / (1.0 + 0.2 * iter)
        delta = damping * jnp.clip(delta, -0.5, 0.5)
        
        nodes_new = nodes - delta
        nodes_new = jnp.maximum(nodes_new, 0.0)
        
        if jnp.any(nodes_new[1:] - nodes_new[:-1] <= 0):
            damping *= 0.5
            nodes_new = nodes - damping * delta
        
        if jnp.max(jnp.abs(nodes_new - nodes)) < tol:
            nodes = nodes_new
            break
            
        nodes = nodes_new
    
    idx = jnp.argsort(nodes)
    nodes = nodes[idx]
    
    L_n, _ = laguerre_function_final(n, nodes, alpha)
    log_weights = (
        -0.5 * nodes
        + jnp.log(nodes)
        + jnp.log(gamma(n + alpha + 1))
        - jnp.log(n * gamma(n))
        - 2 * jnp.log(jnp.abs(L_n))
    )
    
    max_log = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log)
    weights = weights / jnp.sum(weights)
    
    cond_num = jnp.max(weights) / jnp.min(weights)
    if cond_num > 1e12:
        weights = jnp.sqrt(weights)
        weights = weights / jnp.sum(weights)
    
    return nodes, weights


def generalized_laguerre_quadrature(f, n_points, alpha=0):
    """
    Performs Gaussian-Laguerre quadrature with stability improvements.

    Parameters:
    f : callable
        The integrand function
    n_points : int
        Number of quadrature points
    alpha : float, optional
        Parameter for generalized Laguerre polynomial

    Returns:
    float
        Approximated integral
    """
    if n_points > 100:
        raise ValueError("n_points > 100 may lead to numerical instability")
    
    nodes, weights = compute_gauss_laguerre_points_final(n_points, alpha)
    f_vals = jnp.array([f(x) for x in nodes])
    f_vals = jnp.where(jnp.isfinite(f_vals), f_vals, 0.0)
    
    result = 0.0
    correction = 0.0
    for w, v in zip(weights, f_vals):
        term = w * v - correction
        temp = result + term
        correction = (temp - result) - term
        result = temp
    
    return result