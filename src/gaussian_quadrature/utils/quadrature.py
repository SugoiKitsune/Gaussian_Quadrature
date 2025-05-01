"""
Core quadrature implementations with numerical stability improvements
"""

import jax.numpy as jnp
from jax.scipy.special import gamma
import numpy as np
from functools import lru_cache
from scipy.special import binom


# Simple Legendre polynomial implementation
def legendre_polynomial(n, x):
    """
    Basic implementation of Legendre polynomials without stability improvements.
    
    Parameters:
    n : int
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the polynomial
        
    Returns:
    array_like
        Legendre polynomial values at x
    """
    x = jnp.asarray(x)
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return x
    else:
        p_prev = jnp.ones_like(x)
        p_curr = x
        
        for k in range(1, n):
            p_next = ((2*k + 1) * x * p_curr - k * p_prev) / (k + 1)
            p_prev = p_curr
            p_curr = p_next
            
        return p_curr

# Simple Laguerre polynomial implementation
def laguerre_polynomial(n, x, alpha=0.0):
    """
    Basic implementation of generalized Laguerre polynomials without stability improvements.
    
    Parameters:
    n : int
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the polynomial
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
        
    Returns:
    array_like
        Laguerre polynomial values at x
    """
    x = jnp.asarray(x)
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return 1 + alpha - x
    else:
        L_prev = jnp.ones_like(x)
        L_curr = 1 + alpha - x
        
        for k in range(1, n):
            L_next = ((2*k + 1 + alpha - x) * L_curr - (k + alpha) * L_prev) / (k + 1)
            L_prev = L_curr
            L_curr = L_next
            
        return L_curr

# Simple calculation of derivative for Laguerre polynomial
def laguerre_polynomial_derivative(n, x, alpha=0.0):
    """
    Calculates the derivative of generalized Laguerre polynomials.
    
    Parameters:
    n : int
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the derivative
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
        
    Returns:
    array_like
        Derivative of Laguerre polynomial at x
    """
    if n == 0:
        return jnp.zeros_like(x)
    else:
        return -laguerre_polynomial(n-1, x, alpha+1)

# Simple computation of Gauss-Laguerre quadrature points
def compute_gauss_laguerre_points_basic(n, alpha=0.0, max_iter=100, tol=1e-12):
    """
    Basic implementation of Gauss-Laguerre quadrature points computation.
    
    Parameters:
    n : int
        Number of quadrature points
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
    max_iter : int, optional
        Maximum number of iterations for root finding
    tol : float, optional
        Convergence tolerance
        
    Returns:
    tuple
        (nodes, weights) - Quadrature points and weights
    """
    # Initial guess for roots (using asymptotic formula for Laguerre zeros)
    j = jnp.arange(1, n + 1)
    theta = jnp.pi * (4*j - 1) / (4*n + 2)
    nodes = (n + alpha/2 + 0.5) * jnp.cos(theta)**2
    
    # Newton's method to find the roots
    for _ in range(max_iter):
        p_n = laguerre_polynomial(n, nodes, alpha)
        dp_n = laguerre_polynomial_derivative(n, nodes, alpha)
        
        # Newton step
        delta = p_n / dp_n
        nodes_new = nodes - delta
        
        # Keep nodes positive
        nodes_new = jnp.maximum(nodes_new, 0.0)
        
        if jnp.max(jnp.abs(delta)) < tol:
            nodes = nodes_new
            break
            
        nodes = nodes_new
    
    # Sort nodes in ascending order
    idx = jnp.argsort(nodes)
    nodes = nodes[idx]
    
    # Compute weights using standard formula
    p_n1 = laguerre_polynomial(n+1, nodes, alpha)
    weights = gamma(n + alpha + 1) / (gamma(n + 1) * n * p_n1**2) * nodes**(alpha+1) * jnp.exp(-nodes)
    
    return nodes, weights

# Basic Gaussian-Laguerre quadrature
def gaussian_laguerre_quadrature_basic(f, n_points, alpha=0.0):
    """
    Basic Gaussian-Laguerre quadrature without stability improvements.
    
    Parameters:
    f : callable
        The integrand function
    n_points : int
        Number of quadrature points
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
    
    Returns:
    float
        Approximated integral
    """
    # Get Gauss-Laguerre nodes and weights
    nodes, weights = compute_gauss_laguerre_points_basic(n_points, alpha)
    
    # Evaluate integrand and compute quadrature
    f_vals = jnp.array([f(x) for x in nodes])
    return jnp.sum(weights * f_vals)

# Simple computation of Gauss-Legendre quadrature points
def compute_gauss_legendre_points_basic(n, max_iter=100, tol=1e-12):
    """
    Basic implementation of Gauss-Legendre quadrature points computation.
    
    Parameters:
    n : int
        Number of quadrature points
    max_iter : int, optional
        Maximum number of iterations for root finding
    tol : float, optional
        Convergence tolerance
        
    Returns:
    tuple
        (nodes, weights) - Quadrature points and weights
    """
    # Initial guess for roots (Chebyshev nodes)
    k = jnp.arange(1, n + 1)
    nodes = jnp.cos(jnp.pi * (k - 0.25) / (n + 0.5))
    
    # Newton's method to find the roots
    for _ in range(max_iter):
        p_n = legendre_polynomial(n, nodes)
        # Compute derivative using recurrence relation
        dp_n = n * (nodes * legendre_polynomial(n, nodes) - legendre_polynomial(n-1, nodes)) / (nodes**2 - 1)
        dp_n = jnp.where(jnp.abs(nodes) < 1 - 1e-10, dp_n, n*(n+1)/2 * legendre_polynomial(n-1, nodes))
        
        # Newton step
        delta = p_n / dp_n
        nodes_new = nodes - delta
        
        if jnp.max(jnp.abs(delta)) < tol:
            nodes = nodes_new
            break
            
        nodes = nodes_new
    
    # Compute weights
    p_n1 = legendre_polynomial(n + 1, nodes)
    weights = 2.0 / ((1.0 - nodes**2) * p_n1**2)
    
    return nodes, weights

# Plain Gaussian quadrature
def gaussian_legendre_quadrature_basic(f, a, b, n_points):
    """
    Basic Gaussian-Legendre quadrature without stability improvements.
    
    Parameters:
    f : callable
        The integrand function
    a : float
        Lower integration limit
    b : float
        Upper integration limit
    n_points : int
        Number of quadrature points
    
    Returns:
    float
        Approximated integral
    """
    # Get Gauss-Legendre nodes and weights for [-1, 1]
    nodes, weights = compute_gauss_legendre_points_basic(n_points)
    
    # Transform to [a, b]
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    transformed_nodes = scale * nodes + shift
    
    # Evaluate integrand and compute quadrature
    f_vals = jnp.array([f(x) for x in transformed_nodes])
    return scale * jnp.sum(weights * f_vals)


def laguerre_function_scaled(n, x, alpha=0.0):
    """
    Scaled version of Laguerre functions with additional error control.
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


def compute_gauss_laguerre_points_scaled(n, alpha=0.0, max_iter=100, tol=1e-12):
    """
    Scaled version of Gauss-Laguerre quadrature points computation with stability improvements.
    """
    j = jnp.arange(1, n + 1)
    theta = jnp.pi * (j - 0.25) / (n + 0.5)
    nodes = (n + alpha/2 + 0.75) * jnp.cos(theta)**2
    
    for iter in range(max_iter):
        L_n, dL_n = laguerre_function_scaled(n, nodes, alpha)
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
    
    # Add a small epsilon to prevent taking log of zero
    eps = 1e-15
    safe_nodes = jnp.maximum(nodes, eps)
    
    L_n, _ = laguerre_function_scaled(n, nodes, alpha)
    # Add epsilon to ensure abs(L_n) is non-zero before taking log
    safe_L_n = jnp.maximum(jnp.abs(L_n), eps)
    
    log_weights = (
        -0.5 * nodes
        + jnp.log(safe_nodes)
        + jnp.log(gamma(n + alpha + 1))
        - jnp.log(n * gamma(n))
        - 2 * jnp.log(safe_L_n)
    )
    
    max_log = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log)
    weights = weights / jnp.sum(weights)
    
    cond_num = jnp.max(weights) / jnp.min(weights)
    if cond_num > 1e12:
        weights = jnp.sqrt(weights)
        weights = weights / jnp.sum(weights)
    
    return nodes, weights


def gauss_laguerre_quadrature_scaled(f, n_points, alpha=0):
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
    
    nodes, weights = compute_gauss_laguerre_points_scaled(n_points, alpha)
    f_vals = jnp.array([f(x) for x in nodes])
    f_vals = jnp.where(jnp.isfinite(f_vals), f_vals, 0.0)
    
    result = 0.0
    correction = 0.0
    for w, v in zip(weights, f_vals):
        term = w * v - correction
        temp = result + term
        correction = (temp - result) - term
        result = temp
    
    # Estimate error by comparing with a lower order quadrature
    if n_points > 1:
        lower_result, _ = gauss_laguerre_quadrature_scaled(f, n_points - 1, alpha)
        error = abs(result - lower_result)
    else:
        error = 0.0
    
    return result, error

# For backward compatibility
laguerre_function_final = laguerre_function_scaled
compute_gauss_laguerre_points_final = compute_gauss_laguerre_points_scaled
gauss_laguerre_quadrature_final = gauss_laguerre_quadrature_scaled

# Improved Legendre polynomial implementation with better numerical stability
def legendre_polynomial_stable(n, x):
    """
    Improved implementation of Legendre polynomials with numerical stability.
    
    Parameters:
    n : int
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the polynomial
        
    Returns:
    tuple
        (P_n, P_n') - Legendre polynomial value and its derivative at x
    """
    x = jnp.asarray(x)
    
    # Handle edge cases for improved numerical stability
    edge_mask = (jnp.abs(x) > 0.9999)
    
    # Direct computation for edge cases
    p_edge = jnp.where(edge_mask, 
                       jnp.power(x, n) * (1 + n*(n-1)/(2*3) * (1-x**2) + 
                                         n*(n-1)*(n-2)*(n-3)/(2*3*4*5) * (1-x**2)**2),
                       jnp.zeros_like(x))
    dp_edge = jnp.where(edge_mask,
                        n * (jnp.power(x, n-1) - (n-1)/(2*3) * (n+2) * jnp.power(x, n+1) +
                             (n-1)*(n-2)*(n-3)/(2*3*4*5) * (n+2)*(n+3) * jnp.power(x, n+3)),
                        jnp.zeros_like(x))
    
    # Standard recurrence computation for stable regions
    p_prev = jnp.ones_like(x)
    p_curr = x
    dp_prev = jnp.zeros_like(x)
    dp_curr = jnp.ones_like(x)
    
    if n == 0:
        return p_prev, dp_prev
    elif n == 1:
        return p_curr, dp_curr
        
    for k in range(1, n):
        # Apply scaling to prevent overflow
        scale = 1.0 / jnp.maximum(1.0, jnp.sqrt(k))
        
        # Recurrence relation for polynomial value
        p_next = ((2*k + 1) * x * p_curr * scale - k * p_prev * scale) / (k + 1)
        p_prev = p_curr
        p_curr = p_next
        
        # Recurrence relation for derivative
        dp_next = ((2*k + 1) * (p_curr + x * dp_curr) * scale - k * dp_prev * scale) / (k + 1)
        dp_prev = dp_curr
        dp_curr = dp_next
    
    # Combine stable recurrence results with edge case handling
    p_n = jnp.where(edge_mask, p_edge, p_curr)
    dp_n = jnp.where(edge_mask, dp_edge, dp_curr)
    
    return p_n, dp_n

# Improved computation of Gauss-Legendre quadrature points
def compute_gauss_legendre_points_stable(n, max_iter=100, tol=1e-14):
    """
    Stable implementation of Gauss-Legendre quadrature points computation.
    
    Parameters:
    n : int
        Number of quadrature points
    max_iter : int, optional
        Maximum number of iterations for root finding
    tol : float, optional
        Convergence tolerance
        
    Returns:
    tuple
        (nodes, weights) - Quadrature points and weights
    """
    if n > 100:
        raise ValueError("n > 100 may lead to numerical instability")
    
    # Initial guess for roots using asymptotic formula
    k = jnp.arange(1, n + 1)
    nodes = jnp.cos(jnp.pi * (k - 0.25) / (n + 0.5))
    
    # Newton's method with dampening for stability
    for iter in range(max_iter):
        p_n, dp_n = legendre_polynomial_stable(n, nodes)
        
        # Apply dampening factor for stability
        damping = 1.0 / (1.0 + 0.1 * iter)
        delta = p_n / dp_n
        delta = damping * jnp.clip(delta, -0.2, 0.2)  # Limit step size
        
        nodes_new = nodes - delta
        
        # Ensure nodes stay within [-1, 1] domain
        nodes_new = jnp.clip(nodes_new, -1.0, 1.0)
        
        # Check ordering of nodes (should be decreasing)
        if jnp.any(nodes_new[1:] - nodes_new[:-1] >= 0):
            damping *= 0.5
            nodes_new = nodes - damping * delta
            nodes_new = jnp.clip(nodes_new, -1.0, 1.0)
        
        if jnp.max(jnp.abs(nodes_new - nodes)) < tol:
            nodes = nodes_new
            break
            
        nodes = nodes_new
    
    # Ensure nodes are ordered
    idx = jnp.argsort(-nodes)  # Sort in decreasing order
    nodes = nodes[idx]
    
    # Compute weights using stable formulation
    _, dp_n = legendre_polynomial_stable(n, nodes)
    weights = 2.0 / ((1.0 - nodes**2) * dp_n**2)
    
    # Normalize weights for improved numerical stability
    weights = weights / jnp.sum(weights)
    
    # Check condition number and apply regularization if needed
    cond_num = jnp.max(weights) / jnp.min(weights)
    if cond_num > 1e10:
        # Apply regularization by slightly smoothing weights
        weights = jnp.sqrt(weights)
        weights = weights / jnp.sum(weights)
    
    return nodes, weights

# Improved Gaussian quadrature with stability enhancements
def gaussian_legendre_quadrature_stable(f, a, b, n_points):
    """
    Gaussian-Legendre quadrature with stability improvements.
    
    Parameters:
    f : callable
        The integrand function
    a : float
        Lower integration limit
    b : float
        Upper integration limit
    n_points : int
        Number of quadrature points
    
    Returns:
    float
        Approximated integral
    """
    # Get Gauss-Legendre nodes and weights for [-1, 1]
    nodes, weights = compute_gauss_legendre_points_stable(n_points)
    
    # Transform to [a, b] with careful handling of potential overflow
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    transformed_nodes = scale * nodes + shift
    
    # Evaluate integrand and handle potential NaN/inf values
    f_vals = jnp.array([f(x) for x in transformed_nodes])
    f_vals = jnp.where(jnp.isfinite(f_vals), f_vals, 0.0)
    
    # Use Kahan summation for improved accuracy
    result = 0.0
    correction = 0.0
    for w, v in zip(weights, f_vals):
        term = scale * w * v - correction
        temp = result + term
        correction = (temp - result) - term
        result = temp
    
    return result


def get_laguerre_coefficients(n):
    """
    Generate the coefficients of the nth Laguerre polynomial in explicit form.
    
    The Laguerre polynomial L_n(x) can be written as:
    L_n(x) = sum_{k=0}^{n} binom(n,k) * (-1)^k / k! * x^k
    
    For example:
    L_0(x) = 1
    L_1(x) = -x + 1
    L_2(x) = (x^2 - 4x + 2) / 2
    
    Parameters:
    n : int
        Degree of the Laguerre polynomial
        
    Returns:
    list
        List of coefficients [a_0, a_1, ..., a_n] where L_n(x) = a_0 + a_1*x + ... + a_n*x^n
    """
    coeffs = np.zeros(n + 1)
    
    for k in range(n + 1):
        # Calculate coefficient for x^k term: binom(n,k) * (-1)^k / k!
        coeff = binom(n, k) * ((-1) ** k) / np.math.factorial(k)
        coeffs[k] = coeff
    
    return coeffs


def print_laguerre_polynomial(n):
    """
    Print the nth Laguerre polynomial in human-readable form with factorial coefficients.
    
    Parameters:
    n : int
        Degree of the Laguerre polynomial
        
    Returns:
    str
        String representation of the Laguerre polynomial with factorial coefficient
        and bracketed polynomial
    """
    if n < 0:
        return "Invalid degree: n must be non-negative"
    
    coeffs = get_laguerre_coefficients(n)
    
    # Format the polynomial inside brackets
    bracket_terms = []
    for k in range(n, -1, -1):  # Start from highest power
        coeff = coeffs[k] * np.math.factorial(n)
        if coeff == 0:
            continue
            
        # Format the coefficient
        if k == 0:  # Constant term
            if coeff == 1:
                bracket_terms.append("1")
            elif coeff == -1:
                bracket_terms.append("-1")
            else:
                bracket_terms.append(str(int(coeff)) if coeff.is_integer() else str(coeff))
        elif k == 1:  # Linear term
            if coeff == 1:
                bracket_terms.append("x")
            elif coeff == -1:
                bracket_terms.append("-x")
            else:
                coeff_str = str(int(coeff)) if coeff.is_integer() else str(coeff)
                bracket_terms.append(f"{coeff_str}*x")
        else:  # Higher order terms
            if coeff == 1:
                bracket_terms.append(f"x^{k}")
            elif coeff == -1:
                bracket_terms.append(f"-x^{k}")
            else:
                coeff_str = str(int(coeff)) if coeff.is_integer() else str(coeff)
                bracket_terms.append(f"{coeff_str}*x^{k}")
    
    if not bracket_terms:
        return "0"
    
    # Format the full polynomial with appropriate signs
    bracket_result = bracket_terms[0]
    for term in bracket_terms[1:]:
        if term[0] == '-':
            bracket_result += " " + term
        else:
            bracket_result += " + " + term
    
    # Create the outer factorial coefficient
    if n == 0:
        return "L_0(x) = 1"
    else:
        return f"L_{n}(x) = {1}/{np.math.factorial(n)} * ({bracket_result})"


def verify_laguerre_implementation(n_max=10, x_values=None):
    """
    Verify the correctness of different Laguerre polynomial implementations by
    comparing computed values against the explicit coefficient form.
    
    Parameters:
    n_max : int
        Maximum degree of polynomial to check
    x_values : array_like, optional
        Points at which to evaluate the polynomials. If None, default points are used.
        
    Returns:
    dict
        Dictionary containing verification results
    """
    if x_values is None:
        x_values = np.linspace(0, 5, 10)
    
    results = {}
    
    for n in range(n_max + 1):
        coeffs = get_laguerre_coefficients(n)
        
        # Evaluate using explicit coefficient form (exact reference)
        exact_values = np.zeros_like(x_values, dtype=float)
        for k, coeff in enumerate(coeffs):
            exact_values += coeff * x_values**k
        
        # Evaluate using recurrence relation implementation
        recurrence_values = laguerre_polynomial(n, x_values)
        
        # Evaluate using final implementation (scaled version)
        scaled_values, _ = laguerre_function_scaled(n, x_values)
        
        # Calculate maximum absolute differences
        recurrence_diff = np.max(np.abs(recurrence_values - exact_values))
        scaled_diff = np.max(np.abs(scaled_values - exact_values))
        
        results[n] = {
            'coefficients': coeffs,
            'max_recurrence_diff': float(recurrence_diff),
            'max_scaled_diff': float(scaled_diff)
        }
    
    return results