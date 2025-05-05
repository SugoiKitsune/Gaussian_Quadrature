"""
Core quadrature implementations with numerical stability improvements
"""

import jax.numpy as jnp
from jax.scipy.special import gamma
import numpy as np
from functools import lru_cache
from scipy.special import binom


# Simple Laguerre polynomial implementation
def laguerre_polynomial(n, x, alpha=0.0):
    """
    Basic implementation of generalized Laguerre polynomials without stability improvements.
    
    Parameters:
    n : int
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the polynomial
    alpha : float, optionalcompute_gauss_laguerre_points_basic(
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

# Get Laguerre polynomial coefficients
def get_laguerre_coefficients(n, alpha=0.0):
    """
    Compute coefficients of Laguerre polynomial.
    
    Parameters:
    n : int
        Degree of the polynomial
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
        
    Returns:
    array
        Coefficients of the Laguerre polynomial, lowest power first
    """
    coeffs = np.zeros(n+1)
    for k in range(n+1):
        # L_n^(α)(x) = Σ(k=0 to n) (-1)^k * binom(n+α,n-k) * x^k / k!
        binom_val = gamma(n+alpha+1) / (gamma(k+alpha+1) * gamma(n-k+1))
        coeffs[k] = (-1)**k * binom_val / gamma(k+1)
    
    return coeffs

# Improved computation of Gauss-Laguerre quadrature points with enhanced initial guesses
def compute_gauss_laguerre_points_basic(n, alpha=0.0, max_iter=100, tol=1e-12):
    """
    Improved implementation of Gauss-Laguerre quadrature points computation.
    
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
    if n <= 0:
        return jnp.array([]), jnp.array([])
    
    # Enhanced initial guess for roots using more accurate asymptotic formulas
    if n == 1:
        # For n=1, the root is exactly n+alpha
        nodes = jnp.array([1.0 + alpha])
    else:
        # For n>1, use improved initial guesses based on asymptotic formulas
        j = jnp.arange(1, n + 1)
        
        # Use more accurate Stroud-Secrest approach for initial guesses
        aj = (2 * j - 1) * jnp.pi / (4 * n + 2)
        r1 = (n + (alpha + 1) / 2) + (n + (alpha + 1) / 2 - 0.25)**0.5
        r2 = (n + (alpha + 1) / 2) - (n + (alpha + 1) / 2 - 0.25)**0.5
        
        nodes = (r1 + r2) / 2 + (r1 - r2) / 2 * jnp.cos(2 * aj)
        
        # Sort initial guesses in ascending order (important for root separation)
        nodes = jnp.sort(nodes)
    
    # Ensure initial nodes are distinct with proper separation
    min_sep = 1e-6 * n  # Scale separation based on degree
    
    # Apply separation for initial guesses
    for i in range(1, n):
        if nodes[i] - nodes[i-1] < min_sep:
            nodes = nodes.at[i].set(nodes[i-1] + min_sep)
    
    # Newton's method with proper damping and handling of each root individually
    for iter_count in range(max_iter):
        converged = True
        
        # Process each root individually to avoid interference
        for i in range(n):
            x = nodes[i]
            p_n = laguerre_polynomial(n, x, alpha)
            dp_n = laguerre_polynomial_derivative(n, x, alpha)
            
            # Skip this root if derivative is too small (avoid division by near-zero)
            if jnp.abs(dp_n) < 1e-14:
                continue
                
            # Newton step with adaptive damping
            delta = p_n / dp_n
            
            # Apply stronger damping for larger corrections
            damping = 1.0 / (1.0 + 0.1 * iter_count + 0.1 * jnp.abs(delta))
            delta = damping * delta
            
            # Apply the correction
            x_new = x - delta
            
            # Keep roots positive (Laguerre roots are always positive)
            x_new = jnp.maximum(x_new, 0.0)
            
            # Update the node
            nodes = nodes.at[i].set(x_new)
            
            # Check convergence for this root
            if jnp.abs(delta) > tol:
                converged = False
        
        # Apply separation constraints after each iteration
        for i in range(1, n):
            if nodes[i] - nodes[i-1] < min_sep:
                # If roots get too close, push them apart
                mid = (nodes[i] + nodes[i-1]) / 2
                nodes = nodes.at[i-1].set(mid - min_sep/2)
                nodes = nodes.at[i].set(mid + min_sep/2)
                converged = False
        
        # Resort nodes after each iteration to maintain order
        nodes = jnp.sort(nodes)
        
        if converged:
            break
    
    # Compute weights using standard formula
    p_n1 = laguerre_polynomial(n+1, nodes, alpha)
    weights = gamma(n + alpha + 1) / (gamma(n + 1) * n * p_n1**2) * nodes**(alpha) * jnp.exp(-nodes)
    
    return nodes, weights

# Simple Gaussian-Laguerre quadrature
def gauss_laguerre_quadrature_basic(f, n_points, alpha=0.0):
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
        delta = jnp.clip(delta, -0.5, 0.5)
        
        nodes_new = nodes - delta
        nodes_new = jnp.maximum(nodes_new, 0.0)
        
        if jnp.any(nodes_new[1:] - nodes_new[:-1] <= 0):
            damping *= 0.5
            nodes_new = nodes - damping * delta
            nodes_new = jnp.maximum(nodes_new, 0.0)
            
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

def generalized_laguerre_quadrature(f, n_points, alpha=0.0, scale=1.0):
    """
    Performs generalized Gaussian-Laguerre quadrature for integrals of the form:
    
    ∫ x^alpha * exp(-x) * f(x) dx from 0 to infinity
    
    with optional scaling parameter for the integration variable.
    
    Parameters:
    f : callable
        The integrand function (excluding the weight function)
    n_points : int
        Number of quadrature points
    alpha : float, optional
        Parameter for generalized Laguerre polynomial (weight function x^alpha)
    scale : float, optional
        Scaling factor for the integration variable (x → x/scale)
        
    Returns:
    float
        Approximated integral value
    float
        Error estimate
    """
    if n_points > 100:
        raise ValueError("n_points > 100 may lead to numerical instability")
    
    # Get Gauss-Laguerre nodes and weights
    nodes, weights = compute_gauss_laguerre_points_final(n_points, alpha)
    
    # Apply scaling to nodes
    if scale != 1.0:
        scaled_nodes = nodes * scale
        scaled_weights = weights * (scale ** (alpha + 1))
    else:
        scaled_nodes = nodes
        scaled_weights = weights
    
    # Evaluate integrand and handle potential NaN/inf values
    f_vals = jnp.array([f(x) for x in scaled_nodes])
    f_vals = jnp.where(jnp.isfinite(f_vals), f_vals, 0.0)
    
    # Use Kahan summation for improved accuracy
    result = 0.0
    correction = 0.0
    for w, v in zip(scaled_weights, f_vals):
        term = w * v - correction
        temp = result + term
        correction = (temp - result) - term
        result = temp
    
    # Estimate error by comparing with a lower order quadrature
    if n_points > 1:
        lower_result, _ = generalized_laguerre_quadrature(f, n_points - 1, alpha, scale)
        error = abs(result - lower_result)
    else:
        error = 0.0
    
    return result, error

def laguerre_root_finding_method(poly_func, poly_deriv, poly_deriv2, initial_guess, n, max_iter=100, tol=1e-12):
    """
    Implements Laguerre's method for finding roots of polynomials.
    This method has cubic convergence and is more robust than Newton's method.
    
    Parameters:
    poly_func : callable
        Function that evaluates the polynomial at a point
    poly_deriv : callable
        Function that evaluates the first derivative of the polynomial
    poly_deriv2 : callable
        Function that evaluates the second derivative of the polynomial
    initial_guess : float or array_like
        Initial guess for the root
    n : int
        Degree of the polynomial
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance
        
    Returns:
    float or array_like
        The approximated root
    """
    x = jnp.asarray(initial_guess)
    
    for i in range(max_iter):
        p = poly_func(x)
        
        # Check for convergence
        if jnp.abs(p) < tol:
            return x
            
        # Compute first and second derivatives
        dp = poly_deriv(x)
        d2p = poly_deriv2(x)
        
        # Compute G and H for Laguerre's method
        G = dp / p
        H = G**2 - d2p / p
        
        # Determine the denominator with appropriate sign to maximize magnitude
        discriminant = jnp.sqrt(jnp.abs((n - 1) * ((n * H) - G**2)))
        denom1 = G + discriminant
        denom2 = G - discriminant
        
        # Choose denominator with larger magnitude to improve stability
        denom = jnp.where(jnp.abs(denom1) > jnp.abs(denom2), denom1, denom2)
        
        # Prevent division by zero
        denom = jnp.where(jnp.abs(denom) < 1e-10, 1e-10 * jnp.sign(denom), denom)
        
        # Apply Laguerre's formula
        delta = n / denom
        
        # Apply damping for better stability in early iterations
        damping = 1.0 / (1.0 + 0.1 * i)
        delta = damping * jnp.clip(delta, -1.0, 1.0)
        
        x_new = x - delta
        
        # Check for convergence
        if jnp.abs(x_new - x) < tol:
            return x_new
            
        x = x_new
        
    return x  # Return best estimate even if not fully converged

# Function to compute second derivative of Laguerre polynomial
def laguerre_polynomial_second_derivative(n, x, alpha=0.0):
    """
    Calculates the second derivative of generalized Laguerre polynomials.
    
    Parameters:
    n : int
        Degree of the polynomial
    x : array_like
        Points at which to evaluate the second derivative
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
        
    Returns:
    array_like
        Second derivative of Laguerre polynomial at x
    """
    if n <= 1:
        return jnp.zeros_like(x)
    else:
        return laguerre_polynomial(n-2, x, alpha+2)

# Enhanced computation of Gauss-Laguerre quadrature points using Laguerre's method
def compute_gauss_laguerre_points_laguerre_method(n, alpha=0.0, max_iter=100, tol=1e-12):
    """
    Enhanced implementation of Gauss-Laguerre quadrature points computation 
    using Laguerre's root-finding method for improved stability and convergence.
    
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
    # Initial guess for roots (using better asymptotic formula for Laguerre zeros)
    j = jnp.arange(1, n + 1)
    theta = jnp.pi * (j - 0.25) / (n + 0.5)
    nodes = (n + alpha/2 + 0.75) * jnp.cos(theta)**2
    
    # Ensure initial nodes are distinct with minimum separation
    min_sep = 1e-6
    for i in range(1, n):
        if nodes[i] - nodes[i-1] < min_sep:
            nodes = nodes.at[i].set(nodes[i-1] + min_sep)
    
    # Define wrappers to make function signatures compatible with Laguerre's method
    def poly_func(x):
        return laguerre_polynomial(n, x, alpha)
        
    def poly_deriv(x):
        return laguerre_polynomial_derivative(n, x, alpha)
        
    def poly_deriv2(x):
        return laguerre_polynomial_second_derivative(n, x, alpha)
    
    # Find roots using Laguerre's method
    refined_nodes = jnp.array([
        laguerre_root_finding_method(poly_func, poly_deriv, poly_deriv2, node, n, max_iter, tol)
        for node in nodes
    ])
    
    # Sort nodes in ascending order
    idx = jnp.argsort(refined_nodes)
    refined_nodes = refined_nodes[idx]
    
    # Compute weights using standard formula
    p_n1 = laguerre_polynomial(n+1, refined_nodes, alpha)
    weights = gamma(n + alpha + 1) / (gamma(n + 1) * n * p_n1**2) * refined_nodes**(alpha+1) * jnp.exp(-refined_nodes)
    
    return refined_nodes, weights

# Gaussian-Laguerre quadrature using Laguerre's root finding method
def gauss_laguerre_quadrature_laguerre_method(f, n_points, alpha=0.0):
    """
    Gaussian-Laguerre quadrature using Laguerre's method for root finding.
    
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
    # Get Gauss-Laguerre nodes and weights using Laguerre's method
    nodes, weights = compute_gauss_laguerre_points_laguerre_method(n_points, alpha)
    
    # Evaluate integrand and compute quadrature
    f_vals = jnp.array([f(x) for x in nodes])
    return jnp.sum(weights * f_vals)

# Eigenvalue-based method for Laguerre roots
def compute_gauss_laguerre_points_eigenvalue(n, alpha=0.0):
    """
    Computes Gauss-Laguerre quadrature points using eigenvalue method.
    
    The eigenvalue method is more accurate than iterative root-finding methods
    and guarantees distinct roots without duplication. This method constructs
    the Jacobi matrix for Laguerre polynomials and finds its eigenvalues,
    which correspond to the quadrature points.
    
    Parameters:
    n : int
        Number of quadrature points
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
        
    Returns:
    tuple
        (nodes, weights) - Quadrature points and weights
    """
    if n <= 0:
        return jnp.array([]), jnp.array([])
    
    # Construct the Jacobi matrix for Laguerre polynomials
    # For Laguerre polynomials, the recurrence coefficients are:
    # a_i = 2*i + alpha + 1
    # b_i = i*(i+alpha)
    #
    # The Jacobi matrix J is tridiagonal with diagonal entries a_i and off-diagonal entries sqrt(b_i)
    
    # Diagonal entries
    diag = jnp.array([2*i + alpha + 1 for i in range(n)])
    
    # Off-diagonal entries (upper and lower)
    off_diag = jnp.array([jnp.sqrt(i*(i+alpha)) for i in range(1, n)])
    
    # Construct the Jacobi matrix as a numpy array first
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2*i + alpha + 1
    
    for i in range(n-1):
        J[i, i+1] = np.sqrt((i+1)*(i+1+alpha))
        J[i+1, i] = np.sqrt((i+1)*(i+1+alpha))
    
    # Find eigenvalues and eigenvectors of the Jacobi matrix
    # The eigenvalues are the nodes of the quadrature rule
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    
    # Convert to JAX arrays and sort by eigenvalues
    nodes = jnp.array(eigenvalues)
    idx = jnp.argsort(nodes)
    nodes = nodes[idx]
    
    # The weights are given by the square of the first component of each eigenvector
    # multiplied by the zeroth moment (which is gamma(alpha+1) for Laguerre)
    eigenvectors = jnp.array(eigenvectors)
    weights = gamma(alpha+1) * eigenvectors[0, :]**2
    weights = weights[idx]
    
    return nodes, weights

def gauss_laguerre_quadrature_eigenvalue(f, n_points, alpha=0.0):
    """
    Gaussian-Laguerre quadrature using eigenvalue method for root finding.
    
    Parameters:
    f : callable
        The integrand function
    n_points : int
        Number of quadrature points
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
    
    Returns:
    tuple
        (result, error) - Approximated integral and error estimate
    """
    # Get Gauss-Laguerre nodes and weights using eigenvalue method
    nodes, weights = compute_gauss_laguerre_points_eigenvalue(n_points, alpha)
    
    # Evaluate integrand and compute quadrature
    f_vals = jnp.array([f(x) for x in nodes])
    result = jnp.sum(weights * f_vals)
    
    # Estimate error by comparing with a lower order quadrature
    if n_points > 1:
        lower_result, _ = gauss_laguerre_quadrature_eigenvalue(f, n_points - 1, alpha)
        error = abs(result - lower_result)
    else:
        error = 0.0
        
    return result, error


def compute_gauss_laguerre_points_eigen(n, alpha=0.0):
    """
    Compute Gauss-Laguerre quadrature points using the eigenvalue method.
    
    This is the most accurate and stable method for finding the roots of Laguerre
    polynomials, based on the fact that the roots are eigenvalues of a tridiagonal
    symmetric matrix constructed from the recurrence coefficients.
    
    Parameters:
    n : int
        Number of quadrature points
    alpha : float, optional
        Parameter for generalized Laguerre polynomials
        
    Returns:
    tuple
        (nodes, weights) - Quadrature points and weights
    """
    if n <= 0:
        return jnp.array([]), jnp.array([])
    
    # Compute the recurrence coefficients for Laguerre polynomials
    # For Laguerre polynomials, the recurrence coefficients are:
    # a_i = 2*i + 1 + alpha
    # b_i = sqrt(i * (i + alpha))
    
    # Create the diagonal elements
    diag = jnp.array([2*i + 1 + alpha for i in range(n)])
    
    # Create the off-diagonal elements
    off_diag = jnp.array([jnp.sqrt(i * (i + alpha)) for i in range(1, n)])
    
    # Convert to numpy for eigenvalue computation
    diag_np = np.array(diag)
    off_diag_np = np.array(off_diag)
    
    # Find eigenvalues (these are the roots)
    nodes = np.sort(scipy.linalg.eigvalsh_tridiagonal(diag_np, off_diag_np))
    
    # Convert back to jax array
    nodes = jnp.array(nodes)
    
    # Compute weights using standard formula
    # For Laguerre polynomials, the weights are given by:
    # w_i = gamma(n + alpha + 1) / (gamma(n + 1) * L_{n+1}(x_i)^2) * x_i^alpha * exp(-x_i)
    
    # We need L_{n+1}(x_i), we can use the recurrence relation:
    # L_{n+1}(x) = ((2n+1+alpha-x)L_n(x) - (n+alpha)L_{n-1}(x)) / (n+1)
    
    # For the eigenvalue method, we can compute the weights directly using:
    # w_i = v_i[0]^2 where v_i is the normalized eigenvector corresponding to eigenvalue x_i
    
    # Compute the first component of each eigenvector
    # For Laguerre polynomials, this is:
    # v_i[0]^2 = gamma(n + alpha + 1) / (n! * L_{n+1}(x_i)^2)
    
    # This is a more stable way to compute weights:
    weights = jnp.zeros_like(nodes)
    for i in range(n):
        x = nodes[i]
        # Compute L_{n+1}(x_i) using recurrence
        if n == 1:
            L_np1 = ((1 + alpha + 1 - x) - (1 + alpha)) / 2
        else:
            L_nm1 = 1.0
            L_n = 1 + alpha - x
            for k in range(1, n):
                L_np1 = ((2*k + 1 + alpha - x) * L_n - (k + alpha) * L_nm1) / (k + 1)
                L_nm1 = L_n
                L_n = L_np1
            
            L_np1 = ((2*n + 1 + alpha - x) * L_n - (n + alpha) * L_nm1) / (n + 1)
        
        # Compute weight
        weights = weights.at[i].set(gamma(n + alpha + 1) / (gamma(n + 1) * n * L_np1**2) * x**(alpha) * jnp.exp(-x))
    
    return nodes, weights

def gauss_laguerre_quadrature_eigen(f, n_points, alpha=0):
    """
    Performs Gaussian-Laguerre quadrature using the eigenvalue method for root finding.
    
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
    
    nodes, weights = compute_gauss_laguerre_points_eigen(n_points, alpha)
    f_vals = jnp.array([f(x) for x in nodes])
    f_vals = jnp.where(jnp.isfinite(f_vals), f_vals, 0.0)
    
    result = jnp.sum(weights * f_vals)
    
    # Estimate error by comparing with a lower order quadrature
    if n_points > 1:
        lower_nodes, lower_weights = compute_gauss_laguerre_points_eigen(n_points - 1, alpha)
        lower_f_vals = jnp.array([f(x) for x in lower_nodes])
        lower_f_vals = jnp.where(jnp.isfinite(lower_f_vals), lower_f_vals, 0.0)
        lower_result = jnp.sum(lower_weights * lower_f_vals)
        error = abs(result - lower_result)
    else:
        error = 0.0
        
    return result, error

# Selected methods
laguerre_function_final = laguerre_function_scaled
compute_gauss_laguerre_points_final = compute_gauss_laguerre_points_eigenvalue
gauss_laguerre_quadrature_final = gauss_laguerre_quadrature_eigenvalue
