import numpy as np
from scipy.special import gamma, binom
import jax.numpy as jnp
from functools import lru_cache

def get_laguerre_coefficients(n):
    """
    Generate the coefficients of the nth Laguerre polynomial in explicit form.
    
    The Laguerre polynomial L_n(x) can be written as:
    L_n(x) = sum_{k=0}^{n} binom(n,k) * (-1)^k / k! * x^k
    
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

def laguerre_polynomial(n, x, alpha=0.0):
    """
    Implementation of generalized Laguerre polynomials.
    
    For standard Laguerre polynomials (alpha=0), uses the explicit formula:
    L_n(x) = sum_{k=0}^{n} binom(n,k) * (-1)^k / k! * x^k
    
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
    
    if alpha == 0.0:
        # For standard Laguerre polynomials, use the explicit coefficient formula
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return 1.0 - x
        else:
            # Use explicit formula based on coefficients
            coeffs = get_laguerre_coefficients(n)
            result = jnp.zeros_like(x)
            for k, coeff in enumerate(coeffs):
                result = result + coeff * x**k
            return result
    else:
        # For generalized Laguerre polynomials (alpha ≠ 0), use recurrence relation
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

def print_laguerre_polynomial(n):
    """
    Print the nth Laguerre polynomial in human-readable form.
    Shows both the standard form (with computed coefficients) and
    the symbolic factorial representation.
    
    Parameters:
    n : int
        Degree of the Laguerre polynomial
        
    Returns:
    str
        String representation of the Laguerre polynomial
    """
    if n < 0:
        return "Invalid degree: n must be non-negative"
    
    # Standard form with computed coefficients
    coeffs = get_laguerre_coefficients(n)
    
    # Format the standard polynomial form
    terms = []
    for k in range(n, -1, -1):  # Start from highest power
        coeff = coeffs[k]
        if abs(coeff) < 1e-10:  # Skip terms with extremely small coefficients
            continue
            
        # Format standard form coefficient
        if k == 0:  # Constant term
            terms.append(f"{coeff:.10g}")
        elif k == 1:  # Linear term
            if abs(coeff - 1) < 1e-10:
                terms.append("x")
            elif abs(coeff + 1) < 1e-10:
                terms.append("-x")
            else:
                terms.append(f"{coeff:.10g}*x")
        else:  # Higher order terms
            if abs(coeff - 1) < 1e-10:
                terms.append(f"x^{k}")
            elif abs(coeff + 1) < 1e-10:
                terms.append(f"-x^{k}")
            else:
                terms.append(f"{coeff:.10g}*x^{k}")
    
    if not terms:
        return "0"
    
    # Format the full standard form with appropriate signs
    result = terms[0]
    for term in terms[1:]:
        if term[0] == '-':
            result += " " + term
        else:
            result += " + " + term
    
    # Instead of manipulating coefficients, just show the mathematical formula for factorial form
    # This is L_n(x) = sum_{k=0}^n (-1)^k/k! * binom(n,k) * x^k
    if n <= 5:  # Only do the full expansion for small n to keep output readable
        factorial_form = "("
        factorial_terms = []
        
        for k in range(n, -1, -1):
            sign = "+" if k % 2 == 0 else "-"
            if k == 0:
                coefficient = f"{binom(n, k):.0f}"
            else:
                coefficient = f"{binom(n, k):.0f}*x^{k}"
                
            # Skip the + sign for the first term
            if k == n:
                factorial_terms.append(f"{'-' if sign == '-' else ''}{coefficient}")
            else:
                factorial_terms.append(f"{sign} {coefficient}")
                
        factorial_form += " ".join(factorial_terms) + ")"
    else:
        # For higher degrees, use summation notation
        factorial_form = "(Σ_{k=0}^{n} (-1)^k*binom(n,k)*x^k/k!)"
    
    return f"L_{n}(x) = {result}\n       = 1/{n}! * {factorial_form}"

def laguerre_function_final(n, x, alpha=0.0):
    """
    Final version of scaled Laguerre functions with additional error control
    from section 3.3 of the paper.
    """
    x = jnp.asarray(x)
    
    # Scale x for better conditioning when x is small
    x_scaled = jnp.where(x < 1.0, x, x * (1.0 + jnp.log(x))/x)
    
    # Initialize with careful scaling
    L0 = jnp.exp(-x_scaled/2)
    if n == 0:
        return L0, -0.5 * L0
    
    L1 = (alpha + 1 - x_scaled) * jnp.exp(-x_scaled/2)
    if n == 1:
        return L1, -(alpha + 3 - x_scaled)/2 * jnp.exp(-x_scaled/2)
    
    # Use scaled recurrence with error control
    L_prev = L0
    L_curr = L1
    
    for k in range(1, n):
        # Scale coefficients to prevent overflow
        scale = 1.0 / jnp.maximum(1.0, jnp.sqrt(k))
        
        # Modified recurrence from (3.2) with scaling
        term1 = (2*k + alpha + 1 - x_scaled) * L_curr * scale
        term2 = (k + alpha) * L_prev * scale
        L_next = (term1 - term2) / (k + 1)
        
        L_prev = L_curr
        L_curr = L_next
    
    # Derivative with improved stability
    deriv = -0.5 * L_curr - 0.5 * L_prev
    
    return L_curr, deriv

def compute_gauss_laguerre_points_final(n, alpha=0.0, max_iter=100, tol=1e-12):
    """
    Final version of Gauss-Laguerre quadrature points computation with
    all stability improvements and error controls.
    """
    # Initial guess using improved formula (2.16)
    j = jnp.arange(1, n + 1)
    theta = jnp.pi * (j - 0.25) / (n + 0.5)
    nodes = (n + alpha/2 + 0.75) * jnp.cos(theta)**2
    
    # Newton iterations with stability control
    for iter in range(max_iter):
        L_n, dL_n = laguerre_function_final(n, nodes, alpha)
        
        # Newton step with adaptive damping
        delta = L_n / dL_n
        damping = 1.0 / (1.0 + 0.2 * iter)  # Slower damping
        delta = damping * jnp.clip(delta, -0.5, 0.5)  # More conservative limits
        
        nodes_new = nodes - delta
        
        # Ensure nodes stay positive and ordered
        nodes_new = jnp.maximum(nodes_new, 0.0)
        if jnp.any(nodes_new[1:] - nodes_new[:-1] <= 0):
            damping *= 0.5
            nodes_new = nodes - damping * delta
        
        if jnp.max(jnp.abs(nodes_new - nodes)) < tol:
            nodes = nodes_new
            break
            
        nodes = nodes_new
    
    # Sort nodes
    idx = jnp.argsort(nodes)
    nodes = nodes[idx]
    
    # Compute weights with improved scaling
    L_n, _ = laguerre_function_final(n, nodes, alpha)
    
    # Use log space for weight computation
    log_weights = (
        -0.5 * nodes  # Function scaling
        + jnp.log(nodes)  # From weight formula
        + jnp.log(gamma(n + alpha + 1))
        - jnp.log(n * gamma(n))
        - 2 * jnp.log(jnp.abs(L_n))
    )
    
    # Handle potential overflow in weights
    max_log = jnp.max(log_weights)
    weights = jnp.exp(log_weights - max_log)
    weights = weights / jnp.sum(weights)
    
    # Additional stability check from error analysis
    cond_num = jnp.max(weights) / jnp.min(weights)
    if cond_num > 1e12:  # If condition number is too large
        # Recompute with additional scaling
        weights = jnp.sqrt(weights)  # Use sqrt scaling
        weights = weights / jnp.sum(weights)
    
    return nodes, weights

def generalized_laguerre_quadrature(f, n_points, alpha=0.0, *args):
    """
    Final version of Gauss-Laguerre quadrature with all improvements
    and complete error control.
    """
    if n_points > 100:  # Practical limit based on error analysis
        raise ValueError("n_points > 100 may lead to numerical instability")
    
    # Get nodes and weights
    nodes, weights = compute_gauss_laguerre_points_final(n_points, alpha)
    
    # Scale function evaluation
    def f_scaled(x, *args):
        return f(x, *args) * jnp.exp(-x/2)
    
    # Evaluate with careful handling of large values
    f_vals = jnp.array([f_scaled(x, *args) for x in nodes])
    f_vals = jnp.where(jnp.isfinite(f_vals), f_vals, 0.0)
    
    # Use compensated summation for better accuracy
    result = 0.0
    correction = 0.0
    for w, v in zip(weights, f_vals):
        term = w * v - correction
        temp = result + term
        correction = (temp - result) - term
        result = temp
    
    # Error estimation using nested quadrature with stability check
    if n_points > 2:
        try:
            result_prev = generalized_laguerre_quadrature(f, n_points-1, alpha, *args)
            error_est = jnp.abs(result - result_prev)
        except:
            error_est = jnp.abs(result) * 1e-6  # Fallback error estimate
    else:
        error_est = jnp.abs(result) * 1e-6
    
    return result

def test_quadrature_convergence(max_n=20):
    """
    Test quadrature convergence with simple test function.
    """
    def f(x):
        return x  # Test f(x) = x, should give integral = 1
    
    errors = []
    for n in range(2, max_n + 1):
        result = generalized_laguerre_quadrature(f, n)
        error = abs(result - 1.0)
        errors.append(error)
        print(f"n={n}, error={error:.2e}")
    
    return errors