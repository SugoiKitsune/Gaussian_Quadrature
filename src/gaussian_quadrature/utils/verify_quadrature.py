#!/usr/bin/env python
"""
Verification script for Gaussian quadrature methods.
Compares various quadrature implementations against:
1. SciPy integration methods
2. Taylor series approximations
3. Analytical solutions (when available)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from functools import partial
import jax.numpy as jnp
import time
import pandas as pd
from scipy.special import gamma
import scipy.special as special
from scipy.misc import derivative

from src.gaussian_quadrature.utils.quadrature import (
    gauss_laguerre_quadrature_basic,
    gauss_laguerre_quadrature_scaled,
    compute_gauss_laguerre_points_basic,
    compute_gauss_laguerre_points_scaled
)

# Set of test functions with known analytical solutions for Laguerre-weighted integrals
# All functions should represent integrals of the form: \int_0^\infty f(x) * x^alpha * e^(-x) dx

class TestFunction:
    def __init__(self, name, func, analytical_solution, alpha=0.0, description=""):
        self.name = name
        self.func = func
        self.analytical_solution = analytical_solution
        self.alpha = alpha
        self.description = description

# Define test functions
def simple_polynomial(x):
    return x**2

def rational_function(x):
    return 1 / (1 + x**2)

def oscillatory_function(x):
    return np.cos(x)

def composite_function(x):
    return np.exp(-x/2) * np.sin(x)

# Define test functions with analytical solutions
test_functions = [
    TestFunction(
        name="Polynomial",
        func=simple_polynomial,
        analytical_solution=2.0,
        alpha=0.0,
        description="f(x) = x^2, weighted integral = 2.0"
    ),
    TestFunction(
        name="Rational",
        func=rational_function,
        analytical_solution=np.exp(1) * special.expn(1, 1),
        alpha=0.0,
        description="f(x) = 1/(1+x^2), integral requires special functions"
    ),
    TestFunction(
        name="Oscillatory",
        func=oscillatory_function,
        analytical_solution=0.5,
        alpha=0.0,
        description="f(x) = cos(x), weighted integral = 0.5"
    ),
    TestFunction(
        name="Composite",
        func=composite_function,
        analytical_solution=2/5,
        alpha=0.0,
        description="f(x) = e^(-x/2) * sin(x), weighted integral = 2/5"
    )
]

def taylor_series_approximation(f, x0, n_terms, x):
    """
    Compute the Taylor series approximation of a function.
    
    Parameters:
    f : callable
        Function to approximate
    x0 : float
        Expansion point
    n_terms : int
        Number of terms in the Taylor series
    x : array_like
        Points to evaluate the approximation at
        
    Returns:
    array_like
        Taylor series approximation values
    """
    result = np.zeros_like(x, dtype=float)
    for k in range(n_terms):
        # Compute kth derivative at x0
        if k == 0:
            dk = f(x0)
        else:
            dk = derivative(f, x0, n=k, order=k+2)
        
        # Add term to Taylor series
        term = dk / np.math.factorial(k) * (x - x0)**k
        result += term
    
    return result

def integrate_taylor_approximation(f, a, b, n_terms, alpha=0.0):
    """
    Integrate a function using Taylor series approximation for Laguerre-weighted integral.
    
    Parameters:
    f : callable
        Function to integrate
    a : float
        Lower integration bound
    b : float
        Upper integration bound (use large value to approximate infinity)
    n_terms : int
        Number of terms in Taylor series
    alpha : float
        Parameter for Laguerre weight function x^alpha * e^(-x)
        
    Returns:
    float
        Approximated integral value
    """
    # Taylor expansion point (typically 0 for Laguerre-type integrals)
    x0 = 0.0
    
    # Create Taylor approximation of the function
    taylor_approx = lambda x: taylor_series_approximation(f, x0, n_terms, x)
    
    # For each term in the Taylor series, we get integral of x^(k+alpha) * e^(-x)
    # which equals gamma(k+alpha+1)
    result = 0.0
    for k in range(n_terms):
        if k == 0:
            dk = f(x0)
        else:
            dk = derivative(f, x0, n=k, order=k+2)
        
        # Add contribution from this term
        term = dk / np.math.factorial(k) * gamma(k + alpha + 1)
        result += term
    
    return result

def compare_quadrature_methods(test_functions, ns_points=[5, 10, 15, 20, 30, 50], methods=None):
    """
    Compare different quadrature methods for a set of test functions.
    
    Parameters:
    test_functions : list
        List of TestFunction objects
    ns_points : list
        List of numbers of quadrature points to try
    methods : list, optional
        List of method names to compare
        
    Returns:
    pandas.DataFrame
        Results of the comparison
    """
    if methods is None:
        methods = ["gauss_laguerre_basic", "gauss_laguerre_scaled", "scipy_quad", "taylor_series"]
    
    results = []
    
    for test_func in test_functions:
        # Get analytical solution
        exact = test_func.analytical_solution
        
        for n in ns_points:
            for method in methods:
                start_time = time.time()
                
                if method == "gauss_laguerre_basic":
                    approx = gauss_laguerre_quadrature_basic(test_func.func, n, test_func.alpha)
                    
                elif method == "gauss_laguerre_scaled":
                    approx, _ = gauss_laguerre_quadrature_scaled(test_func.func, n, test_func.alpha)
                    
                elif method == "scipy_quad":
                    # Use scipy's quad with weight function explicitly
                    def weighted_func(x):
                        return test_func.func(x) * x**test_func.alpha * np.exp(-x)
                    
                    # Limit upper bound to a large value for infinite integration
                    approx, _ = integrate.quad(weighted_func, 0, 100)
                    
                elif method == "taylor_series":
                    # Use Taylor series approximation with n terms
                    approx = integrate_taylor_approximation(
                        test_func.func, 0, 100, n, test_func.alpha
                    )
                
                elapsed_time = time.time() - start_time
                abs_error = abs(approx - exact)
                rel_error = abs_error / abs(exact) if exact != 0 else abs_error
                
                results.append({
                    "Function": test_func.name,
                    "Method": method,
                    "Points/Terms": n,
                    "Approx. Value": approx,
                    "Exact Value": exact,
                    "Abs. Error": abs_error,
                    "Rel. Error": rel_error,
                    "Time (s)": elapsed_time
                })
    
    return pd.DataFrame(results)

def plot_convergence(results, function_name=None, save_path=None):
    """
    Plot convergence of different methods for a specific function or all functions.
    
    Parameters:
    results : pandas.DataFrame
        Results from compare_quadrature_methods
    function_name : str, optional
        Name of function to plot results for, or None for all functions
    save_path : str, optional
        Path to save the figure to
    """
    plt.figure(figsize=(10, 6))
    
    if function_name:
        df = results[results["Function"] == function_name]
        title = f"Convergence for {function_name} Function"
    else:
        df = results
        title = "Convergence for All Functions"
    
    methods = df["Method"].unique()
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, method in enumerate(methods):
        method_df = df[df["Method"] == method]
        
        for func_name in method_df["Function"].unique():
            func_df = method_df[method_df["Function"] == func_name]
            
            # Sort by number of points/terms
            func_df = func_df.sort_values("Points/Terms")
            
            plt.semilogy(
                func_df["Points/Terms"],
                func_df["Rel. Error"],
                marker=markers[i % len(markers)],
                linestyle='-' if function_name else ':',
                label=f"{method}" if function_name else f"{func_name} - {method}"
            )
    
    plt.grid(True, which="both", ls="--")
    plt.xlabel("Number of Points/Terms")
    plt.ylabel("Relative Error (log scale)")
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_performance(results, save_path=None):
    """
    Plot performance comparison of different methods.
    
    Parameters:
    results : pandas.DataFrame
        Results from compare_quadrature_methods
    save_path : str, optional
        Path to save the figure to
    """
    # Group by Method and get mean time and error
    performance = results.groupby(["Method", "Points/Terms"]).agg({
        "Time (s)": "mean",
        "Rel. Error": "mean"
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    methods = performance["Method"].unique()
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, method in enumerate(methods):
        method_df = performance[performance["Method"] == method]
        method_df = method_df.sort_values("Points/Terms")
        
        plt.semilogy(
            method_df["Points/Terms"],
            method_df["Time (s)"], 
            marker=markers[i % len(markers)],
            linestyle='-',
            label=f"{method}"
        )
    
    plt.grid(True, which="both", ls="--")
    plt.xlabel("Number of Points/Terms")
    plt.ylabel("Computation Time (s) (log scale)")
    plt.title("Performance Comparison")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def create_summary_table(results):
    """
    Create a summary table of results.
    
    Parameters:
    results : pandas.DataFrame
        Results from compare_quadrature_methods
        
    Returns:
    pandas.DataFrame
        Summary table
    """
    # Group by Function, Method and calculate summary statistics
    summary = results.groupby(["Function", "Method"]).agg({
        "Rel. Error": ["min", "max", "mean"],
        "Time (s)": "mean",
        "Points/Terms": "max"
    }).reset_index()
    
    # Format the table
    summary.columns = [
        'Function', 'Method', 
        'Min Error', 'Max Error', 'Mean Error',
        'Avg Time (s)', 'Max Points/Terms'
    ]
    
    return summary

def main():
    """Main function to run the quadrature verification."""
    print("Running quadrature verification...")
    
    # Compare methods
    ns_points = [5, 10, 15, 20, 30, 50]
    methods = ["gauss_laguerre_basic", "gauss_laguerre_scaled", "scipy_quad", "taylor_series"]
    
    results = compare_quadrature_methods(test_functions, ns_points, methods)
    
    # Print summary table
    summary = create_summary_table(results)
    print("\nSummary of Results:")
    print(summary)
    
    # Plot convergence for each function
    for func in test_functions:
        plot_convergence(
            results, 
            function_name=func.name,
            save_path=f"convergence_{func.name.lower()}.png"
        )
    
    # Plot overall convergence
    plot_convergence(results, save_path="convergence_all.png")
    
    # Plot performance comparison
    plot_performance(results, save_path="performance_comparison.png")
    
    print("\nVerification complete. Results and plots saved.")

if __name__ == "__main__":
    main()