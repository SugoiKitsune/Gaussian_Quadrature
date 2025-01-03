import jax.numpy as jnp
from jax.scipy.linalg import eigh_tridiagonal
from jax.scipy.special import gamma
import integrand  # Import the integrand module
import sympy as sp
from sympy import symbols, pprint


# ----------------------------
# Quadrature Functions
# ----------------------------

def generalized_laguerre_quadrature(f, n_points, alpha=0, tau=1.0, K=100.0, Rt=0.05, v=0.2, PtT=1.0):
    """
    Performs Gaussian-Laguerre quadrature to approximate the integral of f(x) * x^alpha * exp(-x) over [0, ∞).

    Parameters:
    f : callable
        The integrand function. Should accept JAX arrays and return JAX arrays.
    n_points : int
        Number of quadrature points (degree of the Laguerre polynomial).
    alpha : float, optional
        Parameter for the generalized Laguerre polynomial (default is 0 for standard Laguerre).
    tau, K, Rt, v, PtT : float, optional
        Additional parameters for the integrand function.

    Returns:
    float
        Approximated integral of f(x) * x^alpha * exp(-x) over [0, ∞).
    """
    # Define the index array
    i = jnp.arange(1, n_points + 1)

    # Compute diagonal and off-diagonal elements
    diagonal = 2 * i + alpha + 1
    off_diagonal = jnp.sqrt(i * (i + alpha))

    # Compute eigenvalues (nodes) and eigenvectors
    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)

    # Nodes are the eigenvalues
    nodes = eigenvalues

    # Weights are the square of the first row of eigenvectors times Gamma(alpha + 1)
    weights = (eigenvectors[0, :] ** 2) * gamma(alpha + 1)

    # Evaluate the integrand at the nodes with given parameters
    fx = f(nodes, tau, K, Rt, v, PtT)

    # Compute the integral as the weighted sum
    integral = jnp.sum(weights * fx)

    return integral


# Optional: Define a standard Laguerre quadrature using the generalized function
def standard_laguerre_quadrature(f, n_points, tau=1.0, K=100.0, Rt=0.05, v=0.2, PtT=1.0):
    """
    Performs Gaussian-Laguerre quadrature to approximate the integral of f(x) * exp(-x) over [0, ∞).

    Parameters:
    f : callable
        The integrand function.
    n_points : int
        Number of quadrature points (degree of the Laguerre polynomial).
    tau, K, Rt, v, PtT : float, optional
        Additional parameters for the integrand function.

    Returns:
    float
        Approximated integral of f(x) * exp(-x) over [0, ∞).
    """
    return generalized_laguerre_quadrature(f, n_points, alpha=0, tau=tau, K=K, Rt=Rt, v=v, PtT=PtT)


# ----------------------------
# Optional: SymPy Functions to Print Generalized Laguerre Polynomials
# ----------------------------

def print_generalized_laguerre(n, alpha_value=None):
    """
    Generates and prints the generalized Laguerre polynomial L_n^(alpha)(x) using SymPy.

    Parameters:
    n (int): Degree of the polynomial.
    alpha_value (int or float, optional): The alpha parameter. If provided, substitutes its value.
    """
    x_sym, alpha_sym = symbols('x alpha')
    # Generate the generalized Laguerre polynomial using SymPy's assoc_laguerre
    L_n_alpha = sp.assoc_laguerre(n, alpha_sym, x_sym)

    # Substitute alpha if a specific value is provided
    if alpha_value is not None:
        L_n_alpha = L_n_alpha.subs(alpha_sym, alpha_value)

    # Pretty-print the polynomial
    print(f"L_{n}^({alpha_value if alpha_value is not None else 'alpha'}) (x) =")
    pprint(L_n_alpha, use_unicode=True)
    print("\n")


# ----------------------------
# Test Script
# ----------------------------

def main():
    """
    Main function to perform and test Gaussian-Laguerre quadrature.
    """
    # Define quadrature parameters
    n_points = 5  # Number of quadrature points
    alpha_val = 2  # Parameter for generalized Laguerre polynomials

    # Optional: Print generalized Laguerre polynomials using SymPy
    print("Generalized Laguerre Polynomials (L_n^(2)(x)):")
    for degree in range(0, 6):
        print_generalized_laguerre(degree, alpha_value=alpha_val)

    # Perform the quadrature using the integrand from integrand.py
    approx_integral = generalized_laguerre_quadrature(
        integrand.f,
        n_points,
        alpha=alpha_val,
        tau=1.0,  # Example value
        K=100.0,  # Example value
        Rt=0.05,  # Example value
        v=0.2,  # Example value
        PtT=1.0  # Example value
    )

    print(f"Approximated Integral using integrand.f: {approx_integral}")

    # ----------------------------
    # Validation: Compare with Exact Integral
    # ----------------------------

    # For validation, define parameters such that the integral can be computed exactly.
    # Example: Let's choose f(x) = x^k where k + alpha + 1 is an integer.
    # The exact integral becomes Gamma(k + alpha + 1).

    # Define k based on the integrand. Since f(x) = x^3 and weight is x^alpha * e^{-x},
    # the integral is Gamma(k + alpha + 1) = Gamma(4 + 2 + 1) = Gamma(7) = 720

    exact_integral = gamma(7.0)  # Gamma(7) = 720

    print(f"Exact Integral (Gamma(7)): {exact_integral}")

    # Compute the absolute error
    error = jnp.abs(approx_integral - exact_integral)
    print(f"Absolute Error: {error}")

    # Assert the accuracy (within a reasonable tolerance)
    if jnp.isclose(approx_integral, exact_integral, rtol=1e-6):
        print("Quadrature validated successfully!")
    else:
        print("Quadrature result is not accurate enough.")


if __name__ == "__main__":
    main()