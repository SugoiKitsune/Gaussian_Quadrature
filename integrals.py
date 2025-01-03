import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import eigh_tridiagonal
from jax.scipy.special import gamma
import sympy as sp  # Optional: For symbolic polynomial printing
from sympy import symbols, pprint


# ----------------------------
# 2. Define Symbolic Variables and Print Laguerre Polynomials (Optional)
# ----------------------------

# Optional: If you want to print generalized Laguerre polynomials symbolically
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


# Example: Print L_0^(2)(x) to L_5^(2)(x) (Optional)
for degree in range(0, 6):
    print_generalized_laguerre(degree, alpha_value=2)


# ----------------------------
# 3. Define Quadrature Functions Using JAX
# ----------------------------

@jit
def generalized_laguerre_quadrature(f, n_points, alpha=0):
    """
    Performs Gaussian-Laguerre quadrature to approximate the integral of f(x) * x^alpha * exp(-x) over [0, ∞).

    Parameters:
    f : callable
        The integrand function. Should accept and return JAX arrays.
    n_points : int
        Number of quadrature points (degree of the Laguerre polynomial).
    alpha : float, optional
        Parameter for the generalized Laguerre polynomial (default is 0 for standard Laguerre).

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

    # Evaluate the integrand at the nodes
    fx = f(nodes)

    # Compute the integral as the weighted sum
    integral = jnp.sum(weights * fx)

    return integral


# Optional: Define a standard Laguerre quadrature using the generalized function
@jit
def standard_laguerre_quadrature(f, n_points):
    """
    Performs Gaussian-Laguerre quadrature to approximate the integral of f(x) * exp(-x) over [0, ∞).

    Parameters:
    f : callable
        The integrand function.
    n_points : int
        Number of quadrature points (degree of the Laguerre polynomial).

    Returns:
    float
        Approximated integral of f(x) * exp(-x) over [0, ∞).
    """
    return generalized_laguerre_quadrature(f, n_points, alpha=0)


# ----------------------------
# 4. Define the Integrand Function
# ----------------------------

def integrand(x):
    return x ** 3  # Example: Integrate x^3 * x^alpha * e^{-x}


# ----------------------------
# 5. Perform Quadrature and Print Results
# ----------------------------

# Quadrature parameters
n_points = 5  # Number of quadrature points
alpha_val = 2  # Parameter for generalized Laguerre polynomials

# Perform the quadrature
approx_integral = generalized_laguerre_quadrature(integrand, n_points, alpha=alpha_val)

print(f"Approximated Integral: {approx_integral}")

# ----------------------------
# 6. Validate the Implementation (Optional)
# ----------------------------

# Compute the exact integral
exact_integral = gamma(alpha_val + 1 + 3)  # Gamma(alpha + 1 + k), where k=3
print(f"Exact Integral: {exact_integral}")

# Compute and print the error
error = jnp.abs(approx_integral - exact_integral)
print(f"Error: {error}")