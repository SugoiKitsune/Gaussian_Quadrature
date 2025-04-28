"""
Test suite for quadrature implementations
"""

import jax.numpy as jnp
from gaussian_quadrature.utils.quadrature import (
    laguerre_function_final,
    compute_gauss_laguerre_points_final,
    generalized_laguerre_quadrature
)
from jax.scipy.special import gamma


def test_laguerre_stability():
    """Test stability of Laguerre function computation"""
    print("\nTesting Laguerre function stability:")
    print("n   |  Max Value  |  Min Value  | Max/Min Ratio")
    print("-" * 50)
    
    x = jnp.linspace(0, 30, 100)
    for n in [5, 10, 20, 50, 100]:
        try:
            L_n, _ = laguerre_function_final(n, x)
            max_val = jnp.max(jnp.abs(L_n))
            nonzero_vals = jnp.abs(L_n)[jnp.abs(L_n) > 1e-300]
            if len(nonzero_vals) > 0:
                min_val = jnp.min(nonzero_vals)
                ratio = max_val / min_val
            else:
                min_val = 0.0
                ratio = float('inf')
            print(f"{n:3d} | {max_val:10.2e} | {min_val:10.2e} | {ratio:10.2e}")
        except Exception as e:
            print(f"{n:3d} | Failed: {str(e)}")


def test_quadrature_precision():
    """Test quadrature precision with simple test function"""
    print("\nTesting quadrature precision:")
    print("n   | Value        | Rel Error")
    print("-" * 40)
    
    def test_func(x):
        return x * jnp.exp(-x/2)  # Should give exact integral = 2
    
    exact = 2.0
    for n in [5, 10, 20, 40]:
        try:
            result = generalized_laguerre_quadrature(test_func, n)
            rel_error = jnp.abs(result - exact) / exact
            print(f"{n:3d} | {result:11.8f} | {rel_error:.2e}")
        except Exception as e:
            print(f"{n:3d} | Failed: {str(e)}")


def test_known_integrals():
    """Test against known integrals with different alpha values"""
    print("\nTesting known integrals with different alpha:")
    
    test_cases = [
        (lambda x: 1.0, 0.0, 1.0, "∫e^(-x)dx = 1"),
        (lambda x: x, 0.0, 1.0, "∫xe^(-x)dx = 1"),
        (lambda x: x**2, 0.0, 2.0, "∫x²e^(-x)dx = 2"),
        (lambda x: x**3, 0.0, 6.0, "∫x³e^(-x)dx = 6"),
        (lambda x: x, 1.0, 2.0, "∫x²e^(-x)dx = 2 (α=1)"),
        (lambda x: jnp.exp(-x/2), 0.0, 2.0, "∫e^(-3x/2)dx = 2/3")
    ]
    
    print("\nα  n  | Integral     | Rel Error")
    print("-" * 40)
    
    for f, alpha, exact, desc in test_cases:
        print(f"\n{desc}:")
        for n in [10, 20, 30]:
            try:
                result = generalized_laguerre_quadrature(f, n, alpha)
                rel_error = jnp.abs(result - exact) / exact
                print(f"{alpha:2.0f} {n:2d} | {result:11.8f} | {rel_error:.2e}")
            except Exception as e:
                print(f"{alpha:2.0f} {n:2d} | Failed: {str(e)}")


if __name__ == "__main__":
    print("Running Laguerre quadrature tests...")
    test_laguerre_stability()
    test_quadrature_precision()
    test_known_integrals()