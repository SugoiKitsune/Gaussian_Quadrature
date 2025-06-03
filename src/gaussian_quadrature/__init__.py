"""
Gaussian Quadrature for Financial Derivatives

This package provides implementations of numerical integration techniques
for pricing financial derivatives, specifically focusing on caplet pricing
using the Bachelier model.

Main Components:
- Bachelier model implementation for caplet pricing
- Gauss-Laguerre quadrature with numerical stability improvements
- Taylor series approximation method
- Visualization tools for integrand analysis
"""

# We'll use a simplified approach with version number only
__version__ = "0.1.0"

# Don't import everything at package level to avoid circular dependencies
# Functions will be imported directly from their modules when needed