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

from .models.bachelier import (
    bachelier_caplet_price_taylor,
    bachelier_caplet_price_laguerre
)
from .pricing.pricer import compare_pricing_models
from .utils.quadrature import (
    generalized_laguerre_quadrature,
    compute_gauss_laguerre_points_final,
    laguerre_function_final
)
from .utils.stats import standard_normal_cdf, standard_normal_pdf
from .utils.transforms import compute_transformed_strike
from .visualization.charting import visualize_integral

__version__ = "0.1.0"

__all__ = [
    # Models
    'bachelier_caplet_price_taylor',
    'bachelier_caplet_price_laguerre',
    
    # Pricing
    'compare_pricing_models',
    
    # Quadrature
    'generalized_laguerre_quadrature',
    'compute_gauss_laguerre_points_final',
    'laguerre_function_final',
    
    # Statistics
    'standard_normal_cdf',
    'standard_normal_pdf',
    
    # Transforms
    'compute_transformed_strike',
    
    # Visualization
    'visualize_integral'
]