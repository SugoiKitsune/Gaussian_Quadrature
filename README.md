# Gaussian Quadrature for Financial Derivatives

This project implements advanced numerical integration techniques for pricing financial derivatives, specifically focusing on caplet pricing using the Bachelier model. The implementation uses JAX for efficient computation and provides both Taylor series and Gauss-Laguerre quadrature approaches.

## Features

- Gaussian-Laguerre quadrature implementation with stability improvements
- Bachelier model implementation for caplet pricing
- Taylor series approximation method
- Numerically stable implementations with careful overflow handling
- Visualization tools for integrands
- Comprehensive test suite

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Pricing a Caplet

```python
from pricer import compare_pricing_models

# Example parameters
RtT = 0.05      # Forward rate
K = 0.04        # Strike rate
sigma_gtT = 0.01 # Volatility
tau = 0.5       # Accrual period
PtT = 0.98      # Discount factor
n_points = 10   # Quadrature points

# Compare pricing methods
results = compare_pricing_models(RtT, K, sigma_gtT, tau, PtT, n_points)
```

### Using Gaussian Quadrature

```python
from quadrature import gauss_laguerre_quadrature_final

def your_integrand(x):
    return x * np.exp(-x/2)  # Example integrand

result, error_est = gauss_laguerre_quadrature_final(your_integrand, n=20)
```

## Project Structure

- `bachelier_model.py`: Implements the Bachelier model for caplet pricing
- `quadrature.py`: Core implementation of Gauss-Laguerre quadrature
- `pricer.py`: High-level pricing interface comparing different methods
- `integrals.py`: General purpose integration utilities
- `charting.py`: Visualization utilities
- `test_quadrature.py`: Test suite for quadrature implementations

## Technical Details

### Numerical Stability

The implementation includes several numerical stability improvements:
- Scaled recurrence relations for Laguerre polynomials
- Careful handling of exponential terms to prevent overflow
- Adaptive damping in Newton iterations
- Error estimation and condition number monitoring

### Performance Optimization

- Uses JAX for automatic differentiation and JIT compilation
- Efficient quadrature node computation
- Compensated summation for improved accuracy

## Testing

Run the test suite:
```bash
python -m pytest test_quadrature.py
```

## License

MIT License - See LICENSE file for details