# Test script for the average rate caplet integrand function
import sys
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the module
from src.gaussian_quadrature.models import average_rate_caplet as avg_cap

# Force reload to ensure we have the latest changes
importlib.reload(avg_cap)

def test_average_rate_caplet_integrand():
    # Define test parameters
    tau = 0.5      # Year fraction (6 months)
    K = 0.025      # Strike rate (2.5%)
    Rt = 0.02      # Forward rate (2%)
    
    # Calculate the variance adjustment for average rate (σ²T₁/3)
    sigma = 0.004  # Bachelier volatility (40 bps)
    T1 = 1.0       # Time to start of averaging period
    T2 = T1 + tau  # Time to end of averaging period
    
    # Adjusted volatility for average rate
    var_avg = (sigma**2 * T1 / 3) * (1 + (T2-T1)/(2*T1))
    v = np.sqrt(var_avg)
    
    PtT = 0.97     # Discount factor
    
    # Create a range of x values to evaluate the integrand
    x_values = np.linspace(0, 5, 50)
    
    # Calculate integrand values
    integrand_values = [avg_cap.f(x, tau, K, Rt, v, PtT) for x in x_values]
    
    # Create weighted integrand values (f(x)*e^(-x))
    weighted_values = [f_val * np.exp(-x) for f_val, x in zip(integrand_values, x_values)]
    
    # Print some values for debugging
    print("Sample integrand values:")
    print(f"x=0: {avg_cap.f(0.0, tau, K, Rt, v, PtT)}")
    print(f"x=1: {avg_cap.f(1.0, tau, K, Rt, v, PtT)}")
    print(f"x=2: {avg_cap.f(2.0, tau, K, Rt, v, PtT)}")
    print(f"x=3: {avg_cap.f(3.0, tau, K, Rt, v, PtT)}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, integrand_values, 'b-', label='Integrand f(x)')
    plt.plot(x_values, weighted_values, 'r--', label='Weighted Integrand f(x)e^(-x)')
    plt.title('Average Rate Caplet Integrand Function')
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('integrand_plot.png')
    plt.show()

if __name__ == "__main__":
    test_average_rate_caplet_integrand()
