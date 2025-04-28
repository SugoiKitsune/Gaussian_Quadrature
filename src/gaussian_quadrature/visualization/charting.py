"""
Visualization tools for integrand analysis
"""

import jax.numpy as jnp
from plotly import graph_objects as go


def visualize_integral(integrand, n_points=100, x_range=(0, 10)):
    """
    Visualize the integrand and its approximation using Laguerre quadrature.

    Parameters:
    integrand : callable
        The integrand function to visualize
    n_points : int
        Number of points for visualization
    x_range : tuple
        (min, max) range for x-axis

    Returns:
    None
        Displays the plot using plotly
    """
    x_values = jnp.linspace(*x_range, n_points)
    y_values = jnp.array([integrand(x) for x in x_values])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values, 
        mode='lines', 
        name='Integrand'
    ))
    
    fig.update_layout(
        title="Integrand Visualization",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_dark",
        showlegend=True
    )
    
    fig.show()