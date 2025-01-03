import numpy as np
import jax.numpy as jnp
from plotly import graph_objects as go
from utils import compute_transformed_strike, standard_normal_pdf

def visualize_integral(integrand, n_points=100, x_range=(0, 10)):
    """
    Visualize the integrand and its approximation using Laguerre quadrature.
    """
    x_values = jnp.linspace(*x_range, n_points)
    y_values = jnp.array([integrand(x) for x in x_values])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Integrand'))
    fig.update_layout(
        title="Integrand Visualization",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_dark"
    )
    fig.show()
