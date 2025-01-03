from bachelier_model import bachelier_caplet_price_taylor, bachelier_caplet_price_laguerre
import time

def compare_pricing_models(RtT, K, sigma_gtT, tau, PtT, n_points=10):
    """
    Compare the caplet price using the Taylor series-based approach and Laguerre Quadrature.

    Parameters:
    RtT : float
        Forward rate at time t for maturity T.
    K : float
        Strike rate of the caplet.
    sigma_gtT : float
        Volatility term (σ * g_t,T).
    tau : float
        Year fraction or accrual period.
    PtT : float
        Discount factor from time t to T.
    n_points : int
        Number of quadrature points for Laguerre Quadrature.

    Returns:
    dict
        Dictionary containing results from both models.
    """
    # Taylor series-based pricing
    start_taylor = time.time()
    price_taylor = bachelier_caplet_price_taylor(RtT, K, sigma_gtT, tau, PtT)
    end_taylor = time.time()

    # Laguerre quadrature-based pricing
    start_laguerre = time.time()
    price_laguerre = bachelier_caplet_price_laguerre(tau, K, RtT, sigma_gtT, PtT, n_points)
    end_laguerre = time.time()

    return {
        "Taylor Series": {"Price": price_taylor, "Time": end_taylor - start_taylor},
        "Laguerre Quadrature": {"Price": price_laguerre, "Time": end_laguerre - start_laguerre}
    }

if __name__ == "__main__":
    RtT = 0.05  # Forward rate
    K = 0.04  # Strike rate
    sigma_gtT = 0.01  # Volatility
    tau = 0.5  # Accrual period
    PtT = 0.98  # Discount factor
    n_points = 10  # Quadrature points

    results = compare_pricing_models(RtT, K, sigma_gtT, tau, PtT, n_points)
    for method, result in results.items():
        print(f"{method}: Price = {result['Price']:.6f}, Time = {result['Time']:.6f} seconds")