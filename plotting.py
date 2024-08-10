import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from greeks import delta, gamma, vega, theta

from pricing_models import black_scholes, heston_price, merton_jump_diffusion, monte_carlo_american_option

def plot_greeks_carousel(S, K, T, r, sigma, option_type):
    prices = np.linspace(S * 0.8, S * 1.2, 50)
    deltas = [delta(price, K, T, r, sigma, option_type) for price in prices]
    gammas = [gamma(price, K, T, r, sigma) for price in prices]
    vegas = [vega(price, K, T, r, sigma) for price in prices]
    thetas = [theta(price, K, T, r, sigma, option_type) for price in prices]

    fig = make_subplots(rows=1, cols=4, subplot_titles=("Delta", "Gamma", "Vega", "Theta"))

    fig.add_trace(go.Scatter(x=prices, y=deltas, name="Delta", line=dict(color="#FF9999")), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices, y=gammas, name="Gamma", line=dict(color="#66B2FF")), row=1, col=2)
    fig.add_trace(go.Scatter(x=prices, y=vegas, name="Vega", line=dict(color="#99FF99")), row=1, col=3)
    fig.add_trace(go.Scatter(x=prices, y=thetas, name="Theta", line=dict(color="#FFCC99")), row=1, col=4)

    fig.update_layout(
        height=300,
        width=1200,
        title_text="Option Greeks",
        showlegend=False
    )

    return fig

def plot_heatmap(S, K, T, r, sigma, option_type, spot_min, spot_max, sigma_min, sigma_max):
    sigma_values = np.linspace(sigma_min, sigma_max, 20)
    spot_values = np.linspace(spot_min, spot_max, 20)
    price_grid = np.array([[black_scholes(spot_val, K, T, r, sigma_val, option_type) for spot_val in spot_values] for sigma_val in sigma_values])

    fig = go.Figure(data=go.Heatmap(
        z=price_grid,
        x=spot_values,
        y=sigma_values,
        colorscale='Viridis',
        text=np.round(price_grid, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo='all',
        hoverongaps=False
    ))

    fig.update_layout(
        title=f'{option_type.capitalize()} Option Price Heatmap',
        xaxis_title='Spot Price',
        yaxis_title='Volatility',
        height=500,
        width=600
    )

    return fig




# Keep the existing plot_greeks_carousel function

def plot_pnl_heatmap(S, K, T, r, sigma, option_type, spot_min, spot_max, sigma_min, sigma_max, purchase_price, model):
    sigma_values = np.linspace(sigma_min, sigma_max, 20)
    spot_values = np.linspace(spot_min, spot_max, 20)

    def calculate_price(S, K, T, r, sigma, option_type):
        if model == "Black-Scholes":
            return black_scholes(S, K, T, r, sigma, option_type)
        elif model == "Heston":
            return heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, option_type)
        elif model == "Jump Diffusion":
            return merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type)
        elif model == "Monte Carlo":
            return monte_carlo_american_option(S, K, T, r, sigma, N, M, option_type)

    pnl_grid = np.array([[calculate_price(spot_val, K, T, r, sigma_val, option_type) - purchase_price
                          for spot_val in spot_values] for sigma_val in sigma_values])

    fig = go.Figure(data=go.Heatmap(
        z=pnl_grid,
        x=spot_values,
        y=sigma_values,
        colorscale='RdYlGn',  # Red for negative, Yellow for neutral, Green for positive
        text=np.round(pnl_grid, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo='all',
        hoverongaps=False
    ))

    fig.update_layout(
        title=f'{option_type.capitalize()} Option PNL Heatmap',
        xaxis_title='Spot Price',
        yaxis_title='Volatility',
        height=500,
        width=600
    )

    return fig

# Keep any other existing functions