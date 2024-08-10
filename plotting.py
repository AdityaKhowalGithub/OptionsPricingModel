import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from greeks import delta, gamma, vega, theta
from pricing_models import black_scholes

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
