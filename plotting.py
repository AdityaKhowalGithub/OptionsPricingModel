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

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=prices, y=deltas, name="Delta", line=dict(color="#FF9999")))
    fig.add_trace(go.Scatter(x=prices, y=gammas, name="Gamma", line=dict(color="#66B2FF")))
    fig.add_trace(go.Scatter(x=prices, y=vegas, name="Vega", line=dict(color="#99FF99")))
    fig.add_trace(go.Scatter(x=prices, y=thetas, name="Theta", line=dict(color="#FFCC99")))

    fig.update_layout(
        height=300,
        width=600,
        title_text="Option Greeks",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 1000, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 300,
                                                 "easing": "quadratic-in-out"}}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}])
            ])]
    )

    fig.frames = [
        go.Frame(data=[go.Scatter(x=prices, y=deltas, name="Delta")]),
        go.Frame(data=[go.Scatter(x=prices, y=gammas, name="Gamma")]),
        go.Frame(data=[go.Scatter(x=prices, y=vegas, name="Vega")]),
        go.Frame(data=[go.Scatter(x=prices, y=thetas, name="Theta")])
    ]

    return fig

# Keep the plot_heatmap function as it was in the previous version
def plot_heatmap(S, K, T, r, sigma, option_type, S_min, S_max, sigma_min, sigma_max):
    sigma_values = np.linspace(sigma_min, sigma_max, 50)
    S_values = np.linspace(S_min, S_max, 50)
    price_grid = np.array([[black_scholes(S_val, K, T, r, sigma_val, option_type) for S_val in S_values] for sigma_val in sigma_values])

    fig = go.Figure(data=go.Heatmap(
        z=price_grid,
        x=S_values,
        y=sigma_values,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title=f'{option_type.capitalize()} Option Price Heatmap',
        xaxis_title='Stock Price',
        yaxis_title='Volatility',
        height=400,
        width=600
    )

    return fig