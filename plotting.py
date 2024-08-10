import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from greeks import delta, gamma, vega, theta
from pricing_models import black_scholes, heston_price, merton_jump_diffusion, monte_carlo_american_option
from database import save_inputs,save_outputs, get_latest_calculation_id
import random

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
        height=400,
        width=1200,
        title_text="Option Greeks",
        showlegend=False
    )

    for i in range(1, 5):
        fig.update_xaxes(title_text="Stock Price", row=1, col=i)
        fig.update_yaxes(title_text="Value", row=1, col=i)

    return fig

def plot_pnl_heatmap(S, K, T, r, sigma, option_type, spot_min, spot_max, sigma_min, sigma_max, model, purchase_price, **kwargs):
    sigma_values = np.linspace(sigma_min, sigma_max, 20)
    spot_values = np.linspace(spot_min, spot_max, 20)

    def calculate_price(S, K, T, r, sigma, option_type):
        if model == "Black-Scholes":
            return black_scholes(S, K, T, r, sigma, option_type)
        elif model == "Heston":
            return heston_price(S, K, T, r, kwargs.get('v0'), kwargs.get('rho'),
                                kwargs.get('kappa'), kwargs.get('theta'), sigma, option_type)
        elif model == "Jump Diffusion":
            return merton_jump_diffusion(S, K, T, r, sigma, kwargs.get('lam'),
                                         kwargs.get('mu_j'), kwargs.get('sigma_j'), option_type)
        elif model == "Monte Carlo":
            return monte_carlo_american_option(S, K, T, r, sigma, kwargs.get('N'), kwargs.get('M'), option_type)

    price_grid = np.array([[calculate_price(spot_val, K, T, r, sigma_val, option_type)
                            for spot_val in spot_values] for sigma_val in sigma_values])
    # Calculate PNL
    pnl_grid = price_grid - purchase_price

    # Save heatmap data to database
    input_id = get_latest_calculation_id()

    for i, sigma_val in enumerate(sigma_values):
        for j, spot_val in enumerate(spot_values):
            volatility_shock = (sigma_val - sigma) / sigma
            stock_price_shock = (spot_val - S) / S
            option_price = price_grid[i, j]
            pnl = pnl_grid[i, j]
            is_call = option_type == "call"
            save_outputs(input_id, volatility_shock, stock_price_shock, option_price, is_call, pnl)

    fig = go.Figure(data=go.Heatmap(
        z=pnl_grid,
        x=spot_values,
        y=sigma_values,
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
        text=np.round(pnl_grid, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo='all',
        hoverongaps=False,
        zmid=0  # This ensures that 0 is always white
    ))

    fig.update_layout(
        title=f'{option_type.capitalize()} Option PNL Heatmap',
        xaxis_title='Spot Price',
        yaxis_title='Volatility',
        height=500,
        width=600
    )

    return fig


def plot_past_calculations(df):
    # Ensure the dataframe has the necessary columns
    if 'volatility' not in df.columns or 'model' not in df.columns:
        return None

    # Create a scatter plot
    fig = px.scatter(df, x='volatility', y='stock_price', color='model',
                     hover_data=['strike_price', 'time_to_expiry', 'risk_free_rate', 'pnl'],
                     labels={
                         'volatility': 'Volatility',
                         'stock_price': 'Stock Price',
                         'model': 'Pricing Model'
                     },
                     title='Past Calculations: Stock Price vs Volatility')

    # Add trend lines for each model
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        fig.add_trace(go.Scatter(
            x=model_data['volatility'],
            y=model_data['stock_price'],
            mode='lines',
            name=f'{model} Trend',
            line=dict(dash='dash')
        ))

    # Customize the layout
    fig.update_layout(
        xaxis_title='Volatility',
        yaxis_title='Stock Price',
        legend_title='Model',
        height=600
    )

    return fig


def generate_random_calculations(num_calculations=100):
    models = ["Black-Scholes", "Heston", "Jump Diffusion", "Monte Carlo"]

    for _ in range(num_calculations):
        model = random.choice(models)
        S = random.uniform(50, 150)
        K = random.uniform(0.8 * S, 1.2 * S)
        T = random.uniform(0.1, 2)
        r = random.uniform(0.01, 0.1)
        sigma = random.uniform(0.1, 0.5)

        input_id = save_inputs(S, K, T, r, sigma, model)

        if model == "Black-Scholes":
            call_price = black_scholes(S, K, T, r, sigma, "call")
            put_price = black_scholes(S, K, T, r, sigma, "put")
        elif model == "Heston":
            v0 = random.uniform(0.01, 0.05)
            rho = random.uniform(-0.9, 0.9)
            kappa = random.uniform(1, 5)
            theta = random.uniform(0.01, 0.05)
            call_price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, "call")
            put_price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, "put")
        elif model == "Jump Diffusion":
            lam = random.uniform(0.1, 2)
            mu_j = random.uniform(-0.5, 0.5)
            sigma_j = random.uniform(0.1, 0.5)
            call_price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, "call")
            put_price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, "put")
        elif model == "Monte Carlo":
            N = 10000
            M = 100
            call_price = monte_carlo_american_option(S, K, T, r, sigma, N, M, "call")
            put_price = monte_carlo_american_option(S, K, T, r, sigma, N, M, "put")

        # Generate random purchase prices
        call_purchase_price = call_price * random.uniform(0.8, 1.2)
        put_purchase_price = put_price * random.uniform(0.8, 1.2)

        save_outputs(input_id, 0, 0, call_price, True, call_price - call_purchase_price)
        save_outputs(input_id, 0, 0, put_price, False, put_price - put_purchase_price)

    print(f"Generated {num_calculations} random calculations.")