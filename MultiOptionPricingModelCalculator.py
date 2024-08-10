import streamlit as st
from pricing_models import black_scholes, heston_price, merton_jump_diffusion, monte_carlo_american_option
from plotting import plot_greeks_carousel, plot_pnl_heatmap

# Streamlit App
st.set_page_config(page_title="Advanced Options Pricing Dashboard", layout="wide")

# Title and input fields
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Advanced Options Pricing Dashboard")

# Input Fields
col1, col2, col3 = st.columns(3)

with col1:
    S = st.number_input("Stock Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)

with col2:
    T = st.number_input("Time to Maturity (T)", value=1.0)
    r = st.number_input("Risk-free Rate (r)", value=0.05)

with col3:
    sigma = st.number_input("Volatility (sigma)", value=0.2)
    model = st.selectbox("Select Model", ("Black-Scholes", "Heston", "Jump Diffusion", "Monte Carlo"))

# Model-Specific Parameters
if model == "Heston":
    col1, col2, col3 = st.columns(3)
    with col1:
        v0 = st.number_input("Initial Variance (v0)", value=0.04)
        rho = st.number_input("Correlation (rho)", value=-0.7)
    with col2:
        kappa = st.number_input("Rate of Mean Reversion (kappa)", value=2.0)
    with col3:
        theta = st.number_input("Long-term Variance (theta)", value=0.04)
elif model == "Jump Diffusion":
    col1, col2, col3 = st.columns(3)
    with col1:
        lam = st.number_input("Intensity of Jumps (lambda)", value=0.75)
    with col2:
        mu_j = st.number_input("Mean of Jump Size (mu_j)", value=-0.2)
    with col3:
        sigma_j = st.number_input("Volatility of Jump Size (sigma_j)", value=0.3)
elif model == "Monte Carlo":
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Number of Simulations (N)", value=10000)
    with col2:
        M = st.number_input("Number of Time Steps (M)", value=100)

# Calculate option prices
if model == "Black-Scholes":
    call_price = black_scholes(S, K, T, r, sigma, "call")
    put_price = black_scholes(S, K, T, r, sigma, "put")
elif model == "Heston":
    call_price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, "call")
    put_price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, "put")
elif model == "Jump Diffusion":
    call_price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, "call")
    put_price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, "put")
elif model == "Monte Carlo":
    call_price = monte_carlo_american_option(S, K, T, r, sigma, N, M, "call")
    put_price = monte_carlo_american_option(S, K, T, r, sigma, N, M, "put")

# Purchase price inputs
col1, col2 = st.columns(2)
with col1:
    call_purchase_price = st.number_input("Call Purchase Price", value=call_price, step=0.01)
    call_pnl = call_price - call_purchase_price
with col2:
    put_purchase_price = st.number_input("Put Purchase Price", value=put_price, step=0.01)
    put_pnl = put_price - put_purchase_price

# Display option prices and PNL
st.markdown("## Option Prices and PNL")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div style="background-color: #28a745; padding: 10px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">Call Price</h3>
            <p style="color: white; font-size: 24px; margin: 0;">${call_price:.2f}</p>
            <h4 style="color: white; margin: 5px 0 0 0;">PNL</h4>
            <p style="color: white; font-size: 20px; margin: 0;">${call_pnl:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div style="background-color: #dc3545; padding: 10px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">Put Price</h3>
            <p style="color: white; font-size: 24px; margin: 0;">${put_price:.2f}</p>
            <h4 style="color: white; margin: 5px 0 0 0;">PNL</h4>
            <p style="color: white; font-size: 20px; margin: 0;">${put_pnl:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Greeks Carousel
st.markdown("## Option Greeks")
greeks_fig = plot_greeks_carousel(S, K, T, r, sigma, "call")

# Display the plotly chart in full width
st.plotly_chart(greeks_fig, use_container_width=True)

# Heatmap sliders
st.markdown("## Option PNL Heatmaps")
col1, col2 = st.columns(2)
with col1:
    spot_min = st.slider("Min Spot Price", min_value=0.5 * S, max_value=S, value=0.8 * S, step=0.01, key="spot_min")
    spot_max = st.slider("Max Spot Price", min_value=S, max_value=1.5 * S, value=1.2 * S, step=0.01, key="spot_max")
with col2:
    sigma_min = st.slider("Min Volatility", min_value=0.05, max_value=sigma, value=0.1, step=0.01, key="sigma_min")
    sigma_max = st.slider("Max Volatility", min_value=sigma, max_value=1.0, value=0.5, step=0.01, key="sigma_max")

# Plot PNL Heatmaps
col1, col2 = st.columns(2)
with col1:
    call_heatmap = plot_pnl_heatmap(S, K, T, r, sigma, "call", spot_min, spot_max, sigma_min, sigma_max, call_purchase_price, model)
    st.plotly_chart(call_heatmap, use_container_width=True)
with col2:
    put_heatmap = plot_pnl_heatmap(S, K, T, r, sigma, "put", spot_min, spot_max, sigma_min, sigma_max, put_purchase_price, model)
    st.plotly_chart(put_heatmap, use_container_width=True)