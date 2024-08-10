import streamlit as st
import plotly.express as px
from pricing_models import black_scholes, heston_price, merton_jump_diffusion, monte_carlo_american_option
from plotting import plot_greeks_carousel, plot_pnl_heatmap, plot_past_calculations, generate_random_calculations
from database import init_db, save_inputs, save_outputs, get_past_calculations, get_latest_calculation_id

# Set page config as the first Streamlit command
st.set_page_config(page_title="Aditya Khowal's Multi Options Pricing Dashboard", layout="wide")

# Initialize the database connection
init_db()

# Initialize session state variables
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
if 'call_price' not in st.session_state:
    st.session_state.call_price = None
if 'put_price' not in st.session_state:
    st.session_state.put_price = None
if 'heatmap_params' not in st.session_state:
    st.session_state.heatmap_params = {}

# Title and input fields
st.title("Aditya Khowal's Advanced Options Pricing Dashboard")

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

# Calculate button
if st.button("Calculate"):
    # Save inputs
    input_id = save_inputs(S, K, T, r, sigma, model)

    # Calculate option prices
    if model == "Black-Scholes":
        st.session_state.call_price = black_scholes(S, K, T, r, sigma, "call")
        st.session_state.put_price = black_scholes(S, K, T, r, sigma, "put")
    elif model == "Heston":
        st.session_state.call_price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, "call")
        st.session_state.put_price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, "put")
    elif model == "Jump Diffusion":
        st.session_state.call_price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, "call")
        st.session_state.put_price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, "put")
    elif model == "Monte Carlo":
        st.session_state.call_price = monte_carlo_american_option(S, K, T, r, sigma, N, M, "call")
        st.session_state.put_price = monte_carlo_american_option(S, K, T, r, sigma, N, M, "put")

    # Prepare parameters for heatmap plotting
    st.session_state.heatmap_params = {
        "S": S, "K": K, "T": T, "r": r, "sigma": sigma,
        "model": model
    }

    if model == "Heston":
        st.session_state.heatmap_params.update({"v0": v0, "rho": rho, "kappa": kappa, "theta": theta})
    elif model == "Jump Diffusion":
        st.session_state.heatmap_params.update({"lam": lam, "mu_j": mu_j, "sigma_j": sigma_j})
    elif model == "Monte Carlo":
        st.session_state.heatmap_params.update({"N": N, "M": M})

    st.session_state.calculated = True

# Display results if calculation has been performed
if st.session_state.calculated:
    # Display option prices and get purchase prices
    st.markdown("## Option Prices and PNL")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background-color: #28a745; padding: 10px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">Call Price</h3>
                <p style="color: white; font-size: 24px; margin: 0;">${st.session_state.call_price:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        call_purchase_price = st.number_input("Call Purchase Price", value=st.session_state.call_price, step=0.01)
        call_pnl = st.session_state.call_price - call_purchase_price
        st.markdown(f"Call PNL: ${call_pnl:.2f}")
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #dc3545; padding: 10px; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">Put Price</h3>
                <p style="color: white; font-size: 24px; margin: 0;">${st.session_state.put_price:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        put_purchase_price = st.number_input("Put Purchase Price", value=st.session_state.put_price, step=0.01)
        put_pnl = st.session_state.put_price - put_purchase_price
        st.markdown(f"Put PNL: ${put_pnl:.2f}")

    # Greeks Carousel
    st.markdown("## Option Greeks")
    greeks_fig = plot_greeks_carousel(S, K, T, r, sigma, "call")
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

    # Update heatmap parameters
    st.session_state.heatmap_params.update({
        "spot_min": spot_min, "spot_max": spot_max,
        "sigma_min": sigma_min, "sigma_max": sigma_max
    })

    # Plot PNL Heatmaps
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner('Generating Call Option PNL Heatmap...'):
            call_heatmap = plot_pnl_heatmap(**st.session_state.heatmap_params, option_type="call",
                                            purchase_price=call_purchase_price)
        st.plotly_chart(call_heatmap, use_container_width=True)
    with col2:
        with st.spinner('Generating Put Option PNL Heatmap...'):
            put_heatmap = plot_pnl_heatmap(**st.session_state.heatmap_params, option_type="put",
                                           purchase_price=put_purchase_price)
        st.plotly_chart(put_heatmap, use_container_width=True)

# Display past calculations
st.markdown("## Past Calculations Analysis")
past_calculations = get_past_calculations()

if not past_calculations.empty:
    st.plotly_chart(plot_past_calculations(past_calculations), use_container_width=True)

    # Add some statistical insights
    st.markdown("### Statistical Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Average Values by Model")
        st.dataframe(past_calculations.groupby('model')[['stock_price', 'volatility', 'time_to_expiry', 'pnl']].mean())
    with col2:
        st.markdown("#### Correlation Matrix")
        correlation_matrix = past_calculations[
            ['stock_price', 'strike_price', 'time_to_expiry', 'risk_free_rate', 'volatility', 'pnl']].corr()
        st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm'))

    # Add a download button for the raw data
    csv = past_calculations.to_csv(index=False)
    st.download_button(
        label="Download Past Calculations CSV",
        data=csv,
        file_name="options_pricing_past_calculations.csv",
        mime="text/csv",
    )
else:
    st.info("No past calculations available yet. Use the calculator above to generate some data.")

# Add button to generate random calculations
if st.button("Generate Random Calculations"):
    generate_random_calculations(100)  # Generate 100 random calculations
    st.success("Generated 100 random calculations and added them to the database.")

# Add a section for model comparison
st.markdown("## Model Comparison")
if not past_calculations.empty:
    model_comparison = past_calculations.groupby('model')[['pnl']].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(model_comparison)

    # Visualize model comparison
    fig = px.box(past_calculations, x='model', y='pnl', title="PNL Distribution by Model")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Generate some calculations to see model comparison.")

# Add a section for PNL distribution
st.markdown("## PNL Distribution")
if not past_calculations.empty:
    fig = px.histogram(past_calculations, x='pnl', nbins=50, title="Overall PNL Distribution")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Generate some calculations to see PNL distribution.")

# Disclaimer
st.markdown("---")
st.markdown("""
    **Disclaimer**: This tool is for educational purposes only. Always consult with a qualified financial advisor before making investment decisions.
    """)