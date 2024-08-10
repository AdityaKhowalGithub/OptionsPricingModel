import streamlit as st
from pricing_models import black_scholes, heston_price, merton_jump_diffusion, monte_carlo_american_option
from plotting import plot_greeks_carousel, plot_heatmap

# Streamlit App
st.set_page_config(page_title="Advanced Options Pricing Dashboard", layout="wide")

# Light/Dark mode switch
if 'light_mode' not in st.session_state:
    st.session_state.light_mode = False

# Custom CSS for the shaking lightbulb
shaking_lightbulb = """
<style>
@keyframes shake {
    0% { transform: rotate(0deg); }
    25% { transform: rotate(5deg); }
    50% { transform: rotate(0deg); }
    75% { transform: rotate(-5deg); }
    100% { transform: rotate(0deg); }
}
.shaking {
    display: inline-block;
    animation: shake 0.5s;
}
</style>
"""

st.markdown(shaking_lightbulb, unsafe_allow_html=True)

# Title and light mode switch
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Advanced Options Pricing Dashboard")
with col2:
    light_mode = st.checkbox("🔦", value=st.session_state.light_mode, key="light_mode_checkbox")
    if light_mode != st.session_state.light_mode:
        st.session_state.light_mode = light_mode
        st.markdown(f"<script>document.querySelector('.stCheckbox').classList.add('shaking');</script>", unsafe_allow_html=True)

# Custom color scheme
if light_mode:
    primary_color = "#4A4E69"
    secondary_color = "#9A8C98"
    accent_color = "#C9ADA7"
    background_color = "#F2E9E4"
    text_color = "#22223B"
else:
    primary_color = "#22223B"
    secondary_color = "#4A4E69"
    accent_color = "#9A8C98"
    background_color = "#2B2D42"
    text_color = "#EDF2F4"

# Apply custom theme
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {background_color};
        color: {text_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {primary_color};
    }}
    .Widget>label {{
        color: {text_color};
    }}
    .stButton>button {{
        color: {background_color};
        background-color: {accent_color};
    }}
    .stSlider>div>div>div>div {{
        background-color: {accent_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Display option prices
st.markdown("## Option Prices")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div style="background-color: #28a745; padding: 10px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">Call Price</h3>
            <p style="color: white; font-size: 24px; margin: 0;">${call_price:.2f}</p>
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
        </div>
        """,
        unsafe_allow_html=True
    )

# Greeks Carousel
st.markdown("## Option Greeks")
greeks_fig = plot_greeks_carousel(S, K, T, r, sigma, "call")

# Custom CSS for horizontal scrolling
st.markdown("""
<style>
.scroll-container {
    width: 100%;
    overflow-x: auto;
    white-space: nowrap;
}
.scroll-container::-webkit-scrollbar {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# Wrap the Plotly chart in a scrollable container
st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
st.plotly_chart(greeks_fig, use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)


# Heatmap sliders
st.markdown("## Option Price Heatmaps")
col1, col2 = st.columns(2)
with col1:
    spot_min = st.slider("Min Spot Price", min_value=0.5*S, max_value=S, value=0.8*S, step=0.01, key="spot_min")
    spot_max = st.slider("Max Spot Price", min_value=S, max_value=1.5*S, value=1.2*S, step=0.01, key="spot_max")
with col2:
    sigma_min = st.slider("Min Volatility", min_value=0.05, max_value=sigma, value=0.1, step=0.01, key="sigma_min")
    sigma_max = st.slider("Max Volatility", min_value=sigma, max_value=1.0, value=0.5, step=0.01, key="sigma_max")

# Plot Heatmaps
col1, col2 = st.columns(2)
with col1:
    call_heatmap = plot_heatmap(S, K, T, r, sigma, "call", spot_min, spot_max, sigma_min, sigma_max)
    st.plotly_chart(call_heatmap, use_container_width=True)
with col2:
    put_heatmap = plot_heatmap(S, K, T, r, sigma, "put", spot_min, spot_max, sigma_min, sigma_max)
    st.plotly_chart(put_heatmap, use_container_width=True)