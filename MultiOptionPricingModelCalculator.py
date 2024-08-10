import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


# Option Pricing Models
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price


def heston_char_func(phi, v0, S0, r, T, rho, kappa, theta, sigma):
    xi = kappa - sigma * rho * 1j * phi
    d = np.sqrt(xi ** 2 + sigma ** 2 * (1j * phi + phi ** 2))
    g = (xi - d) / (xi + d)

    C = r * 1j * phi * T + kappa * theta / sigma ** 2 * ((xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D = (xi - d) / sigma ** 2 * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

    return np.exp(C + D * v0 + 1j * phi * np.log(S0))


def heston_price(S0, K, T, r, v0, rho, kappa, theta, sigma, option_type="call"):
    def integrand(phi, S0, K, T, r, v0, rho, kappa, theta, sigma):
        return np.real(
            np.exp(-1j * phi * np.log(K)) * heston_char_func(phi - 1j, v0, S0, r, T, rho, kappa, theta, sigma) / (
                    1j * phi * heston_char_func(-1j, v0, S0, r, T, rho, kappa, theta, sigma)))

    integral_value = quad(integrand, 0, np.inf, args=(S0, K, T, r, v0, rho, kappa, theta, sigma))[0]
    price = S0 * 0.5 + S0 * integral_value / np.pi

    if option_type == "put":
        price = price - S0 + K * np.exp(-r * T)

    return price


def merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type="call"):
    n_max = 50
    option_price = 0
    for n in range(n_max):
        r_n = r - lam * (mu_j - 0.5 * sigma_j ** 2) + n * np.log(1 + mu_j)
        sigma_n = np.sqrt(sigma ** 2 + (n * sigma_j ** 2) / T)
        poisson_prob = np.exp(-lam * T) * (lam * T) ** n / np.math.factorial(n)
        option_price += poisson_prob * black_scholes(S, K, T, r_n, sigma_n, option_type)

    return option_price


def monte_carlo_american_option(S, K, T, r, sigma, N=10000, M=100, option_type="call"):
    dt = T / M
    discount_factor = np.exp(-r * T)

    payoff = np.zeros(N)
    for i in range(N):
        S_t = S
        for t in range(M):
            S_t *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn())
            if option_type == "call":
                payoff[i] = max(payoff[i], S_t - K)
            elif option_type == "put":
                payoff[i] = max(payoff[i], K - S_t)

    option_price = discount_factor * np.mean(payoff)
    return option_price


def get_real_time_data(ticker):
    data = yf.Ticker(ticker)
    return data.history(period="1d")["Close"][-1]


def plot_greeks(S, K, T, r, sigma, option_type):
    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []
    prices = np.linspace(S * 0.8, S * 1.2, 50)
    for price in prices:
        deltas.append(delta(price, K, T, r, sigma, option_type))
        gammas.append(gamma(price, K, T, r, sigma))
        vegas.append(vega(price, K, T, r, sigma))
        thetas.append(theta(price, K, T, r, sigma, option_type))
        rhos.append(rho(price, K, T, r, sigma, option_type))

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].plot(prices, deltas, label='Delta', color='green' if option_type == "call" else 'red')
    ax[0, 0].set_title('Delta')
    ax[0, 0].grid(True)

    ax[0, 1].plot(prices, gammas, label='Gamma', color='blue')
    ax[0, 1].set_title('Gamma')
    ax[0, 1].grid(True)

    ax[1, 0].plot(prices, vegas, label='Vega', color='purple')
    ax[1, 0].set_title('Vega')
    ax[1, 0].grid(True)

    ax[1, 1].plot(prices, thetas, label='Theta', color='orange')
    ax[1, 1].set_title('Theta')
    ax[1, 1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Heatmap of Greeks for varying stock prices and volatilities
    sigma_values = np.linspace(0.1, 0.5, 50)
    S_values = np.linspace(S * 0.8, S * 1.2, 50)
    delta_grid = np.array(
        [[delta(S_val, K, T, r, sigma_val, option_type) for S_val in S_values] for sigma_val in sigma_values])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(delta_grid, xticklabels=np.round(S_values, 2), yticklabels=np.round(sigma_values, 2), cmap="YlGnBu",
                ax=ax)
    ax.set_title('Delta Heatmap')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Volatility')
    st.pyplot(fig)


# Greeks Calculation Functions
def delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)


def rho(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# Streamlit App
st.set_page_config(page_title="Options Pricing Model", layout="wide")
st.title("Options Pricing Model")
st.write("This application allows you to calculate option prices using different models and visualize the Greeks.")

# Input Fields with default values
st.sidebar.header("Input Parameters")

S = st.sidebar.number_input("Stock Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T)", value=1.0)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2)
option_type = st.sidebar.radio("Option Type", ("call", "put"))

# Model Selection
model = st.sidebar.selectbox("Select Model", ("Black-Scholes", "Heston", "Jump Diffusion", "Monte Carlo"))

# Model-Specific Parameters
if model == "Heston":
    st.sidebar.subheader("Heston Model Parameters")
    v0 = st.sidebar.number_input("Initial Variance (v0)", value=0.04)
    rho = st.sidebar.number_input("Correlation (rho)", value=-0.7)
    kappa = st.sidebar.number_input("Rate of Mean Reversion (kappa)", value=2.0)
    theta = st.sidebar.number_input("Long-term Variance (theta)", value=0.04)
elif model == "Jump Diffusion":
    st.sidebar.subheader("Jump Diffusion Parameters")
    lam = st.sidebar.number_input("Intensity of Jumps (lambda)", value=0.75)
    mu_j = st.sidebar.number_input("Mean of Jump Size (mu_j)", value=-0.2)
    sigma_j = st.sidebar.number_input("Volatility of Jump Size (sigma_j)", value=0.3)
elif model == "Monte Carlo":
    st.sidebar.subheader("Monte Carlo Parameters")
    N = st.sidebar.number_input("Number of Simulations (N)", value=10000)
    M = st.sidebar.number_input("Number of Time Steps (M)", value=100)

# Pre-calculate with default values
if model == "Black-Scholes":
    price = black_scholes(S, K, T, r, sigma, option_type)
elif model == "Heston":
    price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, option_type)
elif model == "Jump Diffusion":
    price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type)
elif model == "Monte Carlo":
    price = monte_carlo_american_option(S, K, T, r, sigma, N, M, option_type)

# Display the result
st.write(f"### Option Price: <span style='color:{'green' if option_type == 'call' else 'red'};'>{price:.2f}</span>", unsafe_allow_html=True)


# Plot Greeks
plot_greeks(S, K, T, r, sigma, option_type)
