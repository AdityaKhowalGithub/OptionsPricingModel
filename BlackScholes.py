import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price


# Test the Black-Scholes implementation
S = 100  # Current stock price
K = 100  # Strike price
T = 1  # Time to maturity in years
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

#



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


# Test the Heston model implementation
v0 = 0.04
rho = -0.7
kappa = 2.0
theta = 0.04

# call_price_heston = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma, "call")
# print(f"Heston Call Price: {call_price_heston}")


def merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type="call"):
    n_max = 50
    option_price = 0
    for n in range(n_max):
        r_n = r - lam * (mu_j - 0.5 * sigma_j ** 2) + n * np.log(1 + mu_j)
        sigma_n = np.sqrt(sigma ** 2 + (n * sigma_j ** 2) / T)
        poisson_prob = np.exp(-lam * T) * (lam * T) ** n / np.math.factorial(n)
        option_price += poisson_prob * black_scholes(S, K, T, r_n, sigma_n, option_type)

    return option_price


# Test the Jump Diffusion model implementation
lam = 0.75
mu_j = -0.2
sigma_j = 0.3

# call_price_jump = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, "call")
# print(f"Jump Diffusion Call Price: {call_price_jump}")


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


# Test the Monte Carlo simulation for American options
# call_price_mc = monte_carlo_american_option(S, K, T, r, sigma, N=10000, M=100, option_type="call")
# # print(f"Monte Carlo American Call Price: {call_price_mc}")

def delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def theta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

def rho(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S/K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# Test Greeks calculations
delta_val = delta(S, K, T, r, sigma, "call")
gamma_val = gamma(S, K, T, r, sigma)
vega_val = vega(S, K, T, r, sigma)
theta_val = theta(S, K, T, r, sigma, "call")
rho_val = rho(S, K, T, r, sigma, "call")

# print(f"Delta: {delta_val}")
# print(f"Gamma: {gamma_val}")
# print(f"Vega: {vega_val}")
# print(f"Theta: {theta_val}")
# print(f"Rho: {rho_val}")


import yfinance as yf

def get_real_time_data(ticker):
    data = yf.Ticker(ticker)
    return data.history(period="1d")["Close"][-1]

# Example usage to fetch real-time data
ticker = "AAPL"
current_price = get_real_time_data(ticker)
print(f"Current Price of {ticker}: {current_price}")

import cProfile
import pstats

def profile_function():
    # Example function to profile
    monte_carlo_american_option(S, K, T, r, sigma, N=10000, M=100, option_type="call")

# Profile the function
profiler = cProfile.Profile()
profiler.enable()
profile_function()
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()


def main():
    print("Options Pricing Model")
    print("1. Black-Scholes")
    print("2. Heston Model")
    print("3. Jump Diffusion")
    print("4. Monte Carlo for American Options")
    choice = input("Select a model (1-4): ")

    S = float(input("Enter current stock price (S): "))
    K = float(input("Enter strike price (K): "))
    T = float(input("Enter time to maturity in years (T): "))
    r = float(input("Enter risk-free rate (r): "))
    sigma = float(input("Enter volatility (sigma): "))

    if choice == "1":
        print("Using Black-Scholes Model")
        option_type = input("Enter option type (call/put): ")
        price = black_scholes(S, K, T, r, sigma, option_type)
    elif choice == "2":
        print("Using Heston Model")
        v0 = float(input("Enter initial variance (v0): "))
        rho = float(input("Enter correlation (rho): "))
        kappa = float(input("Enter rate of mean reversion (kappa): "))
        theta = float(input("Enter long-term variance (theta): "))
        price = heston_price(S, K, T, r, v0, rho, kappa, theta, sigma)
    elif choice == "3":
        print("Using Jump Diffusion Model")
        lam = float(input("Enter intensity of jumps (lam): "))
        mu_j = float(input("Enter mean of jump size (mu_j): "))
        sigma_j = float(input("Enter volatility of jump size (sigma_j): "))
        option_type = input("Enter option type (call/put): ")
        price = merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type)
    elif choice == "4":
        print("Using Monte Carlo Simulation for American Options")
        N = int(input("Enter number of simulations (N): "))
        M = int(input("Enter number of time steps (M): "))
        option_type = input("Enter option type (call/put): ")
        price = monte_carlo_american_option(S, K, T, r, sigma, N, M, option_type)
    else:
        print("Invalid choice")
        return

    print(f"Option Price: {price}")

if __name__ == "__main__":
    main()
