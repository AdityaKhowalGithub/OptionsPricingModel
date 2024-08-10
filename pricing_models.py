import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

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