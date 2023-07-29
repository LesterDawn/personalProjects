"""
Suppose we have a stock with a current price of $100, a constant volatility of 0.2, and a constant drift rate of 0.05.
We want to calculate the expected stock price after a time interval of 1 year.

Using Ito's Lemma, we can model the stock price as a stochastic process governed by the following equation:

dS(t) = μS(t) dt + σS(t) dW(t)

where S(t) is the stock price, μ is the drift rate, σ is the volatility, dt represents a small time increment, and dW(t)
 is the increment of a Wiener process (Brownian motion).

To simulate the stock price, we can discretize the time interval into smaller steps and use the following formula:

S(t + dt) = S(t) + μS(t) dt + σS(t) dW(t)

Let's generate some data using this formula and simulate the stock price over a time interval of 1 year with a step size
of 1 month:
"""
import numpy as np

# Parameters
S0 = 100  # Initial stock price
mu = 0.05  # Drift rate
sigma = 0.2  # Volatility
T = 1  # Time horizon in years
N = 12  # Number of time steps (monthly intervals)
dt = T / N  # Time step size

# Generate stock price path
np.random.seed(0)
t = np.linspace(0, T, N+1)
W = np.random.standard_normal(N) * np.sqrt(dt)
W = np.insert(W, 0, 0.0)  # Insert 0 at the beginning for the initial stock price
S = np.zeros(N+1)
S[0] = S0

for i in range(1, N+1):
    S[i] = S[i-1] + mu*S[i-1]*dt + sigma*S[i-1]*W[i]

# Print the simulated stock prices
print(S)

"""
The output will be an array of simulated stock prices over the specified time interval. Each element in the array 
represents the stock price at a particular time step.

Now, if you're interested in valuing options using Ito's Lemma, we can use the simulated stock price path to calculate 
the option prices. For example, we can use the Black-Scholes model to value European call options. The Black-Scholes 
formula is derived using Ito's Lemma and assumes that the stock price follows a geometric Brownian motion.

Let's consider valuing a European call option with a strike price of $105 and a maturity of 1 year. We can use the 
simulated stock prices to calculate the option prices using the Black-Scholes formula:
"""

from scipy.stats import norm

# Option parameters
K = 105  # Strike price
r = 0.05  # Risk-free interest rate
T = 1  # Time to maturity

# Calculate option prices using Black-Scholes formula
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
call_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Print the option price
print(call_price)
"""
The output will be the estimated price of the European call option based on the simulated stock prices.

This example demonstrates how Ito's Lemma can be applied to simulate stock prices and value options using the 
Black-Scholes model. However, please note that this is a simplified example, and in practice, options pricing can 
involve more complex models and considerations.

Feel free to modify the parameters or ask any further questions regarding the application of Ito's Lemma in valuing stock 
or option prices!"""