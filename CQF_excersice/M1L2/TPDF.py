import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
spx = pd.read_csv('SPX500.csv', parse_dates=True)[['Date', 'Adj Close']]
# spx['Date'] = pd.to_datetime(spx['Date'], format='%d/%m/%Y')
# spx.set_index('Date', inplace=True)

# Calculate log returns
log_returns = np.log(spx['Adj Close']/spx['Adj Close'].shift(1)).dropna()

# Calculate mean and standard deviation of log returns
mu = log_returns.mean()
sigma = log_returns.std()

# Define function to generate random numbers using normal distribution
def random_numbers(mu, sigma, size):
    return np.random.normal(mu, sigma, size)

# Define function to simulate future price
def simulate_price(start_price, mu, sigma, days):
    prices = np.zeros(days)
    prices[0] = start_price
    for i in range(1, days):
        prices[i] = prices[i-1] * np.exp(random_numbers(mu, sigma, 1))
    return prices

# Set start price and number of days to simulate
start_price = spx['Adj Close'].iloc[-14000]
days = 20000

# Simulate future prices
simulated_prices = simulate_price(start_price, mu, sigma, days)

# Plot actual and simulated prices
plt.plot(spx['Adj Close'])
plt.plot(range(len(spx)-14000, len(spx)), simulated_prices)
plt.legend(['Actual', 'Simulated'])
plt.show()
