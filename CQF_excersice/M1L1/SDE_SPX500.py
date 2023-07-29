import pandas as pd
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read the file and compute std and mean of Return
spx_500 = pd.read_csv('SPX500.csv')
std_rtn = spx_500['Return'].std()
mean_rtn = spx_500['Return'].mean()

# Scale the Return to 0 mean and 1 std
spx_500['Scaled Rtn'] = spx_500['Return'].apply(lambda x: (x - mean_rtn) / std_rtn)

"""
1 "
"""
spx_500['5D_Return'] = (spx_500['Adj Close'].shift(-5) - spx_500['Adj Close']) / spx_500['Adj Close']
spx_500['2D_Return'] = (spx_500['Adj Close'].shift(-2) - spx_500['Adj Close']) / spx_500['Adj Close']

# Obtain std_2d and std_5d
std_rtn_2d = spx_500['2D_Return'].std()
std_rtn_5d = spx_500['5D_Return'].std()
"""
std_rtn_2d * 1/math.sqrt(2) = 0.011899422160776173
std_rtn_5d * 1/math.sqrt(5) = 0.011711201920123577
which are similar to std_rtn, 0.01198
"""

"""
2 "
"""
def Q2():
    # Split the dataset into two halves (even and odd observations)
    even_spx_500 = spx_500.iloc[::2]
    odd_spx_500 = spx_500.iloc[1::2]

    # Calculate the mean and standard deviation for each half (1D returns only)
    even_mean = even_spx_500['Return'].mean()
    even_std = even_spx_500['Return'].std()
    odd_mean = odd_spx_500['Return'].mean()
    odd_std = odd_spx_500['Return'].std()
    """
    Even Mean: 0.0003255999405664943
    Even Standard Deviation: 0.012288498578777158
    Odd Mean: 0.0002763576761488403
    Odd Standard Deviation: 0.011682707490220979
    """


"""
3 "
"""
# Create a Q-Q plot for 1D returns
sm.qqplot(spx_500['Return'], line='s')
plt.title('Q-Q Plot for 1D Returns')
plt.show()

# Create a Q-Q plot for 5D returns
sm.qqplot(spx_500['5D_Return'], line='s')
plt.title('Q-Q Plot for 5D Returns')
plt.show()
