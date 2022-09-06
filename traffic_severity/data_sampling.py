import pandas as pd

df = pd.read_csv('./US_Accidents_Dec20.csv')
print("Shape of dataset:", df.shape)
df.sample(500000).to_csv('./sampled_data.csv')