import pandas as pd

# Read all CSVs
btc = pd.read_csv('coin_Bitcoin.csv', parse_dates=['Date'])
eth = pd.read_csv('coin_Ethereum.csv', parse_dates=['Date'])
ltc = pd.read_csv('coin_Litecoin.csv', parse_dates=['Date'])
gold = pd.read_csv('coin_Gold.csv', parse_dates=['Date'])

# Drop unnecessary columns
btc = btc.drop(['Name', 'Symbol', 'Marketcap'], axis=1)
eth = eth.drop(['Name', 'Symbol', 'Marketcap'], axis=1)
ltc = ltc.drop(['Name', 'Symbol', 'Marketcap'], axis=1)
gold = gold.drop(['Name', 'Symbol', 'Marketcap'], axis=1)

# Rename columns for clarity
btc = btc.rename(lambda x: f'BTC_{x}' if x != 'Date' else x, axis=1)
eth = eth.rename(lambda x: f'ETH_{x}' if x != 'Date' else x, axis=1)
ltc = ltc.rename(lambda x: f'LTC_{x}' if x != 'Date' else x, axis=1)
gold = gold.rename(lambda x: f'Gold_{x}' if x != 'Date' else x, axis=1)

# Merge on Date
df = btc.merge(eth, on='Date').merge(ltc, on='Date').merge(gold, on='Date')

# Sort and drop NaN if needed
df = df.sort_values('Date')
df = df.dropna()

df.to_csv("predictive_coins_merged.csv", index=False)
print(df.head())