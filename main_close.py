import pandas as pd
import ccxt
from datetime import datetime
import time

# Fetch Binance OHLCV data for multiple coins
# This function retrieves daily OHLCV data for a specified symbol from Binance
def fetch_binance_daily(symbol, since, until):
    binance = ccxt.binance()
    all_ohlcv = []
    since = int(since.timestamp() * 1000)
    until = int(until.timestamp() * 1000)
    limit = 1000
    while since < until:
        ohlcv = binance.fetch_ohlcv(symbol, '1d', since, limit=limit)
        if not ohlcv:
            break
        last = ohlcv[-1][0]
        all_ohlcv += ohlcv
        since = last + 1
        time.sleep(1)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['Date', 'open', 'high', 'low', 'close', 'volume']]

START_DATE = datetime(2020, 8, 28)
END_DATE = datetime.now()

COINS = {
    'BTCUSDT': 'Bitcoin',
    'ETHUSDT': 'Ethereum',
    'LTCUSDT': 'Litecoin',
}

for symbol, name in COINS.items():
    print(f"Fetching {name}...")
    df = fetch_binance_daily(symbol, START_DATE, END_DATE)
    df['Name'] = name
    df['Symbol'] = symbol.replace('USDT', '')
    df['Marketcap'] = None  # Optional, placeholder for now
    df = df.sort_values('Date')
    df.to_csv(f"coin_{name}.csv", index=False)
    print(f"Saved coin_{name}.csv [{df.shape}]")

print("All done! All CSVs are ready for imputation and merging.")
