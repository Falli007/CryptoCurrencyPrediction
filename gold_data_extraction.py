import ccxt
import pandas as pd
from datetime import datetime
import time

# Fetch Binance OHLCV data for Gold (PAXG/USDT)
# This function retrieves daily OHLCV data for a specified symbol from Binance
def fetch_binance_ohlcv_daily(symbol, since, until):
    binance = ccxt.binance()
    all_ohlcv = []
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    limit = 1000
    while since_ms < until_ms:
        try:
            ohlcv = binance.fetch_ohlcv(symbol, '1d', since_ms, limit=limit)
            if not ohlcv:
                break
            last = ohlcv[-1][0]
            all_ohlcv += ohlcv
            since_ms = last + 86400000  # 1 day in ms
            time.sleep(0.5)
        except Exception as e:
            print(e)
            time.sleep(3)
    if not all_ohlcv:
        print(f"No data found for {symbol}")
        return None
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    # Format columns to match your other coin data
    df['Name'] = 'Gold'
    df['Symbol'] = symbol
    df['Marketcap'] = None
    df = df.reset_index().rename(columns={'timestamp': 'Date'})
    df = df[['Date', 'open', 'high', 'low', 'close', 'volume', 'Name', 'Symbol', 'Marketcap']]
    df.to_csv('coin_Gold.csv', index=False)
    print(f"Saved coin_Gold.csv [{df.shape}]")
    return df

# Example usage
START_DATE = datetime(2019, 1, 1)
END_DATE = datetime.now()
gold_df = fetch_binance_ohlcv_daily('PAXGUSDT', START_DATE, END_DATE)
