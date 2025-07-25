import pandas as pd
import numpy as np
import ccxt
import time

def fetch_binance_ohlcv_30m(symbol, since, until, timeframe='30m'):
    binance = ccxt.binance()
    all_ohlcv = []
    since = int(since.timestamp() * 1000)
    until = int(until.timestamp() * 1000)
    limit = 1000
    while since < until:
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since, limit=limit)
            if not ohlcv:
                break
            last = ohlcv[-1][0]
            all_ohlcv += ohlcv
            since = last + 1
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(10)
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df

def engineer_internal_features(df):
    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols:
        for lag in [1,2,3,6,12,24]:  # Lags: last 30min, 1h, 1.5h, 3h, 6h, 12h
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    for col in cols:
        for w in [3,6,12,24]:  # Rolling: last 1.5h, 3h, 6h, 12h
            df[f"{col}_rollmean_{w}"] = df[col].rolling(w).mean()
            df[f"{col}_rollstd_{w}"] = df[col].rolling(w).std()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['target_close_next'] = df['close'].shift(-1)  # Predict next period's close
    df = df.dropna()
    return df
