import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Price Forecast", layout="centered", page_icon="💸")

# ---- Feature columns ----
feature_cols = [
    'BTC_close_scaled', 'ETH_close_scaled', 'LTC_close_scaled', 'Gold_close_scaled',
    'BTC_close_logret_scaled', 'ETH_close_logret_scaled', 'LTC_close_logret_scaled', 'Gold_close_logret_scaled'
]
coin_map = {
    "BTC": 'BTC_close_scaled',
    "ETH": 'ETH_close_scaled',
    "LTC": 'LTC_close_scaled',
    "Gold": 'Gold_close_scaled'
}
COIN_COLORS = {"BTC": "#F7931A", "ETH": "#3C3C3D", "LTC": "#BEBEBE", "Gold": "#FFD700"}

# ---- Load data and models ----
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(r'C:\Users\allif\Downloads\Crypto_stocks\notebooks\minmax_scaler_close_prices.pkl')
    data = pd.read_csv(
        r'C:\Users\allif\Downloads\Crypto_stocks\data\processed\scaled_predictive_coins.csv', 
        parse_dates=["Date"]
    )
    models = {
        "BTC": load_model(r'C:\Users\allif\Downloads\Crypto_stocks\notebooks\btc_bd_lstm_14d_model.h5'),
        "ETH": load_model(r'C:\Users\allif\Downloads\Crypto_stocks\notebooks\eth_bd_lstm_14d_model.h5'),
        "LTC": load_model(r'C:\Users\allif\Downloads\Crypto_stocks\notebooks\ltc_bd_lstm_14d_model.h5'),
        "Gold": load_model(r'C:\Users\allif\Downloads\Crypto_stocks\notebooks\gold_bd_lstm_14d_model.h5'),
    }
    return scaler, data, models

scaler, data, models = load_artifacts()

# ---- Streamlit UI ----
st.title("💹 Crypto Price Forecast (BD-LSTM)")
st.markdown(
    "Predicting the next **14 days** close prices for Bitcoin, Ethereum, Litecoin, and Gold using deep learning. "
    "All predictions are based on past price trends and recent movements."
)

st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=80)
st.sidebar.header("App Controls")
coin = st.sidebar.selectbox("Select a coin to predict:", models.keys())
lookback = st.sidebar.slider("Lookback Window", 30, 90, 30, 1)
horizon = st.sidebar.slider("Forecast Days", 7, 21, 14, 1)

# ---- Get inputs for prediction ----
window_df = data[feature_cols].iloc[-lookback:].copy()
input_seq = window_df.values.reshape(1, lookback, len(feature_cols))

# ---- Forecast loop ----
model = models[coin]
preds = []
input_seq_temp = input_seq.copy()
for _ in range(horizon):
    next_pred = model.predict(input_seq_temp, verbose=0)
    next_row = input_seq_temp[:, -1, :].copy()
    col_idx = feature_cols.index(coin_map[coin])
    next_row[0, col_idx] = next_pred[0, 0]
    input_seq_temp = np.concatenate([input_seq_temp[:, 1:, :], next_row.reshape(1, 1, len(feature_cols))], axis=1)
    preds.append(next_pred[0, 0])

# ---- Inverse scale ----
to_inverse = np.zeros((len(preds), len(feature_cols)))
to_inverse[:, col_idx] = preds
preds_unscaled = scaler.inverse_transform(to_inverse)[:, col_idx]
last_prices_scaled = input_seq[0, :, col_idx].reshape(-1, 1)
to_inverse_hist = np.zeros((len(last_prices_scaled), len(feature_cols)))
to_inverse_hist[:, col_idx] = last_prices_scaled.flatten()
history_unscaled = scaler.inverse_transform(to_inverse_hist)[:, col_idx]

# ---- Plotly chart ----
last_date = pd.to_datetime(data['Date'].iloc[-1])
pred_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
hist_dates = data['Date'].iloc[-lookback:]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist_dates, y=history_unscaled, mode="lines+markers",
    name="History", line=dict(color=COIN_COLORS.get(coin, "blue"))
))
fig.add_trace(go.Scatter(
    x=pred_dates, y=preds_unscaled, mode="lines+markers",
    name="Forecast", line=dict(color="orange", dash="dash")
))
fig.update_layout(
    title=f"{coin} - {horizon}-Day Close Price Forecast",
    xaxis_title="Date", yaxis_title=f"{coin} Close Price (USD)",
    legend=dict(orientation="h"), hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ---- Show prediction table and download ----
with st.expander("🔍 Show prediction table"):
    df_pred = pd.DataFrame({"date": pred_dates, f"{coin}_predicted_price": preds_unscaled})
    st.dataframe(df_pred)
    st.download_button(f"Download {coin} {horizon}d Forecast (CSV)", df_pred.to_csv(index=False), "predictions.csv")

# ---- More info/stats ----
recent_change = 100 * (history_unscaled[-1] - history_unscaled[-2]) / history_unscaled[-2] if len(history_unscaled) > 1 else 0
st.metric(f"Last Daily Change (%)", f"{recent_change:+.2f}%")
st.write(f"**Latest Actual {coin} Close Price:** ${history_unscaled[-1]:,.2f}")

st.info(
    "Note: This is a demonstration app. Predictions are for educational purposes only and **not financial advice**."
)

st.caption(
    "Powered by Streamlit • BD-LSTM Deep Learning Model • Author: Fredrick Alli"
)

# Optional: Add feedback form, social links, or coin explanations!
