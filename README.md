🚀 Crypto Price Prediction App
A professional, multivariate time series deep learning project to forecast the next 14 days' prices for cryptocurrencies (BTC, ETH, LTC) and Gold, using BD-LSTM models and a modern Streamlit dashboard.

🌟 Overview
This project demonstrates how to build, evaluate, and deploy advanced deep learning models (Bidirectional LSTM and more) for financial time series prediction.
It features:

Automated data extraction, merging, and feature engineering

Multiple deep learning models (LSTM, BD-LSTM, Conv-LSTM, TCN, and more)

Model selection, hyperparameter tuning, and backtesting

A user-friendly Streamlit web app for interactive forecasting and visualization

📈 Features
Multivariate Inputs: Uses multiple cryptocurrencies, Gold, and their log-returns as features for more robust forecasting.

Multiple Models: Benchmarks classic LSTM, BD-LSTM, Conv-LSTM, TCN, and attention-based variants.

14-day Forecast: Predicts the next 14 daily close prices for each asset.

Interactive App: Select your coin, adjust lookback window, visualize history and future predictions, and download results.

Production-Ready: Modular code, scalable architecture, and cloud-ready deployment.

🛠️ Tech Stack
Python 3.9+

Pandas / NumPy for data processing

TensorFlow / Keras for deep learning

Streamlit for web dashboard

Joblib for model/scaler serialization

Matplotlib for visualizations

🏗️ Project Structure

Crypto_stocks/
│
├── data/
│   └── processed/
│       └── scaled_predictive_coins.csv
├── models/
│   └── btc_bd_lstm_14d_model.h5, ...
├── notebooks/
│   └── (EDA, training, tuning, etc.)
├── app/
│   └── app.py  # Streamlit app
├── README.md
└── requirements.txt

🔍 Example Usage
Select a coin (BTC, ETH, LTC, Gold)

Set your lookback window (30–90 days)

View historical and predicted prices on an interactive chart

Download 14-day predictions as a CSV

🤖 Model Details
BD-LSTM: Bidirectional LSTM for better context in sequential data

Multivariate: Leverages correlated assets and their log-returns

Windowed Forecasting: Uses rolling windows for robust prediction

Want to compare with other models? See the included notebooks for TCN, Conv-LSTM, Encoder-Decoder, and attention-based variants!

📊 Results
BD-LSTM outperformed baseline LSTM, Conv-LSTM, and ED-LSTM on several metrics (MAE, RMSE, R²).

Consistent, realistic 14-day forecasts—ideal for both EDA and production use.

📝 Credits
Developed by Fredrick Alli

Inspired by: Wu et al. (2024). Review of deep learning models for crypto price prediction.

Open source libraries: TensorFlow, Streamlit, Pandas

⚠️ Disclaimer
This app and its predictions are for educational purposes only.
It is not financial advice and should not be used for trading decisions.

📬 Contact
Questions or want to collaborate? Open an issue or email me.

Pro tip: You can further customize the README for badges, screenshots, or links to papers/notebooks!