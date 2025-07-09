# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 🎨 Page config
st.set_page_config(layout="wide", page_title="📊 Stock Market Dashboard")

# 🏷️ Title
st.title("📈 Stock Market Analysis Dashboard")
st.markdown("Visualize stock trends, technical indicators, and download data")

# 📌 Sidebar Inputs
st.sidebar.header("🛠 Configuration")
stocks = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Wipro": "WIPRO.NS"
}
selected_stocks = st.sidebar.multiselect("Select Stock(s)", list(stocks.keys()), default=["TCS"])

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-07-07"))

show_close = st.sidebar.checkbox("Show Close Price", value=True)
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=False)
show_vol = st.sidebar.checkbox("Show Volatility", value=False)

# 📈 Main Analysis Section
for stock in selected_stocks:
    ticker = stocks[stock]
    st.subheader(f"📊 {stock} ({ticker})")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    data.reset_index(inplace=True)
    data.ffill(inplace=True)
    data['Daily Return'] = data['Close'].pct_change()
    data['SMA_30'] = data['Close'].rolling(30).mean()
    data['SMA_100'] = data['Close'].rolling(100).mean()
    data['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
    data['Volatility'] = data['Daily Return'].rolling(window=21).std()

    # Bollinger Bands
    data['BB_MA20'] = data['Close'].rolling(window=20).mean()
    data['BB_STD'] = data['Close'].rolling(window=20).std()
    data['Upper'] = data['BB_MA20'] + 2 * data['BB_STD']
    data['Lower'] = data['BB_MA20'] - 2 * data['BB_STD']

    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # 📊 Charts
    if show_close or show_ma or show_bbands:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Date'], data['Close'], label='Close Price', alpha=0.5)
        if show_ma:
            ax.plot(data['Date'], data['SMA_30'], label='SMA 30', color='green')
            ax.plot(data['Date'], data['SMA_100'], label='SMA 100', color='orange')
        if show_bbands:
            ax.plot(data['Date'], data['Upper'], label='Upper Band', linestyle='--', color='red')
            ax.plot(data['Date'], data['Lower'], label='Lower Band', linestyle='--', color='blue')
        ax.set_title(f'{stock} - Price & Indicators')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    if show_rsi:
        fig2, ax2 = plt.subplots(figsize=(10, 2))
        ax2.plot(data['Date'], data['RSI'], label='RSI', color='purple')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_title(f'{stock} - RSI')
        ax2.grid(True)
        st.pyplot(fig2)

    if show_vol:
        fig3, ax3 = plt.subplots(figsize=(10, 2))
        ax3.plot(data['Date'], data['Volatility'], label='Volatility', color='darkblue')
        ax3.set_title(f'{stock} - Volatility (21-day Rolling Std Dev)')
        ax3.grid(True)
        st.pyplot(fig3)

    # 📦 Download section
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name=f"{stock}_cleaned_stock_data.csv",
        mime='text/csv'
    )

    # 📋 Show last 10 rows
    with st.expander("🔍 View Data Table"):
        st.dataframe(data.tail(10))

