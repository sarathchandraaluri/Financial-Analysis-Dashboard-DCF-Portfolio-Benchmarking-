import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📊 Financial Analysis Dashboard")

# -----------------------------
# USER INPUT
# -----------------------------
stocks = st.multiselect(
    "Select Stocks",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
    default=["RELIANCE.NS", "TCS.NS"]
)

benchmark = "^NSEI"

start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

# -----------------------------
# FETCH DATA
# -----------------------------
data = yf.download(stocks + [benchmark], start=start_date, end=end_date)["Adj Close"]

stock_data = data[stocks]
benchmark_data = data[benchmark]

# -----------------------------
# RETURNS
# -----------------------------
returns = stock_data.pct_change().dropna()
benchmark_returns = benchmark_data.pct_change().dropna()

# -----------------------------
# METRICS
# -----------------------------
def sharpe_ratio(returns):
    return (returns.mean() / returns.std()) * np.sqrt(252)

def volatility(returns):
    return returns.std() * np.sqrt(252)

sharpe = sharpe_ratio(returns)
vol = volatility(returns)

st.subheader("📈 Risk Metrics")
st.write("Sharpe Ratio:", sharpe)
st.write("Volatility:", vol)

# -----------------------------
# BENCHMARK COMPARISON
# -----------------------------
st.subheader("📊 Performance vs Benchmark")

normalized = stock_data / stock_data.iloc[0]
benchmark_norm = benchmark_data / benchmark_data.iloc[0]

plt.figure(figsize=(10,5))
for col in normalized.columns:
    plt.plot(normalized[col], label=col)

plt.plot(benchmark_norm, label="NIFTY 50", linestyle="--")
plt.legend()
plt.title("Stock vs Benchmark")
st.pyplot(plt)

# -----------------------------
# DCF SECTION
# -----------------------------
st.subheader("💰 DCF Valuation")

discount_rate = st.slider("Discount Rate (WACC)", 0.05, 0.15, 0.10)
growth_rate = st.slider("Growth Rate", 0.01, 0.08, 0.04)

cash_flows = [100, 110, 120, 130, 140]

def dcf(cash_flows, discount_rate, growth_rate):
    value = 0
    for i, cf in enumerate(cash_flows):
        value += cf / ((1 + discount_rate) ** (i+1))
    
    terminal_value = (cash_flows[-1] * (1 + growth_rate)) / (discount_rate - growth_rate)
    terminal_value /= ((1 + discount_rate) ** len(cash_flows))
    
    return value + terminal_value

valuation = dcf(cash_flows, discount_rate, growth_rate)

st.write("Estimated Intrinsic Value:", valuation)

# -----------------------------
# INTERPRETATION
# -----------------------------
st.subheader("🧠 Insights")

st.write("""
- Higher Sharpe ratio indicates better risk-adjusted returns  
- Volatility measures risk level  
- Benchmark comparison shows relative performance  
- DCF helps estimate intrinsic value based on future cash flows  
""")
