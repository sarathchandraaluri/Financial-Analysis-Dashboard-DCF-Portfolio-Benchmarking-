import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Dashboard", layout="wide")

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
# VALIDATION
# -----------------------------
if len(stocks) == 0:
    st.warning("Please select at least one stock.")
    st.stop()

# -----------------------------
# FETCH DATA (SAFE)
# -----------------------------
try:
    raw_data = yf.download(stocks + [benchmark], start=start_date, end=end_date)
    
    if raw_data.empty:
        st.error("No data fetched. Check internet or ticker symbols.")
        st.stop()

    data = raw_data["Adj Close"].dropna()

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# -----------------------------
# SPLIT DATA
# -----------------------------
try:
    stock_data = data[stocks].copy()
    benchmark_data = data[[benchmark]].copy()
except KeyError:
    st.error("Stock symbols not found in data. Try different stocks.")
    st.write("Available columns:", data.columns)
    st.stop()

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
col1, col2 = st.columns(2)

with col1:
    st.write("### Sharpe Ratio")
    st.dataframe(sharpe)

with col2:
    st.write("### Volatility")
    st.dataframe(vol)

# -----------------------------
# BENCHMARK COMPARISON
# -----------------------------
st.subheader("📊 Performance vs Benchmark")

normalized = stock_data / stock_data.iloc[0]
benchmark_norm = benchmark_data / benchmark_data.iloc[0]

fig, ax = plt.subplots()

for col in normalized.columns:
    ax.plot(normalized.index, normalized[col], label=col)

ax.plot(benchmark_norm.index, benchmark_norm[benchmark], label="NIFTY 50", linestyle="--")

ax.legend()
ax.set_title("Stock vs Benchmark Growth")

st.pyplot(fig)

# -----------------------------
# DCF SECTION
# -----------------------------
st.subheader("💰 DCF Valuation")

discount_rate = st.slider("Discount Rate (WACC)", 0.05, 0.15, 0.10)
growth_rate = st.slider("Growth Rate", 0.01, 0.08, 0.04)

cash_flows = st.text_input(
    "Enter projected cash flows (comma separated)",
    "100,110,120,130,140"
)

try:
    cash_flows = [float(x.strip()) for x in cash_flows.split(",")]
except:
    st.error("Invalid cash flow input format")
    st.stop()

def dcf(cash_flows, discount_rate, growth_rate):
    value = 0
    for i, cf in enumerate(cash_flows):
        value += cf / ((1 + discount_rate) ** (i + 1))
    
    terminal_value = (cash_flows[-1] * (1 + growth_rate)) / (discount_rate - growth_rate)
    terminal_value /= ((1 + discount_rate) ** len(cash_flows))
    
    return value + terminal_value

valuation = dcf(cash_flows, discount_rate, growth_rate)

st.success(f"Estimated Intrinsic Value: {round(valuation, 2)}")

# -----------------------------
# INSIGHTS
# -----------------------------
st.subheader("🧠 Insights")

st.markdown("""
- **Sharpe Ratio** → Higher means better risk-adjusted returns  
- **Volatility** → Measures risk (higher = more risky)  
- **Benchmark Comparison** → Shows performance vs NIFTY 50  
- **DCF Valuation** → Estimates intrinsic value based on future cash flows  
""")
