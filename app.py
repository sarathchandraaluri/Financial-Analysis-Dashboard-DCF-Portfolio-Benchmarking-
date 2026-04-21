import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Dashboard", layout="wide")

st.title("📊 Financial Analysis Dashboard")

# -----------------------------
# STOCK LIST
# -----------------------------
top_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS"
]

stocks = st.multiselect("Select Stocks", top_stocks, default=top_stocks[:3])
benchmark = "^NSEI"

start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

if len(stocks) == 0:
    st.warning("Select at least one stock.")
    st.stop()

# -----------------------------
# FETCH DATA (ROBUST)
# -----------------------------
try:
    raw_data = yf.download(stocks + [benchmark], start=start_date, end=end_date)

    if raw_data.empty:
        st.error("No data fetched. Check internet or ticker symbols.")
        st.stop()

    # Handle different column formats
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" in raw_data.columns.levels[0]:
            data = raw_data["Adj Close"]
        else:
            data = raw_data["Close"]
    else:
        if "Adj Close" in raw_data.columns:
            data = raw_data["Adj Close"]
        else:
            data = raw_data["Close"]

    data = data.dropna()

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.write("Columns received:", raw_data.columns if 'raw_data' in locals() else "No data")
    st.stop()

# -----------------------------
# SPLIT DATA
# -----------------------------
try:
    stock_data = data[stocks].copy()
    benchmark_data = data[[benchmark]].copy()
except KeyError:
    st.error("Some selected stocks not found in data.")
    st.write("Available columns:", data.columns)
    st.stop()

# -----------------------------
# RETURNS
# -----------------------------
returns = stock_data.pct_change().dropna()
benchmark_returns = benchmark_data.pct_change().dropna()

# -----------------------------
# BASIC METRICS
# -----------------------------
def sharpe_ratio(r):
    return (r.mean() / r.std()) * np.sqrt(252)

def volatility(r):
    return r.std() * np.sqrt(252)

st.subheader("📈 Risk Metrics")

col1, col2 = st.columns(2)

with col1:
    st.write("### Sharpe Ratio")
    st.dataframe(sharpe_ratio(returns))

with col2:
    st.write("### Volatility")
    st.dataframe(volatility(returns))

# -----------------------------
# BENCHMARK COMPARISON
# -----------------------------
st.subheader("📊 Performance vs Benchmark")

normalized = stock_data / stock_data.iloc[0]
benchmark_norm = benchmark_data / benchmark_data.iloc[0]

fig, ax = plt.subplots()

for col in normalized.columns:
    ax.plot(normalized.index, normalized[col], label=col)

ax.plot(benchmark_norm.index, benchmark_norm[benchmark], linestyle="--", label="NIFTY 50")

ax.legend()
ax.set_title("Stock vs Benchmark")

st.pyplot(fig)

# -----------------------------
# BETA
# -----------------------------
st.subheader("📉 Beta")

beta = {}
market_var = np.var(benchmark_returns.squeeze())

for stock in returns.columns:
    cov = np.cov(returns[stock], benchmark_returns.squeeze())[0][1]
    beta[stock] = cov / market_var

beta = pd.Series(beta)
st.dataframe(beta)

# -----------------------------
# CAPM
# -----------------------------
st.subheader("📊 CAPM Expected Return")

risk_free_rate = st.slider("Risk-Free Rate", 0.03, 0.08, 0.06)

market_return = benchmark_returns.mean() * 252
capm = risk_free_rate + beta * (market_return - risk_free_rate)

st.dataframe(capm)

# -----------------------------
# PORTFOLIO WEIGHTS
# -----------------------------
st.subheader("⚖️ Portfolio Allocation")

weights = []

for stock in stocks:
    weight = st.slider(f"{stock} Weight", 0.0, 1.0, 1.0/len(stocks))
    weights.append(weight)

weights = np.array(weights)

if weights.sum() == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

weights = weights / weights.sum()

st.write("Normalized Weights:", weights)

# -----------------------------
# PORTFOLIO RETURN
# -----------------------------
portfolio_return = np.sum(returns.mean() * weights) * 252
st.write("📈 Portfolio Expected Return:", round(portfolio_return, 4))

# -----------------------------
# ALPHA
# -----------------------------
st.subheader("⭐ Alpha")

alpha = (returns.mean() * 252) - capm
st.dataframe(alpha)

# -----------------------------
# DCF
# -----------------------------
st.subheader("💰 DCF Valuation")

discount_rate = st.slider("Discount Rate", 0.05, 0.15, 0.10)
growth_rate = st.slider("Growth Rate", 0.01, 0.08, 0.04)

cash_input = st.text_input("Cash Flows (comma separated)", "100,110,120,130,140")

try:
    cash_flows = [float(x.strip()) for x in cash_input.split(",")]
except:
    st.error("Invalid cash flow format.")
    st.stop()

def dcf(cfs, r, g):
    value = 0
    for i, cf in enumerate(cfs):
        value += cf / ((1 + r) ** (i + 1))
    
    tv = (cfs[-1] * (1 + g)) / (r - g)
    tv /= ((1 + r) ** len(cfs))
    
    return value + tv

valuation = dcf(cash_flows, discount_rate, growth_rate)

st.success(f"Intrinsic Value: {round(valuation,2)}")

# -----------------------------
# INSIGHTS
# -----------------------------
st.subheader("🧠 Insights")

st.markdown("""
- Sharpe Ratio → risk-adjusted return  
- Beta → sensitivity to market  
- CAPM → expected return  
- Alpha → excess return  
- DCF → intrinsic value estimation  
""")
