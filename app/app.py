import sys
import warnings
import random
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from typing import List
import datetime

warnings.filterwarnings("ignore")

# ==========================================================
# Config & Seeds
# ==========================================================
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 10
SIMS_PER_SEED = 2000
FORECAST_DAYS = 21  # 1-month horizon base

# ==========================================================
# Basic Helpers
# ==========================================================
def to_weights(raw: List[float]) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to positive.")
    return arr / s


def annualized_return_daily(r):
    r = r.dropna()
    if r.empty:
        return np.nan
    compounded = np.exp(r.sum())
    years = len(r) / 252.0
    return compounded ** (1 / years) - 1 if years > 0 else np.nan


def annualized_vol_daily(r):
    r = r.dropna()
    return r.std(ddof=0) * np.sqrt(252) if len(r) > 1 else np.nan


def annualized_sharpe_daily(r, rf_daily=0.0):
    r = r.dropna()
    if r.empty:
        return np.nan
    excess = r - rf_daily
    mu, sigma = excess.mean(), excess.std(ddof=0)
    return (mu / sigma) * np.sqrt(252) if sigma and sigma > 0 else np.nan


def max_drawdown_from_rets(returns):
    cum = np.exp(returns.cumsum())
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).min()

# ==========================================================
# Data Fetch
# ==========================================================
def fetch_prices_daily(tickers, start=DEFAULT_START, include_dividends=True):
    data = yf.download(
        tickers, start=start, interval="1d",
        auto_adjust=include_dividends, progress=False, threads=False
    )
    if data.empty:
        raise ValueError("No price data returned.")

    if include_dividends:
        if isinstance(data, pd.DataFrame):
            if "Close" in data.columns:
                close = data["Close"].copy()
            else:
                close = data.xs("Close", axis=1, level=0)
        else:
            raise ValueError("Unexpected data format for adjusted prices.")
    else:
        if isinstance(data.columns, pd.MultiIndex):
            for f in ["Close", "Adj Close"]:
                if f in data.columns.get_level_values(0):
                    close = data[f].copy()
                    break
            else:
                raise ValueError("No valid price field found.")
        else:
            close = data.copy()

    close = close.ffill().astype(np.float32)
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(1)
    return close.dropna(how="all")


def portfolio_log_returns_daily(prices, weights):
    prices = prices.ffill().dropna().astype(np.float64)
    weights = np.array(weights, dtype=np.float64)
    if len(weights) != prices.shape[1]:
        raise ValueError("Weight count mismatch")
    rets = np.log(prices / prices.shift(1)).dropna()
    port_rets = (rets * weights).sum(axis=1)
    return port_rets.astype(np.float32)

# ==========================================================
# Optimal Block Length Estimation
# ==========================================================
def estimate_optimal_block_length(residuals):
    res2 = residuals ** 2
    if len(res2) < 2:
        return 5
    rho1 = np.corrcoef(res2[:-1], res2[1:])[0, 1]
    T = len(res2)
    b_opt = 1.5 * ((T / (1 - rho1**2 + 1e-9)) ** (1 / 3))
    return int(np.clip(b_opt, 5, 100))

# ==========================================================
# Monte Carlo Simulation
# ==========================================================
def run_monte_carlo_paths(residuals, sims_per_seed, rng, base_mean, total_days, block_length):
    n_res = len(residuals)
    n_blocks = int(np.ceil(total_days / block_length))
    starts = rng.integers(0, n_res - block_length, size=(sims_per_seed, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[..., None] + offsets).reshape(sims_per_seed, -1)
    idx = np.mod(idx, n_res)[:, :total_days]
    eps = residuals[idx]
    log_paths = np.cumsum(base_mean + eps, axis=1, dtype=np.float32)
    return np.exp(log_paths - log_paths[:, [0]])

# ==========================================================
# Mean-Like Path
# ==========================================================
def compute_medoid_path(paths):
    log_paths = np.log(paths)
    mean_path = np.mean(log_paths, axis=0, dtype=np.float32)
    distances = np.linalg.norm(log_paths - mean_path, axis=1)
    idx = np.argmin(distances)
    return paths[idx]

# ==========================================================
# Forecast-Based Walk-Forward Directional Accuracy
# ==========================================================
def compute_oos_directional_accuracy_walkforward(prices, weights, resample_rule, horizon_days):
    port_rets = portfolio_log_returns_daily(prices, weights)
    agg_returns = port_rets.resample(resample_rule).sum()
    dates = agg_returns.index
    preds, acts = [], []
    for i in range(4, len(dates) - 1):
        sub_prices = prices.loc[:dates[i]]
        sub_rets = portfolio_log_returns_daily(sub_prices, weights)
        mu = sub_rets.mean()
        sigma = sub_rets.std(ddof=0)
        base_mean = mu - 0.5 * sigma ** 2
        residuals = (sub_rets - mu).to_numpy(dtype=np.float32)
        residuals -= residuals.mean()
        b_opt = estimate_optimal_block_length(residuals)
        all_paths = []
        for seed in range(ENSEMBLE_SEEDS):
            rng = np.random.default_rng(GLOBAL_SEED + seed)
            sims = run_monte_carlo_paths(residuals, SIMS_PER_SEED, rng, base_mean,
                                         horizon_days, block_length=b_opt)
            all_paths.append(sims)
        paths = np.vstack(all_paths)
        medoid = compute_medoid_path(paths)
        forecast_ret = np.log(medoid[-1] / medoid[0])
        preds.append(forecast_ret)
        acts.append(agg_returns.iloc[i + 1])
    preds, acts = np.array(preds), np.array(acts)
    dir_acc = np.mean(np.sign(preds) == np.sign(acts))
    return dir_acc, len(preds)

# ==========================================================
# Stats + Plot
# ==========================================================
def compute_forecast_stats_from_path(path, start_cap, last_date):
    norm = path / path[0]
    idx = pd.date_range(start=last_date, periods=len(norm) + 1, freq="B")
    price = pd.Series(norm, index=idx[:-1]) * start_cap
    rets = np.log(price / price.shift(1)).dropna()
    return {
        "CAGR": annualized_return_daily(rets),
        "Volatility": annualized_vol_daily(rets),
        "Sharpe": annualized_sharpe_daily(rets),
        "Max Drawdown": max_drawdown_from_rets(rets),
    }

# ==========================================================
# Plot Forecasts (Main + Benchmark)
# ==========================================================
def plot_forecasts(port_rets, start_cap, central, paths, bench_central=None, bench_paths=None):
    port_cum = np.exp(port_rets.cumsum()) * start_cap
    last = port_cum.index[-1]
    dates = pd.date_range(start=last, periods=len(central), freq="B")

    terminal_vals = paths[:, -1]
    low_cut, high_cut = np.percentile(terminal_vals, [16, 84])
    mask = (terminal_vals >= low_cut) & (terminal_vals <= high_cut)
    filtered = paths[mask]

    # ---- Plot 1 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, color="black", lw=2, label="Portfolio Backtest")
    for sim in filtered[:100]:
        ax.plot(dates, port_cum.iloc[-1] * sim / sim[0], color="gray", alpha=0.05)
    ax.plot(dates, port_cum.iloc[-1] * central / central[0],
            color="blue", lw=2, label="Forecast")
    if bench_central is not None:
        ax.plot(dates, port_cum.iloc[-1] * bench_central / bench_central[0],
                color="orange", lw=2.5, label="Benchmark Forecast")
    ax.legend(); ax.set_title("Forecast"); ax.set_ylabel("Portfolio Value ($)")
    st.pyplot(fig)

    # ---- Plot 2 ----
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for sim in filtered[:100]:
        ax2.plot(dates, start_cap * sim / sim[0], color="gray", alpha=0.05)
    ax2.plot(dates, start_cap * central / central[0],
             color="blue", lw=2, label="Forecast")
    if bench_central is not None:
        ax2.plot(dates, start_cap * bench_central / bench_central[0],
                 color="orange", lw=2.5, label="Benchmark Forecast")
    ax2.set_title("Forecast (Horizon View)")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()
    st.pyplot(fig2)

# ==========================================================
# Streamlit App
# ==========================================================
def main():
    st.title("Portfolio Forecasting Tool")
    tickers = st.text_input("Tickers","VTI,AGG")
    weights_str = st.text_input("Weights","0.6,0.4")

    bench_tickers = st.text_input("Benchmark Tickers (optional)","SPY")
    bench_weights_str = st.text_input("Benchmark Weights","1.0")

    start_cap = st.number_input("Starting Value ($)",1000.0,1_000_000.0,10_000.0,1000.0)
    forecast_years = st.selectbox("Forecast Horizon (Years)", list(range(1,21)), index=0)
    enable_oos = st.selectbox("Out-of-sample Testing",["No","Yes"],index=0)
    div_mode = st.selectbox("Reinvest Dividends", ["No", "Yes"], index=1)
    backtest_start = st.date_input("Backtest Start Date",
        value=datetime.date(2000,1,1),
        min_value=datetime.date(1924,1,1),
        max_value=datetime.date.today()
    )

    if st.button("Run"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_daily(tickers, backtest_start.strftime("%Y-%m-%d"), include_dividends=(div_mode == "Yes"))
            port_rets = portfolio_log_returns_daily(prices, weights)

            mu, sigma = port_rets.mean(), port_rets.std(ddof=0)
            base_mean = mu - 0.5*sigma**2
            residuals = (port_rets-mu).to_numpy(dtype=np.float32); residuals -= residuals.mean()
            b_opt = estimate_optimal_block_length(residuals)
            total_days = 20*252

            all_paths=[]
            for i in range(ENSEMBLE_SEEDS):
                rng=np.random.default_rng(GLOBAL_SEED+i)
                sims=run_monte_carlo_paths(residuals,SIMS_PER_SEED,rng,base_mean,total_days,b_opt)
                all_paths.append(sims)
            paths_full=np.vstack(all_paths)
            medoid_full=compute_medoid_path(paths_full)
            forecast_days=forecast_years*252
            paths=paths_full[:,:forecast_days]; final=medoid_full[:forecast_days]

            bench_central=None
            if bench_tickers.strip():
                try:
                    bench_weights = to_weights([float(x) for x in bench_weights_str.split(",")])
                    bench_list = [t.strip() for t in bench_tickers.split(",") if t.strip()]
                    bench_prices = fetch_prices_daily(bench_list, backtest_start.strftime("%Y-%m-%d"), include_dividends=(div_mode == "Yes"))
                    bench_rets = portfolio_log_returns_daily(bench_prices, bench_weights)
                    bench_mu, bench_sigma = bench_rets.mean(), bench_rets.std(ddof=0)
                    bench_base_mean = bench_mu - 0.5*bench_sigma**2
                    bench_resid = (bench_rets - bench_mu).to_numpy(dtype=np.float32); bench_resid -= bench_resid.mean()
                    bench_b = estimate_optimal_block_length(bench_resid)
                    bench_paths=[]
                    for i in range(ENSEMBLE_SEEDS):
                        rng=np.random.default_rng(GLOBAL_SEED+i)
                        sims=run_monte_carlo_paths(bench_resid,SIMS_PER_SEED,rng,bench_base_mean,total_days,bench_b)
                        bench_paths.append(sims)
                    bench_full=np.vstack(bench_paths)
                    bench_medoid=compute_medoid_path(bench_full)
                    bench_central=bench_medoid[:forecast_days]
                except Exception as e:
                    st.warning(f"Benchmark forecast failed: {e}")

            plot_forecasts(port_rets,start_cap,final,paths,bench_central=bench_central)

            if enable_oos=="Yes":
                w_acc,w_n=compute_oos_directional_accuracy_walkforward(prices,weights,"W",5)
                m_acc,m_n=compute_oos_directional_accuracy_walkforward(prices,weights,"M",21)
                q_acc,q_n=compute_oos_directional_accuracy_walkforward(prices,weights,"Q",63)
                s_acc,s_n=compute_oos_directional_accuracy_walkforward(prices,weights,"2Q",126)
                a_acc,a_n=compute_oos_directional_accuracy_walkforward(prices,weights,"Y",252)
                st.write("Out-of-Sample Directional Accuracy:")
                st.write(f"Weekly: {w_acc:.2%} | Monthly: {m_acc:.2%} | Quarterly: {q_acc:.2%} | Semiannual: {s_acc:.2%} | Annual: {a_acc:.2%}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__=="__main__":
    main()