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
FORECAST_DAYS = 252

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
def fetch_prices_daily(tickers, start=DEFAULT_START):
    data = yf.download(
        tickers, start=start, interval="1d",
        auto_adjust=False, progress=False, threads=False
    )
    if data.empty:
        raise ValueError("No price data returned.")
    if isinstance(data.columns, pd.MultiIndex):
        for f in ["Adj Close", "Close"]:
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
# Monte Carlo Simulation (Uniform Sampling of Residuals)
# ==========================================================
def run_monte_carlo_paths(residuals, sims_per_seed, rng, base_mean, total_days):
    """Random draw of residuals (uniform) for each step independently."""
    eps = rng.choice(residuals, size=(sims_per_seed, total_days), replace=True)
    log_paths = np.cumsum(base_mean + eps, axis=1, dtype=np.float32)
    return np.exp(log_paths - log_paths[:, [0]])

# ==========================================================
# Medoid Path
# ==========================================================
def compute_medoid_path(paths):
    log_paths = np.log(paths)
    diffs = np.diff(log_paths, axis=1)
    normed = diffs - diffs.mean(axis=1, keepdims=True)
    mean_traj = normed.mean(axis=0)
    dots = np.sum(normed * mean_traj, axis=1)
    norms = np.linalg.norm(normed, axis=1) * np.linalg.norm(mean_traj)
    sims = dots / (norms + 1e-9)
    medoid_idx = np.argmax(sims)
    return paths[medoid_idx]

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


def plot_forecasts(port_rets, start_cap, central, paths):
    port_cum = np.exp(port_rets.cumsum()) * start_cap
    last = port_cum.index[-1]
    dates = pd.date_range(start=last, periods=len(central), freq="B")

    terminal_vals = paths[:, -1]
    low_cut, high_cut = np.percentile(terminal_vals, [5, 95])
    mask = (terminal_vals >= low_cut) & (terminal_vals <= high_cut)
    filtered_paths = paths[mask]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, color="black", lw=2, label="Portfolio Backtest")

    for sim in filtered_paths[:100]:
        ax.plot(dates, port_cum.iloc[-1] * sim / sim[0], color="gray", alpha=0.05)

    ax.plot(dates, port_cum.iloc[-1] * central / central[0],
            color="red", lw=2, label="Forecast (Medoid Path)")

    ax.set_title("Monte Carlo Forecast (Medoid Path + 5–95% Fan Lines)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    st.pyplot(fig)

    percentiles_end = np.percentile(terminal_vals, [5,95])
    df = pd.DataFrame({
        "Percentile": ["P5","P95"],
        "Terminal Value ($)": [f"${v * start_cap:,.2f}" for v in percentiles_end]
    })
    st.subheader("Forecasted Terminal Portfolio Values")
    st.table(df)

# ==========================================================
# Rebalancing Logic (no-op, unchanged)
# ==========================================================
def apply_rebalance_snapback(port_paths, rebalance_freq, weights):
    n_sims, total_days = port_paths.shape
    dates = pd.date_range(start=0, periods=total_days, freq="B")
    df_dummy = pd.Series(0, index=dates)
    freq_map = {
        "Daily": "B",
        "Weekly": "W-FRI",
        "Monthly": "M",
        "Quarterly": "Q",
        "Semiannually": "2Q",
        "Annually": "A"
    }
    rb = freq_map.get(rebalance_freq, "M")
    rebalance_dates = pd.Series(1, index=dates).resample(rb).first().dropna().index
    rebalance_mask = np.isin(dates, rebalance_dates).astype(int)
    return port_paths

# ==========================================================
# Streamlit App
# ==========================================================
def main():
    st.title("Portfolio Forecasting Tool")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", 1000.0, 1_000_000.0, 10_000.0, 1000.0)
    forecast_years = st.selectbox("Forecast Horizon (Years)", [1, 2, 3, 4, 5], index=0)
    rebalance_freq = st.selectbox("Rebalancing Frequency", ["Daily", "Weekly", "Monthly", "Quarterly", "Semiannually", "Annually"])

    if st.button("Run"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_daily(tickers, DEFAULT_START)
            port_rets = portfolio_log_returns_daily(prices, weights)

            mu = port_rets.mean()
            sigma = port_rets.std(ddof=0)
            base_mean = mu - 0.5 * sigma ** 2
            residuals = (port_rets - mu).to_numpy(dtype=np.float32)
            residuals -= residuals.mean()

            total_days = 5 * 252
            all_paths = []
            bar2 = st.progress(0)
            txt2 = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(residuals, SIMS_PER_SEED, rng, base_mean, total_days)
                all_paths.append(sims)
                bar2.progress((i + 1) / ENSEMBLE_SEEDS)
                txt2.text(f"Running forecasts... {int((i + 1) / ENSEMBLE_SEEDS * 100)}%")
            bar2.empty()
            txt2.empty()

            paths_full = np.vstack(all_paths)
            # Compute medoid path ONCE using the full 5-year horizon
            medoid_full = compute_medoid_path(paths_full)

            # Slice forecast horizon without recomputing medoid
            forecast_days = forecast_years * 252
            paths = paths_full[:, :forecast_days]
            paths = apply_rebalance_snapback(paths, rebalance_freq, weights)
            final = medoid_full[:forecast_days]

            stats = compute_forecast_stats_from_path(final, start_cap, port_rets.index[-1])
            back = {
                "CAGR": annualized_return_daily(port_rets),
                "Volatility": annualized_vol_daily(port_rets),
                "Sharpe": annualized_sharpe_daily(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets),
            }

            st.subheader("Results")
            c1, c2 = st.columns(2)
            for (col, data, label) in [(c1, back, "Backtest"), (c2, stats, "Forecast")]:
                with col:
                    st.markdown(f"**{label}**")
                    for k, v in data.items():
                        st.metric(k, f"{v:.2%}" if "Sharpe" not in k else f"{v:.2f}")

            st.metric("Forecasted Portfolio Value", f"${final[-1] * start_cap:,.2f}")
            plot_forecasts(port_rets, start_cap, final, paths)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()