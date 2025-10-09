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
ENSEMBLE_SEEDS = 8
SIMS_PER_SEED = 2000
FORECAST_DAYS = 252  # 1 trading year
P_STATIONARY = 0.1   # Probability of new block in bootstrap

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
    if r.empty: return np.nan
    compounded = np.exp(r.sum())
    years = len(r) / 252.0
    return compounded ** (1 / years) - 1 if years > 0 else np.nan


def annualized_vol_daily(r):
    r = r.dropna()
    return r.std(ddof=0) * np.sqrt(252) if len(r) > 1 else np.nan


def annualized_sharpe_daily(r, rf_daily=0.0):
    r = r.dropna()
    if r.empty: return np.nan
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
    data = yf.download(tickers, start=start, interval="1d",
                       auto_adjust=False, progress=False, threads=False)
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
# Stationary Bootstrap (Vectorized)
# ==========================================================
def stationary_bootstrap_residuals(residuals, size, sims, p=P_STATIONARY, rng=None):
    """Vectorized stationary bootstrap for many paths."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(residuals)
    idx = rng.integers(0, n, size=(sims, size))
    # geometric block continuation
    cont = rng.random((sims, size)) > p
    for t in range(1, size):
        idx[:, t] = np.where(cont[:, t], (idx[:, t - 1] + 1) % n, idx[:, t])
    return residuals[idx]

# ==========================================================
# Monte Carlo Simulation
# ==========================================================
def run_monte_carlo_paths(residuals, sims_per_seed, rng, base_mean):
    """Fast simulation with constant drift and empirical residuals."""
    eps = stationary_bootstrap_residuals(residuals, FORECAST_DAYS, sims_per_seed, rng=rng)
    drift = base_mean
    log_paths = np.cumsum(drift + eps, axis=1, dtype=np.float32)
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


def plot_forecasts(port_rets, start_cap, central):
    port_cum = np.exp(port_rets.cumsum()) * start_cap
    last = port_cum.index[-1]
    fore = port_cum.iloc[-1] * (central / central[0])
    dates = pd.date_range(start=last, periods=len(central), freq="B")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, label="Portfolio Backtest")
    ax.plot([last, *dates], [port_cum.iloc[-1], *fore],
            label="Forecast (Medoid Path)", lw=2)
    ax.legend()
    st.pyplot(fig)

# ==========================================================
# Streamlit App
# ==========================================================
def main():
    st.title("Pure Monte Carlo Forecast (Constant Drift + Empirical Residuals)")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", 1000.0, 1000000.0, 10000.0, 1000.0)

    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_daily(tickers, DEFAULT_START)
            port_rets = portfolio_log_returns_daily(prices, weights)

            # Core stats
            back_cagr = annualized_return_daily(port_rets)
            base_mean = np.log(1.0 + back_cagr) / 252.0
            residuals = port_rets - port_rets.mean()
            residuals = residuals.to_numpy(dtype=np.float32)
            residuals -= residuals.mean()

            # Simulations
            all_paths = []
            bar = st.progress(0)
            txt = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(residuals, SIMS_PER_SEED, rng, base_mean)
                all_paths.append(sims)
                bar.progress((i + 1) / ENSEMBLE_SEEDS)
                txt.text(f"Running forecasts... {int((i + 1) / ENSEMBLE_SEEDS * 100)}%")
            bar.empty()
            txt.empty()

            paths = np.vstack(all_paths)
            final = compute_medoid_path(paths)

            stats = compute_forecast_stats_from_path(final, start_cap, port_rets.index[-1])
            back = {
                "CAGR": annualized_return_daily(port_rets),
                "Volatility": annualized_vol_daily(port_rets),
                "Sharpe": annualized_sharpe_daily(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets),
            }

            st.subheader("Results")
            c1, c2 = st.columns(2)
            for (col, data, label) in [(c1, back, "Backtest"), (c2, stats, "Forecast (Medoid)")]:
                with col:
                    st.markdown(f"**{label}**")
                    for k, v in data.items():
                        st.metric(k, f"{v:.2%}" if "Sharpe" not in k else f"{v:.2f}")

            st.metric("Forecasted Portfolio Value", f"${final[-1] * start_cap:,.2f}")
            plot_forecasts(port_rets, start_cap, final)

            # Distribution summary
            terminal_vals = paths[:, -1]
            p10, p50, p90 = np.percentile(terminal_vals, [10, 50, 90])
            st.write(f"12-month terminal value percentiles: "
                     f"P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()







































