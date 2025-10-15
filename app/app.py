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
SIMS_PER_SEED = 10000
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
# Optimal Block Length Estimation (Volatility Clustering)
# ==========================================================
def estimate_optimal_block_length(residuals):
    """Estimate optimal block length from volatility autocorrelation."""
    res2 = residuals ** 2
    if len(res2) < 2:
        return 5
    rho1 = np.corrcoef(res2[:-1], res2[1:])[0, 1]
    T = len(res2)
    b_opt = 1.5 * ((T / (1 - rho1**2 + 1e-9)) ** (1/3))
    return int(np.clip(b_opt, 5, 100))

# ==========================================================
# Monte Carlo Simulation (Block Bootstrap Sampling)
# ==========================================================
def run_monte_carlo_paths(residuals, sims_per_seed, rng, base_mean, total_days, block_length):
    """Fixed-length block bootstrap preserving volatility clustering (vectorized)."""
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
# Mean-Like Path (Closest to Ensemble Mean in Log Space)
# ==========================================================
def compute_medoid_path(paths):
    """Select the simulated path closest to the ensemble mean trajectory (vectorized)."""
    log_paths = np.log(paths)
    mean_path = np.mean(log_paths, axis=0, dtype=np.float32)
    distances = np.linalg.norm(log_paths - mean_path, axis=1)
    idx = np.argmin(distances)
    return paths[idx]

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
    low_cut, high_cut = np.percentile(terminal_vals, [16, 84])
    mask = (terminal_vals >= low_cut) & (terminal_vals <= high_cut)
    filtered_paths = paths[mask]

    # ----------------------------
    # Full Historical + Forecast Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, color="black", lw=2, label="Portfolio Backtest")
    for sim in filtered_paths[:100]:
        ax.plot(dates, port_cum.iloc[-1] * sim / sim[0], color="gray", alpha=0.05)
    ax.plot(dates, port_cum.iloc[-1] * central / central[0],
            color="red", lw=2, label="Forecast")
    ax.set_title("Forecast")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # Forecast-Only (Zoomed, Reset to Starting Capital)
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for sim in filtered_paths[:100]:
        ax2.plot(dates, start_cap * sim / sim[0], color="gray", alpha=0.05)
    ax2.plot(dates, start_cap * central / central[0],
             color="red", lw=2, label="Forecast")
    ax2.set_title("Forecast (Horizon View, Reset to Starting Capital)")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()
    st.pyplot(fig2)

    # ----------------------------
    # Histogram of Terminal Portfolio Values (Final — P5 Left, P75 Right)
    # ----------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    terminal_vals = paths[:, -1] * start_cap

    # Key percentiles
    percentiles = [5, 25, 50, 75, 95]
    p_values = np.percentile(terminal_vals, percentiles)

    # Skew & kurtosis
    mean_val = np.mean(terminal_vals)
    std_val = np.std(terminal_vals)
    skew = np.mean(((terminal_vals - mean_val) / (std_val + 1e-12)) ** 3)
    kurt = np.mean(((terminal_vals - mean_val) / (std_val + 1e-12)) ** 4) - 3

    # Histogram
    counts, bins, patches = ax3.hist(
        terminal_vals, bins=60, color="lightgray", edgecolor="black", alpha=0.6
    )
    ax3.set_title("Distribution of Terminal Portfolio Values", fontsize=13)
    ax3.set_xlabel("Final Portfolio Value ($)", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)

    max_y = counts.max() if len(counts) else 1.0
    ax3.set_ylim(0, max_y * 1.25)

    # Colors
    colors = {5: "red", 25: "orange", 50: "blue", 75: "green", 95: "darkgreen"}

    # Axis range for spacing
    x_min, x_max = ax3.get_xlim()
    x_range = x_max - x_min

    for i, (p, v) in enumerate(zip(percentiles, p_values)):
        bin_idx = np.searchsorted(bins, v) - 1
        bin_idx = np.clip(bin_idx, 0, len(counts) - 1)
        y_val = counts[bin_idx]

        # Tick mark
        ax3.plot([v, v], [y_val - 0.02 * max_y, y_val + 0.02 * max_y],
                 color=colors[p], lw=3, solid_capstyle="round")

        # Default offsets
        side_offset = 0.02 * x_range
        y_offset = 0.04 * max_y
        ha_pos = "left"

        # Custom placement for P5 and P75
        if p == 5:
            side_offset = -0.02 * x_range  # shift left
            ha_pos = "right"
        elif p == 75:
            side_offset = 0.02 * x_range   # shift right
            ha_pos = "left"
        elif p in [25, 95]:
            side_offset *= (1 if i % 2 == 0 else -1)
            ha_pos = "left" if side_offset > 0 else "right"

        ax3.text(v + side_offset, y_val + y_offset,
                 f"P{p}  ${v:,.0f}",
                 ha=ha_pos, va="bottom",
                 color=colors[p], fontsize=10, fontweight="bold",
                 clip_on=False)

    # Legend (top-right corner)
    handles = [plt.Line2D([0], [0], color=colors[p], lw=3, label=f"P{p}") for p in percentiles]
    ax3.legend(handles=handles, title="Percentiles",
               loc="upper right", bbox_to_anchor=(1.0, 1.0),
               frameon=True, facecolor="white", framealpha=0.9)

    # Skew/Kurt box (fully bottom-right corner)
    ax3.text(
        0.99, 0.02,
        f"Skew: {skew:.2f}\nKurt: {kurt:.2f}",
        transform=ax3.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9)
    )

    plt.tight_layout()
    st.pyplot(fig3)

# ==========================================================
# Streamlit App
# ==========================================================
def main():
    st.title("Portfolio Forecasting Tool")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", 1000.0, 1_000_000.0, 10_000.0, 1000.0)
    forecast_years = st.selectbox("Forecast Horizon (Years)", [1, 2, 3, 4, 5], index=0)

    backtest_start = st.date_input(
        "Backtest Start Date",
        value=datetime.date(2000, 1, 1),
        min_value=datetime.date(1980, 1, 1),
        max_value=datetime.date.today()
    )

    if st.button("Run"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_daily(tickers, backtest_start.strftime("%Y-%m-%d"))
            port_rets = portfolio_log_returns_daily(prices, weights)

            mu = port_rets.mean()
            sigma = port_rets.std(ddof=0)
            base_mean = mu - 0.5 * sigma ** 2
            residuals = (port_rets - mu).to_numpy(dtype=np.float32)
            residuals -= residuals.mean()
            b_opt = estimate_optimal_block_length(residuals)

            total_days = 5 * 252
            all_paths = []
            bar2 = st.progress(0)
            txt2 = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(
                    residuals, SIMS_PER_SEED, rng, base_mean, total_days, block_length=b_opt
                )
                all_paths.append(sims)
                bar2.progress((i + 1) / ENSEMBLE_SEEDS)
                txt2.text(f"Running forecasts... {int((i + 1) / ENSEMBLE_SEEDS * 100)}%")
            bar2.empty()
            txt2.empty()

            paths_full = np.vstack(all_paths)
            medoid_full = compute_medoid_path(paths_full)
            forecast_days = forecast_years * 252
            paths = paths_full[:, :forecast_days]
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