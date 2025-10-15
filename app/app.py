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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # <-- Added r2_score

warnings.filterwarnings("ignore")

# ==========================================================
# Config & Seeds
# ==========================================================
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 20
SIMS_PER_SEED = 3000
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
# Rolling OOS Metrics
# ==========================================================
def compute_oos_metrics(port_rets):
    """Compute rolling 5-year (60-month) OOS directional accuracy and normalized errors."""
    monthly = port_rets.resample("M").sum()
    if len(monthly) < 120:  # need at least 10 years (5 train + 5 test)
        return np.nan, np.nan, np.nan, np.nan, np.nan

    preds = []
    actuals = []
    for t in range(60, len(monthly) - 1):  # start after 5 years
        train = monthly.iloc[t-60:t]
        pred = train.mean()  # forecast next month as mean of prior 5 years
        preds.append(pred)
        actuals.append(monthly.iloc[t+1])
    preds = np.array(preds)
    actuals = np.array(actuals)

    dir_acc = np.mean(np.sign(preds) == np.sign(actuals))
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)  # <-- Added R² calculation
    monthly_vol = monthly.std(ddof=0)
    norm_mae = mae / monthly_vol if monthly_vol > 0 else np.nan
    norm_rmse = rmse / monthly_vol if monthly_vol > 0 else np.nan
    return dir_acc, norm_mae, norm_rmse, monthly_vol, r2

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

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for sim in filtered_paths[:100]:
        ax2.plot(dates, start_cap * sim / sim[0], color="gray", alpha=0.05)
    ax2.plot(dates, start_cap * central / central[0],
             color="red", lw=2, label="Forecast")
    ax2.set_title("Forecast (Horizon View)")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()
    st.pyplot(fig2)

    terminal_vals = paths[:, -1] * start_cap
    percentiles = [5, 25, 50, 75, 95]
    p_values = np.percentile(terminal_vals, percentiles)
    cvar_cutoff = np.percentile(terminal_vals, 5)
    cvar = terminal_vals[terminal_vals <= cvar_cutoff].mean()
    p_returns = (p_values / start_cap) - 1

    rows = [
        ("CVaR", f"${cvar:,.0f}", f"{(cvar / start_cap - 1) * 100:.2f}%")
    ] + [
        (f"P{p}", f"${v:,.0f}", f"{r * 100:.2f}%")
        for p, v, r in zip(percentiles, p_values, p_returns)
    ]

    html = """
    <style>
    table.custom, table.custom tr, table.custom th, table.custom td {
        border: none !important;
        border-collapse: collapse !important;
        border-spacing: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    table.custom th, table.custom td {
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif !important;
        font-size: 15px !important;
        padding: 3px 10px !important;
        text-align: left !important;
    }
    table.custom th { font-weight: 700 !important; }
    table.custom td { font-weight: 400 !important; }
    tr, td, th { border: none !important; border-bottom: none !important; }
    </style>
    <table class="custom">
        <tr><th>Percentile</th><th>Terminal Value ($)</th><th>Return (%)</th></tr>
    """
    for row in rows:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
    html += "</table>"

    st.subheader("Forecast Distribution")
    st.markdown(html, unsafe_allow_html=True)

# ==========================================================
# Streamlit App
# ==========================================================
def main():
    st.title("Portfolio Forecasting Tool")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", 1000.0, 1_000_000.0, 10_000.0, 1000.0)
    forecast_years = st.selectbox("Forecast Horizon (Years)", [1, 2, 3, 4, 5], index=0)
    enable_oos = st.selectbox("Out-Of-Sample Testing", ["No", "Yes"], index=0)
    backtest_start = st.date_input("Backtest Start Date",
        value=datetime.date(2000, 1, 1),
        min_value=datetime.date(1980, 1, 1),
        max_value=datetime.date.today()
    )

    col_run, col_val = st.columns([1, 3])
    with col_run:
        run_pressed = st.button("Run")

    if st.session_state.get("last_tickers") != st.session_state.get("curr_tickers", ""):
        st.session_state.pop("forecast_val", None)
    if st.session_state.get("last_weights") != st.session_state.get("curr_weights", ""):
        st.session_state.pop("forecast_val", None)

    st.session_state["curr_tickers"] = tickers
    st.session_state["curr_weights"] = weights_str
    st.session_state["last_tickers"] = tickers
    st.session_state["last_weights"] = weights_str

    if run_pressed:
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
                sims = run_monte_carlo_paths(residuals, SIMS_PER_SEED, rng, base_mean, total_days, block_length=b_opt)
                all_paths.append(sims)
                bar2.progress((i + 1) / ENSEMBLE_SEEDS)
                txt2.text(f"Running forecasts... {int((i + 1)/ENSEMBLE_SEEDS*100)}%")
            bar2.empty()
            txt2.empty()

            paths_full = np.vstack(all_paths)
            medoid_full = compute_medoid_path(paths_full)
            forecast_days = forecast_years * 252
            paths = paths_full[:, :forecast_days]
            final = medoid_full[:forecast_days]
            st.session_state["forecast_val"] = final[-1] * start_cap

            stats = compute_forecast_stats_from_path(final, start_cap, port_rets.index[-1])
            back = {
                "CAGR": annualized_return_daily(port_rets),
                "Volatility": annualized_vol_daily(port_rets),
                "Sharpe": annualized_sharpe_daily(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets),
            }

            st.markdown(
                f"<p style='color:white; font-size:27px; font-weight:bold; margin-top:17px;'>Forecasted Portfolio Value ~ <span style='font-weight:300;'>${final[-1] * start_cap:,.2f}</span></p>",
                unsafe_allow_html=True
            )

            rows = [
                ("CAGR", f"{back['CAGR']:.2%}", f"{stats['CAGR']:.2%}"),
                ("Volatility", f"{back['Volatility']:.2%}", f"{stats['Volatility']:.2%}"),
                ("Sharpe", f"{back['Sharpe']:.2f}", f"{stats['Sharpe']:.2f}"),
                ("Max Drawdown", f"{back['Max Drawdown']:.2%}", f"{stats['Max Drawdown']:.2%}")
            ]

            html = """
            <style>
            table.results, table.results tr, table.results th, table.results td {
                border: none !important;
                border-collapse: collapse !important;
                border-spacing: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
                outline: none !important;
            }
            table.results th, table.results td {
                color: white !important;
                font-family: 'Helvetica Neue', sans-serif !important;
                font-size: 15px !important;
                padding: 3px 10px !important;
                text-align: left !important;
            }
            table.results th { font-weight: 700 !important; }
            table.results td { font-weight: 400 !important; }
            tr, td, th { border: none !important; border-bottom: none !important; }
            </style>
            <table class="results">
                <tr><th>Metric</th><th>Backtest</th><th>Forecast</th></tr>
            """
            for row in rows:
                html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
            html += "</table>"

            st.subheader("Performance Comparison")
            st.markdown(html, unsafe_allow_html=True)
            plot_forecasts(port_rets, start_cap, final, paths)

            # Add OOS table with header if enabled
            if enable_oos == "Yes":
                dir_acc, norm_mae, norm_rmse, monthly_vol, r2 = compute_oos_metrics(port_rets)
                oos_html = f"""
                <h3 style='color:white; font-size:22px; font-weight:700; margin-top:25px;'>
                    Out-Of-Sample Testing Results
                </h3>
                <table class='results'>
                    <tr><th>OOS Metric</th><th>Value</th></tr>
                    <tr><td>Directional Accuracy (Monthly)</td><td>{dir_acc:.2%}</td></tr>
                    <tr><td>Normalized MAE (MAE / Monthly Vol)</td><td>{norm_mae:.3f}</td></tr>
                    <tr><td>Normalized RMSE (RMSE / Monthly Vol)</td><td>{norm_rmse:.3f}</td></tr>
                    <tr><td>R² (Goodness of Fit)</td><td>{r2:.3f}</td></tr>
                    <tr><td>Monthly Volatility</td><td>{monthly_vol:.4f}</td></tr>
                </table>
                """
                st.markdown(oos_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()