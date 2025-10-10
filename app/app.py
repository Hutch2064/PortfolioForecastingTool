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
import optuna

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
FORECAST_DAYS = 252  # 1 trading year
DEFAULT_BLOCK = 21   # fallback if tuning disabled

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
    port_rets = np.dot(rets, weights)
    return pd.Series(port_rets, index=rets.index, dtype=np.float32)

# ==========================================================
# Stationary Bootstrap (Vectorized)
# ==========================================================
def stationary_bootstrap_residuals(residuals, size, sims, block_length, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(residuals)
    p = 1.0 / block_length
    idx = rng.integers(0, n, size=(sims, size))
    cont = rng.random((sims, size)) > p
    roll = (idx[:, :-1] + 1) % n
    rand_new = rng.integers(0, n, size=(sims, size - 1))
    idx[:, 1:] = np.where(cont[:, 1:], roll, rand_new)
    return residuals[idx]

# ==========================================================
# Monte Carlo Simulation
# ==========================================================
def run_monte_carlo_paths(residuals, sims_per_seed, rng, base_mean, block_length):
    eps = stationary_bootstrap_residuals(
        residuals, FORECAST_DAYS, sims_per_seed, block_length, rng=rng
    )
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
    sims = np.sum(normed * mean_traj, axis=1) / (np.linalg.norm(normed, axis=1) * np.linalg.norm(mean_traj) + 1e-9)
    return paths[np.argmax(sims)]

# ==========================================================
# OOS Evaluation (Directional Accuracy)
# ==========================================================
def evaluate_block_length(port_rets, block_length):
    years = port_rets.index.year.unique()
    if len(years) < 12:
        return -999, pd.DataFrame()

    results = []
    for i in range(10, len(years) - 1):
        train = port_rets[port_rets.index.year <= years[i - 1]]
        test = port_rets[port_rets.index.year == years[i]]
        mu, sigma = train.mean(), train.std(ddof=0)
        base_mean = mu - 0.5 * sigma ** 2
        residuals = (train - mu).to_numpy(dtype=np.float32)
        residuals -= residuals.mean()
        all_paths = []
        for s in range(ENSEMBLE_SEEDS):
            rng = np.random.default_rng(GLOBAL_SEED + s)
            all_paths.append(run_monte_carlo_paths(residuals, SIMS_PER_SEED, rng, base_mean, block_length))
        paths = np.vstack(all_paths)
        medoid = compute_medoid_path(paths)
        steps = min(len(test), FORECAST_DAYS)
        forecast_series = pd.Series(medoid[:steps], index=test.index[:steps])
        forecast_rets = np.log(forecast_series / forecast_series.shift(1)).dropna()
        actual_rets = test.iloc[:len(forecast_rets)]
        acc = np.mean(np.sign(forecast_rets.resample("M").sum()) == np.sign(actual_rets.resample("M").sum()))
        results.append((years[i], acc))
    df = pd.DataFrame(results, columns=["Year", "Directional_Accuracy"])
    return df["Directional_Accuracy"].mean(skipna=True), df

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
    low_cut, high_cut = np.percentile(terminal_vals, [1, 99])
    mask = (terminal_vals >= low_cut) & (terminal_vals <= high_cut)
    filtered_paths = paths[mask]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, color="black", lw=2, label="Portfolio Backtest")
    for sim in filtered_paths[:100]:
        ax.plot(dates, port_cum.iloc[-1] * sim / sim[0], color="gray", alpha=0.05)
    ax.plot(dates, port_cum.iloc[-1] * central / central[0], color="red", lw=2, label="Forecast (Medoid Path)")
    ax.set_title("Monte Carlo Forecast (Medoid Path + 1–99% Fan Lines)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    st.pyplot(fig)

# ==========================================================
# Streamlit App
# ==========================================================
def main():
    st.title("Portfolio Forecasting Tool")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", min_value=0.01, value=10000.0, step=100.0)
    run_oos = st.selectbox("Out-Of-Sample Testing", ["Yes", "No"])

    if st.button("Run"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_daily(tickers, DEFAULT_START)
            port_rets = portfolio_log_returns_daily(prices, weights)

            best_block = DEFAULT_BLOCK
            if run_oos == "Yes":
                st.write("Tuning block length…")
                bar = st.progress(0)
                txt = st.empty()
                total_trials = 25

                def objective(trial):
                    blk = trial.suggest_int("block_length", 5, 63)
                    acc, _ = evaluate_block_length(port_rets, blk)
                    bar.progress((trial.number + 1) / total_trials)
                    txt.text(f"Tuning... {(trial.number + 1) / total_trials:.0%}")
                    return -acc

                study = optuna.create_study(direction="minimize",
                                            sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
                study.optimize(objective, n_trials=total_trials, show_progress_bar=False)
                bar.empty(); txt.empty()
                best_block = study.best_params["block_length"]

            mu, sigma = port_rets.mean(), port_rets.std(ddof=0)
            base_mean = mu - 0.5 * sigma ** 2
            residuals = (port_rets - mu).to_numpy(dtype=np.float32)
            residuals -= residuals.mean()
            # Clip residuals (1–99%)
            low, high = np.percentile(residuals, [1, 99])
            residuals = np.clip(residuals, low, high)

            all_paths = []
            bar2 = st.progress(0); txt2 = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                all_paths.append(run_monte_carlo_paths(residuals, SIMS_PER_SEED, rng, base_mean, best_block))
                bar2.progress((i + 1) / ENSEMBLE_SEEDS)
            bar2.empty(); txt2.empty()
            paths = np.vstack(all_paths)
            final = compute_medoid_path(paths)

            # === Plot chart first ===
            plot_forecasts(port_rets, start_cap, final, paths)

            # === Stats ===
            stats = compute_forecast_stats_from_path(final, start_cap, port_rets.index[-1])
            back = {
                "CAGR": annualized_return_daily(port_rets),
                "Volatility": annualized_vol_daily(port_rets),
                "Sharpe": annualized_sharpe_daily(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets),
            }

            forecast_ending = final[-1] * start_cap
            forecast_total_return = (forecast_ending / start_cap) - 1

            # Horizontal format
            st.subheader("Performance Summary")
            cols = st.columns(len(back))
            for i, k in enumerate(back.keys()):
                cols[i].markdown(f"**{k}**")
            for i, (k, v) in enumerate(back.items()):
                cols[i].markdown(f"Backtest: {v:.2%}" if k != "Sharpe" else f"Backtest: {v:.2f}")
            for i, (k, v) in enumerate(stats.items()):
                cols[i].markdown(f"Forecast: {v:.2%}" if k != "Sharpe" else f"Forecast: {v:.2f}")

            # Forecast totals
            st.write(f"**Total Forecast Return:** {forecast_total_return:.2%}")
            st.write(f"**Forecast Ending Value:** ${forecast_ending:,.2f}")

            # True terminal values (5–95%)
            terminal_vals = paths[:, -1]
            percentiles_end = np.percentile(terminal_vals, [5, 95])
            df = pd.DataFrame({
                "Percentile": ["P5", "P95"],
                "Terminal Value ($)": [f"${v * start_cap:,.2f}" for v in percentiles_end]
            })
            st.subheader("Forecasted Terminal Portfolio Values (5–95%)")
            st.table(df)

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()