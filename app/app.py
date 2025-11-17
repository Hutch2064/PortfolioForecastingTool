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

    # ============================================
    # Logic for price selection
    # - If include_dividends=True → Yahoo already adjusts "Close"
    # - If include_dividends=False → use raw "Close" or "Adj Close" fallback
    # ============================================
    if include_dividends:
        # Yahoo returns only "Close" column when auto_adjust=True (dividends + splits included)
        if isinstance(data, pd.DataFrame):
            if "Close" in data.columns:
                close = data["Close"].copy()
            else:
                # Handle rare MultiIndex case
                close = data.xs("Close", axis=1, level=0)
        else:
            raise ValueError("Unexpected data format for adjusted prices.")
    else:
        # Normal mode: unadjusted prices
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
# Most Likely Path
# ==========================================================
def compute_medoid_path(paths):
    log_paths = np.log(paths)
    mean_path = np.mean(log_paths, axis=0, dtype=np.float32)
    distances = np.linalg.norm(log_paths - mean_path, axis=1)
    idx = np.argmin(distances)
    return paths[idx]

# ==========================================================
# Forecast-Based Walk-Forward Directional Accuracy (Generalized)
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
# Plot Forecasts (with Optimal Horizon)
# ==========================================================
def plot_forecasts(port_rets, start_cap, central, paths, median_path):
    port_cum = np.exp(port_rets.cumsum()) * start_cap
    last = port_cum.index[-1]
    dates = pd.date_range(start=last, periods=len(central), freq="B")

    terminal_vals = paths[:, -1]
    low_cut, high_cut = np.percentile(terminal_vals, [16, 84])
    mask = (terminal_vals >= low_cut) & (terminal_vals <= high_cut)
    filtered = paths[mask]

    # ---- Best/Worst 1-year ----
    log_rets = np.log(central[1:] / central[:-1])
    roll_sum = pd.Series(log_rets).rolling(252).sum()
    best_idx, worst_idx = roll_sum.idxmax(), roll_sum.idxmin()
    if not np.isnan(best_idx) and not np.isnan(worst_idx):
        best_start, best_end = int(best_idx - 252), int(best_idx)
        worst_start, worst_end = int(worst_idx - 252), int(worst_idx)
        best_return = np.exp(roll_sum.max()) - 1
        worst_return = np.exp(roll_sum.min()) - 1
    else:
        best_start = best_end = worst_start = worst_end = None
        best_return = worst_return = 0.0

    # ---- Compute Optimal Horizon ----
    backtest_years = len(port_rets) / 252
    opt_years = 0.2 * backtest_years
    opt_days = int(opt_years * 252)

    # ---- Plot 1 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, color="black", lw=2, label="Portfolio Backtest")
    for sim in filtered[:100]:
        ax.plot(dates, port_cum.iloc[-1] * sim / sim[0], color="gray", alpha=0.05)

    ax.plot(dates[:opt_days], port_cum.iloc[-1] * central[:opt_days] / central[0],
            color="blue", lw=2, label=f"Forecast (Optimal ≤ {opt_years:.1f} yrs)")
    ax.plot(dates[opt_days:], port_cum.iloc[-1] * central[opt_days:] / central[0],
            color="#6fa8dc", lw=2, label="Beyond Optimal Horizon")
   
    # ---- Median Path (Plot 1) ----
    median_norm = median_path / median_path[0]
    ax.plot(dates, port_cum.iloc[-1] * median_norm,
            color="orange", lw=2, label="Median Path")

    if best_start is not None:
        ax.plot(dates[best_start:best_end],
                port_cum.iloc[-1] * central[best_start:best_end] / central[0],
                color="limegreen", lw=3, label=f"Best 1-Year Period ~ {best_return*100:.1f}%")
    if worst_start is not None:
        ax.plot(dates[worst_start:worst_end],
                port_cum.iloc[-1] * central[worst_start:worst_end] / central[0],
                color="red", lw=3, label=f"Worst 1-Year Period ~ {worst_return*100:.1f}%")
    ax.legend(); ax.set_title("Forecast"); ax.set_ylabel("Portfolio Value ($)")
    st.pyplot(fig)

    # ---- Plot 2 ----
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for sim in filtered[:100]:
        ax2.plot(dates, start_cap * sim / sim[0], color="gray", alpha=0.05)

    ax2.plot(dates[:opt_days], start_cap * central[:opt_days] / central[0],
             color="blue", lw=2, label=f"Forecast (Optimal ≤ {opt_years:.1f} yrs)")
    ax2.plot(dates[opt_days:], start_cap * central[opt_days:] / central[0],
             color="#6fa8dc", lw=2, label="Beyond Optimal Horizon")

    # ---- Median Path (Plot 2) ----
    median_norm = median_path / median_path[0]
    ax2.plot(dates, start_cap * median_norm,
             color="orange", lw=2, label="Median Path")

    if best_start is not None:
        ax2.plot(dates[best_start:best_end],
                 start_cap * central[best_start:best_end] / central[0],
                 color="limegreen", lw=3,
                 label=f"Best 1-Year Period ~ {best_return*100:.1f}%")
    if worst_start is not None:
        ax2.plot(dates[worst_start:worst_end],
                 start_cap * central[worst_start:worst_end] / central[0],
                 color="red", lw=3,
                 label=f"Worst 1-Year Period ~ {worst_return*100:.1f}%")
    ax2.set_title("Forecast (Horizon View)")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()
    st.pyplot(fig2)

    # ---- Percentile Table ----
    terminal_vals = paths[:, -1] * start_cap
    percentiles = [5, 25, 50, 75, 95]
    p_vals = np.percentile(terminal_vals, percentiles)
    cvar_cutoff = np.percentile(terminal_vals, 5)
    cvar = terminal_vals[terminal_vals <= cvar_cutoff].mean()
    p_rets = (p_vals / start_cap) - 1
    rows = [("CVaR", f"${cvar:,.0f}", f"{(cvar/start_cap-1)*100:.2f}%")] + [
        (f"P{p}", f"${v:,.0f}", f"{r*100:.2f}%") for p, v, r in zip(percentiles, p_vals, p_rets)
    ]
    html = """
    <style>
    table.custom, table.custom tr, table.custom th, table.custom td {
        border:none!important;border-collapse:collapse!important;
        background:transparent!important;box-shadow:none!important;
    }
    table.custom th,table.custom td {
        color:white!important;font-family:'Helvetica Neue',sans-serif!important;
        font-size:15px!important;padding:3px 10px!important;text-align:left!important;
    }
    </style>
    <table class="custom"><tr><th>Percentile</th><th>Terminal Value ($)</th><th>Total Return (%)</th></tr>
    """ + "".join(
        [f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows]
    ) + "</table>"
    st.subheader("Forecast Distribution")
    st.markdown(html, unsafe_allow_html=True)

# ==========================================================
# Streamlit App
# ==========================================================
def main():
    st.title("Portfolio Forecasting Tool")
    tickers = st.text_input("Tickers","VTI,AGG")
    weights_str = st.text_input("Weights","0.6,0.4")
    start_cap = st.number_input("Starting Value ($)",1000.0,1_000_000.0,10_000.0,1000.0)
    forecast_years = st.selectbox("Forecast Horizon (Years)", list(range(1,21)), index=0)
    enable_oos = st.selectbox("Out-of-sample Testing",["No","Yes"],index=0)
    div_mode = st.selectbox("Reinvest Dividends", ["No", "Yes"], index=1)
    backtest_start = st.date_input("Backtest Start Date",
        value=datetime.date(2000,1,1),
        min_value=datetime.date(1924,1,1),
        max_value=datetime.date.today()
    )

    col_run, col_val = st.columns([1,3])
    with col_run: run_pressed = st.button("Run")

    if st.session_state.get("last_tickers")!=st.session_state.get("curr_tickers",""):
        st.session_state.pop("forecast_val",None)
    if st.session_state.get("last_weights")!=st.session_state.get("curr_weights",""):
        st.session_state.pop("forecast_val",None)
    st.session_state["curr_tickers"]=tickers
    st.session_state["curr_weights"]=weights_str
    st.session_state["last_tickers"]=tickers
    st.session_state["last_weights"]=weights_str

    if run_pressed:
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
            all_paths=[]; bar2=st.progress(0); txt2=st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng=np.random.default_rng(GLOBAL_SEED+i)
                sims=run_monte_carlo_paths(residuals,SIMS_PER_SEED,rng,base_mean,total_days,b_opt)
                all_paths.append(sims); bar2.progress((i+1)/ENSEMBLE_SEEDS)
                txt2.text(f"Running forecasts... {int((i+1)/ENSEMBLE_SEEDS*100)}%")
            bar2.empty(); txt2.empty()
            paths_full=np.vstack(all_paths)
            medoid_full=compute_medoid_path(paths_full)
            forecast_days=forecast_years*252
            paths=paths_full[:,:forecast_days]; final=medoid_full[:forecast_days]
            median_path = np.median(paths, axis=0).astype(np.float32)
            st.session_state["forecast_val"]=final[-1]*start_cap
            stats=compute_forecast_stats_from_path(final,start_cap,port_rets.index[-1])
            back={
                "CAGR":annualized_return_daily(port_rets),
                "Volatility":annualized_vol_daily(port_rets),
                "Sharpe":annualized_sharpe_daily(port_rets),
                "Max Drawdown":max_drawdown_from_rets(port_rets),
            }
            st.markdown(
                f"<p style='color:white;font-size:27px;font-weight:bold;margin-top:17px;'>"
                f"Forecasted Portfolio Value ~ <span style='font-weight:300;'>${final[-1]*start_cap:,.2f}</span></p>",
                unsafe_allow_html=True)
            rows=[("CAGR",f"{back['CAGR']:.2%}",f"{stats['CAGR']:.2%}"),
                  ("Volatility",f"{back['Volatility']:.2%}",f"{stats['Volatility']:.2%}"),
                  ("Sharpe",f"{back['Sharpe']:.2f}",f"{stats['Sharpe']:.2f}"),
                  ("Max Drawdown",f"{back['Max Drawdown']:.2%}",f"{stats['Max Drawdown']:.2%}")]
            html=("""
            <style>table.results,table.results tr,table.results th,table.results td{
            border:none!important;border-collapse:collapse!important;background:transparent!important;}
            table.results th,table.results td{color:white!important;font-family:'Helvetica Neue',sans-serif!important;
            font-size:15px!important;padding:3px 10px!important;text-align:left!important;}
            </style><table class='results'><tr><th>Metric</th><th>Backtest</th><th>Forecast</th></tr>"""+
            "".join([f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a,b,c in rows])+"</table>")
            st.subheader("Performance Comparison")
            st.markdown(html, unsafe_allow_html=True)
            plot_forecasts(port_rets,start_cap,final,paths,median_path)

            if enable_oos=="Yes":
                w_acc,w_n=compute_oos_directional_accuracy_walkforward(prices,weights,"W",5)
                m_acc,m_n=compute_oos_directional_accuracy_walkforward(prices,weights,"M",21)
                q_acc,q_n=compute_oos_directional_accuracy_walkforward(prices,weights,"Q",63)
                s_acc,s_n=compute_oos_directional_accuracy_walkforward(prices,weights,"2Q",126)
                a_acc,a_n=compute_oos_directional_accuracy_walkforward(prices,weights,"Y",252)

                oos_html=f"""
                <h3 style='color:white;font-size:22px;font-weight:700;margin-top:25px;'>
                    Out-Of-Sample Testing Results
                </h3>
                <table class='results'>
                    <tr><th>OOS Horizon</th><th>Directional Accuracy</th><th>Sample Size</th></tr>
                    <tr><td>Weekly</td><td>{w_acc:.2%}</td><td>{w_n}</td></tr>
                    <tr><td>Monthly</td><td>{m_acc:.2%}</td><td>{m_n}</td></tr>
                    <tr><td>Quarterly</td><td>{q_acc:.2%}</td><td>{q_n}</td></tr>
                    <tr><td>Semiannual</td><td>{s_acc:.2%}</td><td>{s_n}</td></tr>
                    <tr><td>Annual</td><td>{a_acc:.2%}</td><td>{a_n}</td></tr>
                </table>
                """
                st.markdown(oos_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__=="__main__":
    main()
