import sys
import warnings
import random
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ==========================================================
#  Global Configuration
# ==========================================================
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 8        # reduced for speed and convergence
SIMS_PER_SEED = 1000
FORECAST_YEARS = 1
P_STATIONARY = 0.1666     # expected block length ≈ 6 months

# ==========================================================
#  Helper Functions
# ==========================================================
def to_weights(raw: List[float]) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to positive.")
    return arr / s


def annualized_return_monthly(m):
    m = m.dropna()
    if m.empty:
        return np.nan
    compounded = np.exp(m.sum())
    years = len(m) / 12.0
    return compounded ** (1 / years) - 1 if years > 0 else np.nan


def annualized_vol_monthly(m):
    m = m.dropna()
    return m.std(ddof=0) * np.sqrt(12) if len(m) > 1 else np.nan


def annualized_sharpe_monthly(m, rf_monthly=0.0):
    m = m.dropna()
    if m.empty:
        return np.nan
    excess = m - rf_monthly
    mu, sigma = excess.mean(), excess.std(ddof=0)
    return (mu / sigma) * np.sqrt(12) if sigma and sigma > 0 else np.nan


def max_drawdown_from_rets(returns):
    cum = np.exp(returns.cumsum())
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).min()


def compute_current_drawdown(returns):
    cum = np.exp(returns.cumsum())
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).astype(np.float32)


def realized_vol(returns, window):
    return returns.rolling(window).std(ddof=0)

# ==========================================================
#  Data Fetch
# ==========================================================
def fetch_prices_monthly(tickers, start=DEFAULT_START):
    data = yf.download(tickers, start=start, interval="1mo",
                       auto_adjust=False, progress=False, threads=False)
    if data.empty:
        raise ValueError("No price data returned.")
    if isinstance(data.columns, pd.MultiIndex):
        for f in ["Adj Close", "Close"]:
            if f in data.columns.get_level_values(0):
                close = data[f].copy()
                break
        else:
            raise ValueError("No valid price field.")
    else:
        close = data.copy()
    close = close.ffill().astype(np.float32)
    df = pd.DataFrame(close)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    return df.dropna(how="all")


def fetch_macro_features(start=DEFAULT_START):
    # Keep only most relevant indicators (VIX, YC spread)
    tickers = ["^VIX", "^TNX", "^IRX"]
    data = yf.download(tickers, start=start, interval="1mo",
                       progress=False, threads=False)
    close = data["Close"].ffill().astype(np.float32)
    df = pd.DataFrame(index=close.index)
    df["VIX"] = close["^VIX"]
    df["YC_Spread"] = close["^TNX"] - close["^IRX"]
    return df

# ==========================================================
#  Portfolio Construction
# ==========================================================
def portfolio_log_returns_monthly(prices, weights, rebalance="M"):
    prices = prices.ffill().dropna().astype(np.float64)
    weights = np.array(weights, dtype=np.float64)
    rets = np.log(prices / prices.shift(1)).dropna()
    if len(weights) != rets.shape[1]:
        raise ValueError("Weight count mismatch.")

    # Monthly rebalance (fixed weights each month)
    port_rets = (rets * weights).sum(axis=1)
    return port_rets.astype(np.float32)

# ==========================================================
#  Feature Engineering
# ==========================================================
def build_features(returns):
    df = pd.DataFrame(index=returns.index)
    df["mom_3m"] = returns.rolling(3).sum()
    df["mom_6m"] = returns.rolling(6).sum()
    df["mom_12m"] = returns.rolling(12).sum()
    df["vol_3m"] = realized_vol(returns, 3)
    df["vol_6m"] = realized_vol(returns, 6)
    df["vol_12m"] = realized_vol(returns, 12)
    df["dd_state"] = compute_current_drawdown(returns)

    macro = fetch_macro_features()
    df = df.join(macro, how="left").ffill()

    # Drop early NaNs after rolling windows
    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index()])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ==========================================================
#  Stationary Bootstrap
# ==========================================================
def stationary_bootstrap_residuals(residuals, size, p=P_STATIONARY, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(residuals)
    out = np.empty(size, dtype=np.float32)
    idx = rng.integers(0, n)
    for t in range(size):
        out[t] = residuals[idx]
        if rng.random() < p:
            idx = rng.integers(0, n)
        else:
            idx = (idx + 1) % n
    return out

# ==========================================================
#  Monte Carlo Simulation
# ==========================================================
def run_monte_carlo_paths(model, X_base, residuals, sims_per_seed, rng):
    horizon = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon), dtype=np.float32)
    state = np.repeat(X_base.iloc[[-1]].values, sims_per_seed, axis=0).astype(np.float32)

    for t in range(horizon):
        mu_t = model.predict(state)
        eps = stationary_bootstrap_residuals(residuals, sims_per_seed, p=P_STATIONARY, rng=rng)
        log_paths[:, t] = (log_paths[:, t - 1] if t > 0 else 0) + mu_t + eps

        # Evolve state (momentum, vol, drawdown) dynamically
        df_temp = pd.DataFrame(log_paths[:, :t + 1])
        inc = df_temp.diff(axis=1)
        mom_3m = df_temp.diff(3, axis=1).iloc[:, -1].fillna(0).values
        mom_6m = df_temp.diff(6, axis=1).iloc[:, -1].fillna(0).values
        mom_12m = df_temp.diff(12, axis=1).iloc[:, -1].fillna(0).values
        vol_3m = inc.iloc[:, -3:].std(axis=1, ddof=0).values if t >= 2 else np.zeros(sims_per_seed)
        vol_6m = inc.iloc[:, -6:].std(axis=1, ddof=0).values if t >= 5 else np.zeros(sims_per_seed)
        vol_12m = inc.iloc[:, -12:].std(axis=1, ddof=0).values if t >= 11 else np.zeros(sims_per_seed)
        cum_vals = np.exp(df_temp)
        dd_state = cum_vals.values[:, -1] / np.maximum.accumulate(cum_vals.values, axis=1)[:, -1] - 1

        # Assign back into state array
        for name, vals in zip(
            ["mom_3m", "mom_6m", "mom_12m", "vol_3m", "vol_6m", "vol_12m", "dd_state"],
            [mom_3m, mom_6m, mom_12m, vol_3m, vol_6m, vol_12m, dd_state]
        ):
            if name in X_base.columns:
                col_idx = list(X_base.columns).index(name)
                state[:, col_idx] = vals

        # Keep macros constant with small drift
        for macro_feat in ("VIX", "YC_Spread"):
            if macro_feat in X_base.columns:
                col_idx = list(X_base.columns).index(macro_feat)
                state[:, col_idx] *= 0.99 + rng.normal(0, 0.01)

    return np.exp(log_paths - log_paths[:, [0]], dtype=np.float32)

# ==========================================================
#  Medoid Path (Ensemble Consensus)
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
#  Stats
# ==========================================================
def compute_forecast_stats_from_path(path, start_cap, last_date):
    norm = path / path[0]
    idx = pd.date_range(start=last_date, periods=len(norm) + 1, freq="M")
    price = pd.Series(norm, index=idx[:-1]) * start_cap
    rets = np.log(price / price.shift(1)).dropna()
    return {
        "CAGR": annualized_return_monthly(rets),
        "Volatility": annualized_vol_monthly(rets),
        "Sharpe": annualized_sharpe_monthly(rets),
        "Max Drawdown": max_drawdown_from_rets(rets),
    }

# ==========================================================
#  SHAP Attribution
# ==========================================================
def plot_feature_attributions(model, X, medoid_states):
    expl = shap.TreeExplainer(model)
    shap_hist = np.abs(expl.shap_values(X)).mean(axis=0)
    shap_fore_all = []
    for s in medoid_states:
        val = np.abs(expl.shap_values(pd.DataFrame([s], columns=X.columns)))
        shap_fore_all.append(val.reshape(-1))
    shap_fore = np.mean(shap_fore_all, axis=0)
    feats = X.columns
    pos = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pos - 0.2, shap_hist / shap_hist.sum() * 100, width=0.4, label="Backtest (avg)")
    ax.bar(pos + 0.2, shap_fore / shap_fore.sum() * 100, width=0.4, label="Forecast (medoid)")
    ax.set_xticks(pos)
    ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.set_ylabel("% contribution to explained variance")
    ax.set_title("Feature Contributions (Backtest vs Forecast)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================================
#  Plot Forecast Paths
# ==========================================================
def plot_forecasts(port_rets, start_cap, central):
    port_cum = np.exp(port_rets.cumsum()) * start_cap
    last = port_cum.index[-1]
    fore = port_cum.iloc[-1] * (central / central[0])
    dates = pd.date_range(start=last, periods=len(central), freq="M")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, label="Portfolio Backtest")
    ax.plot([last, *dates], [port_cum.iloc[-1], *fore],
            label="Forecast (Medoid Path)", lw=2)
    ax.legend()
    st.pyplot(fig)

# ==========================================================
#  Main Streamlit App
# ==========================================================
def main():
    st.title("State-Dependent Monte Carlo (SDMC) Portfolio Forecaster")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", 1000.0, 1000000.0, 10000.0, 1000.0)

    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_monthly(tickers, DEFAULT_START)
            port_rets = portfolio_log_returns_monthly(prices, weights)
            df = build_features(port_rets)
            Y = port_rets.loc[df.index].astype(np.float32)
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]

            # Train single LightGBM model (no Optuna)
            model = LGBMRegressor(
                n_estimators=800, learning_rate=0.02, max_depth=3,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=GLOBAL_SEED, n_jobs=1
            )
            model.fit(X, Y)

            # Compute and rescale residuals to match historical vol
            res = (Y.values - model.predict(X)).astype(np.float32)
            res = np.ascontiguousarray(res[~np.isnan(res)])
            hist_vol = annualized_vol_monthly(Y)
            res *= hist_vol / (np.std(res) * np.sqrt(12))

            # Monte Carlo ensemble
            all_paths = []
            bar = st.progress(0)
            txt = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(model, X, res, SIMS_PER_SEED, rng)
                all_paths.append(sims)
                bar.progress((i + 1) / ENSEMBLE_SEEDS)
                txt.text(f"Running simulations... {int((i + 1)/ENSEMBLE_SEEDS * 100)}%")
            bar.empty()
            txt.empty()

            paths = np.vstack(all_paths)
            final = compute_medoid_path(paths)
            stats = compute_forecast_stats_from_path(final, start_cap, port_rets.index[-1])
            back = {
                "CAGR": annualized_return_monthly(port_rets),
                "Volatility": annualized_vol_monthly(port_rets),
                "Sharpe": annualized_sharpe_monthly(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets),
            }

            st.subheader("Results")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Backtest**")
                for k, v in back.items():
                    st.metric(k, f"{v:.2%}" if "Sharpe" not in k else f"{v:.2f}")
            with c2:
                st.markdown("**Forecast (Medoid Path)**")
                for k, v in stats.items():
                    st.metric(k, f"{v:.2%}" if "Sharpe" not in k else f"{v:.2f}")

            st.metric("Forecasted Portfolio Value", f"${final[-1] * start_cap:,.2f}")
            plot_forecasts(port_rets, start_cap, final)

            medoid_states = np.repeat(X.iloc[[-1]].values, FORECAST_YEARS * 12, axis=0)
            plot_feature_attributions(model, X, medoid_states)

            terminal_vals = paths[:, -1]
            p10, p50, p90 = np.percentile(terminal_vals, [10, 50, 90])
            st.write(f"12-month terminal value percentiles: "
                     f"P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()





































