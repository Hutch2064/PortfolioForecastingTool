import sys
import warnings
import random
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import shap
import streamlit as st
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ---------- Global Seed Fix ----------
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------- Config ----------
DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 12
SIMS_PER_SEED = 2000
FORECAST_DAYS = 252  # ~1 trading year
P_STATIONARY = 0.1  # Probability of new block for bootstrap

# ---------- Helpers ----------
def to_weights(raw: List[float]) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
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


def compute_current_drawdown(returns):
    cum = np.exp(returns.cumsum())
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).astype(np.float32)


def realized_vol(returns, window):
    return returns.rolling(window).std(ddof=0)

# ---------- Data Fetch ----------
def fetch_prices_daily(tickers, start=DEFAULT_START):
    data = yf.download(tickers, start=start, interval="1d", auto_adjust=False, progress=False, threads=False)
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
    df = pd.DataFrame(close)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    return df.dropna(how="all")


def fetch_macro_features_daily(start=DEFAULT_START):
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX", "^SKEW", "DX-Y.NYB"]
    data = yf.download(tickers, start=start, interval="1d", progress=False, threads=False)
    close = data["Close"].ffill().astype(np.float32)
    df = pd.DataFrame(index=close.index)
    df["VIX"] = close["^VIX"]
    df["MOVE"] = close["^MOVE"]
    df["YC_Spread"] = close["^TNX"] - close["^IRX"]
    df["SKEW"] = close["^SKEW"]
    df["DXY"] = close["DX-Y.NYB"]
    df = df.interpolate().ffill().bfill()
    return df

# ---------- Portfolio ----------
def portfolio_log_returns_daily(prices, weights):
    prices = prices.ffill().dropna().astype(np.float64)
    weights = np.array(weights, dtype=np.float64)
    if len(weights) != prices.shape[1]:
        raise ValueError("Weight count mismatch")
    rets = np.log(prices / prices.shift(1)).dropna()
    port_rets = (rets * weights).sum(axis=1)
    return port_rets.astype(np.float32)

# ---------- Features ----------
def build_features(returns):
    df = pd.DataFrame(index=returns.index)
    df["mom_21d"] = returns.rolling(21).sum()
    df["mom_63d"] = returns.rolling(63).sum()
    df["mom_126d"] = returns.rolling(126).sum()
    df["vol_21d"] = realized_vol(returns, 21)
    df["vol_63d"] = realized_vol(returns, 63)
    df["vol_126d"] = realized_vol(returns, 126)
    df["dd_state"] = compute_current_drawdown(returns)
    macro = fetch_macro_features_daily()
    df = df.join(macro, how="left").ffill()
    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index()])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ---------- Indicator Models ----------
def train_indicator_models(X, feats):
    models = {}
    for f in feats:
        if f not in X.columns:
            continue
        y = X[f].shift(-1)
        df = pd.concat([X, y.rename("target")], axis=1).dropna()
        if len(df) < 60:
            continue
        mdl = LGBMRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=GLOBAL_SEED, n_jobs=1
        )
        mdl.fit(df[X.columns], df["target"])
        models[f] = mdl
    return models

# ---------- Stationary Bootstrap ----------
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

# ---------- Monte Carlo Simulation ----------
def run_monte_carlo_paths(model, X_base, residuals, sims_per_seed, rng,
                          indicator_models=None, base_mean=None):
    horizon = FORECAST_DAYS
    log_paths = np.zeros((sims_per_seed, horizon), dtype=np.float32)
    state = np.repeat(X_base.iloc[[-1]].values, sims_per_seed, axis=0).astype(np.float32)

    for t in range(horizon):
        # Predict conditional volatility
        sigma_t = np.abs(model.predict(state))
        # Draw bootstrapped standardized residuals
        eps = stationary_bootstrap_residuals(residuals, sims_per_seed, p=P_STATIONARY, rng=rng)
        # Compute returns with fixed drift + evolving volatility
        r_t = base_mean + sigma_t * eps
        log_paths[:, t] = (log_paths[:, t - 1] if t > 0 else 0) + r_t

        # Evolve state features dynamically
        df_temp = pd.DataFrame(log_paths[:, :t + 1])
        inc = df_temp.diff(axis=1)
        mom_21 = df_temp.diff(21, axis=1).iloc[:, -1].fillna(0).values
        mom_63 = df_temp.diff(63, axis=1).iloc[:, -1].fillna(0).values
        mom_126 = df_temp.diff(126, axis=1).iloc[:, -1].fillna(0).values
        vol_21 = inc.iloc[:, -21:].std(axis=1, ddof=0).values if t >= 20 else np.zeros(sims_per_seed)
        vol_63 = inc.iloc[:, -63:].std(axis=1, ddof=0).values if t >= 62 else np.zeros(sims_per_seed)
        vol_126 = inc.iloc[:, -126:].std(axis=1, ddof=0).values if t >= 125 else np.zeros(sims_per_seed)
        cum_vals = np.exp(df_temp)
        dd_state = cum_vals.values[:, -1] / np.maximum.accumulate(cum_vals.values, axis=1)[:, -1] - 1

        for name, vals in zip(
            ["mom_21d", "mom_63d", "mom_126d", "vol_21d", "vol_63d", "vol_126d", "dd_state"],
            [mom_21, mom_63, mom_126, vol_21, vol_63, vol_126, dd_state]
        ):
            if name in X_base.columns:
                col_idx = list(X_base.columns).index(name)
                state[:, col_idx] = vals

        # Update macro indicators via submodels
        if indicator_models:
            for feat in ("VIX", "MOVE", "YC_Spread", "SKEW", "DXY"):
                if feat in indicator_models:
                    col_idx = list(X_base.columns).index(feat)
                    preds = indicator_models[feat].predict(state)
                    state[:, col_idx] = preds

    return np.exp(log_paths - log_paths[:, [0]], dtype=np.float32)

# ---------- Medoid Path ----------
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

# ---------- Stats ----------
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

# ---------- SHAP ----------
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
    ax.set_ylabel("% of explained volatility variance (feature-level)")
    ax.set_title("Feature Contribution to Conditional Volatility (Backtest vs Forecast)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ---------- Plot ----------
def plot_forecasts(port_rets, start_cap, central):
    port_cum = np.exp(port_rets.cumsum()) * start_cap
    last = port_cum.index[-1]
    fore = port_cum.iloc[-1] * (central / central[0])
    dates = pd.date_range(start=last, periods=len(central), freq="B")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, label="Portfolio Backtest")
    ax.plot([last, *dates], [port_cum.iloc[-1], *fore], label="Forecast (Medoid Path)", lw=2)
    ax.legend()
    st.pyplot(fig)

# ---------- Streamlit ----------
def main():
    st.title("Daily Conditional Volatility Forecasting Tool")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", 1000.0, 1000000.0, 10000.0, 1000.0)
    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_daily(tickers, DEFAULT_START)
            port_rets = portfolio_log_returns_daily(prices, weights)
            df = build_features(port_rets)
            Y = port_rets.loc[df.index].astype(np.float32)
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]

            # Train volatility model
            vol_target = Y.rolling(21).std().shift(-1).dropna()
            valid_idx = vol_target.index.intersection(X.index)
            X = X.loc[valid_idx]
            vol_target = vol_target.loc[valid_idx]

            model = LGBMRegressor(
                n_estimators=600, learning_rate=0.02, max_depth=3,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=GLOBAL_SEED, n_jobs=1
            )
            model.fit(X, vol_target)

            # Compute standardized residuals
            sigma_hat = model.predict(X)
            res = (Y.loc[X.index].values / (sigma_hat + 1e-8)).astype(np.float32)
            res = np.ascontiguousarray(res[~np.isnan(res)])
            base_mean = Y.mean()

            indicators = train_indicator_models(X, ["VIX", "MOVE", "YC_Spread", "SKEW", "DXY"])
            all_paths = []
            bar = st.progress(0)
            txt = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(
                    model, X, res, SIMS_PER_SEED, rng, indicators, base_mean
                )
                all_paths.append(sims)
                bar.progress((i + 1) / ENSEMBLE_SEEDS)
                txt.text(f"Running forecasts... {int((i + 1) / ENSEMBLE_SEEDS * 100)}%/100%")
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
            medoid_states = np.repeat(X.iloc[[-1]].values, FORECAST_DAYS, axis=0)
            plot_feature_attributions(model, X, medoid_states)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

























