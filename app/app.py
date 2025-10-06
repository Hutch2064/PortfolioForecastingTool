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
import optuna

warnings.filterwarnings("ignore")

# ---------- Global Seed Fix ----------
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------- Config ----------
DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 100
SIMS_PER_SEED = 1000
FORECAST_YEARS = 1

# ---------- Helpers ----------
def to_weights(raw: List[float]) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return arr / s

def annualized_return_monthly(monthly_returns: pd.Series) -> float:
    m = monthly_returns.dropna()
    if m.empty:
        return np.nan
    compounded = (1 + m).prod()
    years = len(m) / 12.0
    return compounded ** (1 / years) - 1 if years > 0 else np.nan

def annualized_vol_monthly(monthly_returns: pd.Series) -> float:
    m = monthly_returns.dropna()
    return m.std(ddof=0) * np.sqrt(12) if len(m) > 1 else np.nan

def annualized_sharpe_monthly(monthly_returns: pd.Series, rf_monthly: float = 0.0) -> float:
    m = monthly_returns.dropna()
    if m.empty:
        return np.nan
    excess = m - rf_monthly
    mu, sigma = excess.mean(), excess.std(ddof=0)
    return (mu / sigma) * np.sqrt(12) if sigma and sigma > 0 else np.nan

def max_drawdown_from_rets(returns: pd.Series) -> float:
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    return dd.min()

def compute_current_drawdown(returns: pd.Series) -> pd.Series:
    cum = (1 + returns.fillna(0)).cumprod()
    roll_max = cum.cummax()
    return (cum / roll_max - 1.0).astype(np.float32)

# ---------- Data Fetch ----------
def fetch_prices_monthly(tickers: List[str], start=DEFAULT_START) -> pd.DataFrame:
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False, interval="1mo", threads=False)
    if data.empty:
        raise ValueError("No price data returned from Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        for field in ["Adj Close", "Close"]:
            if field in data.columns.get_level_values(0):
                close = data[field].copy()
                break
        else:
            raise ValueError("Could not find Close/Adj Close in Yahoo data.")
    else:
        colname = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = pd.DataFrame(data[colname])
        close.columns = tickers
    close = close.ffill().dropna(how="all").astype(np.float32)
    first_valids = [close[col].first_valid_index() for col in close.columns]
    valid_starts = [d for d in first_valids if d is not None]
    if not valid_starts:
        raise ValueError("No valid price history found for tickers.")
    non_na_start = max(valid_starts)
    return close.loc[non_na_start:]

def fetch_macro_features(start=DEFAULT_START) -> pd.DataFrame:
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX"]
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False, interval="1mo", threads=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data.copy()
    close = close.ffill().astype(np.float32)
    df = pd.DataFrame(index=close.index)
    df["VIX"] = close["^VIX"]
    df["MOVE"] = close["^MOVE"]
    df["YC_Spread"] = close["^TNX"] - close["^IRX"]
    return df

# ---------- Portfolio ----------
def portfolio_returns_monthly(prices: pd.DataFrame, weights: np.ndarray, rebalance: str) -> pd.Series:
    rets = prices.pct_change().dropna(how="all").astype(np.float32)
    if rebalance == "N":
        vals = (1 + rets).cumprod()
        port_vals = vals.dot(weights)
        port_vals = port_vals / port_vals.iloc[0]
        return port_vals.pct_change().fillna(0.0).astype(np.float32)
    else:
        freq_map = {"M": "M", "Q": "Q", "S": "2Q", "Y": "A"}
        rule = freq_map.get(rebalance)
        if rule is None:
            raise ValueError("Invalid rebalance option")
        port_val, port_vals, current_weights = 1.0, [], weights.copy()
        rebalance_dates = rets.resample(rule).last().index
        for i, date in enumerate(rets.index):
            if i > 0:
                port_val *= (1 + (rets.iloc[i] @ current_weights))
            port_vals.append(port_val)
            if date in rebalance_dates:
                current_weights = weights.copy()
        return pd.Series(port_vals, index=rets.index, name="Portfolio").pct_change().fillna(0.0).astype(np.float32)

# ---------- Feature Builders ----------
def build_features(returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame()
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1 + x).prod() - 1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1 + x).prod() - 1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1 + x).prod() - 1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)
    macro = fetch_macro_features()
    df = df.join(macro, how="left").ffill()
    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index() is not None])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ---------- Block Bootstrap ----------
def block_bootstrap_residuals(residuals, size, block_len, rng):
    n = len(residuals)
    num_blocks = max(1, int(np.ceil(size / block_len)))
    starts = rng.integers(0, n - block_len + 1, size=num_blocks)
    blocks = [residuals[s:s + block_len] for s in starts]
    flat = np.concatenate(blocks, axis=0)
    if len(flat) < size:
        flat = np.concatenate([flat, rng.choice(residuals, size=size - len(flat), replace=True)])
    elif len(flat) > size:
        flat = flat[:size]
    return np.ascontiguousarray(flat, dtype=np.float32)

# ---------- Indicator Models ----------
def train_indicator_models(X, feats):
    models = {}
    for f in feats:
        if f not in X.columns:
            continue
        y = X[f].shift(-1)
        idx = y.dropna().index.intersection(X.index)
        if len(idx) < 24:
            continue
        mdl = LGBMRegressor(n_estimators=500, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=GLOBAL_SEED, n_jobs=1)
        X_fit = np.ascontiguousarray(X.loc[idx].values.astype(np.float32))
        y_fit = np.ascontiguousarray(y.loc[idx].values.astype(np.float32))
        mdl.fit(X_fit, y_fit)
        models[f] = mdl
    return models

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, residuals, sims_per_seed, rng, block_len=12, indicator_models=None):
    horizon = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon), dtype=np.float32)
    ar_returns = np.zeros_like(log_paths)
    feature_cols = list(X_base.columns)
    state = pd.Series(X_base.iloc[-1].values, index=feature_cols).astype(np.float32)
    cum_val = np.ones(sims_per_seed, dtype=np.float32)
    cum_max = np.ones_like(cum_val)

    def _mom_med(k, t):
        start = max(0, t - k + 1)
        window = ar_returns[:, start:t+1]
        if window.shape[1] == 0:
            return 0.0
        prod = np.prod(1 + window, axis=1) - 1
        return float(np.median(prod))

    for t in range(horizon):
        mu_t = float(model.predict(np.ascontiguousarray(state.values.reshape(1, -1), dtype=np.float32))[0])
        shocks = block_bootstrap_residuals(residuals, sims_per_seed, block_len, rng)
        log_step = mu_t + shocks
        log_paths[:, t] = (log_paths[:, t-1] if t > 0 else 0) + log_step
        ar_t = np.expm1(log_step)
        ar_returns[:, t] = ar_t
        cum_val *= (1 + ar_t)
        cum_max = np.maximum(cum_max, cum_val)
        dd_med = float(np.median(cum_val / cum_max - 1))
        state["mom_3m"] = np.float32(_mom_med(3, t))
        state["mom_6m"] = np.float32(_mom_med(6, t))
        state["mom_12m"] = np.float32(_mom_med(12, t))
        state["dd_state"] = np.float32(dd_med)
        if indicator_models:
            for feat in ("VIX", "MOVE", "YC_Spread"):
                if feat in indicator_models:
                    pred_val = indicator_models[feat].predict(np.ascontiguousarray(state.values.reshape(1, -1), dtype=np.float32))[0]
                    state[feat] = np.float32(pred_val)

    return np.exp(log_paths, dtype=np.float32)

# ---------- Vol-Matched Path ----------
def find_single_vol_closest_path(paths, hist_vol):
    diffs = np.diff(np.log(paths), axis=1)
    vols = diffs.std(axis=1) * np.sqrt(12)
    return paths[np.argmin(np.abs(vols - hist_vol))]

# ---------- Streamlit App ----------
def main():
    st.title("Portfolio Forecasting Tool – Fixed Block Bootstrap Version")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value ($)", 1000.0, 1000000.0, 10000.0, 1000.0)
    freq_map = {"M": "Monthly", "Q": "Quarterly", "S": "Semiannual", "Y": "Yearly", "N": "None"}
    reb_label = st.selectbox("Rebalance", list(freq_map.values()), index=0)
    reb = [k for k, v in freq_map.items() if v == reb_label][0]

    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_monthly(tickers, DEFAULT_START)
            port_rets = portfolio_returns_monthly(prices, weights, reb)
            df = build_features(port_rets)

            Y = np.log(1 + port_rets.loc[df.index]).astype(np.float32)
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]
            X = X.replace([np.inf, -np.inf], np.nan).dropna().astype(np.float32)
            Y = Y.loc[X.index]

            model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4, random_state=GLOBAL_SEED, n_jobs=1)
            model.fit(np.ascontiguousarray(X.values), np.ascontiguousarray(Y.values))
            residuals = (Y.values - model.predict(np.ascontiguousarray(X.values))).astype(np.float32)
            indicator_models = train_indicator_models(X, ["VIX", "MOVE", "YC_Spread"])
            hist_vol = annualized_vol_monthly(port_rets.iloc[-12:])

            all_paths = []
            sim_bar = st.progress(0)
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(model, X, residuals, SIMS_PER_SEED, rng, 12, indicator_models)
                all_paths.append(sims)
                sim_bar.progress((i + 1) / ENSEMBLE_SEEDS)
            sim_bar.empty()

            paths = np.vstack(all_paths)
            final_path = find_single_vol_closest_path(paths, hist_vol)

            st.metric("Forecasted Value", f"${final_path[-1] * start_cap:,.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()




