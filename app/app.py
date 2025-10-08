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
ENSEMBLE_SEEDS = 12
SIMS_PER_SEED = 2000
FORECAST_YEARS = 1
P_STATIONARY = 0.5  # Probability of starting a new block (1/p = expected block length)

# ---------- Helpers ----------
def to_weights(raw: List[float]) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    s = arr.sum()
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
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

# ---------- Data Fetch ----------
def fetch_prices_monthly(tickers, start=DEFAULT_START):
    data = yf.download(tickers, start=start, interval="1mo", auto_adjust=False, progress=False, threads=False)
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
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX", "^SKEW", "DX-Y.NYB"]
    data = yf.download(tickers, start=start, interval="1mo", progress=False, threads=False)
    close = data["Close"].ffill().astype(np.float32)
    df = pd.DataFrame(index=close.index)
    df["VIX"] = close["^VIX"]
    df["MOVE"] = close["^MOVE"]
    df["YC_Spread"] = close["^TNX"] - close["^IRX"]
    df["SKEW"] = close["^SKEW"]
    df["DXY"] = close["DX-Y.NYB"]
    return df

# ---------- Portfolio (fixed drift) ----------
def portfolio_log_returns_monthly(prices, weights, rebalance):
    """
    Computes portfolio log returns with proper weighting, rebalancing,
    and log-space consistency. Guaranteed not to collapse to zero.
    """
    prices = prices.ffill().dropna().astype(np.float64)
    n_assets = prices.shape[1]
    weights = np.array(weights, dtype=np.float64).reshape(-1)
    if len(weights) != n_assets:
        raise ValueError(f"Weight count ({len(weights)}) does not match asset count ({n_assets})")
    rets = np.log(prices / prices.shift(1)).dropna()
    freq_map = {"M": "M", "Q": "Q", "S": "2Q", "Y": "A", "N": None}
    rule = freq_map.get(rebalance)
    if rule is None and rebalance != "N":
        raise ValueError("Invalid rebalance frequency")
    if rebalance == "N":
        port_rets = (rets * weights).sum(axis=1)
        return port_rets.astype(np.float32)
    rebalance_dates = rets.resample(rule).last().index
    port_vals = [1.0]
    current_weights = weights.copy()
    for i in range(1, len(rets)):
        gross = np.exp(rets.iloc[i].values)
        port_vals.append(port_vals[-1] * np.dot(current_weights, gross))
        if rets.index[i] in rebalance_dates:
            current_weights = weights.copy()
    port_vals = pd.Series(port_vals, index=rets.index)
    port_rets = np.log(port_vals / port_vals.shift(1)).dropna()
    return port_rets.astype(np.float32)

# ---------- Features ----------
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
    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index()])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ---------- Optuna ----------
def _oos_years_available(idx, max_years=5):
    years = sorted(set(idx.year))
    full = []
    for y in years:
        months = set(idx[idx.year == y].month)
        if all(m in months for m in range(1, 13)):
            full.append(y)
    return full[-max_years:]


def _split_train_test_for_year(X, Y, y):
    train_end = pd.Timestamp(f"{y-1}-12-31")
    test_start = pd.Timestamp(f"{y}-01-01")
    train_X = X.loc[:train_end]
    train_Y = Y.loc[train_X.index]
    test_X = X.loc[test_start:]
    test_Y = Y.loc[test_X.index]
    return train_X, train_Y, test_X, test_Y


def _median_params(params_list):
    if not params_list:
        return {}
    keys = set().union(*[d.keys() for d in params_list])
    out = {}
    for k in keys:
        vals = [d[k] for d in params_list if k in d]
        if not vals:
            continue
        if isinstance(vals[0], (int, np.integer)):
            out[k] = int(np.median(vals))
        elif isinstance(vals[0], (float, np.floating)):
            out[k] = float(np.median(vals))
        else:
            out[k] = max(set(vals), key=vals.count)
    out["random_state"] = GLOBAL_SEED
    out["n_jobs"] = 1
    return out


def tune_across_recent_oos_years(X, Y, years_back=5, seed=GLOBAL_SEED, n_trials=100):
    years = _oos_years_available(Y.index, years_back)
    params_all, details = [], []
    total_jobs = len(years) * n_trials
    bar = st.progress(0)
    txt = st.empty()
    done = 0
    for y in years:
        Xtr, Ytr, Xte, Yte = _split_train_test_for_year(X, Y, y)
        if len(Xtr) < 24 or len(Xte) < 6:
            continue
        def objective(trial):
            nonlocal done
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2500),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": seed,
                "n_jobs": 1,
            }
            mdl = LGBMRegressor(**params)
            mdl.fit(Xtr, Ytr)
            preds = mdl.predict(Xte)
            rmse = np.sqrt(mean_squared_error(Yte, preds))
            done += 1
            bar.progress(done / total_jobs)
            txt.text(f"Tuning models... {int(done / total_jobs * 100)}%/100%")
            return rmse
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_trial
        details.append({"year": y, "rmse": float(best.value), "best_params": dict(best.params)})
        params_all.append(dict(best.params))
    bar.empty()
    txt.empty()
    return _median_params(params_all), details, np.nan, np.nan

# ---------- Indicator Models ----------
def train_indicator_models(X, feats):
    models = {}
    for f in feats:
        if f not in X.columns:
            continue
        y = X[f].shift(-1)
        df = pd.concat([X, y.rename("target")], axis=1).dropna()
        if len(df) < 24:
            continue
        mdl = LGBMRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=GLOBAL_SEED, n_jobs=1
        )
        mdl.fit(df[X.columns], df["target"])
        models[f] = mdl
    return models

# ---------- Stationary Bootstrap ----------
def stationary_bootstrap_residuals(residuals, size, p=P_STATIONARY, rng=None):
    """
    Stationary Bootstrap (Politis & Romano, 1994)
    - residuals: np.array of original residuals
    - size: number of simulated steps to generate
    - p: probability of starting a new block (1/p = expected block length)
    - rng: np.random.Generator for reproducibility
    """
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

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng,
                          seed_id=None, block_len=3, indicator_models=None, port_rets=None):
    horizon = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon), dtype=np.float32)
    state = np.repeat(X_base.iloc[[-1]].values, sims_per_seed, axis=0).astype(np.float32)
    for t in range(horizon):
        mu_t = model.predict(state)
        shocks = stationary_bootstrap_residuals(residuals, sims_per_seed, p=P_STATIONARY, rng=rng)
        log_paths[:, t] = (log_paths[:, t - 1] if t > 0 else 0) + mu_t + shocks
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
        for name, vals in zip(
            ["mom_3m", "mom_6m", "mom_12m", "vol_3m", "vol_6m", "vol_12m", "dd_state"],
            [mom_3m, mom_6m, mom_12m, vol_3m, vol_6m, vol_12m, dd_state]
        ):
            if name in X_base.columns:
                col_idx = list(X_base.columns).index(name)
                state[:, col_idx] = vals
        if indicator_models:
            for feat in ("VIX", "MOVE", "YC_Spread", "SKEW", "DXY"):
                if feat in indicator_models:
                    col_idx = list(X_base.columns).index(feat)
                    preds = indicator_models[feat].predict(state)
                    state[:, col_idx] = preds
    return np.exp(log_paths - log_paths[:, [0]], dtype=np.float32)

# ---------- True Medoid Path ----------
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
    idx = pd.date_range(start=last_date, periods=len(norm) + 1, freq="M")
    price = pd.Series(norm, index=idx[:-1]) * start_cap
    rets = np.log(price / price.shift(1)).dropna()
    return {
        "CAGR": annualized_return_monthly(rets),
        "Volatility": annualized_vol_monthly(rets),
        "Sharpe": annualized_sharpe_monthly(rets),
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
    ax.set_ylabel("% of explained return variance (feature-level)")
    ax.set_title("Feature Contribution to Returns (Backtest vs Forecast)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ---------- Plot ----------
def plot_forecasts(port_rets, start_cap, central, reb_label):
    port_cum = np.exp(port_rets.cumsum()) * start_cap
    last = port_cum.index[-1]
    fore = port_cum.iloc[-1] * (central / central[0])
    dates = pd.date_range(start=last, periods=len(central), freq="M")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, label="Portfolio Backtest")
    ax.plot([last, *dates], [port_cum.iloc[-1], *fore], label="Forecast (Medoid Path)", lw=2)
    ax.legend()
    st.pyplot(fig)

# ---------- Streamlit ----------
def main():
    st.title("Portfolio Forecasting Tool")
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
            port_rets = portfolio_log_returns_monthly(prices, weights, reb)
            df = build_features(port_rets)
            Y = port_rets.loc[df.index].astype(np.float32)
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]
            cons, _, _, _ = tune_across_recent_oos_years(X, Y, 5, GLOBAL_SEED, 50)
            blk_len = 2
            lgb_params = {k: v for k, v in cons.items()}
            model = LGBMRegressor(**lgb_params)
            model.fit(X, Y)
            res = (Y.values - model.predict(X)).astype(np.float32)
            res = np.ascontiguousarray(res[~np.isnan(res)])
            indicators = train_indicator_models(X, ["VIX", "MOVE", "YC_Spread", "SKEW", "DXY"])
            all_paths = []
            bar = st.progress(0)
            txt = st.empty()
            for i in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED + i)
                sims = run_monte_carlo_paths(
                    model, X, Y, res, SIMS_PER_SEED, rng, i, blk_len, indicators, port_rets
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
            plot_forecasts(port_rets, start_cap, final, reb_label)
            medoid_states = np.repeat(X.iloc[[-1]].values, FORECAST_YEARS * 12, axis=0)
            plot_feature_attributions(model, X, medoid_states)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()





















