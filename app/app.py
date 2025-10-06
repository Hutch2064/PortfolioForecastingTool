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
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")

# ---------- Global Seed Fix ----------
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------- Config ----------
DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 100
SIMS_PER_SEED = 10000
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
    if m.empty: return np.nan
    compounded = (1 + m).prod()
    years = len(m) / 12.0
    return compounded ** (1 / years) - 1 if years > 0 else np.nan

def annualized_vol_monthly(monthly_returns: pd.Series) -> float:
    m = monthly_returns.dropna()
    return m.std(ddof=0) * np.sqrt(12) if len(m) > 1 else np.nan

def annualized_sharpe_monthly(monthly_returns: pd.Series, rf_monthly: float = 0.0) -> float:
    m = monthly_returns.dropna()
    if m.empty: return np.nan
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
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        interval="1mo",
        threads=False
    )
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
        close = pd.DataFrame(data[colname]); close.columns = tickers
    close = close.ffill().dropna(how="all").astype(np.float32)
    first_valids = [close[col].first_valid_index() for col in close.columns]
    valid_starts = [d for d in first_valids if d is not None]
    if not valid_starts:
        raise ValueError("No valid price history found for tickers.")
    non_na_start = max(valid_starts)
    return close.loc[non_na_start:]

def fetch_macro_features(start=DEFAULT_START) -> pd.DataFrame:
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX"]
    data = yf.download(
        tickers, start=start, auto_adjust=False, progress=False, interval="1mo", threads=False
    )
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
        if rule is None: raise ValueError("Invalid rebalance option")
        port_val, port_vals, current_weights = 1.0, [], weights.copy()
        rebalance_dates = rets.resample(rule).last().index
        for i, date in enumerate(rets.index):
            if i > 0: port_val *= (1 + (rets.iloc[i] @ current_weights))
            port_vals.append(port_val)
            if date in rebalance_dates: current_weights = weights.copy()
        return pd.Series(port_vals, index=rets.index, name="Portfolio").pct_change().fillna(0.0).astype(np.float32)

# ---------- Feature Builders ----------
def build_features(returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame()
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)
    macro = fetch_macro_features()
    df = df.join(macro, how="left").ffill()
    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index() is not None])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ---------- Optuna ----------
def _oos_years_available(idx: pd.DatetimeIndex, max_years=5) -> List[int]:
    years = sorted(set(idx.year))
    complete_years = []
    for y in years:
        months = set(idx[idx.year == y].month)
        if all(m in months for m in range(1, 13)):
            complete_years.append(y)
    return complete_years[-max_years:] if len(complete_years) > max_years else complete_years

def _split_train_test_for_year(X: pd.DataFrame, Y: pd.Series, test_year: int):
    train_end = pd.Timestamp(f"{test_year-1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    train_X = X.loc[:train_end]
    train_Y = Y.loc[train_X.index]
    test_X = X.loc[test_start:test_end]
    test_Y = Y.loc[test_X.index]
    return train_X, train_Y, test_X, test_Y

def _median_params(param_dicts: List[dict]) -> dict:
    if not param_dicts: return {}
    all_keys = set().union(*[d.keys() for d in param_dicts])
    consensus = {}
    for k in all_keys:
        vals = [d[k] for d in param_dicts if k in d]
        if not vals: continue
        if isinstance(vals[0], (int, np.integer)):
            consensus[k] = int(np.round(np.median(vals)))
        elif isinstance(vals[0], (float, np.floating)):
            consensus[k] = float(np.median(vals))
        else:
            counts = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            consensus[k] = max(counts.items(), key=lambda x: x[1])[0]
    consensus["random_state"] = GLOBAL_SEED
    consensus["n_jobs"] = 1
    return consensus

def tune_across_recent_oos_years(X: pd.DataFrame, Y: pd.Series, years_back: int = 5, seed: int = GLOBAL_SEED, n_trials: int = 50):
    years = _oos_years_available(Y.index, max_years=years_back)
    param_runs, details = [], []
    total_jobs = len(years) * n_trials
    tuning_bar = st.progress(0)
    tuning_status = st.empty()
    completed = 0
    for y in years:
        train_X, train_Y, test_X, test_Y = _split_train_test_for_year(X, Y, y)
        if len(train_X) < 24 or len(test_X) < 6:
            continue
        def objective(trial):
            nonlocal completed
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 1000, 8000),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "df": trial.suggest_int("df", 3, 30),
                "random_state": seed,
                "n_jobs": 1
            }
            model_params = {k:v for k,v in params.items() if k != "df"}
            model = LGBMRegressor(**model_params)
            model.fit(train_X, train_Y)
            preds = model.predict(test_X)
            actual_cum = (1 + test_Y).cumprod()
            pred_cum = (1 + preds).cumprod()
            rmse = np.sqrt(mean_squared_error(actual_cum, pred_cum))
            actual_dir = np.sign(test_Y.values)
            pred_dir = np.sign(preds)
            directional_acc = (actual_dir == pred_dir).mean()
            completed += 1
            tuning_bar.progress(completed / total_jobs)
            tuning_status.text(f"Tuning models... {int((completed/total_jobs)*100)}%")
            return rmse, -directional_acc
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_trials[0]
        details.append({"year": y, "rmse": float(best.values[0]), "da": -float(best.values[1]), "best_params": dict(best.params)})
        param_runs.append(dict(best.params))
    tuning_bar.empty(); tuning_status.empty()
    consensus_params = _median_params(param_runs)
    return consensus_params, details, np.nan, np.nan

# ---------- Indicator Models (for exogenous features) ----------
def train_indicator_models(X: pd.DataFrame, feats: List[str]) -> Dict[str, LGBMRegressor]:
    models: Dict[str, LGBMRegressor] = {}
    for f in feats:
        if f not in X.columns:  # safety
            continue
        y = X[f].shift(-1)
        idx = y.dropna().index.intersection(X.index)
        if len(idx) < 24:  # need minimal samples
            continue
        y_fit = y.loc[idx]
        X_fit = X.loc[idx]
        mdl = LGBMRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=GLOBAL_SEED, n_jobs=1
        )
        mdl.fit(X_fit, y_fit)
        models[f] = mdl
    return models

# ---------- Monte Carlo (Dynamic drift via evolving VIX/MOVE/YC_Spread; endogenous features mechanical) ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, seed_id=None, df=5, indicator_models: Dict[str, LGBMRegressor]=None):
    """
    - Drift at month t is model.predict(current_features_t)
    - Exogenous features (VIX, MOVE, YC_Spread) evolve via their one-step-ahead ML models (deterministic).
    - Endogenous features (mom_3m, mom_6m, mom_12m, dd_state) evolve mechanically from simulated returns (median across sims fed back to features).
    - Returns simulated in log space; output is price index path (starts ~1.0).
    """
    horizon_months = FORECAST_YEARS * 12
    # arrays per simulation
    log_paths = np.zeros((sims_per_seed, horizon_months), dtype=np.float32)
    ar_returns = np.zeros((sims_per_seed, horizon_months), dtype=np.float32)  # arithmetic returns per step
    # residual stats
    mu_eps = float(residuals.mean())
    sigma_eps = float(residuals.std(ddof=0))
    # feature state (single vector used for predicting drift, evolves each month)
    # Start from last observed feature row (already lagged properly via X construction)
    feature_cols = list(X_base.columns)
    current_state = pd.Series(X_base.iloc[-1].values, index=feature_cols).astype(np.float32)

    # running cumulative value & max for drawdown (per sim)
    cum_val = np.ones(sims_per_seed, dtype=np.float32)
    cum_max = np.ones(sims_per_seed, dtype=np.float32)

    # helper to compute momentum median across sims for last k months (arithmetic)
    def _momentum_median(k: int, t: int) -> float:
        start = max(0, t - k + 1)
        if start > t: 
            return 0.0
        window = ar_returns[:, start:t+1]
        if window.shape[1] == 0:
            return 0.0
        prod = np.prod(1.0 + window, axis=1, dtype=np.float64) - 1.0
        return float(np.median(prod))

    for t in range(horizon_months):
        # 1) Predict drift given current (evolving) features
        mu_t = float(model.predict(current_state.values.reshape(1, -1)).astype(np.float32)[0])

        # 2) Simulate shocks (Student-t) around drift (log-return space)
        shocks = student_t.rvs(df, loc=mu_eps, scale=sigma_eps, size=sims_per_seed, random_state=rng).astype(np.float32)
        log_step = mu_t + shocks  # log(1+r_t)
        log_paths[:, t] = (log_paths[:, t-1] if t > 0 else 0.0) + log_step

        # 3) Update arithmetic returns, cumulative, drawdown
        ar_t = np.expm1(log_step).astype(np.float32)  # convert to arithmetic return
        ar_returns[:, t] = ar_t
        cum_val *= (1.0 + ar_t)
        cum_max = np.maximum(cum_max, cum_val)
        dd_now = (cum_val / np.maximum(cum_max, 1e-12) - 1.0)  # per-sim drawdown
        dd_med = float(np.median(dd_now))

        # 4) Evolve endogenous (portfolio) features mechanically (use cross-sim medians)
        #    These feed into next month's feature vector for drift prediction
        current_state["mom_3m"] = np.float32(_momentum_median(3, t))
        current_state["mom_6m"] = np.float32(_momentum_median(6, t))
        current_state["mom_12m"] = np.float32(_momentum_median(12, t))
        current_state["dd_state"] = np.float32(dd_med)

        # 5) Evolve exogenous features via their ML one-step models (deterministic)
        if indicator_models:
            for feat in ("VIX", "MOVE", "YC_Spread"):
                if feat in indicator_models and feat in current_state.index:
                    # Predict next month level given current state vector
                    next_val = float(indicator_models[feat].predict(current_state.values.reshape(1, -1))[0])
                    current_state[feat] = np.float32(next_val)

    # Return price index paths (start near 1.0)
    return np.exp(log_paths, dtype=np.float32)

# ---------- Vol-Matched Single Path ----------
def find_single_vol_closest_path(paths: np.ndarray, hist_vol: float):
    diffs = np.diff(np.log(paths), axis=1)
    path_vols = diffs.std(axis=1, ddof=0) * np.sqrt(12)
    best_idx = np.argmin(np.abs(path_vols - hist_vol))
    return paths[best_idx]

# ---------- Forecast Stats ----------
def compute_forecast_stats_from_path(path: np.ndarray, start_capital: float, last_date: pd.Timestamp):
    norm_path = path / path[0]
    forecast_index = pd.date_range(start=last_date, periods=len(norm_path)+1, freq="M")
    price = pd.Series(norm_path, index=forecast_index[:-1]) * start_capital
    monthly = price.pct_change().dropna()
    return {
        "CAGR": annualized_return_monthly(monthly),
        "Volatility": annualized_vol_monthly(monthly),
        "Sharpe": annualized_sharpe_monthly(monthly),
        "Max Drawdown": max_drawdown_from_rets(monthly)
    }

# ---------- SHAP ----------
def plot_feature_attributions(model, X, final_X):
    explainer = shap.TreeExplainer(model)
    shap_values_hist = explainer.shap_values(X)
    shap_mean_hist = np.abs(shap_values_hist).mean(axis=0)
    shap_values_fore = explainer.shap_values(final_X)
    shap_mean_fore = np.abs(shap_values_fore).reshape(-1)
    features = X.columns
    x_pos = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - 0.2, shap_mean_hist, width=0.4, label="Backtest Avg")
    ax.bar(x_pos + 0.2, shap_mean_fore, width=0.4, label="Forecast Snapshot")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Average |SHAP Value|")
    ax.set_title("Feature Contributions: Backtest vs Forecast Snapshot")
    ax.legend()
    st.pyplot(fig)

# ---------- Plot ----------
def plot_forecasts(port_rets, start_capital, central, rebalance_label):
    port_cum = (1 + port_rets).cumprod() * start_capital
    last_date = port_cum.index[-1]
    forecast_path = port_cum.iloc[-1] * (central / central[0])
    forecast_dates = pd.date_range(start=last_date, periods=len(central), freq="M")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, label="Portfolio Backtest")
    ax.plot([last_date, *forecast_dates], [port_cum.iloc[-1], *forecast_path], linewidth=2, label="Forecast (Vol-Matched)")
    ax.set_title("Portfolio Forecast (Backtest + 1Y Vol-Matched Forecast)")
    ax.set_xlabel("Date"); ax.set_ylabel("Balance ($)")
    ax.legend()
    st.pyplot(fig)

# ---------- Streamlit App ----------
def main():
    st.title("Portfolio Forecasting Tool – Volatility-Matched Path")
    tickers = st.text_input("Tickers (comma-separated, e.g. VTI,AGG)", "VTI,AGG")
    weights_str = st.text_input("Weights (comma-separated, must sum > 0)", "0.6,0.4")
    start_capital = st.number_input("Starting Value ($)", min_value=1000.0, value=10000.0, step=1000.0)
    freq_map = {"M": "Monthly","Q": "Quarterly","S": "Semiannual","Y": "Yearly","N": "None"}
    rebalance_label = st.selectbox("Rebalance", list(freq_map.values()), index=0)
    rebalance_choice = [k for k,v in freq_map.items() if v == rebalance_label][0]

    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]
            prices = fetch_prices_monthly(tickers, start=DEFAULT_START)
            port_rets = portfolio_returns_monthly(prices, weights, rebalance_choice)
            df = build_features(port_rets)
            if df.empty:
                st.error("Feature engineering returned no data.")
                return
            Y = np.log(1 + port_rets.loc[df.index]).astype(np.float32)
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]

            consensus_params, oos_details, last_rmse, last_da = tune_across_recent_oos_years(
                X, Y, years_back=5, seed=GLOBAL_SEED, n_trials=50
            )
            st.json(consensus_params)

            df_opt = int(consensus_params.get("df", 5))
            lgbm_params = {k:v for k,v in consensus_params.items() if k != "df"}
            lgbm_params["random_state"] = GLOBAL_SEED
            lgbm_params["n_jobs"] = 1

            final_model = LGBMRegressor(**lgbm_params)
            final_model.fit(X, Y)
            preds = final_model.predict(X).astype(np.float32)
            residuals = (Y.values - preds).astype(np.float32)

            # --- Train indicator models for exogenous features (deterministic one-step evolution) ---
            indicator_models = train_indicator_models(X, ["VIX", "MOVE", "YC_Spread"])

            hist_vol = annualized_vol_monthly(port_rets.iloc[-12:])

            all_paths = []
            sim_bar = st.progress(0)
            sim_status = st.empty()
            for i, seed in enumerate(range(ENSEMBLE_SEEDS)):
                rng = np.random.default_rng(GLOBAL_SEED + seed)
                sims = run_monte_carlo_paths(
                    final_model, X, Y, residuals,
                    SIMS_PER_SEED, rng, seed_id=seed, df=df_opt,
                    indicator_models=indicator_models
                )
                all_paths.append(sims)
                sim_bar.progress((i+1)/ENSEMBLE_SEEDS)
                sim_status.text(f"Running forecasts... {i+1}/{ENSEMBLE_SEEDS}")
            sim_bar.empty(); sim_status.empty()

            all_paths = np.vstack(all_paths)
            final_path = find_single_vol_closest_path(all_paths, hist_vol)

            stats = compute_forecast_stats_from_path(final_path, start_capital, port_rets.index[-1])
            backtest_stats = {
                "CAGR": annualized_return_monthly(port_rets),
                "Volatility": annualized_vol_monthly(port_rets),
                "Sharpe": annualized_sharpe_monthly(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets)
            }

            st.subheader("Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Backtest**")
                for k,v in backtest_stats.items(): st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            with col2:
                st.markdown("**Forecast (Vol-Matched)**")
                for k,v in stats.items(): st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            ending_value = float(final_path[-1]) * start_capital
            st.metric("Forecasted Portfolio Value", f"${ending_value:,.2f}")

            plot_forecasts(port_rets, start_capital, final_path, rebalance_label)
            final_X = X.iloc[[-1]]
            plot_feature_attributions(final_model, X, final_X)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()


