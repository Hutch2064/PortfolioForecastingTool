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
import math
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
ENSEMBLE_SEEDS = 50        # number of seeds in the ensemble
SIMS_PER_SEED = 2000       # simulations per seed
FORECAST_YEARS = 1         # 12-month horizon

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
@st.cache_data(show_spinner=False)
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
    non_na_start = max(valid_starts)
    return close.loc[non_na_start:]

@st.cache_data(show_spinner=False)
def fetch_macro_features(start=DEFAULT_START) -> pd.DataFrame:
    # VIX, MOVE, 10y and 3m Treasury yields
    tickers = ["^VIX", "^MOVE", "^TNX", "^IRX"]
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False, interval="1mo")
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
        port_val, port_vals, current_weights = 1.0, [], weights.copy()
        rebalance_dates = rets.resample(rule).last().index
        for i, date in enumerate(rets.index):
            if i > 0: port_val *= (1 + (rets.iloc[i] @ current_weights))
            port_vals.append(port_val)
            if date in rebalance_dates: current_weights = weights.copy()
        return pd.Series(port_vals, index=rets.index, name="Portfolio").pct_change().fillna(0.0).astype(np.float32)

# ---------- Feature Builders ----------
def build_features(returns: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=returns.index)
    df["mom_3m"]  = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"]  = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)

    macro_aligned = macro.reindex(df.index).ffill()
    df = pd.concat([df, macro_aligned], axis=1)

    valid_start = max([df[c].first_valid_index() for c in df.columns if df[c].first_valid_index() is not None])
    df = df.loc[valid_start:].dropna()
    return df.astype(np.float32)

# ---------- Optuna Hyperparameter Tuning ----------
def tune_and_fit_best_model(X: pd.DataFrame, Y: pd.Series, seed=GLOBAL_SEED):
    train_X, test_X = X.iloc[:-12], X.iloc[-12:]
    train_Y, test_Y = Y.iloc[:-12], Y.iloc[-12:]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "block_length": trial.suggest_int("block_length", 1, 12),
            "random_state": seed,
            "n_jobs": 1
        }
        model_params = {k: v for k, v in params.items() if k != "block_length"}
        model = LGBMRegressor(**model_params)
        model.fit(train_X, train_Y)
        preds = model.predict(test_X)
        actual_cum = (1 + test_Y).cumprod()
        pred_cum = (1 + preds).cumprod()
        return np.sqrt(mean_squared_error(actual_cum, pred_cum))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_params = study.best_trial.params
    block_length = int(best_params.pop("block_length"))
    final_model = LGBMRegressor(**best_params, random_state=seed, n_jobs=1)
    final_model.fit(X, Y)
    preds = final_model.predict(X).astype(np.float32)
    residuals = (Y.values - preds).astype(np.float32)
    return final_model, residuals, preds, X, Y, block_length, study.best_value

# ---------- Median-Ending Medoid ----------
def find_median_ending_medoid(paths: np.ndarray):
    endings = paths[:, -1]
    median_ending = np.median(endings)
    idx = np.argmin(np.abs(endings - median_ending))
    return paths[idx]

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, block_length):
    horizon_months = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon_months))
    n_res = len(residuals)
    snapshot_X = X_base.iloc[[-1]].values
    base_pred = model.predict(snapshot_X).astype(np.float32)[0]
    for s in range(sims_per_seed):
        t = 0
        while t < horizon_months:
            block = rng.choice(n_res, size=block_length, replace=True)
            for b in range(block_length):
                step = base_pred + residuals[block[b]]
                log_paths[s, t] = (log_paths[s, t-1] if t>0 else 0) + step
                t += 1
                if t >= horizon_months: break
    return np.exp(log_paths)

# ---------- Forecast Stats ----------
def compute_forecast_stats(path: np.ndarray, start_capital: float, last_date: pd.Timestamp):
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
    shap_hist = explainer.shap_values(X)
    shap_fore = explainer.shap_values(final_X)
    shap_mean_hist = np.abs(shap_hist).mean(axis=0)
    shap_mean_fore = np.abs(shap_fore).reshape(-1)
    features = X.columns
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(features, shap_mean_hist, alpha=0.6, label="Backtest Avg")
    ax.bar(features, shap_mean_fore, alpha=0.6, label="Forecast Snapshot")
    ax.legend(); ax.set_title("Feature Contributions")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ---------- Streamlit App ----------
def main():
    st.title("Portfolio Forecasting with Yahoo Macro Features")

    tickers = st.text_input("Tickers (comma-separated)", "VTI,AGG")
    weights_str = st.text_input("Weights (comma-separated)", "0.6,0.4")
    start_capital = st.number_input("Starting Value ($)", value=10000.0)

    freq_map = {"M": "Monthly","Q": "Quarterly","S": "Semiannual","Y": "Yearly","N": "None"}
    rebalance_label = st.selectbox("Rebalance", list(freq_map.values()))
    rebalance_choice = [k for k,v in freq_map.items() if v == rebalance_label][0]

    if st.button("Run Forecast"):
        weights = to_weights([float(x) for x in weights_str.split(",")])
        tickers = [t.strip() for t in tickers.split(",")]
        prices = fetch_prices_monthly(tickers)
        port_rets = portfolio_returns_monthly(prices, weights, rebalance_choice)
        macro = fetch_macro_features()
        df = build_features(port_rets, macro)

        Y = np.log1p(port_rets.loc[df.index])
        X = df.shift(1).dropna(); Y = Y.loc[X.index]

        model, residuals, preds, X_full, Y_full, block_length, rmse = tune_and_fit_best_model(X, Y)
        st.write("Best RMSE:", rmse)

        rng = np.random.default_rng(GLOBAL_SEED)
        medoids = []
        for seed in range(ENSEMBLE_SEEDS):
            sims = run_monte_carlo_paths(model, X_full, Y_full, residuals, SIMS_PER_SEED, rng, block_length)
            medoids.append(find_median_ending_medoid(sims))
        final_medoid = find_median_ending_medoid(np.vstack(medoids))

        stats = compute_forecast_stats(final_medoid, start_capital, port_rets.index[-1])
        st.write("Forecast Stats:", stats)

        plot_feature_attributions(model, X_full, X_full.iloc[[-1]])

if __name__ == "__main__":
    main()