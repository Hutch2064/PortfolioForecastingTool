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
from pandas_datareader import data as pdr  # for FRED sentiment

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
    if not valid_starts:
        raise ValueError("No valid price history found for tickers.")
    non_na_start = max(valid_starts)
    return close.loc[non_na_start:]

@st.cache_data(show_spinner=False)
def fetch_umich_sentiment(start=DEFAULT_START) -> pd.Series:
    try:
        umcsent = pdr.DataReader("UMCSENT", "fred", start)
        umcsent = umcsent.resample("M").last().ffill()
        umcsent.name = "umcsent"
        return umcsent.astype(np.float32).squeeze()
    except Exception as e:
        st.error(f"UMCSENT fetch failed: {e}")
        return pd.Series(dtype=np.float32, name="umcsent")

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
    df = pd.DataFrame(index=returns.index)
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)

    umcsent = fetch_umich_sentiment(start=returns.index.min().to_pydatetime().date())
    if not umcsent.empty:
        umcsent = umcsent.loc[df.index.min():df.index.max()]
        df["umcsent"] = umcsent.reindex(df.index).astype(np.float32).fillna(method="ffill").fillna(method="bfill")

    # Debug print
    st.write("DEBUG: build_features columns:", df.columns)
    st.write("DEBUG: UMCSENT head:", df["umcsent"].head() if "umcsent" in df else "MISSING")

    return df.dropna().astype(np.float32)

# ---------- Optuna Hyperparameter Tuning ----------
def tune_and_fit_best_model(X: pd.DataFrame, Y: pd.Series, seed=GLOBAL_SEED):
    train_X, test_X = X.iloc[:-12], X.iloc[-12:]
    train_Y, test_Y = Y.iloc[:-12], Y.iloc[-12:]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 8000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "block_length": trial.suggest_int("block_length", 1, 12),
            "random_state": seed,
            "n_jobs": 1
        }
        model_params = {k: v for k, v in params.items() if k != "block_length"}
        model = LGBMRegressor(**model_params)
        model.fit(train_X, train_Y, feature_name=list(train_X.columns))
        preds = model.predict(test_X)
        actual_cum = (1 + test_Y).cumprod()
        pred_cum = (1 + preds).cumprod()
        rmse = np.sqrt(mean_squared_error(actual_cum, pred_cum))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_trial = study.best_trial
    best_params_full = dict(best_trial.params)
    best_params_full["block_length"] = int(best_params_full["block_length"])
    lgbm_params = {k: v for k, v in best_params_full.items() if k != "block_length"}
    best_rmse = float(best_trial.value)

    final_model = LGBMRegressor(**lgbm_params, random_state=seed, n_jobs=1)
    final_model.fit(X, Y, feature_name=list(X.columns))

    # Debug print
    st.write("DEBUG: Model features after fit:", final_model.booster_.feature_name())

    preds = final_model.predict(X).astype(np.float32)
    residuals = (Y.values - preds).astype(np.float32)
    return final_model, residuals, preds, X.astype(np.float32), Y.astype(np.float32), best_params_full, best_rmse

# ---------- SHAP ----------
def plot_feature_attributions(model, X, final_X):
    explainer = shap.TreeExplainer(model)
    shap_values_hist = explainer.shap_values(X)
    features = X.columns

    # Debug print
    st.write("DEBUG: SHAP features:", list(features))
    st.write("DEBUG: SHAP values shape:", np.array(shap_values_hist).shape)

    shap_mean_hist = np.abs(shap_values_hist).mean(axis=0)
    shap_values_fore = explainer.shap_values(final_X)
    shap_mean_fore = np.abs(shap_values_fore).reshape(-1)

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

# ---------- Main ----------
def main():
    st.title("Snapshot Portfolio Forecasting Tool with UMCSENT Debug")

    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_capital = st.number_input("Starting Value ($)", min_value=1000.0, value=10000.0, step=1000.0)

    rebalance_choice = "N"

    if st.button("Run Forecast"):
        prices = fetch_prices_monthly([t.strip() for t in tickers.split(",")], start=DEFAULT_START)
        port_rets = portfolio_returns_monthly(prices, to_weights([float(x) for x in weights_str.split(",")]), rebalance_choice)

        df = build_features(port_rets)
        Y = np.log(1 + port_rets.loc[df.index]).astype(np.float32)
        X = df.shift(1).dropna()
        Y = Y.loc[X.index]

        # Debug check
        st.write("DEBUG: Final X columns:", list(X.columns))
        if "umcsent" in X.columns:
            st.write("DEBUG: UMCSENT non-NA count:", X["umcsent"].count())
            st.write("DEBUG: UMCSENT sample values:", X["umcsent"].head())

        model, residuals, preds, X_full, Y_full, best_params, best_rmse = tune_and_fit_best_model(X, Y)
        final_X = X_full.iloc[[-1]]
        plot_feature_attributions(model, X_full, final_X)

if __name__ == "__main__":
    main()