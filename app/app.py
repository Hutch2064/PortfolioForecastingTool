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
ENSEMBLE_SEEDS = 100        # number of seeds in the ensemble
SIMS_PER_SEED = 10000       # simulations per seed
FORECAST_YEARS = 1          # 12-month horizon

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
    """
    Fetch University of Michigan Sentiment Index (UMCSENT) from FRED, monthly.
    Do NOT lag here; lag is applied uniformly with X.shift(1).
    """
    try:
        umcsent = pdr.DataReader("UMCSENT", "fred", start)
        umcsent = umcsent.resample("M").last().ffill()
        umcsent.name = "umcsent"
        return umcsent.astype(np.float32).squeeze()
    except Exception as e:
        st.error(f"UMCSENT fetch failed: {e}.")
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
        df["umcsent"] = umcsent.reindex(df.index).astype(np.float32)

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
        model.fit(train_X, train_Y)
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
    final_model.fit(X, Y)
    preds = final_model.predict(X).astype(np.float32)
    residuals = (Y.values - preds).astype(np.float32)
    return final_model, residuals, preds, X.astype(np.float32), Y.astype(np.float32), best_params_full, best_rmse

# ---------- Median-Ending Subset Medoid ----------
def find_median_ending_medoid(paths: np.ndarray):
    endings = paths[:, -1]
    median_ending = np.median(endings)
    tol = 0.01 * median_ending
    subset_idx = np.where(np.abs(endings - median_ending) <= tol)[0]
    if len(subset_idx) == 0:
        subset_idx = np.argsort(np.abs(endings - median_ending))[:max(1, len(paths)//20)]
    subset = paths[subset_idx]
    median_series = np.median(subset, axis=0)
    diffs = np.abs(subset - median_series)
    closest = np.argmin(diffs, axis=0)
    scores = np.bincount(closest, minlength=subset.shape[0])
    best_idx = np.argmax(scores)
    return subset[best_idx]

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, block_length, seed_id=None):
    horizon_months = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon_months), dtype=np.float32)
    n_res = len(residuals)
    n_blocks = math.ceil(horizon_months / block_length)
    snapshot_X = X_base.iloc[[-1]].values.astype(np.float32)
    last_X = np.repeat(snapshot_X, sims_per_seed, axis=0)
    block_starts = rng.integers(0, max(1, n_res - block_length), size=(sims_per_seed, n_blocks))
    hist_vol = Y_base.std(ddof=0)
    t = 0
    base_pred = model.predict(last_X).astype(np.float32)
    for j in range(n_blocks):
        block_len = min(block_length, horizon_months - t)
        for b in range(block_len):
            shocks = residuals[(block_starts[:, j] + b) % n_res]
            raw_step = base_pred + shocks
            step_vol = raw_step.std(ddof=0)
            if step_vol > 0:
                raw_step *= (hist_vol / step_vol)
            log_paths[:, t] = (log_paths[:, t-1] if t > 0 else 0) + raw_step
            t += 1
            if t >= horizon_months: break
    return np.exp(log_paths, dtype=np.float32)

# ---------- Forecast Stats ----------
def compute_forecast_stats_from_path(path: np.ndarray, start_capital: float, last_date: pd.Timestamp):
    if path is None or len(path) == 0:
        return {"CAGR": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}
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

# ---------- Plot Forecasts ----------
def plot_forecasts(port_rets, start_capital, central, rebalance_label):
    port_cum = (1 + port_rets).cumprod() * start_capital
    last_date = port_cum.index[-1]
    forecast_path = port_cum.iloc[-1] * (central / central[0])
    forecast_dates = pd.date_range(start=last_date, periods=len(central), freq="M")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, label="Portfolio Backtest")
    ax.plot([last_date, *forecast_dates], [port_cum.iloc[-1], *forecast_path],
            linewidth=2, label="Forecast")
    ax.set_title(f"Portfolio Forecast (Backtest + 1Y Snapshot Forecast)")
    ax.set_xlabel("Date"); ax.set_ylabel("Balance ($)")
    ax.legend()
    st.pyplot(fig)

# ---------- Streamlit App ----------
def main():
    st.title("Snapshot Portfolio Forecasting Tool with Median-Ending Medoid + UMCSENT")

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

            # Apply uniform one-month lag across ALL features
            Y = np.log(1 + port_rets.loc[df.index]).astype(np.float32)
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]

            # Debug print: confirm UMCSENT is included
            st.write("Features used:", list(X.columns))

            if "umcsent" not in X.columns:
                st.error("UMCSENT missing from feature matrix. Check FRED fetch/alignment.")
                return

            model, residuals, preds, X_full, Y_full, best_params, best_rmse = tune_and_fit_best_model(X, Y)
            block_length = int(best_params.get("block_length", 3))

            st.write("**Best Params:**")
            st.json(best_params)
            st.write("**OOS 12m RMSE:**", f"{best_rmse:.6f}")
            st.write("**Optimal Block Length:**", block_length)

            seed_medoids = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, seed in enumerate(range(ENSEMBLE_SEEDS)):
                rng = np.random.default_rng(GLOBAL_SEED + seed)
                sims = run_monte_carlo_paths(model, X_full, Y_full, residuals,
                                             SIMS_PER_SEED, rng, block_length, seed_id=seed)
                seed_medoids.append(find_median_ending_medoid(sims))
                progress = (i+1)/ENSEMBLE_SEEDS
                progress_bar.progress(progress)
                status_text.text(f"Running forecasts... {i+1}/{ENSEMBLE_SEEDS}")
            progress_bar.empty()
            final_medoid = find_median_ending_medoid(np.vstack(seed_medoids))

            stats = compute_forecast_stats_from_path(final_medoid, start_capital, port_rets.index[-1])
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
                for k,v in backtest_stats.items():
                    st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            with col2:
                st.markdown("**Forecast**")
                for k,v in stats.items():
                    st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")

            ending_value = float(final_medoid[-1]) * start_capital
            st.metric("Forecasted Portfolio Value", f"${ending_value:,.2f}")

            plot_forecasts(port_rets, start_capital, final_medoid, rebalance_label)

            final_X = X_full.iloc[[-1]]
            plot_feature_attributions(model, X_full, final_X)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()