import sys
import warnings
import random
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import shap
import math
import streamlit as st

warnings.filterwarnings("ignore")

# ---------- Global Seed Fix ----------
GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ---------- Config ----------
DEFAULT_START = "2000-01-01"
ENSEMBLE_SEEDS = 10        # number of seeds in the ensemble
SIMS_PER_SEED = 2000       # simulations per seed
FORECAST_YEARS = 1         # 12-month horizon
BLOCK_LENGTH = 6           # block length for residual bootstrap

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
        threads=False   # 👈 fix for Streamlit Cloud
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
    return df.dropna().astype(np.float32)

# ---------- Forecast Model ----------
def run_forecast_model(X: pd.DataFrame, Y: pd.Series):
    base_model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        random_state=GLOBAL_SEED
    )
    model = base_model.fit(X, Y)
    preds = model.predict(X).astype(np.float32)
    residuals = (Y.values - preds).astype(np.float32)
    return model, residuals, preds, X.astype(np.float32), Y.astype(np.float32)

# ---------- Medoid ----------
def find_medoid(paths: np.ndarray):
    median_series = np.median(paths, axis=0)
    diffs = np.abs(paths - median_series)
    closest = np.argmin(diffs, axis=0)
    scores = np.bincount(closest, minlength=paths.shape[0])
    best_idx = np.argmax(scores)
    return paths[best_idx]

# ---------- Monte Carlo (pure block bootstrap) ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, seed_id=None):
    horizon_months = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon_months), dtype=np.float32)

    # Compute historical drift (monthly log return from CAGR)
    hist_cagr = annualized_return_monthly(Y_base)
    mu_hist_monthly = (1 + hist_cagr) ** (1/12) - 1
    mu_hist_log = np.log(1 + mu_hist_monthly)

    n_res = len(residuals)
    n_blocks = math.ceil(horizon_months / BLOCK_LENGTH)

    # Pre-sample block starts
    block_starts = rng.integers(0, n_res - BLOCK_LENGTH, size=(sims_per_seed, n_blocks))

    last_X = np.repeat(X_base.iloc[[-1]].values.astype(np.float32), sims_per_seed, axis=0)

    t = 0
    for j in range(n_blocks):
        block_len = min(BLOCK_LENGTH, horizon_months - t)
        for b in range(block_len):
            base_preds = model.predict(last_X).astype(np.float32).reshape(-1, 1)

            # Drift normalization
            mean_pred_drift = np.mean(base_preds[:, 0])
            drift_scaled = base_preds[:, 0] * (mu_hist_log / mean_pred_drift) if mean_pred_drift != 0 else mu_hist_log

            # Raw residuals
            shocks = residuals[(block_starts[:, j] + b) % n_res]
            log_return_step = drift_scaled + shocks

            log_paths[:, t] = (log_paths[:, t-1] if t > 0 else 0) + log_return_step
            last_X = np.repeat(X_base.iloc[[-1]].values.astype(np.float32), sims_per_seed, axis=0)
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

# ---------- Plot Forecasts ----------
def plot_forecasts(port_rets, start_capital, central, rebalance_label):
    port_cum = (1 + port_rets).cumprod() * start_capital
    last_date = port_cum.index[-1]
    forecast_path = port_cum.iloc[-1] * (central / central[0])
    forecast_dates = pd.date_range(start=last_date, periods=len(central), freq="M")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(port_cum.index, port_cum.values, color="blue", label="Portfolio Backtest")
    ax.plot([last_date, *forecast_dates], [port_cum.iloc[-1], *forecast_path],
            color="red", linewidth=2, label="Forecast")
    ax.set_title(f"Portfolio Forecast (Backtest + 1Y Snapshot Forecast)")
    ax.set_xlabel("Date"); ax.set_ylabel("Balance ($)")
    ax.legend()
    st.pyplot(fig)

# ---------- Streamlit App ----------
def main():
    st.title("Snapshot Portfolio Forecasting Tool")

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

            Y = np.log(1 + port_rets.loc[df.index]).astype(np.float32)  # target = log returns
            X = df.shift(1).dropna()
            Y = Y.loc[X.index]

            model, residuals, preds, X_full, Y_full = run_forecast_model(X, Y)

            seed_medoids = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, seed in enumerate(range(ENSEMBLE_SEEDS)):
                rng = np.random.default_rng(GLOBAL_SEED + seed)
                sims = run_monte_carlo_paths(model, X_full, Y_full, residuals, SIMS_PER_SEED, rng, seed_id=seed)
                seed_medoids.append(find_medoid(sims))
                progress = (i+1)/ENSEMBLE_SEEDS
                progress_bar.progress(progress)
                status_text.text(f"Running forecasts... {i+1}/{ENSEMBLE_SEEDS}")

            progress_bar.empty()
            final_medoid = find_medoid(np.vstack(seed_medoids))

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
                for k,v in backtest_stats.items(): st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            with col2:
                st.markdown("**Forecast**")
                for k,v in stats.items(): st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")

            ending_value = float(final_medoid[-1]) * start_capital
            st.metric("Forecasted Portfolio Value", f"${ending_value:,.2f}")

            plot_forecasts(port_rets, start_capital, final_medoid, rebalance_label)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()