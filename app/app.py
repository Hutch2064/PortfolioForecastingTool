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
from sklearn.neighbors import NearestNeighbors
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
K_NEIGHBORS = 17           # fixed k for snapshot mode

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
    df = pd.DataFrame({"ret": np.log(1 + returns).astype(np.float32)})
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True).astype(np.float32)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True).astype(np.float32)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True).astype(np.float32)
    df["dd_state"] = compute_current_drawdown(returns)
    df["vol_3m"] = returns.rolling(3).std().astype(np.float32)
    df["vol_6m"] = returns.rolling(6).std().astype(np.float32)
    df["vol_12m"] = returns.rolling(12).std().astype(np.float32)
    return df.dropna().astype(np.float32)

# ---------- Volatility Selection ----------
def choose_best_vol_indicator(Y: pd.DataFrame) -> str:
    candidates = ["vol_3m", "vol_6m", "vol_12m"]
    best_col, best_score = None, -np.inf
    realized_var = (Y["ret"] ** 2).dropna()
    for col in candidates:
        if col not in Y.columns:
            continue
        common = Y[[col]].join(realized_var, how="inner").dropna()
        if common.empty:
            continue
        r = np.corrcoef(common[col], common["ret"] ** 2)[0,1]
        score = r**2
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_col else "vol_12m"

# ---------- Forecast Model ----------
def run_forecast_model(X: pd.DataFrame, Y: pd.DataFrame):
    base_model = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=3,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        random_state=GLOBAL_SEED,
        bagging_seed=GLOBAL_SEED,
        feature_fraction_seed=GLOBAL_SEED,
        data_random_seed=GLOBAL_SEED
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X, Y)
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

# ---------- Monte Carlo (Snapshot w/ Backtest Vol Scaling) ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, best_vol_col, seed_id=None):
    horizon_months = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon_months), dtype=np.float32)

    # Snapshot features (frozen at last point)
    snapshot_X = X_base.iloc[[-1]].values.astype(np.float32)

    # Predicted return (frozen)
    base_preds = model.predict(snapshot_X).astype(np.float32).flatten()
    pred_ret = base_preds[0]

    # Nearest neighbor conditioning (residual pool)
    nn_model = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="euclidean", algorithm="ball_tree", n_jobs=1)
    nn_model.fit(X_base.values.astype(np.float32))
    _, neighbor_idxs = nn_model.kneighbors(snapshot_X, n_neighbors=K_NEIGHBORS)
    neighbor_idxs = neighbor_idxs.flatten()

    n_res = len(residuals)

    # Historical unconditional volatility from backtest
    hist_vol = Y_base["ret"].std(ddof=0)

    # Draw block start indices for all paths
    n_blocks = math.ceil((horizon_months - 1) / BLOCK_LENGTH)
    start_indices = rng.choice(neighbor_idxs, size=(sims_per_seed, n_blocks))

    # Build residual draws in one shot
    residual_draws = np.zeros((sims_per_seed, horizon_months-1), dtype=np.float32)
    col = 0
    for j in range(n_blocks):
        block_len = min(BLOCK_LENGTH, horizon_months-1-col)
        idxs = (start_indices[:, j][:, None] + np.arange(block_len)[None, :]) % n_res
        residual_draws[:, col:col+block_len] = residuals[idxs, 0]
        col += block_len
        if col >= horizon_months-1: break

    # Scale residuals to unconditional backtest volatility
    res_vol = residual_draws.std(ddof=0)
    if res_vol > 0:
        shocks = residual_draws * (hist_vol / res_vol)
    else:
        shocks = residual_draws

    # Accumulate log paths
    log_returns = pred_ret + shocks
    log_paths[:, 1:] = np.cumsum(log_returns, axis=1)

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

# ---------- Improvement Calculator ----------
def percent_improvement(forecast, backtest, higher_is_better=True):
    if np.isnan(forecast) or np.isnan(backtest) or backtest == 0:
        return "N/A"
    if higher_is_better:
        improvement = (forecast - backtest) / abs(backtest)
    else:
        improvement = (backtest - forecast) / abs(backtest)
    sign = "+" if improvement >= 0 else ""
    return f"{sign}{improvement*100:.1f}%"

# ---------- Feature Attribution ----------
def plot_feature_attributions(model, X, final_X):
    np.random.seed(GLOBAL_SEED)
    explainer = shap.TreeExplainer(model.estimators_[0])
    shap_values_hist = explainer.shap_values(X)
    shap_mean_hist = np.abs(shap_values_hist).mean(axis=0)
    shap_values_fore = explainer.shap_values(final_X)
    shap_mean_fore = np.abs(shap_values_fore).mean(axis=0)
    features = X.columns
    x_pos = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - 0.2, shap_mean_hist, width=0.4, color="blue", label="Backtest Avg")
    ax.bar(x_pos + 0.2, shap_mean_fore, width=0.4, color="red", label="Forecast Avg")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Average |SHAP Value|")
    ax.set_title("Feature Contributions: Backtest vs Forecast")
    ax.legend()
    st.pyplot(fig)

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
    ax.set_title(f"Portfolio Forecast (Backtest + 1Y Forecast)")
    ax.set_xlabel("Date"); ax.set_ylabel("Balance ($)")
    ax.legend()
    st.pyplot(fig)

# ---------- Streamlit App ----------
def main():
    st.title("Portfolio Forecasting Tool")

    tickers = st.text_input("Tickers (comma-separated, e.g. VTI,AGG)", "VTI,AGG")
    weights_str = st.text_input("Weights (comma-separated, must sum > 0)", "0.6,0.4")
    start_capital = st.number_input("Starting Value ($)", min_value=1000.0, value=10000.0, step=1000.0)

    freq_map = {
        "M": "Monthly",
        "Q": "Quarterly",
        "S": "Semiannual",
        "Y": "Yearly",
        "N": "None"
    }
    rebalance_label = st.selectbox("Rebalance", list(freq_map.values()), index=0)
    rebalance_choice = [k for k,v in freq_map.items() if v == rebalance_label][0]

    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",") if t.strip()]

            prices = fetch_prices_monthly(tickers, start=DEFAULT_START)
            port_rets = portfolio_returns_monthly(prices, weights, rebalance_choice)

            df = build_features(port_rets)
            if df is None or df.empty:
                st.error("Feature engineering returned no data. Check tickers or date range.")
                return

            Y = df
            X = Y.shift(1).dropna()
            Y = Y.loc[X.index]

            if X.empty or Y.empty:
                st.error("Not enough samples after feature building.")
                return

            model, residuals, preds, X_full, Y_full = run_forecast_model(X, Y)
            best_vol_col = choose_best_vol_indicator(Y_full)

            seed_medoids = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, seed in enumerate(range(ENSEMBLE_SEEDS)):
                rng = np.random.default_rng(GLOBAL_SEED + seed)
                sims = run_monte_carlo_paths(model, X_full, Y_full, residuals, SIMS_PER_SEED, rng, best_vol_col, seed_id=seed)
                seed_medoids.append(find_medoid(sims))

                progress = (i + 1) / ENSEMBLE_SEEDS
                progress_bar.progress(progress)
                status_text.text(f"Running forecasts... {i+1}/{ENSEMBLE_SEEDS} seeds complete")

            progress_bar.empty()

            seed_medoids = np.vstack(seed_medoids)
            final_medoid = find_medoid(seed_medoids)

            stats = compute_forecast_stats_from_path(final_medoid, start_capital, port_rets.index[-1])

            backtest_stats = {
                "CAGR": annualized_return_monthly(port_rets),
                "Volatility": annualized_vol_monthly(port_rets),
                "Sharpe": annualized_sharpe_monthly(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets)
            }

            st.subheader("Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Backtest**")
                st.metric("CAGR", f"{backtest_stats['CAGR']:.2%}")
                st.metric("Volatility", f"{backtest_stats['Volatility']:.2%}")
                st.metric("Sharpe", f"{backtest_stats['Sharpe']:.2f}")
                st.metric("Max Drawdown", f"{backtest_stats['Max Drawdown']:.2%}")

            with col2:
                st.markdown("**Forecast**")
                st.metric("CAGR", f"{stats['CAGR']:.2%}")
                st.metric("Volatility", f"{stats['Volatility']:.2%}")
                st.metric("Sharpe", f"{stats['Sharpe']:.2f}")
                st.metric("Max Drawdown", f"{stats['Max Drawdown']:.2%}")

            with col3:
                st.markdown("**Comparison**")
                st.metric("CAGR", percent_improvement(stats["CAGR"], backtest_stats["CAGR"], higher_is_better=True))
                st.metric("Volatility", percent_improvement(stats["Volatility"], backtest_stats["Volatility"], higher_is_better=False))
                st.metric("Sharpe", percent_improvement(stats["Sharpe"], backtest_stats["Sharpe"], higher_is_better=True))
                st.metric("Max Drawdown", percent_improvement(stats["Max Drawdown"], backtest_stats["Max Drawdown"], higher_is_better=True))

            ending_value = float(final_medoid[-1]) * start_capital
            st.metric("Forecasted Portfolio Value", f"${ending_value:,.2f}")

            plot_forecasts(port_rets, start_capital, final_medoid, rebalance_label)

            final_X = X_full.iloc[[-1]]
            plot_feature_attributions(model, X_full, final_X)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

