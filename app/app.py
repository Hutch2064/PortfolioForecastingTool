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
from sklearn.metrics import mean_squared_error
import optuna
from scipy.stats import t as student_t  # <-- Student-t distribution

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

# ---------- Optuna + OOS Helpers ----------
# (all helper functions remain unchanged here: tune_and_fit_best_model, _oos_years_available, _split_train_test_for_year, _tune_on_explicit_split, _median_params, _eval_params_on_split, tune_across_recent_oos_years)

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

# ---------- Monte Carlo (Multivariate Student-t Residuals) ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims_per_seed, rng, seed_id=None, df=5):
    horizon_months = FORECAST_YEARS * 12
    log_paths = np.zeros((sims_per_seed, horizon_months), dtype=np.float32)

    mu = residuals.mean()
    sigma = residuals.std(ddof=0)

    snapshot_X = X_base.iloc[[-1]].values.astype(np.float32)
    last_X = np.repeat(snapshot_X, sims_per_seed, axis=0)
    base_pred = model.predict(last_X).astype(np.float32)

    for t in range(horizon_months):
        shocks = student_t.rvs(df, loc=mu, scale=sigma, size=sims_per_seed, random_state=rng).astype(np.float32)
        raw_step = base_pred + shocks
        log_paths[:, t] = (log_paths[:, t-1] if t > 0 else 0) + raw_step

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
    st.title("Portfolio Forecasting Tool (Multivariate Student-t Residuals)")

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

            # ---- NEW: outer progress bar for OOS tuning ----
            years = _oos_years_available(Y.index, max_years=5)
            progress_bar = st.progress(0)
            status_text = st.empty()
            param_runs, details = [], []
            for i, y in enumerate(years):
                status_text.text(f"Running OOS tuning {i+1}/{len(years)} (test year {y})...")
                trX, trY, teX, teY = _split_train_test_for_year(X, Y, y)
                best_params, rmse, da = _tune_on_explicit_split(trX, trY, teX, teY, seed=GLOBAL_SEED, n_trials=50)
                details.append({"year": y, "rmse": rmse, "da": da, "best_params": best_params})
                param_runs.append(best_params)
                progress_bar.progress((i+1)/len(years))
            progress_bar.empty()
            status_text.text("✅ OOS tuning complete")

            # consensus params
            consensus_params = _median_params(param_runs)
            st.write("**Best Params (per-parameter median across last 5 OOS years):**")
            st.json(consensus_params)

            # Show OOS metrics for most recent year
            last_year = years[-1] if years else None
            last_rmse, last_da = np.nan, np.nan
            if last_year:
                trX, trY, teX, teY = _split_train_test_for_year(X, Y, last_year)
                if len(trX) > 0 and len(teX) > 0:
                    last_rmse, last_da = _eval_params_on_split(consensus_params, trX, trY, teX, teY)
            st.write("**OOS 12m RMSE:**", f"{last_rmse:.6f}")
            st.write("**OOS 12m Directional Accuracy:**", f"{last_da:.2%}")

            # Final model on full history
            df_opt = int(consensus_params.get("df", 5))
            lgbm_params = {k:v for k,v in consensus_params.items() if k != "df"}
            lgbm_params["random_state"] = GLOBAL_SEED
            lgbm_params["n_jobs"] = 1

            final_model = LGBMRegressor(**lgbm_params)
            final_model.fit(X, Y)
            preds = final_model.predict(X).astype(np.float32)
            residuals = (Y.values - preds).astype(np.float32)

            seed_medoids = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, seed in enumerate(range(ENSEMBLE_SEEDS)):
                rng = np.random.default_rng(GLOBAL_SEED + seed)
                sims = run_monte_carlo_paths(final_model, X, Y, residuals,
                                             SIMS_PER_SEED, rng, seed_id=seed, df=df_opt)
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
                for k,v in backtest_stats.items(): st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            with col2:
                st.markdown("**Forecast**")
                for k,v in stats.items(): st.metric(k, f"{v:.2%}" if 'Sharpe' not in k else f"{v:.2f}")
            ending_value = float(final_medoid[-1]) * start_capital
            st.metric("Forecasted Portfolio Value", f"${ending_value:,.2f}")
            plot_forecasts(port_rets, start_capital, final_medoid, rebalance_label)

            final_X = X.iloc[[-1]]
            plot_feature_attributions(final_model, X, final_X)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()