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
ENSEMBLE_SEEDS = 10
SIMS_PER_SEED = 2000
FORECAST_YEARS = 1
BLOCK_LENGTH = 6

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
        raise ValueError("No price data returned.")
    if isinstance(data.columns, pd.MultiIndex):
        for field in ["Adj Close", "Close"]:
            if field in data.columns.get_level_values(0):
                close = data[field].copy(); break
        else:
            raise ValueError("No Close/Adj Close found.")
    else:
        colname = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = pd.DataFrame(data[colname]); close.columns = tickers
    close = close.ffill().dropna(how="all").astype(np.float32)
    first_valids = [close[col].first_valid_index() for col in close.columns]
    valid_starts = [d for d in first_valids if d is not None]
    non_na_start = max(valid_starts)
    return close.loc[non_na_start:]

# ---------- Portfolio ----------
def portfolio_returns_monthly(prices: pd.DataFrame, weights: np.ndarray, rebalance: str) -> pd.Series:
    rets = prices.pct_change().dropna(how="all").astype(np.float32)
    if rebalance == "N":
        vals = (1 + rets).cumprod()
        port_vals = vals.dot(weights); port_vals = port_vals / port_vals.iloc[0]
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
        return pd.Series(port_vals, index=rets.index).pct_change().fillna(0.0).astype(np.float32)

# ---------- Feature Builders ----------
def build_features(returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"ret": np.log(1 + returns).astype(np.float32)})
    df["mom_3m"] = returns.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_6m"] = returns.rolling(6).apply(lambda x: (1+x).prod()-1, raw=True)
    df["mom_12m"] = returns.rolling(12).apply(lambda x: (1+x).prod()-1, raw=True)
    df["dd_state"] = compute_current_drawdown(returns)
    df["vol_3m"] = returns.rolling(3).std()
    df["vol_6m"] = returns.rolling(6).std()
    df["vol_12m"] = returns.rolling(12).std()
    return df.dropna().astype(np.float32)

# ---------- Volatility Selection ----------
def choose_best_vol_indicator(Y: pd.DataFrame) -> str:
    candidates = ["vol_3m", "vol_6m", "vol_12m"]
    best_col, best_score = None, -np.inf
    realized_var = (Y["ret"]**2).dropna()
    for col in candidates:
        if col not in Y.columns: continue
        common = Y[[col]].join(realized_var, how="inner").dropna()
        if common.empty: continue
        r = np.corrcoef(common[col], common["ret"]**2)[0,1]
        score = r**2
        if score > best_score: best_score, best_col = score, col
    return best_col if best_col else "vol_12m"

# ---------- Forecast Model ----------
def run_forecast_model(X: pd.DataFrame, Y: pd.DataFrame):
    base = LGBMRegressor(
        n_estimators=5000, learning_rate=0.01, max_depth=3,
        reg_alpha=0.1, reg_lambda=0.1, subsample=0.8,
        colsample_bytree=0.8, n_jobs=1, random_state=GLOBAL_SEED
    )
    model = MultiOutputRegressor(base)
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
    return paths[np.argmax(scores)]

# ---------- Monte Carlo ----------
def run_monte_carlo_paths(model, X_base, Y_base, residuals, sims, rng, best_vol_col, k_neighbors=20):
    horizon_months = FORECAST_YEARS * 12
    log_paths = np.zeros((sims, horizon_months), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean", algorithm="ball_tree", n_jobs=1)
    nn.fit(X_base.values.astype(np.float32))
    _, precomputed_idxs = nn.kneighbors(X_base.values.astype(np.float32))
    precomputed_idxs = np.sort(precomputed_idxs, axis=1)
    last_X = np.repeat(X_base.iloc[[-1]].values.astype(np.float32), sims, axis=0)
    n_res = len(residuals); vol_idx = list(Y_base.columns).index(best_vol_col)
    n_blocks = math.ceil((horizon_months - 1) / BLOCK_LENGTH)
    hist_idx_seq = rng.integers(0, len(precomputed_idxs), size=(sims, n_blocks))
    nn_idx_seq   = rng.integers(0, k_neighbors, size=(sims, n_blocks))
    t = 1
    for j in range(n_blocks):
        if t >= horizon_months: break
        chosen_start = precomputed_idxs[hist_idx_seq[:, j], nn_idx_seq[:, j]]
        block_len = min(BLOCK_LENGTH, horizon_months - t)
        for b in range(block_len):
            base_preds = model.predict(last_X).astype(np.float32)
            shocks = residuals[(chosen_start + b) % n_res]
            pred_vol = base_preds[:, vol_idx]
            hist_vol = Y_base.iloc[(chosen_start + b) % n_res, vol_idx]
            scaling = (pred_vol / hist_vol).clip(0.1, 10.0)
            log_paths[:, t] = log_paths[:, t-1] + base_preds[:, 0] + shocks[:, 0] * scaling
            next_indicators = base_preds[:, 1:] + shocks[:, 1:]
            last_X = np.column_stack([base_preds[:, 0], next_indicators])
            t += 1
            if t >= horizon_months: break
    return np.exp(log_paths, dtype=np.float32)

# ---------- Forecast Stats ----------
def compute_forecast_stats_from_path(path, start_capital, last_date):
    if path is None or len(path) == 0:
        return {"CAGR": np.nan,"Volatility": np.nan,"Sharpe": np.nan,"Max Drawdown": np.nan,"R2": np.nan}
    norm_path = path / path[0]
    forecast_index = pd.date_range(start=last_date, periods=len(norm_path)+1, freq="M")
    price = pd.Series(norm_path, index=forecast_index[:-1]) * start_capital
    monthly = price.pct_change().dropna()
    return {
        "CAGR": annualized_return_monthly(monthly),
        "Volatility": annualized_vol_monthly(monthly),
        "Sharpe": annualized_sharpe_monthly(monthly),
        "Max Drawdown": max_drawdown_from_rets(monthly),
        "R2": np.nan
    }

# ---------- Streamlit App ----------
def main():
    st.title("Portfolio Forecasting Tool")
    tickers = st.text_input("Tickers", "VTI,AGG")
    weights_str = st.text_input("Weights", "0.6,0.4")
    start_cap = st.number_input("Starting Value", min_value=1000.0, value=10000.0)
    freq_map = {"M":"Monthly","Q":"Quarterly","S":"Semiannual","Y":"Yearly","N":"None"}
    rebalance_label = st.selectbox("Rebalance", list(freq_map.values()), index=0)
    rebalance_choice = [k for k,v in freq_map.items() if v==rebalance_label][0]
    run_is = st.radio("Run In-Sample Test?", ["No","Yes"], index=0)
    run_oos = st.radio("Run Out-of-Sample Test?", ["No","Yes"], index=1)

    if st.button("Run Forecast"):
        try:
            weights = to_weights([float(x) for x in weights_str.split(",")])
            tickers = [t.strip() for t in tickers.split(",")]
            prices = fetch_prices_monthly(tickers)
            port_rets = portfolio_returns_monthly(prices, weights, rebalance_choice)
            df = build_features(port_rets)
            Y = df; X = Y.shift(1).dropna(); Y = Y.loc[X.index]
            model, residuals, preds, X_full, Y_full = run_forecast_model(X, Y)
            best_vol_col = choose_best_vol_indicator(Y_full)

            # Ensemble forecast
            seed_medoids = []
            for seed in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(GLOBAL_SEED+seed)
                sims = run_monte_carlo_paths(model, X_full, Y_full, residuals, SIMS_PER_SEED, rng, best_vol_col)
                seed_medoids.append(find_medoid(sims))
            final_medoid = find_medoid(np.vstack(seed_medoids))
            stats = compute_forecast_stats_from_path(final_medoid, start_cap, port_rets.index[-1])

            # Backtest stats
            backtest_stats = {
                "CAGR": annualized_return_monthly(port_rets),
                "Volatility": annualized_vol_monthly(port_rets),
                "Sharpe": annualized_sharpe_monthly(port_rets),
                "Max Drawdown": max_drawdown_from_rets(port_rets)
            }

            st.subheader("Main Forecast")
            st.metric("CAGR", f"{stats['CAGR']:.2%}")
            st.metric("Volatility", f"{stats['Volatility']:.2%}")
            st.metric("Sharpe", f"{stats['Sharpe']:.2f}")
            st.metric("Max Drawdown", f"{stats['Max Drawdown']:.2%}")

            if run_oos=="Yes":
                cutoff = port_rets.index[-13]
                train_rets = port_rets[:cutoff]
                test_rets = port_rets[cutoff:]
                df_train = build_features(train_rets)
                Yt = df_train; Xt = Yt.shift(1).dropna(); Yt = Yt.loc[Xt.index]
                model, residuals, _, Xt_full, Yt_full = run_forecast_model(Xt, Yt)
                best_vol_col = choose_best_vol_indicator(Yt_full)
                medoids=[]
                for seed in range(ENSEMBLE_SEEDS):
                    rng=np.random.default_rng(GLOBAL_SEED+seed)
                    sims=run_monte_carlo_paths(model,Xt_full,Yt_full,residuals,SIMS_PER_SEED,rng,best_vol_col)
                    medoids.append(find_medoid(sims))
                oos_medoid=find_medoid(np.vstack(medoids))
                oos_stats=compute_forecast_stats_from_path(oos_medoid,start_cap,train_rets.index[-1])
                st.subheader("OOS Test vs Backtest")
                st.write("Backtest:", backtest_stats)
                st.write("OOS:", oos_stats)

            if run_is=="Yes":
                st.subheader("In-Sample Test")
                st.write("IS currently runs same ensemble logic on full sample.")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__=="__main__":
    main()

